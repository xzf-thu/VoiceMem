"""左脑组件 LeftBrain。

从 VoiceMem 上帝类里抽出来的**左脑那一整块**——认知图 slot 过滤 / 实体缩窄 /
时间类问题扩候选 / 向量排序 / 查询分类（含动态 slot 下钻）/ LLM 打 slot 标签 /
slot→entity 图层写入 / 子图激活记账与 checkpoint / schema 描述刷新 / 冷记忆归档，
以及围绕它们的左脑侧懒加载单例（repo / extractor / dynamic_slot_store /
graph_entity_store / subgraph_manager）。

参考 mem0 的组合模式：
  * **组件自持零件**——5 个左脑侧懒加载单例（repo / extractor /
    dynamic_slot_store / graph_entity_store / subgraph_manager）连同它们与宿主
    共享的缓存和锁，都在这个组件内部，engine 不再持有各自的 _get_*。
  * **依赖显式注入**——凡是需要用到"文本 embedding / LLM(JSON) / LLM(text) /
    可注入分类器 / 会话追踪器"这些**跨域或运行时**能力的地方，一律在 __init__
    里以 getter/函数引用注入（懒加载语义保持不变），组件内部通过 self._dep() 调用。

logic 一字不改：方法体原样搬运，只改"怎么拿依赖"。

brain.py 不 import engine（避免循环）——模块级辅助（_search_mode / _pool_mode /
_RESCUE_K 等）随本模块落地，engine 反向从这里 import。
"""

from __future__ import annotations

from voicemem.utils.common import space as _space

import os
import threading
from pathlib import Path
from typing import Any, Callable

from voicemem.leftbrain.cognitive_graph.slot_v2 import SLOT_RELATIONS
from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QueryClassification
from voicemem.leftbrain.local_memory_store import MemorySearchHit
from voicemem.rightbrain.brain import _is_en_text


# ── 模块级常量 / 辅助函数 ───────────────────────────────────────────────────────

# 靠词面/时间加分"救回"的记忆最多补几条（在 top_k 之外额外给，不占语义名额）
_RESCUE_K = 3


# ── 时间权重 ──────────────────────────────────────────────────────────────────
# 在这之前，检索排序**只看语义相似度**：三个月前的旧事和昨天说的话完全平权。
# （库里那套 heat 衰减写好了、每次命中也在累加，但它唯一的下游是一个全仓库
#  没人调用的归档函数，对排序零影响。）
#
# 两个刻意的限制，都是为了别把好东西弄坏：
#   · 只在**已经检索到的候选**里重排——不改召回，所以不会拿一条不相关的新记忆
#     去换一条不相关的旧记忆，只在相似度接近时向近期倾斜。
#   · 衰减有下限——"对坚果过敏""不吃辣"这类长期属性几个月才提一次，但每次都关键，
#     不该因为久远就被上周的琐事挤掉。所以最多打到 RECENCY_FLOOR，不会趋近 0。
#
# 用 observed_at（事情**发生**在哪天）而不是入库时间：补记一件上个月的事，
# 该按上个月算，不是按今天。
#: 半衰期（天）。设 0 或负数即关掉时间权重。
RECENCY_HALFLIFE_DAYS = float(os.environ.get("VOICEMEM_RECENCY_HALFLIFE_DAYS", "30"))
#: 衰减的下限，决定时间最多能有多大话语权。
#: 权重范围是 [FLOOR, 1]，所以**能被翻转的相似度差距最大是 1/FLOOR**：
#:   0.60 → 1.67x（相似度高出 67% 的也会被挤掉，太激进）
#:   0.75 → 1.33x（默认：打破近似平局，翻不动明显更相关的）
#:   0.90 → 1.11x（几乎只在同分时起作用）
#: 对记忆系统来说召回错东西比召回旧东西更糟，所以取保守的一档。
RECENCY_FLOOR = float(os.environ.get("VOICEMEM_RECENCY_FLOOR", "0.75"))


#: 只有问"最近怎么样"这类问题时才按时间加权。
#:
#: 一开始是**所有查询**都乘时间权重，实测出事：「我在哪读书」的正确答案
#: （"Jiaqi is an undergraduate in computer science at NUS"，半年前记下的）
#: 相似度 0.810 全场最高，乘上 0.752 之后掉到 0.609，被一条 0.773 的
#: 「用户希望用中文交流」挤到第六。
#: 原因是**长期属性的答案天然是旧的**：专业、过敏、老家、性格——它们没有更新的
#: 版本，罚旧等于罚对。而"最近在忙什么"这类问题才真的该偏向新的。
#: 所以按问句区分：问时效才加权，问属性不加权。
#:
#: 事实相互矛盾时（"在 TikTok 实习" / "在 NVIDIA 实习"）不靠这里解决——
#: 两条都带日期进 prompt，由回复模型按时间取舍，那条路更准也没有副作用。
_RECENCY_CUES = (
    "最近", "近来", "这阵子", "这几天", "今天", "昨天", "刚才", "刚刚",
    "现在", "目前", "当前", "这两天", "最新", "上周", "本周", "这周",
    "recent", "lately", "today", "yesterday", "just now", "right now",
    "currently", "these days", "this week", "last week", "latest",
)


def wants_recency(query: str) -> bool:
    """这个问题问的是「近况」还是「属性」。"""
    q = (query or "").lower()
    return any(c in q for c in _RECENCY_CUES)


def _recency_weight(observed_at: str) -> float:
    """事情发生得越近，权重越高；没有日期就不打折（当作与时间无关的属性）。"""
    if RECENCY_HALFLIFE_DAYS <= 0 or not observed_at:
        return 1.0
    from datetime import date, datetime
    try:
        d = datetime.fromisoformat(str(observed_at)[:10]).date()
    except (TypeError, ValueError):
        return 1.0                       # 日期解析不了就别猜，按不打折处理
    age = (date.today() - d).days
    if age <= 0:                         # 今天或（数据有误）未来
        return 1.0
    return RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * (0.5 ** (age / RECENCY_HALFLIFE_DAYS))


# 候选池构造模式（VOICEMEM_POOL_MODE）：
#   union  —— slot 池 ∪ 宏观关联 slot 池 ∪ 实体池 ∪ 一跳邻居池。
#   strict —— schema routing → entity narrowing → graph expansion：实体命中时取
#             slot 池 ∩ (实体 ∪ 一跳邻居)（交集太小则退回实体池，再空才退回 slot
#             池）；宏观 slot 扩散只在 slot 池小于 _STRICT_MACRO_MIN_POOL 时才开。
_POOL_MODE_ENV = "VOICEMEM_POOL_MODE"
_STRICT_MIN_INTERSECTION = 3      # 交集至少这么多条才用交集
_STRICT_MACRO_MIN_POOL = 30       # slot 池小于这个数才做宏观 slot 扩散


def _pool_mode() -> str:
    return os.environ.get(_POOL_MODE_ENV, "union").strip().lower()


def _search_mode(slot_ids: set, final_ids: set) -> str:
    if not slot_ids and not final_ids:
        return "fallback"
    if final_ids and final_ids < slot_ids:
        return "entity+slot-intersection"
    if final_ids == slot_ids:
        return "slot-only"
    return "entity+slot-union"


# ── LeftBrain 组件 ─────────────────────────────────────────────────────────────

#: 问题在问"你（助手）之前说过什么"时的词面信号。0 LLM——这个判断在投机预取
#: 那条路上，不能为它发一次网络请求。
_ASKS_ASSISTANT = (
    # 中文：句子里同时出现"你"和"说/告诉/推荐/建议/提到"就算
    "你说", "你告诉", "你跟我说", "你和我说", "你之前说", "你刚说", "你不是说",
    "你讲过", "你提过", "你提到", "你推荐", "你建议", "你答应", "你教我",
    "跟我说过", "告诉过我", "说过什么", "说了什么",
    "you said", "you told", "you mentioned", "you recommend", "you suggest",
    "did you say", "did you tell", "you talked about", "you promised",
)


def asks_about_assistant(query: str) -> bool:
    """这个问题是不是在问助手自己说过的话。"""
    q = (query or "").lower()
    return any(w in q or w in (query or "") for w in _ASKS_ASSISTANT)


#: 两条记忆的字符三元组 Jaccard 超过这个值就算"说的是同一件事"。
#: 0.30 是在一个 47 条的真实库上量出来的——所有 ≥0.20 的配对**全是**同一件事的
#: 不同措辞，没有一对是误判；而真正不同的两条落在 0.07 以下。实测样本：
#:   0.658  "今天感到很累，可能与明天下午的会议…" / "最近感到很累，可能与明天下午的会议…"
#:   0.317  "不太喜欢人多的场合，更愿意和一两个朋友安静聊聊" / "不太喜欢人多的场合，聚会待久了会累…"
#:   0.046  "最近开始规律健身…" / "熬了三个通宵，demo 终于跑通了…"        ← 真正不同
#: 取 0.30 而不是更低，是给别的库留余量：那里不同的记忆可能在字面上更像。
_DEDUPE_JACCARD = 0.30


def _trigrams(text: str) -> set[str]:
    t = "".join(ch for ch in str(text or "") if ch.strip())
    return {t[i:i + 3] for i in range(max(0, len(t) - 2))} or {t}


def _dedupe_near(hits: list) -> list:
    """按分数从高到低保留，跟已保留的任一条太像就跳过。

    抽取对同一件事会产出好几条措辞相近的记忆，分数也几乎一样。不去重的话
    top-5 里有两三条说的是同一件事，用户的感受是"它就记得这么点东西"。
    """
    kept, grams = [], []
    for h in hits:
        g = _trigrams(getattr(h, "text", ""))
        dup = False
        for g0 in grams:
            inter = len(g & g0)
            if inter and inter / len(g | g0) >= _DEDUPE_JACCARD:
                dup = True
                break
        if not dup:
            kept.append(h)
            grams.append(g)
    return kept


class LeftBrain:
    """自持零件 + 依赖显式注入的左脑组件。

    构造参数分两类：

    组件自持的**运行参数**（左脑侧路径/身份）::

        memory_root, user_id, base_url, cognitive_db

    显式**注入的跨域/运行时依赖**（全部以 getter/函数引用传入，懒加载语义不变）::

        embed              -> self._embed_text          文本 embedding（slot/图层写入）
        llm_json           -> self._llm_json            LLM(JSON)（打标签/子图/分类兜底）
        llm_text           -> self._llm_text            LLM(text)（schema 描述刷新）
        classifier         -> self._classifier          可注入查询分类器（Classify 用）
        tracker            -> self._get_session_tracker  跨左右脑会话追踪器（记账/checkpoint）

    左脑侧的 5 个懒加载单例（repo / extractor / dynamic_slot_store /
    graph_entity_store / subgraph_manager）连同与宿主共享的缓存/锁由本组件自持，
    见下方 self._get_repo() 等。
    """

    def __init__(
        self,
        *,
        memory_root: Path,
        user_id: str,
        base_url: str | None,
        cognitive_db: Path,
        embedder: Any,
        vector_store: Any,
        embed: Callable[[str], list[float]],
        llm_json: Callable[[str], str],
        llm_text: Callable[..., str],
        classifier: Any,
        tracker: Callable[[], Any],
        cache: dict[str, Any] | None = None,
        lock: Any = None,
    ) -> None:
        # ── 运行参数 ──
        self._memory_root = memory_root
        self._user_id = user_id
        self._base_url = base_url
        self._cognitive_db = cognitive_db
        # repo 构造需要的两个可注入底件（embedder / vector_store），与宿主对齐。
        self._embedder = embedder
        self._vector_store = vector_store

        # ── 注入的跨域/运行时依赖（getter/函数引用）──
        self._embed_text = embed
        self._llm_json = llm_json
        self._llm_text = llm_text
        self._classifier = classifier
        self._get_session_tracker = tracker

        # ── 组件自持的左脑零件缓存 ──
        # 允许宿主共享同一个 cache/lock（左脑侧懒加载单例与宿主 _get_* 落在同一
        # 字典，既存调用点/测试对宿主 _cache 的直接读写与本组件保持一致视图）。
        self._cache: dict[str, Any] = cache if cache is not None else {}
        self._lock = lock if lock is not None else threading.Lock()

    # ── 左脑懒加载单例 ──────────────────────────────────────────────────────────

    def _get_repo(self):
        with self._lock:
            if "repo" not in self._cache:
                from voicemem.leftbrain.cognitive_graph import CognitiveAnnotator, CognitiveAnnotatorConfig
                from voicemem.leftbrain.local_memory_store import OpenAILocalEmbedder, OpenAILocalEmbedderConfig
                from voicemem.leftbrain.memory_repository_v2 import LeftBrainMemoryRepositoryConfig, LeftBrainMemoryRepositoryV2
                annotator = CognitiveAnnotator(CognitiveAnnotatorConfig(base_url=self._base_url))
                embedder  = self._embedder or OpenAILocalEmbedder(OpenAILocalEmbedderConfig(base_url=self._base_url))
                cfg = LeftBrainMemoryRepositoryConfig(
                    json_path=_space.json_path(self._memory_root),
                    db_path=_space.db(self._memory_root),
                    cognitive_db_path=self._cognitive_db,
                    enable_cognitive_graph=True,
                )
                self._cache["repo"] = LeftBrainMemoryRepositoryV2(
                    embedder, config=cfg, cognitive_annotator=annotator,
                    vector_store=self._vector_store,
                )
        return self._cache["repo"]

    def _get_extractor(self):
        with self._lock:
            if "extractor" not in self._cache:
                from voicemem.leftbrain.extract_facts_openai import (
                    OpenAIAdditiveExtractorConfig,
                    OpenAIMem0V3AdditiveExtractor,
                )
                self._cache["extractor"] = OpenAIMem0V3AdditiveExtractor(
                    OpenAIAdditiveExtractorConfig(base_url=self._base_url)
                )
        return self._cache["extractor"]

    def _get_dynamic_slot_store(self):
        with self._lock:
            if "dynamic_slot_store" not in self._cache:
                from voicemem.leftbrain.slot_split import DynamicSlotStore
                self._cache["dynamic_slot_store"] = DynamicSlotStore(
                    _space.db(self._memory_root)
                )
        return self._cache["dynamic_slot_store"]

    def _get_dynamic_slots(self) -> list[tuple[str, str]]:
        """返回该用户已涌现的动态 slot [(name, description), ...]。"""
        try:
            return [(s.name, s.description)
                    for s in self._get_dynamic_slot_store().get_dynamic_slots(self._user_id)]
        except Exception:
            return []

    def _get_graph_entity_store(self):
        with self._lock:
            if "graph_entity_store" not in self._cache:
                from voicemem.leftbrain.slot_split import GraphEntityStore
                self._cache["graph_entity_store"] = GraphEntityStore(
                    _space.db(self._memory_root)
                )
        return self._cache["graph_entity_store"]

    def _get_subgraph_manager(self):
        graph_store = self._get_graph_entity_store()   # 在 lock 外先拿，避免嵌套 acquire
        dyn_store = self._get_dynamic_slot_store()
        with self._lock:
            if "subgraph_manager" not in self._cache:
                from voicemem.leftbrain.slot_split import SubgraphManager

                def _tag_new_slot(user_id: str, memory_id: str, slot_name: str) -> None:
                    cog_store = self._get_repo()._cognitive_store
                    if cog_store is not None and hasattr(cog_store, "upsert_memory_tags"):
                        cog_store.upsert_memory_tags(memory_id, user_id, [(slot_name, 0.9)])

                self._cache["subgraph_manager"] = SubgraphManager(
                    graph_store, dyn_store, llm_fn=self._llm_json, tag_fn=_tag_new_slot,
                )
        return self._cache["subgraph_manager"]

    # ── Step 1: slot 过滤 ──────────────────────────────────────────────────────

    def SearchCogGraph(
        self,
        slots: list[str],
        entities: list[str] | None = None,
        scene_filter: str | None = None,
        speaker_filter: str | None = None,
    ) -> tuple[set[str], QueryClassification]:
        """slot 过滤，返回该 slot 下所有记忆 ID。

        Parameters
        ----------
        slots:
            由语音模块提供的 slot 列表，如 ``["work"]``。
        entities:
            由语音模块提供的实体列表，如 ``["阿里"]``。可为空。
        scene_filter:
            可选场景过滤（audiomem），如 ``"office"``。
        speaker_filter:
            可选说话人过滤（audiomem），传入 person_id（如 ``"person_3a2f1b"``）。
            只返回该说话人说过的记忆。

        Returns
        -------
        (slot_mem_ids, classification)
            ``slot_mem_ids`` — 候选 memory_id 集合。
            ``classification`` — 封装了 slots 和 entities 的数据容器。
        """
        classification = QueryClassification(
            slots=slots,
            entities=entities or [],
        )
        store = self._get_repo()._cognitive_store

        # 所有 slot 的记忆池取并集：Classify() 既返回精确子 slot 也返回宽父 slot。
        slot_mem_ids: set[str] = set()
        if classification.slots and store is not None and hasattr(store, "memory_ids_for_slots_v2"):
            slot_mem_ids = set(store.memory_ids_for_slots_v2(self._user_id, classification.slots))

        # 场景过滤（audiomem）：取 scene:<tag> 标签的记忆与 slot 结果的交集。
        # "unknown" 不是场景，是"没测出来"——防御性地也在这里挡一道，免得别的
        # 调用方又把它当成有效值传进来。
        if scene_filter and scene_filter != "unknown" and slot_mem_ids:
            try:
                if store and hasattr(store, "memory_ids_for_slots_v2"):
                    scene_ids = set(
                        store.memory_ids_for_slots_v2(
                            self._user_id, [f"scene:{scene_filter}"]
                        )
                    )
                    narrowed = slot_mem_ids & scene_ids
                    if narrowed:
                        slot_mem_ids = narrowed
            except Exception:
                pass

        # 说话人过滤（audiomem）：取 speaker:<person_id> 标签与 slot 结果的交集。
        # slot_mem_ids 为空时（未指定 slot）直接把该说话人的记忆当作基础候选池。
        if speaker_filter:
            try:
                if store and hasattr(store, "memory_ids_for_slots_v2"):
                    spk_ids = set(
                        store.memory_ids_for_slots_v2(
                            self._user_id, [f"speaker:{speaker_filter}"]
                        )
                    )
                    if slot_mem_ids:
                        narrowed = slot_mem_ids & spk_ids
                        if narrowed:
                            slot_mem_ids = narrowed
                    elif spk_ids:
                        slot_mem_ids = spk_ids
            except Exception:
                pass

        return slot_mem_ids, classification

    # ── Step 2: 实体匹配（纯认知图，不碰向量） ──────────────────────────────────

    def SearchData(
        self,
        slot_mem_ids: set[str],
        classification: QueryClassification,
    ) -> set[str]:
        """在 slot_mem_ids 基础上用实体名称做交集缩窄，返回最终候选 ID 集合。

        纯认知图操作，不调用向量库，不需要原始 query。

        Parameters
        ----------
        slot_mem_ids:
            SearchCogGraph 返回的 slot 候选 ID 集合。
        classification:
            SearchCogGraph 返回的分类结果（使用其中的 entities 字段）。

        Returns
        -------
        set[str]
            最终候选 ID 集合。
            - 有实体 → slot ∪ entity
            - 无实体 → 直接返回 slot_mem_ids
        """
        final_ids, _activated_names = self._search_data_impl(slot_mem_ids, classification)
        return final_ids

    def _search_data_impl(
        self, slot_mem_ids: set[str], classification: QueryClassification,
    ) -> tuple[set[str], list[str]]:
        """SearchData() 的真正实现，多返回一个"左脑真正激活的实体名字列表"
        （含模糊匹配命中 + 一跳邻居扩散），供 Search() 内部传给右脑用。

        这跟 classification.entities（query 文本里的字面实体提及）不同——右脑
        依赖的是左脑检索管线真正确认/扩散出来的实体集合。SearchData() 公开方法
        只返回 memory id，维持原有 step-by-step 管线契约不变。
        """
        store = self._get_repo()._cognitive_store
        if not classification.entities or store is None:
            return set(slot_mem_ids), []

        entity_mids: set[str] = set()
        matched_entity_ids: set[str] = set()
        activated_names: list[str] = []
        if hasattr(store, "find_entities_by_name_fuzzy"):
            for ent_name in classification.entities:
                ents = store.find_entities_by_name_fuzzy(self._user_id, ent_name)
                if ents:
                    ids = [e.id for e in ents]
                    matched_entity_ids.update(ids)
                    activated_names.extend(e.name for e in ents)
                    mids = store.memory_ids_for_entities(ids)
                    entity_mids.update(mids)

        # 一跳邻居扩散：把直接匹配实体的一跳邻居（entity_edges）的记忆也并进候选池，
        # 一视同仁不加权，排序交给 Rank() 的向量相似度；邻居也计入 activated_names。
        if matched_entity_ids and hasattr(store, "neighbor_entity_ids"):
            neighbor_ids = store.neighbor_entity_ids(self._user_id, list(matched_entity_ids))
            if neighbor_ids:
                entity_mids.update(store.memory_ids_for_entities(neighbor_ids))
                for nid in neighbor_ids:
                    ne = store.get_entity(nid)
                    if ne:
                        activated_names.append(ne.name)

        if not entity_mids:
            return set(slot_mem_ids), activated_names

        if slot_mem_ids:
            if _pool_mode() == "strict":
                # entity narrowing：实体池对 slot 池做交集缩窄；交集太小就信实体不信 slot。
                inter = entity_mids & slot_mem_ids
                if len(inter) >= _STRICT_MIN_INTERSECTION:
                    return inter, activated_names
                return entity_mids, activated_names
            return entity_mids | slot_mem_ids, activated_names
        return entity_mids, activated_names

    # ── Step 2.5: 时间类问题扩候选 ────────────────────────────────────────────

    def _widen_for_time_question(self, query: str, final_ids: set[str]) -> set[str]:
        """问"多久 / 什么时候"时，把库里含时长或日期表达的记忆并进候选池。

        entity 和 slot 都按语义内容建索引，抓不住时间表达。这里按问题类型补一次
        正则扫库，把含时长/日期表达的记忆并进候选。final_ids 为空时走全库兜底，不扩。
        """
        if not final_ids:
            return final_ids
        from voicemem.leftbrain.local_memory_store import time_question_kind

        kind = time_question_kind(query)
        if kind is None:
            return final_ids
        store = self._get_repo()._vector_store
        if not hasattr(store, "memory_ids_with_time_expr"):
            return final_ids
        extra = store.memory_ids_with_time_expr(self._user_id, kind=kind)
        return (final_ids | extra) if extra else final_ids

    # ── Step 3: 向量排序 ──────────────────────────────────────────────────────

    def Rank(
        self,
        query: str,
        candidate_ids: set[str],
        top_k: int = 5,
        speaker_filter: str | None = None,
    ) -> list[MemorySearchHit]:
        """在 candidate_ids 范围内做向量相似度排序，返回 top-N 记忆。"""
        fetch_k = max(top_k * 3, 20)   # 全库兜底时多拉候选
        repo = self._get_repo()
        # 助手说过的话默认不召回；只有问题本身在问"你之前说过什么"才放进来。
        want_assistant = asks_about_assistant(query)

        if candidate_ids:
            # 名额选择交给存储层：top_k 个按纯余弦发，额外补最多 _RESCUE_K 条被
            # 词面/时间加分救回来的（必须在完整候选集上做，避免二次截断丢分）。
            hits = repo._vector_store.search(
                query,
                user_id=self._user_id,
                top_k=top_k * 3,          # 多拉一些，去重之后再截（见 _dedupe_near）
                rescue_k=_RESCUE_K,
                memory_id_filter=candidate_ids,
                include_assistant=want_assistant,
            )
            # 不足时从全库补齐——但按人过滤时不能这样做（会把其他人的记忆混进来），
            # 这种情况下宁可结果数少于 top_k。
            if len(hits) < top_k and not speaker_filter:
                seen = {h.memory_id for h in hits}
                for h in repo.search(query, user_id=self._user_id, top_k=fetch_k,
                                     include_assistant=want_assistant):
                    if h.memory_id not in seen:
                        hits.append(h)
                        seen.add(h.memory_id)
                        if len(hits) >= top_k:
                            break
        else:
            hits = repo.search(query, user_id=self._user_id, top_k=fetch_k,
                               include_assistant=want_assistant)[:top_k]

        # 去掉近重复再截断。抽取会对同一件事产出好几条措辞相近的记忆
        # （"今天感到很累，可能与明天下午的会议有关" / "最近感到很累，可能与明天
        # 下午的会议有关"），它们分数也几乎一样，于是 top-5 里有两三条说的是同一
        # 件事——用户的感受是"它就记得这么点东西"。
        # 去重之后截断。问「近况」的才按时间加权重排（见 wants_recency 上面那段），
        # 问「属性」的保持纯相似度——长期属性的正确答案天然是旧的，罚旧等于罚对。
        # 重排只在候选池内部发生，召回不变。
        deduped = _dedupe_near(hits)
        if wants_recency(query):
            deduped = sorted(deduped,
                             key=lambda h: h.score * _recency_weight(h.observed_at),
                             reverse=True)
        final_hits = deduped[:top_k]
        # 记忆生命周期：检索命中增加热度，读取时按 last_hit_at 指数衰减、低热度归档。
        cog_store = repo._cognitive_store
        if cog_store is not None and hasattr(cog_store, "record_memory_hits"):
            try:
                cog_store.record_memory_hits([h.memory_id for h in final_hits])
            except Exception as e:
                print(f"[MemoryHeat] 记录失败: {e}")
        return final_hits

    # ── v5：LLM 打标签（替代 embedding 相似度） ───────────────────────────────

    # base-7 slot 的中文别名——用于构造 "english / 中文" 短锚点文本算 embedding。
    # 短标签对短标签的余弦相似度才够高，能让翻译变体折叠回同一个 slot。
    _BASE_SLOT_ALIASES: dict[str, str] = {
        "work": "工作", "finance": "财务", "relationships": "关系",
        "health": "健康", "goals": "目标", "daily_life": "日常生活",
        "knowledge": "知识",
    }

    def _get_slot_base_embeddings(self) -> dict[str, list[float]]:
        """base-7 slot 的 embedding，缓存一次。key 用字面枚举值（"relationships"），
        不能直接 str(枚举成员)——SlotV2.RELATIONSHIPS 的 __str__ 是 "SlotV2.RELATIONSHIPS"
        不是 "relationships"，会导致折叠命中后写回一个不存在的 slot 名字。"""
        with self._lock:
            if "slot_base_embeddings" not in self._cache:
                self._cache["slot_base_embeddings"] = {
                    value: self._embed_text(f"{value} / {alias}")
                    for value, alias in self._BASE_SLOT_ALIASES.items()
                }
        return self._cache["slot_base_embeddings"]

    def _get_slot_dyn_embeddings(self, dynamic: list[tuple[str, str]]) -> dict[str, list[float]]:
        """已涌现动态 slot 的 embedding，增量缓存（新 slot 出现才补算）。"""
        with self._lock:
            cache = self._cache.setdefault("slot_dyn_embeddings", {})
        for name, desc in dynamic:
            if name not in cache:
                cache[name] = self._embed_text(f"{name}：{desc}" if desc else name)
        return cache

    def _normalize_slot_name(
        self, candidate: str, known_all: set[str], dynamic: list[tuple[str, str]],
        threshold: float = 0.65,
    ) -> str:
        """精确匹配失败时按语义相似度把候选 slot 折叠回最接近的已知 slot（避免翻译/
        措辞漂移把同一类别拆成两份），只有真正找不到相近的才当作全新 slot。
        """
        if candidate in known_all:
            return candidate

        from voicemem.leftbrain.slot_split.split_manager import cosine_sim
        cand_emb = self._embed_text(candidate)

        best_name, best_sim = None, -1.0
        for name, emb in self._get_slot_base_embeddings().items():
            sim = cosine_sim(cand_emb, emb)
            if sim > best_sim:
                best_sim, best_name = sim, name
        for name, emb in self._get_slot_dyn_embeddings(dynamic).items():
            sim = cosine_sim(cand_emb, emb)
            if sim > best_sim:
                best_sim, best_name = sim, name

        return best_name if best_name is not None and best_sim >= threshold else candidate

    def _llm_tag_memories(self, text: str, memory_ids: list[str]) -> list[str]:
        """用 LLM 给这批记忆打 slot 标签，只能从已知 slot（固定 + 已建好的动态
        slot）里选 1-2 个，不允许 LLM 自造新类别（新 slot 只能由 SubgraphManager
        的共现子图判定产生）。返回实际打上的 slot 名称列表。
        """
        import json as _json
        from voicemem.leftbrain.cognitive_graph.slot_v2 import ALL_SLOT_V2_VALUES, SLOT_V2_DESCRIPTIONS

        dynamic = self._get_dynamic_slots()  # [(name, description), ...]
        dyn_names = {n for n, _ in dynamic}
        known_all = set(ALL_SLOT_V2_VALUES) | dyn_names

        # 构建 slot 列表描述
        slot_lines = [f"- {s}: {SLOT_V2_DESCRIPTIONS[s][:60]}" for s in ALL_SLOT_V2_VALUES]
        if dynamic:
            slot_lines += [f"- {n}: {d}" for n, d in dynamic]
        slot_desc = "\n".join(slot_lines)

        prompt = (
            f"用户说了这句话：\n「{text}」\n\n"
            f"请从下面列表里选最贴近的 1-2 个生活领域（必须选列表里已有的，"
            f"选最接近的即可，不要自创新类别）：\n{slot_desc}\n\n"
            '只输出 JSON：{"slots": ["类别1", "类别2"]}'
        )
        raw = self._llm_json(prompt)
        if not raw:
            return []

        try:
            slots = _json.loads(raw).get("slots", [])
        except Exception:
            return []

        slots = [s.strip() for s in slots if s.strip()][:2]
        if not slots:
            return []

        # 精确匹配失败的候选先按语义相似度折叠回已知 slot；折叠后仍不在已知列表
        # 里的（LLM 自造了新名字）直接丢弃——新 slot 的创造完全交给子图机制。
        slots = [self._normalize_slot_name(s, known_all, dynamic) for s in slots]
        slots = [s for s in slots if s in known_all]
        slots = list(dict.fromkeys(slots))  # 去重保序
        if not slots:
            return []

        cog_store = self._get_repo()._cognitive_store

        # 覆盖写入标签（覆盖 embedding 打的旧标签）
        if cog_store and hasattr(cog_store, "upsert_memory_tags"):
            for mid in memory_ids:
                cog_store.upsert_memory_tags(
                    mid, self._user_id, [(s, 0.95) for s in slots]
                )
        return slots

    # ── 查询分类（含动态 slot） ────────────────────────────────────────────────

    def Classify(self, query: str) -> QueryClassification:
        """LLM 分类 query → slots + entities，分层进行：
        1. 先只在 base-7 里选（不摊平全部动态 slot，避免列表越滚越长）。
        2. 每选中一个 slot，就往它的子 slot（子图机制分裂出来的）再钻一层，
           有比当前层更精确的子 slot 就往下钻，没有就停。
        3. 钻到的子 slot 追加进结果，父 slot 保留不丢——父 slot 兜住召回，
           子 slot 提供指向性，检索端对多 slot 取并集。
        entities 只在第 1 步提取一次。
        """
        from voicemem.leftbrain.cognitive_graph.query_slot_classifier import (
            QuerySlotClassifier, SlotClassifierConfig, QueryClassification,
        )
        # 可注入分类器（默认内置 LLM 版）。和 embedder 对称：传本地实现即切成
        # 本地模型，不碰 LLM/网络——这一步（抽 slot + entity）从此可 OpenAI 可本地。
        clf = self._classifier or QuerySlotClassifier(SlotClassifierConfig(base_url=self._base_url))
        top = clf.classify(query)

        dyn_store = self._get_dynamic_slot_store()
        final_slots = []

        def _add(name: str) -> None:
            if name not in final_slots:
                final_slots.append(name)

        # 子 slot 下钻需要分类器支持 classify_child（本地版没有就整体跳过）。
        _emergence_on = hasattr(clf, "classify_child")
        for slot in top.slots:
            _add(slot)
            if not _emergence_on:
                continue
            current = slot
            seen = {current}
            while True:
                children = dyn_store.get_children(self._user_id, current)
                children = [c for c in children if c.name not in seen]
                if not children:
                    break
                choice = clf.classify_child(
                    query, current, [(c.name, c.description) for c in children]
                )
                if choice is None:
                    break
                current = choice
                seen.add(current)
                _add(current)

        return QueryClassification(slots=final_slots, entities=top.entities)

    _SUBGRAPH_POOL_NS = "subgraph_pool"

    def _record_subgraph_activation(self, hits: list) -> None:
        """检索结果记账：把命中 memory 对应的 graph_entity 记进 session 的子图
        候选池 + 查询激活历史（供簇涌现的密度公式用）。便宜，无 LLM 调用。
        由 Search() 本体每次真实检索后自动执行。
        """
        memory_ids = {h.memory_id for h in hits}
        if not memory_ids:
            return
        tracker = self._get_session_tracker()
        for mid in memory_ids:
            tracker.touch(self._user_id, self._SUBGRAPH_POOL_NS, mid)
        try:
            graph_store = self._get_graph_entity_store()
            activated: set[str] = set()
            for mid in memory_ids:
                activated.update(e.id for e in graph_store.get_entities_for_memory(self._user_id, mid))
            if activated:
                import uuid as _uuid
                session_id = self._get_session_tracker().get_current_session(self._user_id)
                graph_store.record_query_activation(
                    self._user_id, _uuid.uuid4().hex, list(activated), session_id=session_id,
                )
        except Exception as e:
            print(f"[QueryActivation] 记录失败: {e}")

    def RunSubgraphCheckpoint(self) -> dict:
        """把攒下的 memory_id 名单整个取出（并清空），做一次真正的建图→判断——
        这是子图判定"贵"的那一步，真实产品里应在每个 session 结束时调一次。
        """
        tracker = self._get_session_tracker()
        memory_ids = set(tracker.pop_touched(self._user_id, self._SUBGRAPH_POOL_NS))
        if not memory_ids:
            return {"status": "no_memories"}

        cog_store = self._get_repo()._cognitive_store

        def _mem_lookup(mid: str) -> str | None:
            if cog_store is None:
                return None
            rec = cog_store.get_memory_record(mid)
            return rec.content if rec else None

        session_id = self._get_session_tracker().get_current_session(self._user_id)
        return self._get_subgraph_manager().run_for_retrieved_pool(
            self._user_id, memory_ids, memory_content_lookup=_mem_lookup, session_id=session_id,
        )

    def ArchiveColdMemories(
        self, *, min_age_days: float = 30.0, heat_threshold: float | None = None,
    ) -> dict:
        """记忆生命周期的归档一步：扫衰减后热度低于阈值、且存在够久的记忆，
        调 mem0 的 expiration_date 归档（mem0 的 search()/get_all() 会自动隐藏
        过期记忆）。判定在 list_archivable_memories，这里只负责执行；显式调用
        的批处理操作，不在每次 Ingest()/Search() 里自动跑。
        """
        cog_store = self._get_repo()._cognitive_store
        if cog_store is None or not hasattr(cog_store, "list_archivable_memories"):
            return {"status": "no_cognitive_store", "archived": []}

        from voicemem.leftbrain.cognitive_graph.store import ARCHIVE_HEAT_THRESHOLD
        threshold = ARCHIVE_HEAT_THRESHOLD if heat_threshold is None else heat_threshold

        candidate_ids = cog_store.list_archivable_memories(
            self._user_id, min_age_days=min_age_days, heat_threshold=threshold,
        )
        if not candidate_ids:
            return {"status": "nothing_to_archive", "archived": []}

        vector_store = self._get_repo()._vector_store
        archived: list[str] = []
        for mid in candidate_ids:
            try:
                if hasattr(vector_store, "archive_memory") and vector_store.archive_memory(mid):
                    archived.append(mid)
            except Exception as e:
                print(f"[Archive] {mid} 归档失败: {e}")
        return {"status": "archived" if archived else "archive_failed", "archived": archived}

    # ── 左脑检索总入口（Search 里的左脑那半）────────────────────────────────────

    def search(
        self,
        query: str,
        slots: list[str] | None,
        entities: list[str] | None,
        scene_filter: str | None,
        speaker_filter: str | None,
    ) -> dict:
        """左脑检索段：SearchCogGraph → SearchData → 时间扩候选 → 相关槽摘要。

        不含向量排序（Rank）——Rank 与右脑并发跑的编排结构留在 engine.Search，
        engine 拿到这里返回的 final_ids/activated_names 后并发 self._left.rank ‖
        self._right.search。返回一个字典，含 engine 组装 SearchResult 所需的左脑
        字段与各步时间戳。logic 与原 Search() 左脑那几步一字不改。
        """
        import time

        # ① slot 过滤
        t0 = time.time()
        slot_mem_ids, classification = self.SearchCogGraph(
            slots or [], entities, scene_filter=scene_filter, speaker_filter=speaker_filter,
        )
        t1 = time.time()

        # ② 实体缩窄——先于右脑跑完。右脑依赖左脑"已激活"的实体集合，必须等
        # _search_data_impl() 产出真正在左脑图里查到/扩散出来的那批实体。
        final_ids, activated_names = self._search_data_impl(slot_mem_ids, classification)
        final_ids = self._widen_for_time_question(query, final_ids)
        t2 = time.time()

        # 相关槽摘要：优先用从数据共现自动学出来的宏观连接；学出来的关联不够
        # （冷启动）时退回静态表兜底（动态 slot 静态表查不到，关联回其父 slot）。
        primary = classification.primary_slot()
        related_summaries: dict[str, str] = {}
        if primary:
            store = self._get_repo()._cognitive_store
            # 路由到的全部 slot + 主 slot 的 ≤3 个强连接 slot，各附一句 schema 描述。
            wanted: list[str] = list(classification.slots or [primary])
            related_slots: list[str] = []
            if store is not None and hasattr(store, "get_macro_related_slots"):
                related_slots = store.get_macro_related_slots(self._user_id, primary)
            if not related_slots:
                if primary in SLOT_RELATIONS:
                    related_slots = SLOT_RELATIONS[primary]
                else:
                    related_slots = self._get_dynamic_slot_store().get_parent_slots(self._user_id, primary)
            for r_ in related_slots[:3]:
                if r_ not in wanted:
                    wanted.append(r_)
            if wanted and store is not None and hasattr(store, "get_slot_summaries"):
                got = store.get_slot_summaries(self._user_id, wanted)
                related_summaries = {s_: got[s_] for s_ in wanted if got.get(s_)}

        return {
            "slot_mem_ids": slot_mem_ids,
            "final_ids": final_ids,
            "activated_names": activated_names,
            "classification": classification,
            "related_summaries": related_summaries,
            "t0": t0, "t1": t1, "t2": t2,
        }

    def rank(
        self, query: str, candidate_ids: set[str], top_k: int = 5,
        speaker_filter: str | None = None,
    ) -> list[MemorySearchHit]:
        """向量排序（Search 编排里与右脑并发的那半），转发到 Rank。"""
        return self.Rank(query, candidate_ids, top_k, speaker_filter=speaker_filter)

    def record_activation(self, hits: list) -> None:
        """检索结果记账，转发到 _record_subgraph_activation。"""
        return self._record_subgraph_activation(hits)

    # ── 左脑写入 ────────────────────────────────────────────────────────────────

    def ingest_facts(self, vi, *, registry, session_id, extra_metadata):
        """左脑事实抽取 + 入库：ingest_voice_input（合成消息→抽取→写库）。
        registry 是音频侧的声纹姓名映射，由 engine 编排时注入（跨域，不自持）。"""
        from voicemem.utils.common.voice_input import ingest_voice_input
        return ingest_voice_input(
            vi, self._user_id,
            registry=registry,
            repo=self._get_repo(),
            extractor=self._get_extractor(),
            session_id=session_id,
            extra_metadata=extra_metadata,
        )

    def write(self, result, text) -> None:
        """左脑写入段：LLM 打 slot 标签 + slot→entity 图层写入。"""
        if not result.memory_ids:
            return
        # LLM 打标签（覆盖 embedding 标签）
        try:
            llm_slots = self._llm_tag_memories(text, result.memory_ids)
            primary_slot = llm_slots[0] if llm_slots else None
        except Exception as e:
            print(f"[v5] LLM 打标签失败: {e}", flush=True)
            primary_slot = None
            llm_slots = []

        # 语义簇宏观连接：这条记忆同时打了 2 个以上 slot 标签，说明这几个 slot
        # 之间存在真实关联，从数据共现自动学，不是人工写死的关系表。
        if len(llm_slots) >= 2:
            try:
                self._get_repo()._cognitive_store.record_slot_cooccurrence(
                    self._user_id, llm_slots
                )
            except Exception as e:
                print(f"[SlotMacro] 共现记录失败: {e}", flush=True)

        # 左脑 slot→entity 图层：把这条记忆挂到对应slot下的entity节点
        # （entity 名字复用 CognitiveAnnotator 已经抽取好的实体，不额外调LLM）
        try:
            cog_store = self._get_repo()._cognitive_store
            graph_store = self._get_graph_entity_store()
            if cog_store is not None and primary_slot:
                for mid in result.memory_ids:
                    for eid in cog_store.entity_ids_for_memory(mid):
                        ent = cog_store.get_entity(eid)
                        if ent is None:
                            continue
                        ent_emb = self._embed_text(ent.name)
                        g_ent, _created = graph_store.get_or_create_entity_semantic(
                            self._user_id, primary_slot, ent.name, ent_emb,
                        )
                        graph_store.link_memory(g_ent.id, self._user_id, mid)
        except Exception as e:
            print(f"[GraphEntity] 左脑图层写入失败: {e}")

    # ── schema 描述刷新 ─────────────────────────────────────────────────────────

    _SCHEMA_DESC_MIN_NEW = 1      # slot 新增 ≥N 条记忆才重写描述
    _SCHEMA_DESC_MAX_FACTS = 80   # 摘要输入上限（最近的在前）

    def refresh_schema(self) -> None:
        """转发到 _refresh_schema_descriptions（对外命名）。"""
        return self._refresh_schema_descriptions()

    def _refresh_schema_descriptions(self) -> None:
        """给记忆数有变化的 slot 重写一句综合描述，写入 cognitive store 的 slot_summaries。
        描述语言跟随记忆语言；带该领域最近一条记忆的日期，避免 temporal 题被无日期的
        概括带偏。"""
        repo = self._get_repo()
        cog = repo._cognitive_store
        if cog is None or not hasattr(cog, "memory_ids_for_slots_v2"):
            return
        entries = {}
        try:
            for e in repo._vector_store.list_entries(user_id=self._user_id):
                entries[e["id"]] = e
        except Exception:
            entries = {}
        slots = list(SLOT_RELATIONS.keys())
        try:
            slots += [d.name for d in self._get_dynamic_slot_store().get_dynamic_slots(self._user_id)]
        except Exception:
            pass
        for slot in dict.fromkeys(slots):
            try:
                mids = cog.memory_ids_for_slots_v2(self._user_id, [slot])
                n = len(mids)
                if n < 3:
                    continue
                last = cog.get_slot_summary_mem_count(self._user_id, slot) if hasattr(cog, "get_slot_summary_mem_count") else 0
                if n - last < self._SCHEMA_DESC_MIN_NEW:
                    continue
                facts = []
                for mid in mids:
                    e = entries.get(mid)
                    if e is not None and e["text"]:
                        facts.append((e["date"], e["text"]))
                    else:
                        rec = cog.get_memory_record(mid) if hasattr(cog, "get_memory_record") else None
                        if rec and rec.content:
                            facts.append(("", rec.content))
                if len(facts) < 3:
                    continue
                facts.sort(key=lambda t: t[0], reverse=True)
                facts = facts[: self._SCHEMA_DESC_MAX_FACTS]
                latest = next((d for d, _ in facts if d), "")
                sample = "\n".join(f"- {('[' + d + '] ') if d else ''}{t}" for d, t in facts)
                en = _is_en_text(" ".join(t for _, t in facts[:10]))
                prompt = (
                    f"Below are memory facts about a user, all under the life domain '{slot}'.\n"
                    "Write ONE concise sentence (max 40 words) summarizing the overall picture in this domain: "
                    "the main people, ongoing situations, and how things changed over time. Plain and factual, "
                    "no fluff. Mention the most recent date if relevant. Output only the sentence.\n\n"
                    if en else
                    f"以下是用户在「{slot}」这个维度上的记忆事实。\n"
                    "请用一句话（40字以内）概括这个维度的整体情况：主要人物、正在进行的事、随时间的变化。"
                    "平实、有依据、不抒情；如相关请带上最近的日期。只输出这一句。\n\n"
                ) + sample
                text = (self._llm_text(prompt) or "").strip()
                if not text:
                    continue
                if latest and latest not in text:
                    text = f"{text} (latest: {latest})" if en else f"{text}（最近：{latest}）"
                cog.upsert_slot_summary(self._user_id, slot, text, n)
            except Exception as e:
                print(f"[SchemaDesc] {slot} 失败: {e}")

    # ── 用户名提取 ──────────────────────────────────────────────────────────────

    def _get_user_name(self) -> str | None:
        """从左脑记忆中提取用户名字，命中后缓存。"""
        with self._lock:
            if "user_name" in self._cache:
                return self._cache["user_name"]

        import re, sqlite3 as _sql
        name: str | None = None
        try:
            db_path = _space.db(self._memory_root)
            if db_path.exists():
                conn = _sql.connect(db_path)
                rows = conn.execute(
                    "SELECT text FROM memories WHERE user_id=? LIMIT 300",
                    (self._user_id,),
                ).fetchall()
                conn.close()
                patterns = [
                    r"我叫([^\s，。！？,.]{1,6})",
                    r"叫我([^\s，。！？,.]{1,6})",
                    r"我的名字[叫是]([^\s，。！？,.]{1,6})",
                    r"[Mm]y name is ([A-Za-z]{2,15})",
                    r"[Ii]'?m ([A-Z][a-z]{1,14})",
                ]
                for (text,) in rows:
                    for pat in patterns:
                        m = re.search(pat, text)
                        if m:
                            name = m.group(1).strip()
                            break
                    if name:
                        break
        except Exception:
            pass

        with self._lock:
            self._cache["user_name"] = name
        return name


__all__ = ["LeftBrain", "_search_mode", "_pool_mode"]
