"""右脑组件 RightBrain。

从 VoiceMem 上帝类里抽出来的**右脑那一整块**——heartnote 情感记忆写入、
内心 OS 生成、右脑图层（情绪/关系/性格特质）写入、右脑检索（情境指导 rb_directive
的结构化 top-N），以及围绕它们的 LLM 清洁（重复删除 / 矛盾 supersede）。

参考 mem0 的组合模式：
  * **组件自持零件**——3 个右脑侧懒加载单例（rb_repo / rb_graph_store /
    attribution_manager）连同它们与宿主共享的缓存和锁，都在这个组件内部，
    engine 不再持有各自的 _get_*。
  * **依赖显式注入**——凡是需要用到"文本 embedding / LLM(JSON) / LLM(text) /
    会话追踪器 / 左脑仓库 / 内心OS 生成 / 特质抽取"这些**非右脑本域**能力的地方，
    一律在 __init__ 里以 getter/函数引用注入（懒加载语义保持不变），组件内部
    通过 self._dep() 调用。

logic 一字不改：方法体原样搬运，只改"怎么拿依赖"。

brain.py 不 import engine（避免循环）——RightBrainHit 数据类与模块级 _rb_*
辅助函数都落在本模块，engine 反向从这里 import。
"""

from __future__ import annotations

from voicemem.utils.common import space as _space

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ── 结果容器 ───────────────────────────────────────────────────────────────────

@dataclass
class RightBrainHit:
    """右脑检索的单条结构化结果。rb_directive 由这个结构化列表渲染而来。"""
    content: str
    source: str                          # response_experience | situation_pattern | relation | emotion_trait | profile
    priority: float
    metadata: dict = field(default_factory=dict)


# ── 语言判定辅助 ───────────────────────────────────────────────────────────────

def _is_en_text(text: str) -> bool:
    """True if text is predominantly English (low CJK ratio)."""
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    alpha = sum(1 for c in text if c.isalpha())
    return alpha > 0 and cjk / max(alpha, 1) < 0.3


def _rb_lang(rb_ctx) -> bool:
    """Return True if right brain content is predominantly English."""
    samples = [m.content for m in rb_ctx.situation_patterns[:2]]
    samples += [m.content for m in rb_ctx.response_experiences[:1]]
    return _is_en_text(" ".join(samples))


def _rb_mem_date(m) -> str:
    """created_at ISO → '[YYYY-MM-DD] ' 前缀；没有日期返回空串（供时序推理用）。"""
    d = (getattr(m, "created_at", "") or "")[:10]
    return f"[{d}] " if d else ""


def _rb_blended_priority(m) -> float:
    """静态 priority + 本次检索的锚点相关度（归一化后加权）。

    anchor_score = SUM(link.weight*confidence)，用 s/(1+s) 压到 [0,1) 再乘 0.5：
    强命中锚点的具体证据能与 relation/emotion_trait 竞争，没命中的保持原状。"""
    s = getattr(m, "anchor_score", 0.0) or 0.0
    return m.priority + 0.5 * (s / (1.0 + s))


def _rb_ctx_to_hits(rb_ctx) -> list["RightBrainHit"]:
    """rb_ctx（heartnote / response_experience 检索结果）→ 结构化 hit 列表。
    当前信号（不满意/纠正/情绪提示）是本轮实时状态，给个偏高的固定优先级。"""
    en = _rb_lang(rb_ctx)
    hits: list[RightBrainHit] = []
    for m in rb_ctx.response_experiences:
        meta = m.metadata or {}
        failed = meta.get("previous_failure", False)
        prefix = (
            ("⚠ " + ("Avoid repeating: " if en else "避免重复："))
            if failed else
            ("✓ " + ("Effective approach: " if en else "有效方式："))
        )
        # content 只说了"当时怎么做的"，next_time_policy 才是可执行的那半句。
        # 用户为什么那么反应不在这里拼——那是用户特征，走图层的 profile hit。
        body = m.content
        policy = str(meta.get("next_time_policy") or "").strip()
        if policy:
            body += f" (next time: {policy})" if en else f"（下次：{policy}）"
        hits.append(RightBrainHit(
            content=f"{_rb_mem_date(m)}{prefix}{body}", source="response_experience",
            priority=_rb_blended_priority(m),
            metadata={"failed": failed, "anchor_score": getattr(m, "anchor_score", 0.0),
                      "next_time_policy": policy},
        ))
    for m in rb_ctx.situation_patterns:
        meta = m.metadata or {}
        prefix = "Emotional note: " if en else "情感记录："
        inner = str(meta.get("inner_os") or "").strip()
        # content 存原话，inner_os 作为补充渲染拼在后面。超长原话（>400）会挤爆
        # prompt，降级用内心 OS 摘要，没有就截断。
        if len(m.content) > 400:
            body = inner if inner else (m.content[:400] + "…")
        else:
            body = m.content
            if inner and inner != m.content:
                body += f" (inner note: {inner})" if en else f"（内心OS：{inner}）"
        content = f"{_rb_mem_date(m)}{prefix}{body}"
        priority = _rb_blended_priority(m)
        # 被后续记录取代的旧况：保留但标注 + 降权，避免模型把旧况当现状。
        if meta.get("superseded_by"):
            until = str(meta.get("superseded_at") or "")[:10]
            tag = (
                f" [outdated{f', changed around {until}' if until else ''} — see newer note]"
                if en else
                f"【旧况{f'，约 {until} 已变化' if until else ''}，以更新的记录为准】"
            )
            content += tag
            priority *= 0.75
        hits.append(RightBrainHit(
            content=content, source="situation_pattern", priority=priority,
            # emotion 也带出来。它一直存在这条记忆的 metadata 里，但从没进过 hit——
            # 调用方（web demo 的情绪标签）拿不到，只能去 content 里抠正则。
            metadata={"anchor_score": getattr(m, "anchor_score", 0.0),
                      "emotion": str(meta.get("emotion") or "")},
        ))
    sigs = rb_ctx.current_signals
    now: list[str] = []
    if sigs.dissatisfaction_signal:
        now.append("user is dissatisfied; get to the point" if en else "用户不满意，直接说重点")
    if sigs.correction_signal:
        now.append("user is correcting; just accept it" if en else "用户在纠正，接受即可")
    if sigs.affect_hint:
        now.append(f"current emotion={sigs.affect_hint}" if en else f"当前情绪={sigs.affect_hint}")
    if now:
        sep = "; " if en else "；"
        head = "Current signals: " if en else "当前信号："
        hits.append(RightBrainHit(
            content=head + sep.join(now), source="current_signal", priority=0.95,
            # 这是**本轮**的情绪（affect_hint），比检索回来的旧记忆上那个更该显示。
            metadata={"emotion": str(sigs.affect_hint or "")},
        ))
    return hits


# ── 用户对 agent 上一句的反应 ─────────────────────────────────────────────────
# 检索路径要 0 LLM（README 读写分离），所以这里纯词面判定。只认明说的反应词，
# 且必须有 agent 上一句——用户情绪不好 ≠ agent 说错话。
_DISSATISFIED_CUES = (
    "不对", "不是这个", "不是这样", "不是我要", "你没懂", "没听懂", "听不懂",
    "答非所问", "别说了", "无语", "敷衍", "废话", "没用",
    "not what i", "that's not", "thats not", "you don't get", "you dont get",
    "never mind", "nevermind", "forget it", "useless", "not helpful",
    "didn't ask", "didnt ask",
)
_CORRECTION_CUES = (
    "我说的是", "我是说", "我不是说", "我没说过", "记错", "搞错", "弄错", "纠正",
    "i said", "i meant", "you got it wrong", "that's wrong", "thats wrong",
    "no, i ", "actually, i", "actually i ",
)
_APPRECIATION_CUES = (
    "谢谢", "多谢", "太好了", "就是这个", "正是", "懂我", "有帮助", "帮大忙", "说得对",
    "thanks", "thank you", "exactly", "perfect", "that helps", "helpful",
    "good point", "appreciate",
)


def _hits_any(text: str, cues: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(c in low for c in cues)


def _reaction_signals(user_text: str, agent_reply: str = ""):
    """(用户这句, agent 上一句) → CurrentSignals：用户在不满/纠正 agent 吗？"""
    from voicemem.rightbrain.types import CurrentSignals
    if not (agent_reply or "").strip() or not (user_text or "").strip():
        return CurrentSignals()
    return CurrentSignals(
        dissatisfaction_signal=_hits_any(user_text, _DISSATISFIED_CUES),
        correction_signal=_hits_any(user_text, _CORRECTION_CUES),
    )


#: 值不值得花 LLM 去分析——内心 OS 和 4 类特质抽取各是一次 api 调用，
#: 而语音场景里大量输入是试麦、应答、打断（"哎呀"/"能听到吗"/"等一下"）。
#: 这些照样存 heartnote（原话 + 情绪标签，情绪是模型每轮打好的，不额外花钱），
#: 只是不再为它们编一段共情旁白——prompt 要求"被打动"，模型给不出"没情绪"
#: 这个答案，于是试麦克风被读成"心里一定很复杂"，噪音还会顺着情绪 slot
#: 进到人格描述里。
_FILLER = {
    "哎呀", "哎", "唉", "啊", "嗯", "嗯嗯", "哦", "噢", "诶", "欸", "喂", "喂喂", "啊喂",
    "等一下", "等一等", "等等", "稍等", "打断一下",
    "能听到吗", "听得到吗", "听得见吗", "在吗", "有人吗",
    "好", "好的", "行", "行吧", "可以", "对", "是", "没有", "不是",
    "测试", "test", "hello", "hi", "ok", "okay",
}
#: 短于这个长度、又没能让左脑抽出任何事实的，一律当填充语。
_MIN_ANALYZE_CHARS = 8


def _worth_analyzing(text: str, has_fact: bool) -> bool:
    """这句话值不值得再花 LLM。0 成本，只看词表和长度。

    短句不能一律砍——"崩溃""我很伤心"只有 2-4 个字，却正是最该分析的。所以在
    长度之前先问 anchor_router 的情绪关键词表：文本里有明确情绪词就放行（"还好"
    "不行""我知道了"匹配不上，"崩溃""伤心"匹配得上，两类分得很干净）。
    """
    s = re.sub(r"[^\w\u4e00-\u9fff]", "", text or "")
    if not s:
        return False
    if s.lower() in _FILLER:
        return False
    if has_fact or len(s) >= _MIN_ANALYZE_CHARS:
        return True
    from voicemem.rightbrain.anchor_router import normalize_emotion_strict
    return normalize_emotion_strict(text) is not None      # 短，但把情绪说出来了


def _rb_graph_hits(rb_graph, user_id: str) -> list["RightBrainHit"]:
    """右脑图(情绪/喜好与厌恶/表达风格/思维模式/应对方式)的 slot description，
    每个有描述的 slot 各是一条画像观察。"""
    return [
        RightBrainHit(
            content=f"{slot.name}：{slot.description}",
            source="profile", priority=0.5, metadata={"slot_name": slot.name},
        )
        for slot in rb_graph.list_slots(user_id)
        if slot.description
    ]


def _rb_emotion_trait_hit(rb_graph, user_id: str, emotion: str | None) -> "RightBrainHit | None":
    """检索和当前情绪相近的用户固有性格节点：情绪是8个固定规范标签，按当前
    情绪精确查"情绪"slot 下同名的那一个 entity，不用向量、不扫描其余7个。"""
    if not emotion:
        return None
    from voicemem.rightbrain.anchor_router import normalize_emotion_strict
    emo_slot = rb_graph.get_slot_by_name(user_id, "情绪")
    if emo_slot is None:
        return None
    canonical = normalize_emotion_strict(emotion)
    if canonical is None:
        return None
    ent = rb_graph.get_entity_by_name(user_id, emo_slot.id, canonical)
    if ent is None or not ent.description:
        return None
    prefix = (
        "Trait observed around this emotion: " if _is_en_text(ent.description)
        else "当前情绪相关的性格观察："
    )
    return RightBrainHit(
        content=f"{prefix}{ent.description}", source="emotion_trait", priority=0.75,
    )


def _rb_relation_hits(rb_graph, user_id: str, anchors) -> list["RightBrainHit"]:
    """关系节点检索：左脑这次问题触发的实体（anchors 里带真实 entity.id 的
    锚点），直接按 ID 查对应的右脑关系节点——纯索引查表，不扫描、不算向量。"""
    from voicemem.rightbrain.anchor_router import _ENTITY_TYPE_TO_ANCHOR
    entity_anchor_types = set(_ENTITY_TYPE_TO_ANCHOR.values())
    rel_slot = rb_graph.get_slot_by_name(user_id, "人物地点态度")
    if rel_slot is None:
        return []
    hits: list[RightBrainHit] = []
    seen: set[str] = set()
    for a in anchors:
        if a.anchor_type not in entity_anchor_types or not a.anchor_id:
            continue
        if a.anchor_id in seen:
            continue
        seen.add(a.anchor_id)
        ent = rb_graph.get_entity_by_source_id(user_id, rel_slot.id, a.anchor_id)
        if ent is not None and ent.description:
            content = (
                f"Impression of {ent.name}: {ent.description}"
                if _is_en_text(ent.description)
                else f"对 {ent.name} 的印象：{ent.description}"
            )
            hits.append(RightBrainHit(
                content=content,
                source="relation", priority=0.85, metadata={"entity_name": ent.name},
            ))
    return hits


#: 各来源在 top-N 里最多占几席。没列的不限。
#:
#: 为什么要配额：这两类的 priority 都是**查询无关**的常数，谁都竞争不过它们——
#:   · response_experience 记的是"助手上次怎么答的"（"✓ 有效方式：助手用轻松的
#:     语气引导用户展开对话"），是给回复层看的内部笔记，不是对用户其人的认识；
#:   · profile 是每个 slot 的静态描述（``_rb_graph_hits`` 把所有带描述的 slot
#:     无条件全返回，priority 固定 0.5），**每一轮都是同样那几条**。
#: 实测这两类合起来能吃掉 top-5 的全部席位，于是右脑对每个问题给的东西都一样：
#: 回复里读不出"它记得我这件事"，脑图上每次检索射向的也永远是同一批节点。
#: 真正随问题变的是 situation_pattern（情感记录）和 relation（对某人的印象）——
#: 得给它们留出位置。
_SOURCE_QUOTA = {
    "response_experience": max(0, int(os.environ.get("VOICEMEM_RB_RESPONSE_MAX", "1"))),
    "profile": max(0, int(os.environ.get("VOICEMEM_RB_PROFILE_MAX", "2"))),
}


def _apply_source_quota(hits: list["RightBrainHit"]) -> list["RightBrainHit"]:
    """按来源限席，超额的**直接丢弃**。

    原来是"往后挪不丢弃"，但候选常常本来就只有五六条——挪到后面照样进 top-5。
    实测「我好累啊」返回的五条里三条是 response_experience（助手上次怎么答的
    内部笔记）、两条是静态 profile，一条真正的情感记忆都没有，而且每轮都一样。
    宁可只给三条真东西，也不要用内部笔记把位置占满。
    """
    kept, used = [], {}
    for h in hits:
        src = getattr(h, "source", "")
        cap = _SOURCE_QUOTA.get(src)
        if cap is None:
            kept.append(h)
            continue
        used[src] = used.get(src, 0) + 1
        if used[src] <= cap:
            kept.append(h)
    return kept


def _render_rb_directive(hits: list["RightBrainHit"]) -> str:
    """结构化 top-N → 拼进 prompt 的文本块。"""
    return "\n".join(h.content for h in hits) if hits else ""


# ── RightBrain 组件 ────────────────────────────────────────────────────────────

class RightBrain:
    """自持零件 + 依赖显式注入的右脑组件。

    构造参数分两类：

    组件自持的**运行参数**（右脑侧路径/身份）::

        memory_root, user_id, base_url, cognitive_db

    显式**注入的非右脑依赖**（全部以 getter/函数引用传入，懒加载语义不变）::

        embed              -> self._embed_text          文本 embedding（特质/图层写入）
        llm_json           -> self._llm_json            LLM(JSON)（特质抽取）
        llm_text           -> self._llm_text            LLM(text)（归因管理器）
        tracker            -> self._get_session_tracker 跨左右脑会话追踪器（touch）
        repo               -> self._get_repo            左脑仓库（写入时查左脑实体链接）
        generate_inner_os  -> self._generate_inner_os   内心OS 生成（延迟解析，可被测试 patch）
        extract_rb_traits  -> self._extract_rb_traits   特质抽取（延迟解析，可被测试 patch）

    右脑侧的 3 个懒加载单例（rb_repo / rb_graph_store / attribution_manager）
    连同与宿主共享的缓存/锁由本组件自持，见下方 self._rb_repo() 等。
    """

    def __init__(
        self,
        *,
        memory_root: Path,
        user_id: str,
        base_url: str | None,
        cognitive_db: Path,
        embed: Callable[[str], list[float]],
        llm_json: Callable[[str], str],
        llm_text: Callable[..., str],
        tracker: Callable[[], Any],
        repo: Callable[[], Any],
        generate_inner_os: Callable[..., str],
        extract_rb_traits: Callable[..., list[tuple[str, str]]],
        cache: dict[str, Any] | None = None,
        lock: Any = None,
    ) -> None:
        # ── 运行参数 ──
        self._memory_root = memory_root
        self._user_id = user_id
        self._base_url = base_url
        self._cognitive_db = cognitive_db

        # ── 注入的非右脑依赖（getter/函数引用）──
        self._embed = embed
        self._llm_json = llm_json
        self._llm_text = llm_text
        self._tracker = tracker
        self._repo = repo
        # generate_inner_os / extract_rb_traits 延迟解析（getter 形式），
        # 让既有测试对宿主实例的 patch.object 生效（write 走宿主暴露的入口）。
        self._generate_inner_os = generate_inner_os
        self._extract_rb_traits = extract_rb_traits

        # ── 组件自持的右脑零件缓存 ──
        # 允许宿主共享同一个 cache/lock（右脑侧懒加载单例与宿主 _get_* 落在同一
        # 字典，既存调用点/测试对宿主 _cache 的直接读写与本组件保持一致视图）。
        self._cache: dict[str, Any] = cache if cache is not None else {}
        self._lock = lock if lock is not None else threading.Lock()

    # ── 右脑懒加载单例 ──────────────────────────────────────────────────────────

    def _rb_repo(self):
        with self._lock:
            if "rb_repo" not in self._cache:
                from voicemem.leftbrain.cognitive_graph import CognitiveGraphStore
                from voicemem.rightbrain import ExperienceRepository
                cog_store = CognitiveGraphStore(self._cognitive_db)
                self._cache["rb_repo"] = ExperienceRepository.create(
                    _space.db(self._memory_root),
                    cognitive_store=cog_store,
                )
        return self._cache["rb_repo"]

    def _rb_graph_store(self):
        with self._lock:
            if "rb_graph_store" not in self._cache:
                from voicemem.rightbrain import RightBrainGraphStore
                store = RightBrainGraphStore(_space.db(self._memory_root))
                store.ensure_seed_slots(self._user_id)
                self._cache["rb_graph_store"] = store
        return self._cache["rb_graph_store"]

    def _attribution_manager(self):
        rb_graph = self._rb_graph_store()
        rb_repo = self._rb_repo()
        with self._lock:
            if "attribution_manager" not in self._cache:
                from voicemem.rightbrain import AttributionManager
                self._cache["attribution_manager"] = AttributionManager(
                    rb_graph, rb_repo._store, llm_fn=self._llm_text,
                )
        return self._cache["attribution_manager"]

    # ── 右脑检索 ────────────────────────────────────────────────────────────────

    def search(
        self, query: str, activated_names: list[str], emotion: str | None, top_k: int,
        agent_reply: str = "",
    ) -> tuple[list["RightBrainHit"], str]:
        """右脑并发检索段（原 Search() 里的 _run_rb 闭包）：build_query_plan →
        retrieve → 关系/情绪特质/画像图层 → 按 priority 排序截断，返回
        (rb_hits, rb_directive)。异常时降级为无右脑（空列表 + 空指导）。

        ``agent_reply``：agent 上一轮那句。用户正在回应它，两处用到——① 反应信号
        （不满/纠正 → retrieve 多给一条回应经验名额）；② 它提到的实体以 context
        角色、半权重进锚点。纯词面判定，检索路径依旧 0 LLM。
        """
        try:
            rb_repo = self._rb_repo()
            signals = _reaction_signals(query, agent_reply)
            # 右脑接收左脑"已激活实体"作锚点（联合检索：右脑依赖左脑激活结果）
            plan    = rb_repo.build_query_plan(
                query, self._user_id,
                signals=signals,
                entities=activated_names or None,
                emotion=emotion,
                context=agent_reply or None,
            )
            rb_ctx = rb_repo.retrieve(plan)
            collected: list[RightBrainHit] = _rb_ctx_to_hits(rb_ctx) if not rb_ctx.is_empty() else []

            rb_graph = self._rb_graph_store()
            collected.extend(_rb_relation_hits(rb_graph, self._user_id, plan.anchors))
            trait_hit = _rb_emotion_trait_hit(rb_graph, self._user_id, emotion)
            if trait_hit is not None:
                collected.append(trait_hit)
            collected.extend(_rb_graph_hits(rb_graph, self._user_id))

            # 按 priority 排序截断，rb_directive 从截断后的列表渲染，保证一致。
            collected.sort(key=lambda h: h.priority, reverse=True)
            # 右脑结构化 top-N（默认 5，VOICEMEM_RB_TOPN 可调）。
            try:
                _rb_topn = max(1, int(os.environ.get("VOICEMEM_RB_TOPN", "5")))
            except ValueError:
                _rb_topn = 5
            rb_hits = _apply_source_quota(collected)[:_rb_topn]
            return rb_hits, _render_rb_directive(rb_hits)
        except Exception as e:
            import traceback as _tb
            print(f"[Search] 右脑检索失败（本轮降级为无右脑）: {e}\n{_tb.format_exc()}", flush=True)
            return [], ""

    # ── 右脑写入 ────────────────────────────────────────────────────────────────

    def write(self, emotion, result, text, entities, observed_at,
              agent_reply: str = "") -> str | None:
        """右脑写入段：每条 utterance 一条 heartnote，挂 emotion + entity anchors +
        关系节点 + 右脑 slot→entity 图层。gate 只看 emotion（不绑 result.memory_ids）——
        纯情绪句左脑可能抽不出事实但情绪仍值得记；mid 为空时不挂证据、不查左脑实体
        链接，但情绪锚点 + 文本实体名锚点仍正常写。

        ``agent_reply``：agent 上一轮那句。同一句"行吧"，跟在共情后面和跟在甩方案
        后面是两种情绪——喂给内心 OS 生成，并存进 metadata 留证据。
        """
        if not emotion:
            return
        try:
            from voicemem.rightbrain.types import MemoryAnchor
            rb_repo = self._rb_repo()
            mid = result.memory_ids[0] if result.memory_ids else None

            # content 存原话，inner_os 进 metadata（渲染时作为补充拼在原话
            # 后面，见 _rb_ctx_to_hits）——避免共情改写抹掉数字/名字/时间等细节。
            # 填充语（试麦/应答/打断）不值得为它编一段内心 OS，见 _worth_analyzing。
            worth = _worth_analyzing(text, has_fact=bool(mid))
            inner_os = (self._generate_inner_os(text, emotion, entities or [], agent_reply)
                        if worth else "")
            content = text

            # 事件时间用 observed_at（与左脑 time_start 同源），不用写入墙钟
            _obs = str(observed_at) if observed_at and re.match(r"^\d{4}-\d{2}-\d{2}", str(observed_at)) else None
            rb_mem = rb_repo._store.upsert_memory(
                user_id=self._user_id,
                memory_class="heartnote",
                content=content,
                metadata={"emotion": emotion, "entities": entities or [],
                          "left_memory_id": mid, "inner_os": inner_os or "",
                          # 情绪的触发上下文：这句话是接在 agent 的哪句后面说的
                          "agent_reply": (agent_reply or "").strip()},
                evidence_memory_ids=[mid] if mid else [],
                created_at=_obs,
            )
            # emotion anchor：按情感检索。strict 版，识别不出的情绪词不挂锚点。
            from voicemem.rightbrain.anchor_router import normalize_emotion_strict
            canonical_emotion = normalize_emotion_strict(emotion)
            if canonical_emotion is not None:
                rb_repo._store.link_anchor(
                    rb_mem.id, self._user_id,
                    MemoryAnchor(anchor_type="emotion", anchor_id=canonical_emotion,
                                 role="trigger", weight=1.0, confidence=1.0),
                )
            # entity anchors：优先用左脑这条记忆真正链上的 entity.id（稳定），
            # name 字符串锚点做兜底。同一循环里顺手给每个实体建/更新一个专属
            # 关系节点（source_entity_id 精确匹配），这里只负责挂证据 + touch，
            # description 的提炼交给 AttributionManager.run_short_term。
            try:
                from voicemem.rightbrain.anchor_router import _ENTITY_TYPE_TO_ANCHOR
                cog_store = self._repo()._cognitive_store
                if mid and cog_store is not None:
                    rb_graph = self._rb_graph_store()
                    tracker  = self._tracker()
                    rel_slot = rb_graph.get_or_create_slot(self._user_id, "人物地点态度")
                    for eid in cog_store.entity_ids_for_memory(mid):
                        ent = cog_store.get_entity(eid)
                        if ent is None:
                            continue
                        rb_repo._store.link_anchor(
                            rb_mem.id, self._user_id,
                            MemoryAnchor(
                                anchor_type=_ENTITY_TYPE_TO_ANCHOR.get(ent.entity_type.value, "knowledge"),
                                anchor_id=ent.id, role="subject",
                                weight=1.0, confidence=ent.confidence,
                            ),
                        )
                        rel_ent, _created = rb_graph.get_or_create_entity_by_source_id(
                            self._user_id, rel_slot.id, ent.id, ent.name,
                        )
                        rb_graph.link_memory(rel_ent.id, self._user_id, rb_mem.id)
                        tracker.touch(self._user_id, "rb_entity_short", rel_ent.id)
            except Exception as e:
                print(f"[RBAnchor] 实体ID锚点/关系节点写入失败: {e}")

            for name in (entities or []):
                rb_repo._store.link_anchor(
                    rb_mem.id, self._user_id,
                    MemoryAnchor(anchor_type="entity", anchor_id=name.lower().strip(),
                                 role="subject", weight=0.8, confidence=1.0),
                )

            # 右脑 slot→entity 图层：情绪(精确匹配8个固定标签) + 其余4类(语义匹配)
            try:
                rb_graph = self._rb_graph_store()
                tracker = self._tracker()
                emo_slot = rb_graph.get_slot_by_name(self._user_id, "情绪")
                if emo_slot is not None and canonical_emotion is not None:
                    emo_ent = rb_graph.get_or_create_entity(
                        self._user_id, emo_slot.id, canonical_emotion,
                    )
                    rb_graph.link_memory(emo_ent.id, self._user_id, rb_mem.id)
                    tracker.touch(self._user_id, "rb_entity_short", emo_ent.id)
                    tracker.touch(self._user_id, "rb_slot_long", emo_slot.id)

                # 情绪 slot 上面已无条件挂好（模型每轮都打了标）；4 类特质要再花
                # 一次 LLM，填充语里抽不出真特质，只会往画像里塞噪音。
                if worth:
                    for slot_name, label in self._extract_rb_traits(text, emotion):
                        self._write_trait(slot_name, label, rb_mem.id)
            except Exception as e:
                print(f"[RBGraph] 右脑图层写入失败: {e}")
            return rb_mem.id
        except Exception as e:
            print(f"[Ingest] right brain write skipped: {e}")
            return None

    def _write_trait(self, slot_name: str, label: str, memory_id: str) -> bool:
        """往图层 slot 下挂一个语义去重的特质 entity，并把这条记忆作为证据链上去。
        description 交给 AttributionManager 从证据里归纳，这里只负责挂 + touch。"""
        if not slot_name or not label:
            return False
        rb_graph = self._rb_graph_store()
        slot = rb_graph.get_slot_by_name(self._user_id, slot_name)
        if slot is None:
            return False
        ent, _created = rb_graph.get_or_create_entity_semantic(
            self._user_id, slot.id, label, self._embed(label),
        )
        rb_graph.link_memory(ent.id, self._user_id, memory_id)
        tracker = self._tracker()
        tracker.touch(self._user_id, "rb_entity_short", ent.id)
        tracker.touch(self._user_id, "rb_slot_long", slot.id)
        return True

    # ── 回应成败经验（agent 自己说过的话，被用户的反应打分）──────────────────────

    #: 归因每段的字数上限——这几段每轮都要拼进 system prompt，松了就是固定开销。
    _EXPERIENCE_MAX_CHARS = 60

    _ATTRIBUTION_PROMPT = """你在给助手的回应打分：用户这轮是不是在对助手那句作出反应？只输出 JSON。

助手那句：{reply}
用户这轮：{user}{emotion_line}

{{"significant": bool,
  "assistant_helped": "助手那句帮到用户了(true)还是帮了倒忙(false)",
  "assistant_did": "主语必须是助手：助手那句用了什么做法，可迁移不抄原话，"
                   "如「直接给了分点方案」「先接住情绪再问细节」",
  "next_time": "助手以后遇到类似情形怎么做（可执行的动作）",
  "user_reaction": "主语必须是用户：用户的反应",
  "why": "为什么这么反应（落到助手那句的哪一点）",
  "user_trait": {{"slot": "表达风格|应对方式|思维模式|喜好与厌恶", "label": "这个反应
    透露出的用户长期特征，如「被直接给方案会关闭」「不满时直说不绕弯」；
    只是这一次的情境反应、看不出长期特征就填 null"}}}}

significant 默认 false，只有这几种才 true：
明确不满 / 纠正助手 / 明确道谢认可 / 因为助手那句情绪明显变化 /
敷衍收尾（"算了""随便吧""你说吧"这种把话头推回来的）。
以下一律 false：继续讲自己的事、回答助手的问题、提新要求、寒暄。
用户情绪不好 ≠ 助手说错话。
significant 不管真假，其余字段都要照填（调用方另有判定）。
文本字段各 ≤{n} 字，语言跟用户那句一致。"""

    def _attribute_reaction(self, user_text: str, agent_reply: str, emotion: str) -> dict:
        """(助手上一句 + 用户这轮) → 归因：记不记 + 反应/为什么/下次怎么做。

        分工是刻意的——**该不该记不全交给模型**：小模型在这个判断上会抖，实测同一
        句"不是这个意思，你根本没懂"两次跑出相反结论，漏掉的恰恰是最该记的。
          · 词面命中（不满/纠正/感谢）→ 强制记，好坏也由词面定，模型只写文本；
          · 词面没命中 → 灰区（"……算了，你说吧"）交给模型判 significant。
        """
        sigs = _reaction_signals(user_text, agent_reply)
        forced_failed = bool(sigs.dissatisfaction_signal or sigs.correction_signal)
        forced = forced_failed or _hits_any(user_text, _APPRECIATION_CUES)

        emotion_line = f"\n（情绪识别：{emotion}）" if emotion else ""
        raw = self._llm_json(self._ATTRIBUTION_PROMPT.format(
            reply=agent_reply[:300], user=user_text[:300],
            emotion_line=emotion_line, n=self._EXPERIENCE_MAX_CHARS,
        ))
        data = {}
        if raw:
            try:
                import json as _json
                parsed = _json.loads(raw)
                if isinstance(parsed, dict):
                    data = parsed
            except Exception as e:
                print(f"[RBExperience] 归因解析失败: {e}")

        if forced:
            data["significant"] = True
            data["assistant_helped"] = not forced_failed
            data.setdefault("assistant_did", agent_reply[:self._EXPERIENCE_MAX_CHARS])
            if not str(data.get("user_reaction") or "").strip():
                en = _is_en_text(user_text)
                data["user_reaction"] = (
                    (("user corrected it" if sigs.correction_signal else "user pushed back")
                     if forced_failed else "user said it helped") if en else
                    (("用户纠正了" if sigs.correction_signal else "用户表示不满")
                     if forced_failed else "用户认可"))
        return data if "significant" in data else {"significant": False}

    def learn_from_reaction(self, text: str, emotion: str, entities, agent_reply: str,
                            memory_id: str | None = None, observed_at=None,
                            heartnote_id: str | None = None) -> None:
        """(助手上一句 + 用户这轮) → 情绪归因，有必要才落一条 response_experience。

        产物分两处存，因为它们是两类东西：
          · 助手侧「做法 + 下次怎么做」→ response_experience（content 存可迁移的
            做法，存原话换个话题就用不上；next_time 进 metadata）
          · 用户侧「透露出的长期特征」→ 图层 slot（表达风格/应对方式…），由归因
            归纳成人格描述，跨话题都能用

        不受 write() 那个 ``if not emotion`` 管——"不是这个意思"是行为信号，跟声学
        情绪有没有输出无关（text_mode 下 emotion 常为空）。没有助手上一句直接返回，
        所以第一轮不调 LLM。
        """
        reply = (agent_reply or "").strip()
        if not reply or not (text or "").strip():
            return
        try:
            attribution = self._attribute_reaction(text, reply, emotion or "")
            if not attribution.get("significant"):
                return

            from voicemem.rightbrain.types import MemoryAnchor
            from voicemem.rightbrain.anchor_router import normalize_emotion_strict

            def _clip(v) -> str:
                s = str(v or "").strip()
                return s if len(s) <= self._EXPERIENCE_MAX_CHARS else s[:self._EXPERIENCE_MAX_CHARS] + "…"

            what_i_did = _clip(attribution.get("assistant_did")) or _clip(reply)
            reaction   = _clip(attribution.get("user_reaction"))
            why        = _clip(attribution.get("why"))
            next_time  = _clip(attribution.get("next_time"))
            failed     = not bool(attribution.get("assistant_helped", False))

            en = _is_en_text(text)
            condition = (f"{reaction}{' — ' + why if why else ''}" if en
                         else f"{reaction}{'；' + why if why else ''}")

            # global_style：每个查询计划都带这个兜底锚点，所以这条教训每轮都捞得到
            # ——"别再这么回应"不该等同一话题重现才想起来。情绪/实体再叠话题相关性。
            anchors = [
                MemoryAnchor(anchor_type="global_style", anchor_id="global_style",
                             role="global_profile", weight=1.0, confidence=1.0),
            ]
            canonical = normalize_emotion_strict(emotion) if emotion else None
            if canonical is not None:
                anchors.append(MemoryAnchor(anchor_type="emotion", anchor_id=canonical,
                                            role="trigger", weight=1.0, confidence=1.0))
            for name in (entities or []):
                key = str(name).lower().strip()
                if key:
                    anchors.append(MemoryAnchor(anchor_type="entity", anchor_id=key,
                                                role="subject", weight=0.8, confidence=1.0))

            _obs = (str(observed_at)
                    if observed_at and re.match(r"^\d{4}-\d{2}-\d{2}", str(observed_at))
                    else None)
            exp = self._rb_repo().write_response_experience(
                self._user_id, what_i_did, anchors,
                condition=condition,
                failed=failed,
                # 反应/原话留底当证据，不进 prompt——用户侧的结论在图层那边
                metadata={"next_time_policy": next_time, "why": why, "reaction": reaction,
                          "agent_reply": reply[:300], "user_reaction": text.strip()[:200],
                          "emotion": emotion or ""},
                evidence_memory_ids=[memory_id] if memory_id else [],
                created_at=_obs,
            )
            print(f"[RBExperience] {'失败' if failed else '有效'}：{what_i_did}"
                  f" | 下次：{next_time}", flush=True)

            # 用户侧的观察不留在这条经验里：「这人被直接给方案会关闭」是长期特征，
            # 该沉淀进图层 slot 由归因归纳成人格，否则只有这条经验被检中才看得见。
            trait = attribution.get("user_trait") or {}
            if isinstance(trait, dict):
                slot_name, label = str(trait.get("slot") or ""), _clip(trait.get("label"))
                # 证据挂 heartnote（用户原话）——归因是读证据的 content 来写
                # description 的，挂 exp 就成了"拿助手的做法去描述用户特征"。
                if label and label.lower() not in ("null", "none") and \
                        self._write_trait(slot_name, label, heartnote_id or exp.id):
                    print(f"[RBTrait] {slot_name} ← {label}", flush=True)
        except Exception as e:
            print(f"[RBExperience] 回应经验写入失败: {e}")

    # ── 右脑清洁 ────────────────────────────────────────────────────────────────

    def check_and_cleanup(self) -> None:
        """每增加 50 条 heartnote 触发一次右脑清洁。"""
        try:
            import json as _json
            state = {"last_count": _space.kv_get(self._memory_root, "rb_cleanup_last_count", 0)}

            rb_repo = self._rb_repo()
            all_mems = rb_repo._store.get_all(self._user_id)
            current_count = sum(1 for m in all_mems if m.memory_class == "heartnote")

            if current_count - state.get("last_count", 0) >= 50:
                _space.kv_set(self._memory_root, "rb_cleanup_last_count", current_count)
                self.run_cleanup()
        except Exception as e:
            print(f"[Cleanup] check error: {e}")

    def run_cleanup(self) -> None:
        """用 LLM 清洁右脑 heartnote：重复/无意义 → 删除；矛盾 → 标注 supersede。

        矛盾不"删旧留新"（偏好演化题需要新旧两条 + 先后关系）：旧条目打
        superseded_by/superseded_at 标记保留，渲染时标注"旧况"并降权。"""
        try:
            import json as _json
            import sqlite3

            rb_repo = self._rb_repo()
            heartnotes = [
                m for m in rb_repo._store.get_all(self._user_id)
                if m.memory_class == "heartnote"
            ]
            if len(heartnotes) < 10:
                return

            # 构造紧凑列表发给 LLM（用前8位 ID 节省 token）
            lines = []
            id_map: dict[str, str] = {}  # short_id -> full_id
            for i, m in enumerate(heartnotes):
                short = m.id[:8]
                id_map[short] = m.id
                emotion  = (m.metadata or {}).get("emotion", "")
                entities = (m.metadata or {}).get("entities", [])
                lines.append(
                    f"[{i}] ID:{short} | 情感:{emotion} | 实体:{','.join(entities)} | {m.content}"
                )

            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=self._base_url,
                timeout=60.0,
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "你是记忆清洁助手。分析以下情感记忆列表，做两类判断。\n"
                        "一、删除（宁可少删，不要误删有价值的记录）：\n"
                        "1. 重复：内容高度相似，保留一条，删其余；\n"
                        "2. 无意义：信息量极低（如纯标点、单词、句子残缺）。\n"
                        "二、取代（不删除）：同一实体/同一偏好有前后矛盾的描述，"
                        "序号靠后的是新状态——旧的不删，标记为被新的取代，"
                        "以保留偏好演化轨迹。\n"
                        "返回 JSON：{\"delete_ids\": [\"8位ID\", ...], "
                        "\"supersede\": [{\"old_id\": \"8位ID\", \"new_id\": \"8位ID\"}, ...]}\n"
                        "若无需处理返回 {\"delete_ids\": [], \"supersede\": []}"
                    )},
                    {"role": "user", "content": "\n".join(lines)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )

            result      = _json.loads(resp.choices[0].message.content)
            short_ids   = result.get("delete_ids", [])
            full_ids    = [id_map[s] for s in short_ids if s in id_map]

            if full_ids:
                with sqlite3.connect(rb_repo._store._path) as conn:
                    for mid in full_ids:
                        conn.execute(
                            "DELETE FROM right_brain_anchor_links WHERE right_memory_id=?", (mid,)
                        )
                        conn.execute(
                            "DELETE FROM right_brain_memories WHERE id=?", (mid,)
                        )
                print(f"[Cleanup] 清洁完成，删除 {len(full_ids)} 条右脑记忆")
            else:
                print("[Cleanup] 无需删除")

            # 矛盾对：旧条目打 superseded 标记（保留，渲染时标注+降权）。
            pairs = result.get("supersede", []) or []
            keep_old = True
            marked = 0
            from datetime import datetime, timezone
            now_iso = datetime.now(timezone.utc).isoformat()
            for p in pairs:
                old_full = id_map.get(str(p.get("old_id", "")))
                new_full = id_map.get(str(p.get("new_id", "")))
                if not old_full or not new_full or old_full == new_full:
                    continue
                if old_full in full_ids or new_full in full_ids:
                    continue  # 已被删除的不再标记
                if keep_old:
                    rb_repo._store.merge_metadata(
                        old_full, {"superseded_by": new_full, "superseded_at": now_iso},
                    )
                else:
                    with sqlite3.connect(rb_repo._store._path) as conn:
                        conn.execute(
                            "DELETE FROM right_brain_anchor_links WHERE right_memory_id=?",
                            (old_full,),
                        )
                        conn.execute(
                            "DELETE FROM right_brain_memories WHERE id=?", (old_full,)
                        )
                marked += 1
            if marked:
                action = "标记为 superseded（保留演化轨迹）" if keep_old else "删除（旧行为）"
                print(f"[Cleanup] {marked} 条矛盾旧况已{action}")

            # 更新 last_count
            remaining = sum(
                1 for m in rb_repo._store.get_all(self._user_id)
                if m.memory_class == "heartnote"
            )
            _space.kv_set(self._memory_root, "rb_cleanup_last_count", remaining)

        except Exception as e:
            print(f"[Cleanup] run error: {e}")


__all__ = ["RightBrain", "RightBrainHit"]
