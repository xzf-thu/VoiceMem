"""编排实现（mem0 式）：Orchestrator = 左脑 + 右脑 + 音频感知 + 一组可换的能力(utils)。

这是藏在门面 ``voicemem.core.VoiceMem`` 背后的实现层。门面对外只暴露面向用户的
小写 API，实际的整条 pipeline（Search/Ingest 编排、工具方法、向三组件的转发、
SearchResult 数据类、Utils 能力表）都在本模块。

    左脑  事实记忆：实体 + 认知图（slot 分类/检索），底层 mem0 向量库
    右脑  情绪记忆：每轮 valence-arousal、情绪归因、人格画像
    utils 可插拔能力：embedding / schema(分类) / entity / emotion / voiceprint / asr / memory_engine
          每个都有内置默认，传一个函数就换成自己的（本地模型、别的向量库…）

mode 决定加载哪些能力：left_brain_single / text_mode / multi_modal(带音频)。

Orchestrator 直接持有并构造三个自包含组件（self._left/_right/_audio），编排它们完成
Search/Ingest 等完整 pipeline。

用法（编排层，通常经门面调用）::

    from voicemem.orchestrator import Orchestrator

    o = Orchestrator()

    # 语音模块提供 slot 和 entities，直接传入：
    result = o.Search(query, slots=["work"], entities=["阿里"])

    # 分步调用：
    slot_ids, clf = o.SearchCogGraph(slots=["work"], entities=["阿里"])
    candidate_ids  = o.SearchData(slot_ids, clf)
    hits           = o.Rank(query, candidate_ids, top_k=5)
"""

from __future__ import annotations

from voicemem.utils.common import space as _space

import functools
import inspect
import os
import time
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QueryClassification
from voicemem.leftbrain.local_memory_store import MemorySearchHit
# 左脑那一整块（slot 过滤/实体缩窄/时间扩候选/向量排序/查询分类/LLM 打标签/
# slot→entity 图层/子图记账与 checkpoint/schema 描述刷新/冷记忆归档）搬进了
# LeftBrain 组件；_search_mode 辅助函数随之迁至 voicemem.leftbrain.brain。这里
# 反向 import 回来，供 Search() 组装 SearchResult 时调用。
from voicemem.leftbrain.brain import LeftBrain, _search_mode
from voicemem.utils.audio.perceiver import AudioPerception, AudioPerceiver
# 右脑那一整块（heartnote 写入/内心OS/图层/检索/清洁）搬进了 RightBrain 组件；
# RightBrainHit 数据类与 _rb_* 辅助函数随之迁至 voicemem.rightbrain.brain。
from voicemem.rightbrain.brain import (
    RightBrain,
    RightBrainHit,
    _is_en_text,
    _rb_blended_priority,
    _rb_ctx_to_hits,
    _rb_lang,
    _rb_mem_date,
    _rb_trait_hits,
    _render_rb_directive,
)
from voicemem.utils.defaults import default_utils
from voicemem.llm_config import resolve_api_key, resolve_base_url, resolve_model

# 新增多少条记忆（左脑事实 + 右脑 heartnote）才巩固一次。见 Ingest() 里那段注释：
# 巩固是把历史记忆重新概括，每轮跑既慢（7s）又没什么可总结的。session 结束时无论
# 攒了多少都会清账，所以不会一直积着不处理。
SHORT_TERM_MIN_MEMORIES = int(os.environ.get("VOICEMEM_ATTRIBUTION_MIN_MEMORIES", "20"))

# 每个 mode 需要哪些 util（只加载这些）。
# tts 不在任何一档里：核心链路只到文本为止，出声是可选的一层，谁要出声谁
# utils.get("tts")——搁进来会让没装 piper/voxcpm 的用户在 warmup 就炸。
_NEED = {
    "left_brain_single": ["embedding", "slots", "entity", "memory_engine"],
    "text_mode":         ["embedding", "slots", "entity", "emotion", "memory_engine"],
    "multi_modal":       ["embedding", "slots", "entity", "emotion", "voiceprint", "asr", "memory_engine"],
}


#: 说了"给你听首歌"之后，多久之内的纯声音轮算作那首歌。
#: 短一点更安全：说完一般紧接着就放，隔太久多半是另一回事了。
EXPECT_MUSIC_S = float(os.environ.get("VOICEMEM_EXPECT_MUSIC_S", "120"))


#: 同一个能力的历史别名 → 正式能力名。
#:
#: ``schema`` 其实是"查询槽位分类器"，from_config 里一直叫 ``slots``，两个名字指
#: 同一个东西；``embedder`` / ``vector_store`` / ``classifier`` 是构造参数那一路的
#: 叫法。以前这两路各走各的、在 __init__ 里用 pick() 合并，等于同一件事有两个入口
#: 两套代码。现在在门口就归一到能力名，后面只剩一条路。旧名字继续能用。
_ALIASES = {"schema": "slots", "embedder": "embedding",
            "vector_store": "memory_engine", "classifier": "slots"}


def _canon(overrides: dict) -> dict:
    """把别名归一成正式能力名。同时给了新旧两个名字时，正式名优先。"""
    out = {}
    for k, v in overrides.items():
        out.setdefault(_ALIASES.get(k, k), v)
    for k, v in overrides.items():
        if k not in _ALIASES:
            out[k] = v
    return out


class Utils:
    """能力表：内置默认(见 utils/defaults.py) + 用户覆盖，按需懒加载并缓存。"""
    def __init__(self, mode, base_url, memory_root, overrides):
        self._factory = {**default_utils(base_url, memory_root), **_canon(overrides)}
        self.need = _NEED[mode]
        self._cache = {}
    def get(self, name):
        """能力值可以是**工厂**（函数 / lambda / 类，懒加载，内置默认都是这种），
        也可以直接是**造好的对象**——两种都收，用户不必为了塞一个现成对象去包一层
        lambda。判据是"是不是函数/类"，不是 callable()：组件对象自己可能带
        ``__call__``，用 callable() 会把它当工厂调一次。"""
        if name not in self._cache:
            f = self._factory[name]
            self._cache[name] = f() if (inspect.isfunction(f) or inspect.ismethod(f)
                                        or inspect.isclass(f)
                                        or isinstance(f, functools.partial)) else f
        return self._cache[name]


# ── 结果容器 ───────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """Search() 的完整返回值。"""
    hits: list[MemorySearchHit]
    classification: QueryClassification
    related_summaries: dict[str, str]   # {slot: summary_text}
    slot_mem_ids: set[str]              # SearchCogGraph 返回的原始 slot IDs
    final_candidate_ids: set[str]       # SearchData 实体缩窄后的最终候选 IDs
    search_mode: str = "fallback"
    rb_directive: str = ""              # 右脑情境指导文字（由 rb_hits 渲染而来）
    rb_hits: list[RightBrainHit] = field(default_factory=list)  # 右脑结构化 top-N
    scene_directive: str = ""          # 当前声学场景的回复风格建议
    current_scene: str = ""            # 当前场景 tag，如 "transit"
    timing: dict = None                 # {slot_filter, entity_narrow, rank, rb, total} 单位 ms

    # ── 两个脑各自检索到了什么（面向用户的读法）──────────────────────────────
    # hits / rb_hits 是带分数和元数据的结构化结果，下面两个是"直接能读能打印"的
    # 那一层，对应文档里的 result.result_leftbrain / result.result_rightbrain。

    @property
    def result_leftbrain(self) -> list[str]:
        """左脑检索到的事实（按相关度排好序）。"""
        return [h.text for h in self.hits]

    @property
    def result_rightbrain(self) -> list[str]:
        """右脑检索到的情感/人格上下文。"""
        return [h.content for h in self.rb_hits]


# 左脑那一整块的候选池构造/救回常量（_RESCUE_K / _POOL_MODE_ENV / _pool_mode /
# _STRICT_* 等）与 _search_mode 辅助函数随左脑块迁至 voicemem.leftbrain.brain
# （_search_mode 在本模块顶部导入回来，供 Search() 组装 SearchResult 时调用）。
# RightBrainHit / _rb_* 辅助函数与 _is_en_text 已随右脑块迁至
# voicemem.rightbrain.brain（本模块顶部导入回来，供 Search() 等继续直接调用）；
# AudioPerception 迁至 voicemem.utils.audio.perceiver。


# ── Orchestrator 类 ─────────────────────────────────────────────────────────────

class Orchestrator:
    """Left-brain + right-brain personal memory system（mem0 式编排器）。

    直接持有并构造三个自包含组件（``self._left`` / ``self._right`` /
    ``self._audio``），编排它们完成 Search/Ingest 等完整 pipeline。

    Parameters
    ----------
    api_key:
        OpenAI API key；传入即写进 ``OPENAI_API_KEY`` 环境变量。
    mode:
        ``left_brain_single`` / ``text_mode`` / ``multi_modal``，决定加载哪些 util、
        是否启用音频/情绪能力。
    memory_root:
        Memory storage directory.  Defaults to
        ``<当前工作目录>/voicemem_memory``（env ``VOICEMEM_MEMORY_ROOT`` 可覆盖）。
    user_id:
        Owner of all memories managed by this instance.
    base_url:
        OpenAI-compatible API base URL (e.g. a proxy).  Falls back to the
        ``OPENAI_BASE_URL`` environment variable.
    enable_scene / enable_music / enable_abnormal_sound / enable_voiceprint / enable_emotion:
        5 个能力开关。默认 ``None`` → 由 ``mode`` 推导（``multi_modal`` 全开、其余音频
        项关；情绪项在非 ``left_brain_single`` 下开）。显式传 True/False 覆盖 mode 推导。
    embedder / vector_store / classifier:
        ``embedding`` / ``memory_engine`` / ``slots`` 三个能力的**旧参数名**，等价，
        保留兼容。新代码统一用能力名。
    util_overrides:
        按能力名覆盖内置默认：``embedding`` / ``slots`` / ``entity`` / ``emotion`` /
        ``voiceprint`` / ``asr`` / ``vad`` / ``tts`` / ``memory_engine``（全表见
        ``voicemem/utils/defaults.py``）。值可以是工厂，也可以直接是造好的对象。
    """

    def __init__(
        self,
        api_key: str | None = None,
        mode: str = "text_mode",
        memory_root: Path | str | None = None,
        space: str | None = None,
        user_id: str = "voice_user",
        base_url: str | None = None,
        enable_scene: bool | None = None,
        enable_music: bool | None = None,
        enable_abnormal_sound: bool | None = None,
        enable_voiceprint: bool | None = None,
        enable_emotion: bool | None = None,
        embedder: Any = None,
        vector_store: Any = None,
        classifier: Any = None,
        **util_overrides,
    ) -> None:
        if mode not in _NEED:
            raise ValueError("mode 必须是 " + " / ".join(_NEED))
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        self.mode = mode
        # embedder / vector_store / classifier 是同三个能力的旧参数名，收进来一起
        # 归一（见 _ALIASES）：它们收对象、能力名那路收工厂，Utils.get 两种都认，
        # 所以现在只有一条路，不再需要两套代码各走各的。
        overrides = _canon({**util_overrides, "embedder": embedder,
                            "vector_store": vector_store, "classifier": classifier})
        overrides = {k: v for k, v in overrides.items() if v is not None}
        self.utils = Utils(mode, base_url, memory_root, overrides)

        audio = mode == "multi_modal"
        # 只有被用户覆盖的能力才注入组件；否则组件用自己的默认。
        pick = lambda n: self.utils.get(n) if n in overrides else None

        # 5 个音频能力开关：显式传值优先，否则由 mode 推导
        # （multi_modal 全开，其余音频项关；情绪项在非 left_brain_single 下开）。
        if enable_scene is None:          enable_scene = audio
        if enable_music is None:          enable_music = audio
        if enable_abnormal_sound is None: enable_abnormal_sound = audio
        if enable_voiceprint is None:     enable_voiceprint = audio
        if enable_emotion is None:        enable_emotion = mode != "left_brain_single"
        embedder     = pick("embedding")
        vector_store = pick("memory_engine")
        classifier   = pick("slots")

        self._vector_store = vector_store   # 注入的 memory engine（默认 None → mem0）
        # 默认落在**当前工作目录**下，不是包的安装位置。
        #
        # 之前默认是 <包目录>/results/voice_memory —— pip 装的人记忆会写进
        # site-packages/results/：升级包就没了、系统级 Python 往往只读、而且所有
        # 项目共用一份。数据该跟着项目走，不该跟着程序装在哪走（git/docker/npm
        # 都是这个逻辑）。VOICEMEM_MEMORY_ROOT 可覆盖。
        # 不给 memory_root 时按 space 落到 voicemem_memoryspace/<space>/，
        # 默认 space 叫 demo。见 utils/common/space.py。
        from voicemem.utils.common.space import MemorySpace
        if memory_root or os.environ.get("VOICEMEM_MEMORY_ROOT"):
            self._memory_root = Path(memory_root or os.environ["VOICEMEM_MEMORY_ROOT"])
            self._memory_root.mkdir(parents=True, exist_ok=True)
            self._space = None
        else:
            self._space = MemorySpace(space)
            self._memory_root = self._space.dir
        # 一个 space 一个 sqlite：原来九个库各开各的连接，拆开只会让「拷走一个
        # space」变成「别漏了哪个文件」。表名互不重叠。
        self._db_path = _space.db(self._memory_root)
        self._multi_modal = _space.mm(self._memory_root)
        _space.describe(self._memory_root, user_id=user_id, mode=mode)
        self._multi_modal.mkdir(parents=True, exist_ok=True)
        self._cognitive_db = self._db_path
        self._user_id = user_id
        self._base_url = resolve_base_url(base_url)
        # Official/default is OpenAI embeddings (OpenAILocalEmbedder, built
        # lazily in _get_repo() below); pass a different TextEmbedder-
        # conforming object here to use something else for the left-brain
        # store's raw-fact embedding (used for both ingest and Rank()'s
        # search-time ranking). openai_voice_demo uses this to swap in a
        # local model for speed -- see that demo's local_embedder.py.
        self._embedder = embedder
        # query→slots+entities 的分类器（Classify 用）。默认 None → 内置
        # QuerySlotClassifier（单次 LLM）。传一个 .classify(query)->QueryClassification
        # 的实现即可切成本地模型；可选的 .classify_child(...) 存在时才做子 slot 下钻。
        self._classifier = classifier

        # 5 个音频能力开关：由 mode 决定（multi_modal 全开，否则全关）。
        self._enable_scene = enable_scene
        self._enable_music = enable_music
        self._enable_abnormal_sound = enable_abnormal_sound
        self._enable_voiceprint = enable_voiceprint
        self._enable_emotion = enable_emotion

        self._cache: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._ingest_count = 0

        # 对话往返：只存用户那半边，"那就按你说的办"是悬空的。回复层每说完一句
        # 就 remember_reply() 登记，Ingest 取这轮的（左脑消歧）、Search 和右脑取
        # 上一轮的（情绪归因），所以留两轮就够。
        # 写在回复协程、读在 Search 线程池，读的一侧拷快照（见 last_agent_reply）。
        self._exchanges: deque[tuple[str, str]] = deque(maxlen=2)

        # ── 左脑组件（组合模式：自持左脑零件 + 显式注入跨域/运行时依赖）───────────
        # 左脑那一整块（slot 过滤/实体缩窄/时间扩候选/向量排序/查询分类/LLM 打标签/
        # slot→entity 图层/子图记账与 checkpoint/schema 描述刷新/冷记忆归档）搬进了
        # LeftBrain。它自持 5 个左脑侧懒加载单例（repo/extractor/dynamic_slot_store/
        # graph_entity_store/subgraph_manager，与宿主共享同一 _cache/_lock）；凡是要
        # 用到文本 embedding / LLM(JSON) / LLM(text) / 可注入分类器 / 会话追踪器这些
        # 跨域或运行时能力的地方，一律以 getter/函数引用在此显式注入（懒加载语义不变）。
        # 先于 _audio/_right 构造：本类的 _get_repo 等转发到 self._left，且 _audio/
        # _right 注入的 repo=self._get_repo 会经转发落到这里同一份左脑单例。
        self._left = LeftBrain(
            memory_root=self._memory_root,
            user_id=self._user_id,
            base_url=self._base_url,
            cognitive_db=self._cognitive_db,
            embedder=self._embedder,
            vector_store=self._vector_store,
            embed=self._embed_text,
            llm_json=self._llm_json,
            llm_text=self._llm_text,
            classifier=self._classifier,
            tracker=self._get_session_tracker,
            cache=self._cache,
            lock=self._lock,
        )

        # ── 音频感知组件（组合模式：自持音频零件 + 显式注入左脑依赖）───────────
        # 音频那一整块（场景/声纹/情绪/环境音/audiomem 标签/回放）搬进了
        # AudioPerceiver。它自持 10 个音频侧懒加载单例（env/clap/speaker/vp/
        # emotion/music/routine/place/trigger/audio_archive）连同各自缓存/锁与
        # 说话人绑定状态（_session_person_pin / _person_origin_session）；凡是要
        # 用到左脑存储 / 抽取器 / 声纹姓名映射 / 打标签 / 事实追加 / 语义排序这些
        # 非音频能力的地方，一律以 getter/函数引用在此显式注入（懒加载语义不变）。
        self._audio = AudioPerceiver(
            memory_root=self._memory_root,
            user_id=self._user_id,
            base_url=self._base_url,
            enable_scene=self._enable_scene,
            enable_music=self._enable_music,
            enable_abnormal_sound=self._enable_abnormal_sound,
            enable_voiceprint=self._enable_voiceprint,
            enable_emotion=self._enable_emotion,
            repo=self._get_repo,
            extractor=self._get_extractor,
            registry=self._get_registry,
            tag=self._tag_memories,
            extract_and_append=self._extract_and_append,
            rank=self.Rank,
            ingest_env=lambda: self.IngestEnv,
            cache=self._cache,
            lock=self._lock,
        )

        # ── 右脑组件（组合模式：自持右脑零件 + 显式注入跨域依赖）─────────────────
        # 右脑那一整块（heartnote 情感写入/内心OS/图层/检索/LLM清洁）搬进了
        # RightBrain。它自持 3 个右脑侧懒加载单例（rb_repo/rb_graph_store/
        # attribution_manager，与宿主共享同一 _cache/_lock）；凡是要用到文本
        # embedding / LLM(JSON) / LLM(text) / 会话追踪器 / 左脑仓库 / 内心OS 生成 /
        # 特质抽取这些非右脑本域能力的地方，一律以 getter/函数引用在此显式注入
        # （懒加载语义不变；generate_inner_os / extract_rb_traits 延迟解析以便测试 patch）。
        self._right = RightBrain(
            memory_root=self._memory_root,
            user_id=self._user_id,
            base_url=self._base_url,
            cognitive_db=self._cognitive_db,
            embed=self._embed_text,
            llm_json=self._llm_json,
            llm_text=self._llm_text,
            tracker=self._get_session_tracker,
            repo=self._get_repo,
            generate_inner_os=lambda text, emotion, entities, agent_reply="": (
                self._generate_inner_os(text, emotion, entities, agent_reply)),
            extract_rb_traits=lambda text, emotion: self._extract_rb_traits(text, emotion),
            cache=self._cache,
            lock=self._lock,
        )

        # 左脑/右脑对外句柄：直接指向真组件（它们已有 search/write 等方法）。
        self.left_brain = self._left
        self.right_brain = self._right

    # ── 懒加载单例 ──────────────────────────────────────────────────────────────

    def _get_repo(self):
        # 左脑单例已随左脑块搬进 LeftBrain 组件；转发以维持既有调用点/测试对
        # VoiceMem 实例的直接访问，以及 _audio/_right 注入的 repo=self._get_repo
        # （读写的是共享 _cache 里同一个 "repo"）。
        return self._left._get_repo()

    def _get_rb_repo(self):
        # 右脑单例已随右脑块搬进 RightBrain 组件；转发以维持既有调用点/测试对
        # VoiceMem 实例的直接访问（读写的是共享 _cache 里同一个 "rb_repo"）。
        return self._right._rb_repo()

    def _get_extractor(self):
        # 左脑单例已搬进 LeftBrain 组件；转发（共享 _cache 里同一个 "extractor"）。
        return self._left._get_extractor()

    def _get_registry(self):
        with self._lock:
            if "registry" not in self._cache:
                from voicemem.utils.common.voice_input import VoiceprintRegistry
                # 声纹注册表是声纹数据，按 space 规格进 multi_modal/
                self._cache["registry"] = VoiceprintRegistry(
                    _space.mm(self._memory_root, "voiceprint_registry.json"),
                    entity_resolver=self._person_entity_id,
                )
        return self._cache["registry"]

    def _person_entity_id(self, name: str) -> str:
        """人名 -> 认知图里 person 实体的 id；查不到返回 ""（只读，不建实体）。

        声纹认出「这是谁」和认知图记住「关于这个人的事」本是两套 id，这里把它们接上：
        接上之后 speaker_entity_map 才非空，search(speaker_filter=…) 才有边可走。
        """
        try:
            from voicemem.leftbrain.cognitive_graph.store import normalize_name
            store = self._get_repo()._cognitive_store
            for e in store.find_entities(self._user_id, name_norm=normalize_name(name)):
                # 注意取 .value：EntityType 虽是 str 枚举，3.11+ 的 str() 给的是
                # "EntityType.PERSON" 而不是 "person"。
                if getattr(e.entity_type, "value", e.entity_type) in ("person", "user"):
                    return e.id
        except Exception:
            pass
        return ""

    # ── audiomem：场景 + 声纹相关懒加载单例 ─────────────────────────────────────

    def _get_env_detector(self):
        return self._audio._env_detector()

    def _clap_memory_enabled(self) -> bool:
        # AST always supplies the immediate hint. Once a CLAP checkpoint is
        # configured, the 4s-segmented CLAP pass takes over the background-sound
        # description memory write; set VOICEMEM_ENVIRONMENT_MEMORY_BACKEND=ast
        # to opt back out.
        return (
            os.environ.get("VOICEMEM_ENVIRONMENT_MEMORY_BACKEND", "clap").lower() == "clap"
            and bool(os.environ.get("VOICEMEM_CLAP_CHECKPOINT"))
        )

    def _get_clap_env_detector(self):
        return self._audio._clap_env_detector()

    def _finish_clap_environment(self, *a, **k) -> None:
        return self._audio._finish_clap_environment(*a, **k)

    def _get_trigger_store(self):
        return self._audio._trigger_store()

    def _get_audio_archive(self):
        return self._audio._audio_archive()

    def _get_speaker_encoder(self):
        return self._audio._speaker_encoder()

    def _get_vp_store(self):
        return self._audio._vp_store()

    def _get_emotion_detector(self):
        return self._audio._emotion_detector()

    def _get_music_store(self):
        return self._audio._music_store()

    def _get_routine_store(self):
        return self._audio._routine_store()

    def _get_place_store(self):
        return self._audio._place_store()

    # 说话人绑定状态与声纹回收：状态和逻辑都随音频组件走，这里保留转发以维持
    # 既有调用点/测试对 VoiceMem 实例的直接访问（读写的是同一份底层 dict）。
    @property
    def _session_person_pin(self) -> dict[str, str]:
        return self._audio._session_person_pin

    @property
    def _person_origin_session(self) -> dict[str, str]:
        return self._audio._person_origin_session

    def _claimed_by_other_identity(self, *a, **k) -> bool:
        return self._audio._claimed_by_other_identity(*a, **k)

    def _reconcile_speaker_candidates(self, *a, **k) -> tuple[str, str]:
        return self._audio._reconcile_speaker_candidates(*a, **k)

    # ── audiomem：场景触发提醒 ───────────────────────────────────────────────────

    def CreateSceneTrigger(self, *a, **k) -> dict:
        return self._audio.CreateSceneTrigger(*a, **k)

    def GetOriginalAudio(self, *a, **k) -> dict:
        return self._audio.GetOriginalAudio(*a, **k)

    def TryPlayback(self, *a, **k) -> dict | None:
        return self._audio.TryPlayback(*a, **k)

    # ── Dynamic slot（子图机制涌现的新 slot） ──────────────────────────────────

    def _get_dynamic_slot_store(self):
        # 左脑单例已搬进 LeftBrain 组件；转发（共享 _cache 里同一个 "dynamic_slot_store"）。
        return self._left._get_dynamic_slot_store()

    def _get_dynamic_slots(self) -> list[tuple[str, str]]:
        """返回该用户已涌现的动态 slot [(name, description), ...]，转发到 LeftBrain。"""
        return self._left._get_dynamic_slots()

    # ── slot→entity 图层（左脑：挂在 SlotV2 下；右脑：5个感性slot） ─────────────

    def _get_graph_entity_store(self):
        # 左脑单例已搬进 LeftBrain 组件；转发（共享 _cache 里同一个 "graph_entity_store"）。
        return self._left._get_graph_entity_store()

    def _get_rb_graph_store(self):
        # 右脑单例已搬进 RightBrain 组件；转发（共享 _cache 里同一个 "rb_graph_store"）。
        return self._right._rb_graph_store()

    def _get_session_tracker(self):
        with self._lock:
            if "session_tracker" not in self._cache:
                from voicemem.utils.common.session_tracker import SessionTracker
                self._cache["session_tracker"] = SessionTracker(
                    _space.db(self._memory_root)
                )
        return self._cache["session_tracker"]

    def _get_subgraph_manager(self):
        # 左脑单例已搬进 LeftBrain 组件；转发（共享 _cache 里同一个 "subgraph_manager"）。
        return self._left._get_subgraph_manager()

    def _get_attribution_manager(self):
        # 右脑单例已搬进 RightBrain 组件；转发（共享 _cache 里同一个 "attribution_manager"）。
        return self._right._attribution_manager()

    def _extract_rb_traits(self, text: str, emotion: str) -> list[tuple[str, str]]:
        """LLM 从这句话里判断有没有透露"喜好与厌恶/表达风格/思维模式/应对方式"，
        有就提炼一个简短标签。返回 [(slot_name, label), ...]，可能是空列表。

        抽取那一步已经顺便算过了（见 leftbrain/merged_extraction.py），命中就直接
        用，省掉这次 LLM 往返；没命中（合并关掉、或模型没按格式给）才自己调。
        """
        import json as _json

        from voicemem.lang import is_zh as _is_zh, label_rule as _label_rule

        def _looks_cjk(t: str) -> bool:
            return any("\u4e00" <= ch <= "\u9fff" for ch in (t or ""))
        def _keep(items):
            """语言守卫：标签跟原话不同文种就丢掉。

            prompt 里已经写了"跟随说话人的语言"、示例也按语言换过了，但模型在
            temperature=0 下仍会时不时输出中文标签（实测两次里约一次）。存进去
            的后果比丢掉严重得多：这份画像每轮都拼进 system prompt，英文对话里
            会突然冒出中文；而丢掉只是这一轮少一条特质，下一轮还会再抽。
            跟 attribution_manager 精炼后语言变了就保留原句是同一个取舍。
            """
            want_cjk = _is_zh()
            out = []
            for slot, label in items:
                if _looks_cjk(label) != want_cjk:
                    print(f"[RBTrait] 语言不符，丢弃：{slot} ← {label}", flush=True)
                    continue
                out.append((slot, label))
            return out

        from voicemem.leftbrain import merged_extraction
        if merged_extraction.enabled():
            cached = merged_extraction.take_traits(text)
            if cached is not None:
                valid = {"喜好与厌恶", "表达风格", "思维模式", "应对方式", "情绪"}
                return _keep([(s, l) for s, l in cached if s in valid and l])

        # 中英两套 prompt，按这一轮说的话选。
        #
        # 曾经只有中文这一套：英文用户进来，事实是英文、特质却全是中文——而且
        # 模型是**照抄示例**（存下来的正好是「评审前会紧张」「讨厌被打断」这几个
        # 示例原文）。只加一句"跟随输入语言"的规则没用，示例的牵引力更强，
        # 所以整套都要换。slot 名保持中文：它是内部键，检索/配额/脑图都按它做键。
        if _is_zh():
            prompt = (
                f"用户说了这句话（当前情绪：{emotion or '未知'}）：\n「{text[:300]}」\n\n"
                "判断这句话有没有透露出以下几类主观信息，每类最多提炼一条简短标签"
                "（5-15 字）：\n"
                "- 喜好与厌恶：本能的喜欢/讨厌/偏好\n"
                "- 表达风格：说话/沟通方式和习惯\n"
                "- 思维模式：思考、判断、决策的习惯\n"
                "- 应对方式：面对压力/负面情绪时怎么自我调节\n"
                "- 情绪：什么情况下会有什么情绪。**必须写成一个规律，不是一个情绪词**：\n"
                "  「评审前会紧张」「被打断就烦」「一个人待着会踏实」，不要写「焦虑」「开心」。\n"
                "  这句话会成为脑图上一个节点的标题，光一个情绪词看不出是什么事。\n\n"
                "没有清晰体现的类别就不要输出。\n"
                "**标签的写法**：写成一句短短的规律，5-15 字，不要主语、不要句号：\n"
                "「讨厌被打断」「压力大时想被安抚」「先要结论」。\n"
                "不要写成「用户倾向于详细规划和结构化思考。」这种带主语的整句，也不要\n"
                "把原话或事实抄一遍。\n"
                f"{_label_rule()}\n"
                '只输出 JSON：{"items": [{"slot": "喜好与厌恶", "label": "讨厌被打断"}, ...]}'
                '（items 可以是空列表 []）'
            )
        else:
            prompt = (
                f"The user said this (current emotion: {emotion or 'unknown'}):\n"
                f"\"{text[:300]}\"\n\n"
                "Does it reveal any of these subjective things about the speaker? "
                "At most ONE short label per category (3-8 words):\n"
                "- 喜好与厌恶: gut likes / dislikes / preferences\n"
                "- 表达风格: habits of speaking and communicating\n"
                "- 思维模式: how they think, weigh things, decide\n"
                "- 应对方式: what they do to cope with stress or bad feelings\n"
                "- 情绪: WHEN they feel WHAT. **A pattern, never a bare feeling "
                "word**: \"tense before design reviews\", \"annoyed when "
                "interrupted\", \"calm when alone\" — NOT \"anxious\" / \"happy\". "
                "It becomes the title of a node on a graph; a bare word says nothing.\n\n"
                "Skip any category the utterance does not clearly show.\n"
                "**How to write a label**: a short pattern, no subject, no full stop:\n"
                "  good: hates being interrupted / wants comfort under stress / "
                "conclusion first\n"
                "  bad: The user tends to plan in detail. (a full sentence with a subject)\n"
                "  bad: I major in computer science (copying the utterance / a plain fact)\n"
                "The slot names above are internal keys — keep them exactly as written, "
                "in Chinese. Only the label follows the language rule below.\n"
                f"{_label_rule()}\n"
                'Output JSON only: {"items": [{"slot": "喜好与厌恶", '
                '"label": "hates being interrupted"}, ...]} (items may be [])'
            )
        raw = self._llm_json(prompt)
        if not raw:
            return []
        try:
            items = _json.loads(raw).get("items", [])
        except Exception:
            return []
        valid_slots = {"喜好与厌恶", "表达风格", "思维模式", "应对方式", "情绪"}
        result = []
        for it in items:
            slot = str(it.get("slot", "")).strip()
            label = str(it.get("label", "")).strip()
            if slot in valid_slots and label:
                result.append((slot, label))
        return _keep(result)

    def _embed_text(self, text: str) -> list[float]:
        """图层实体 / slot 锚点 / 右脑判断表用的 embedding。

        **注入了 embedder 就用注入的那个**——这里以前写死 OpenAI，绕过了
        ``VoiceMem(embedding=…)``，于是配了本地模型也只生效一半：记忆向量走本地，
        图层实体和判断表仍然发远程。同一个库里因此并存两种维度（384 / 1536），
        而下面那句"跟左脑共用一份缓存"也从来没兑现过——缓存按模型名分键，
        两条通道用不同模型时一次都命中不了。

        这条还在**查询热路径**上：右脑检索每轮都会调它
        （traits_store.search_scored）。实测本地 E5 查询 embedding 10ms、
        OpenAI 178ms，统一之后每轮省下这一跳，也少一个断网/限流的单点。

        没注入时保持原样（默认就是 OpenAI），所以默认配置的行为和已有向量都不变。
        """
        if self._embedder is not None:
            return self._embedder.embed_query_text(text) if hasattr(
                self._embedder, "embed_query_text") else self._embedder.embed_texts([text])[0]
        # 跟左脑共用一份缓存：这里要的实体（'坚果'/'素食主义者'/'用户'）左脑刚
        # embed 过一轮，一模一样的字符串没必要再发一次。
        from voicemem.utils.common import embed_cache
        model = resolve_model(role="embedding")
        return embed_cache.resolve(model, [text], self._embed_uncached)[0]

    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI
        client = OpenAI(
            api_key=resolve_api_key(),
            base_url=self._base_url,
            timeout=15.0,
        )
        _kw = {
            "model": resolve_model(role="embedding"),
            "input": texts,
            "encoding_format": "float",   # 部分兼容后端不支持 base64
        }
        if "openrouter" in str(resolve_base_url(self._base_url) or "").lower():
            _kw["extra_body"] = {"provider": {"order": ["OpenAI"], "allow_fallbacks": False}}
        resp = client.embeddings.create(**_kw)
        _exp = int(os.environ.get("VOICEMEM_EMBED_DIM", "1536"))
        if len(resp.data[0].embedding) != _exp:
            raise RuntimeError(f"embedding 维度 {len(resp.data[0].embedding)} != {_exp}，供应商被换掉了")
        data = resp.data
        if all(d.index is not None for d in data):
            data = sorted(data, key=lambda d: d.index)
        return [list(map(float, row.embedding)) for row in data]

    def _llm_json(self, prompt: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=resolve_api_key(),
                base_url=self._base_url,
                timeout=15.0,
            )
            resp = client.chat.completions.create(
                model=resolve_model(),
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=512,
            )
            from voicemem.utils.common.cost_log import log_usage
            log_usage("llm_json", resp.model, getattr(resp, "usage", None))
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[SplitMgr] LLM 失败: {e}")
            return ""

    def _llm_text(self, prompt: str, max_tokens: int = 300) -> str:
        """跟 _llm_json 不同：不强制 JSON 输出，给归因总结这类要纯文本的场景用。"""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=resolve_api_key(),
                base_url=self._base_url,
                timeout=15.0,
            )
            resp = client.chat.completions.create(
                model=resolve_model(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            from voicemem.utils.common.cost_log import log_usage
            log_usage("llm_text", resp.model, getattr(resp, "usage", None))
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[Attribution] LLM 失败: {e}")
            return ""

    # ── 左脑检索步骤（转发到 LeftBrain：SearchCogGraph/SearchData/Rank 等）─────

    def SearchCogGraph(self, *a, **k) -> tuple[set[str], QueryClassification]:
        """slot 过滤，转发到 LeftBrain.SearchCogGraph。"""
        return self._left.SearchCogGraph(*a, **k)

    def SearchData(self, *a, **k) -> set[str]:
        """实体缩窄，转发到 LeftBrain.SearchData。"""
        return self._left.SearchData(*a, **k)

    def _search_data_impl(self, *a, **k) -> tuple[set[str], list[str]]:
        """SearchData 真正实现（多返 activated_names），转发到 LeftBrain。"""
        return self._left._search_data_impl(*a, **k)

    def _widen_for_time_question(self, *a, **k) -> set[str]:
        """时间类问题扩候选，转发到 LeftBrain。"""
        return self._left._widen_for_time_question(*a, **k)

    def Rank(self, *a, **k) -> list[MemorySearchHit]:
        """向量相似度排序，转发到 LeftBrain.Rank。"""
        return self._left.Rank(*a, **k)

    # ── v5：LLM 打标签（转发到 LeftBrain）─────────────────────────────────────

    def _get_slot_base_embeddings(self, *a, **k) -> dict[str, list[float]]:
        return self._left._get_slot_base_embeddings(*a, **k)

    def _get_slot_dyn_embeddings(self, *a, **k) -> dict[str, list[float]]:
        return self._left._get_slot_dyn_embeddings(*a, **k)

    def _normalize_slot_name(self, *a, **k) -> str:
        return self._left._normalize_slot_name(*a, **k)

    def _llm_tag_memories(self, *a, **k) -> list[str]:
        return self._left._llm_tag_memories(*a, **k)

    # ── 查询分类（含动态 slot） ────────────────────────────────────────────────

    def Classify(self, *a, **k) -> QueryClassification:
        """LLM 分类 query → slots + entities，转发到 LeftBrain.Classify。"""
        return self._left.Classify(*a, **k)

    def PrimeSubgraphFromQuery(self, query: str, top_k: int = 10) -> dict:
        """Classify()+Search() 的便捷封装，返回这次检索记账的条数。

        子图判定分两层：Search() 每次检索完自动把查到的 memory_id 记进累积名单
        （便宜，无 LLM 调用，见 _record_subgraph_activation）；真正"建图→算密度→
        判断"那步很贵，只在 RunSubgraphCheckpoint() 里攒够一批后才做一次。

        记账是 Search() 自动做的副作用，直接调 Search() 效果一样；这个方法只为
        兼容"一次性 Classify+Search+拿记账条数"的调用方。
        """
        classification = self.Classify(query)
        result = self.Search(
            query=query, slots=classification.slots, entities=classification.entities,
            top_k=top_k,
        )
        return {"status": "recorded", "count": len({h.memory_id for h in result.hits})}

    def _record_subgraph_activation(self, *a, **k) -> None:
        """检索结果记账，转发到 LeftBrain._record_subgraph_activation。"""
        return self._left._record_subgraph_activation(*a, **k)

    def RunSubgraphCheckpoint(self, *a, **k) -> dict:
        """子图 checkpoint（建图→判断），转发到 LeftBrain.RunSubgraphCheckpoint。"""
        return self._left.RunSubgraphCheckpoint(*a, **k)

    def ArchiveColdMemories(self, *a, **k) -> dict:
        """冷记忆归档，转发到 LeftBrain.ArchiveColdMemories。"""
        return self._left.ArchiveColdMemories(*a, **k)

    # ── 完整 pipeline ──────────────────────────────────────────────────────────

    def Search(
        self,
        query: str,
        slots: list[str] | None = None,
        entities: list[str] | None = None,
        emotion: str | None = None,
        top_k: int = 5,
        scene_filter: str | None = None,
        speaker_filter: str | None = None,
    ) -> SearchResult:
        """完整检索 pipeline：SearchCogGraph → SearchData → Rank → 右脑 → 摘要。

        Parameters
        ----------
        query:
            用户语句，用于向量排序。
        slots:
            由语音模块提供的 slot 列表，如 ``["work"]``。空时降级为全库搜索。
        entities:
            由语音模块提供的实体列表，如 ``["阿里"]``。可为空。
        top_k:
            最多返回几条记忆。
        scene_filter:
            可选场景过滤（audiomem），如 ``"office"``。
        speaker_filter:
            可选说话人过滤（audiomem），传入 person_id。
        """
        import time
        import concurrent.futures

        # "下周""明天"这种相对时间词在向量里是死的：抽取把日期归一成了绝对日期
        # （"2026年8月26日（周三）"），而问句里一个绝对日期都没有，够不着。实测
        # "我下周有什么安排"三条下周日程一条都检索不到，换成"8月26号我要干嘛"
        # 三条全中——差别只在问法。这里就地把相对时间词展开成日期拼在后面，
        # 只影响拿去检索的这份文本，不改用户说的话，也不写进记忆。
        from voicemem.leftbrain.time_expand import expand_relative_dates
        query = expand_relative_dates(query)

        # 情景绑定记忆：调用方没显式传 scene_filter 时，先从 query 文本反推场景意图。
        if scene_filter is None:
            from voicemem.utils.audio.environment.scene_classifier import infer_scene_from_text
            inferred_scene = infer_scene_from_text(query)
            if inferred_scene is not None:
                scene_filter = inferred_scene.value

        # query 里也没提到场景时，用当前/最近检测到的场景做软优先（narrow 不出
        # 结果会自动还原，不会真把其它场景的记忆过滤没）。
        if scene_filter is None:
            try:
                current_scene = self._get_trigger_store().get_last_scene(self._user_id)
                # "unknown" 是"不知道在哪"，不是一个场景——拿它去过滤会把检索缩到
                # 恰好被标成 scene:unknown 的那几条上。实测：全库只有 7 条带这个
                # 标签，于是之后**每一次检索**都只在这 7 条里挑，问"我对什么过敏"
                # 返回的是马卡龙和草莓，而过敏那条（纯向量排第一，0.935）根本进不了候选。
                if current_scene and current_scene != "unknown":
                    scene_filter = current_scene
            except Exception:
                pass

        # ①②+摘要 左脑检索段（SearchCogGraph→SearchData→时间扩候选→相关槽摘要）
        # 整段抽进 LeftBrain.search；实体缩窄先于右脑跑完（右脑依赖左脑"已激活"的
        # 实体集合），t0/t1/t2 由组件回传，timing 语义不变。
        left = self._left.search(
            query, slots, entities, scene_filter, speaker_filter,
        )
        slot_mem_ids      = left["slot_mem_ids"]
        final_ids         = left["final_ids"]
        activated_names   = left["activated_names"]
        classification    = left["classification"]
        related_summaries = left["related_summaries"]
        t0, t1, t2 = left["t0"], left["t1"], left["t2"]

        # ③ 右脑（依赖 activated_names）与 Rank（向量排序，依赖 final_ids）并发执行——
        # 两者互不依赖对方输出，可以并发。
        rb_hits: list[RightBrainHit] = []
        rb_directive = ""
        rb_duration  = 0.0

        # 右脑检索段已抽进 RightBrain.search、向量排序已抽进 LeftBrain.rank；这里只
        # 保留 Rank ‖ 右脑 并发跑的 ThreadPoolExecutor 结构，两半都换成组件调用。
        # 用户这轮在回应 agent 上一句，右脑要看得到它（反应信号 + 背景锚点）
        agent_reply = self.last_agent_reply()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            rb_future = pool.submit(
                self._right.search, query, activated_names, emotion, top_k, agent_reply,
            )                                            # 右脑并发开跑

            hits = self._left.rank(query, final_ids, top_k, speaker_filter=speaker_filter)
            t3 = time.time()

            rb_hits, rb_directive = rb_future.result()   # 等右脑完成（通常已经跑完了）
            t4 = time.time()
            rb_duration = t4 - t2                        # 右脑总耗时（从 activated_names 就绪开始）

        # 低置信弃权提示：左右脑都没有针对这个问题的具体证据时（左脑无命中/实体
        # 不存在，右脑只剩泛化 fallback），明确告诉 responder"证据不足就说不知道"。
        _specific_rb = {"response_experience", "situation_pattern", "relation"}
        rb_specific = any(h.source in _specific_rb for h in rb_hits)
        left_weak = (not hits) or (not activated_names)
        if left_weak and not rb_specific:
            hint = (
                "Note: the memory system found no specific evidence for this query "
                "(only generic profile context). If the retrieved content does not "
                "actually answer the question, say you don't know instead of guessing."
                if _is_en_text(query) else
                "注意：记忆系统没有为该问题找到具体证据（只有泛化画像信息）。"
                "若检索内容不能真正回答问题，请直接说不知道，不要猜测。"
            )
            rb_directive = f"{rb_directive}\n{hint}".strip()

        # 场景自适应回复风格（audiomem）：读取用户当前场景，生成 directive
        scene_directive = ""
        current_scene = ""
        try:
            from voicemem.utils.audio.environment.scene_classifier import SceneTag, scene_to_response_directive
            last_scene = self._get_trigger_store().get_last_scene(self._user_id)
            if last_scene:
                current_scene = last_scene
                try:
                    scene_directive = scene_to_response_directive(SceneTag(last_scene))
                except ValueError:
                    pass
        except Exception:
            pass

        # 每次真实检索都自动记账（子图簇涌现需要），见 _record_subgraph_activation。
        self._record_subgraph_activation(hits)

        return SearchResult(
            hits=hits,
            classification=classification,
            related_summaries=related_summaries,
            slot_mem_ids=slot_mem_ids,
            final_candidate_ids=final_ids,
            search_mode=_search_mode(slot_mem_ids, final_ids),
            rb_directive=rb_directive,
            rb_hits=rb_hits,
            scene_directive=scene_directive,
            current_scene=current_scene,
            timing={
                "slot_filter_ms":    round((t1 - t0) * 1000, 1),
                "entity_narrow_ms":  round((t2 - t1) * 1000, 1),
                "rank_ms":           round((t3 - t2) * 1000, 1),
                "rb_ms":             round(rb_duration * 1000, 1),
                "total_ms":          round((t4 - t0) * 1000, 1),
            },
        )

    # ── 写入 ──────────────────────────────────────────────────────────────────

    def _get_user_name(self) -> str | None:
        """从左脑记忆中提取用户名字，转发到 LeftBrain（共享 _cache 里 "user_name"）。"""
        return self._left._get_user_name()

    @staticmethod
    def _is_english(text: str) -> bool:
        """True if text is predominantly English (ASCII letters dominate over CJK)."""
        cjk = sum(1 for c in text if "一" <= c <= "鿿" or "぀" <= c <= "ヿ")
        alpha = sum(1 for c in text if c.isalpha())
        return alpha > 0 and cjk / max(alpha, 1) < 0.3

    def _generate_inner_os(self, text: str, emotion: str, entities: list[str],
                           agent_reply: str = "") -> str:
        """用 LLM 把原句转成 AI 第三人称内心 OS 风格，带情绪标签；失败返回空串。

        ``agent_reply``：用户说这句之前 agent 说的那句。情绪归因离不开它——同一句
        "行吧"，接在共情后面和接在甩方案后面是两种情绪；不给上一句，模型只能凭
        用户这半边猜因果。
        """
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=resolve_api_key(),
                base_url=self._base_url,
                timeout=10.0,
            )
            user_name = self._get_user_name()
            entity_hint = f", involving: {', '.join(entities)}" if entities else ""
            is_chinese = self._is_english(text) is False and any("一" <= c <= "鿿" for c in text)
            pronoun = user_name if user_name else ("用户" if is_chinese else "they")
            if is_chinese:
                system_prompt = (
                    f"你是一个有共情能力的AI助手，用第三人称记录你对用户情绪状态的内心感受。"
                    f"根据用户说的话，写出你（AI）的内心反应——就像你悄悄感受到了TA的情绪并被打动。"
                    f"要求：第三人称（称呼用户为『{pronoun}』），口语化，温暖，15-25字，"
                    f"开头用【情绪词】格式标注情绪。只输出一句话，不加任何解释。"
                    f"示例：\n"
                    f"输入：今天被老板当众批评了，好委屈\n"
                    f"输出：【心疼】{pronoun}强撑着没崩，但被这样当众说，心里一定很难受。\n"
                    f"输入：最好的朋友要搬走了\n"
                    f"输出：【担心】{pronoun}要失去身边最近的人了——以后难过的时候找谁说呢。"
                )
            else:
                system_prompt = (
                    "You are an empathetic AI assistant recording your inner observations about the user's emotional state. "
                    "Based on what the user said, write your (the AI's) internal reaction — "
                    "as if you quietly sensed their emotion and were moved by it. "
                    f"Requirements: third person (refer to the user as '{pronoun}'), "
                    "conversational, warm, 15-25 words, start with [emotion word] in brackets. "
                    f"Examples:\n"
                    f"Input: Got yelled at by my boss today, emotion: sad\n"
                    f"Output: [heartache] {pronoun} is holding it together on the outside, but being called out like that must really sting.\n"
                    f"Input: My best friend is moving away, emotion: longing\n"
                    f"Output: [worried] {pronoun} is losing someone close — once they're gone, who do they call on a hard day?\n"
                    "Output only that one sentence, nothing else."
                )
            reply_line = (agent_reply or "").strip()
            if reply_line:
                prior = ("你（AI）上一句说的是" if is_chinese else "What you (the AI) just said")
                user_content = (f"{prior}: {reply_line[:200]}\n"
                                f"What the user said: {text}\nEmotion: {emotion}{entity_hint}")
            else:
                user_content = f"What the user said: {text}\nEmotion: {emotion}{entity_hint}"

            resp = client.chat.completions.create(
                model=resolve_model(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=80,
                temperature=0.7,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""

    def _detect_scene(self, *a, **k):
        return self._audio._detect_scene(*a, **k)

    def _detect_speaker(self, *a, **k):
        return self._audio._detect_speaker(*a, **k)

    def _bind_self_identity(self, *a, **k):
        return self._audio._bind_self_identity(*a, **k)

    def preprocess(self, *a, **k) -> "AudioPerception":
        """流式预处理（音频感知），转发到 AudioPerceiver.preprocess。"""
        return self._audio.preprocess(*a, **k)

    # ── agent 说过的话（对话的另外半边）────────────────────────────────────────

    def remember_reply(self, user_text: str, reply: str) -> None:
        """登记 agent 刚说了什么。``vm.reply()`` 说完会自动调（见 core.py）；
        回复不走它的（web demo 的 llm_stream / Realtime 事件流）自己调一次，
        或者在 ``ingest(text, agent_reply=...)`` 里给，那边会顺手登记。"""
        reply = (reply or "").strip()
        if not reply:
            return
        pair = ((user_text or "").strip(), reply)
        if self._exchanges and self._exchanges[-1] == pair:
            return                      # 同一轮别登记两次（reply() 已记过，ingest 又显式传了一遍）
        self._exchanges.append(pair)

    def last_agent_reply(self, before_text: str | None = None) -> str:
        """agent 最近说的那句。``before_text`` 给定时跳过对它的回复，返回它之前
        那句——用户正在回应的就是那句。"""
        want = (before_text or "").strip()
        for user_text, reply in reversed(list(self._exchanges)):
            if want and user_text == want:
                continue                # 这是对 before_text 的回复，不是它之前那句
            return reply
        return ""

    def _reply_to(self, text: str) -> str:
        """agent 对「这句用户话」的回复（本轮已经答过时才有），供左脑抽取消歧。"""
        want = (text or "").strip()
        for user_text, reply in reversed(list(self._exchanges)):
            if user_text == want:
                return reply
        return ""

    def Ingest(
        self,
        text: str,
        speaker: str = "Speaker 0",
        emotion: str = "",
        entities: list[str] | None = None,
        session_id: int | str | None = None,
        audio_path: str | None = None,
        observed_at: str | None = None,
        async_facts: bool = False,
        agent_reply: str | None = None,
    ) -> dict:
        """将一条语音输入存入记忆库。

        这是 voicemem 与语音层结合的主入口，内部是一条三步流水线::

            ① preprocess()      text (+可选 audio_path) → AudioPerception
                                （流式预处理：场景/声纹/情绪等全部声学分析）
            ② 组装 ctx          把感知结果 + text/时间戳打包
            ③ _finish_ingest()  事实抽取 + 左脑/右脑写入 + audiomem 打标签

        语音那边只需给出 ``text``（多说话人/情绪等结构化输入见
        ``voice_input.ingest_voice_input``），音频感知全部发生在 ①。

        Parameters
        ----------
        observed_at:
            这句话实际发生的时间（如回填历史对话时传真实日期 "2023-05-08" 或 ISO
            字符串）。不传就用当下时刻；回填历史数据必须显式传，否则时序推理和按
            时间排序会失真。
        agent_reply:
            agent 对这句话的回复。不传就自动取 ``vm.reply()`` 登记过的那句，所以
            "先回复、再存"的标准流程不用改；回复不走 ``vm.reply()`` 的显式给一句。
            左脑拿它给用户那句消歧，右脑拿**上一轮**的回复做情绪归因。

        Returns
        -------
        dict
            ``{facts_count, memory_ids, affect}``
        """
        import time

        ts = observed_at or time.strftime("%H:%M:%S")

        if agent_reply is None:
            agent_reply = self._reply_to(text)       # 这轮的回复，左脑消歧用
        else:
            self.remember_reply(text, agent_reply)   # 自己生成回复的，顺手登记
        prior_reply = self.last_agent_reply(before_text=text)   # 上一轮的，右脑归因用

        # ① 流式预处理：场景/声纹/情绪等全部声学分析都在这一步（见 preprocess）
        p = self.preprocess(text, speaker, emotion, session_id, audio_path)
        speaker          = p.speaker
        emotion          = p.emotion
        environment      = p.environment
        environment_hint = p.environment_hint
        scene_tag        = p.scene_tag
        scene_raw_labels = p.scene_raw_labels
        person_id        = p.person_id
        tune_result      = p.tune_result
        abnormal_hits    = p.abnormal_hits
        detection        = p.detection
        place_result     = None   # 由 _finish_ingest 的场景聚类阶段填充
        new_routine      = None   # 由 _finish_ingest 的规律检测阶段填充

        ctx = {
            "text": text, "speaker": speaker, "emotion": emotion, "entities": entities,
            "session_id": session_id, "audio_path": audio_path, "observed_at": observed_at,
            "ts": ts,
            # AST remains the immediate hint. When CLAP final-memory mode
            # is enabled, don't put that provisional text in the utterance memory.
            "environment": "" if self._clap_memory_enabled() else environment,
            "environment_hint": environment_hint,
            "scene_tag": scene_tag,
            "scene_raw_labels": scene_raw_labels, "person_id": person_id,
            "tune_result": tune_result, "abnormal_hits": abnormal_hits,
            "place_result": place_result, "new_routine": new_routine, "detection": detection,
            # 在这里定死，async_facts=True 的后台线程才不会被后续轮次串掉
            "agent_reply": agent_reply or "", "prior_agent_reply": prior_reply or "",
        }

        if self._clap_memory_enabled() and audio_path is not None:
            threading.Thread(
                target=self._finish_clap_environment,
                args=(audio_path, text, session_id, environment_hint),
                daemon=True,
            ).start()

        # async_facts=True：事实抽取 + 图谱写入（耗时的部分）扔进后台线程，
        # Ingest() 立刻带着已同步算完的 audiomem 字段返回。默认 False。
        if async_facts:
            def _bg() -> None:
                """后台入库。异常必须自己打出来。

                线程里抛出去的异常没有任何人 await，表现是这一轮**凭空消失**——
                日志里连"入库 0 条"都没有，库里也查不到，看着像 LLM 没抽出东西。
                实测六轮连说丢掉两轮就是这么丢的，查了很久才定位到。
                """
                try:
                    self._finish_ingest(ctx)
                except Exception as e:
                    import traceback
                    print(f"[ingest] 这一轮没能入库（{type(e).__name__}: {e}）\n"
                          f"{traceback.format_exc()}", flush=True)

            threading.Thread(target=_bg, daemon=True).start()
            return {
                "facts_count":         None,
                "memory_ids":          [],
                "affect":              None,
                "triggered_reminders": [],
                "proactive_memories":  [],
                "current_scene":       scene_tag or "",
                "environment_hint":    environment_hint,
                "speaker_id":          person_id or "",
                "recognized_tune":     (
                    {"tune_id": tune_result.tune_id, "action": tune_result.action,
                     "heard_count": tune_result.heard_count}
                    if tune_result is not None else None
                ),
                "abnormal_sounds":     [l for l, _ in abnormal_hits],
                "recognized_place":    (
                    {"place_id": place_result.place_id, "action": place_result.action,
                     "visit_count": place_result.visit_count,
                     "previous_visit_at": place_result.previous_visit_at}
                    if place_result is not None else None
                ),
                "familiar_place_prompt": None,
                "new_routine":         None,
            }

        return self._finish_ingest(ctx)

    def _tag_memories(self, memory_ids, tags) -> None:
        """给一批记忆写 memory_tags；tags=[(name, conf),...]。cog store 不支持就跳过。"""
        cog_store = self._get_repo()._cognitive_store
        if cog_store and hasattr(cog_store, "upsert_memory_tags"):
            for mid in memory_ids:
                cog_store.upsert_memory_tags(mid, self._user_id, tags)

    def _extract_and_append(self, messages, instructions, ts, extra_metadata):
        """合成消息 → 抽取原子事实 → 追加入库，返回新 memory_ids（抽不出则空）。
        audiomem 里 routine/music/abnormal/环境音 那几处合成记忆共用这条。"""
        extracted = self._get_extractor().extract(
            new_messages=messages, custom_instructions=instructions,
            observation_date=ts, current_date=ts,
        )
        if not extracted:
            return []
        return self._get_repo().append_extracted(
            extracted, user_id=self._user_id, extra_metadata=extra_metadata)

    def _finish_ingest(self, ctx: dict) -> dict:
        """Ingest() 里事实抽取 + 图谱写入（左脑/右脑）那部分，拆出来是为了让
        async_facts=True 时能扔进后台线程跑。"""
        text = ctx["text"]; speaker = ctx["speaker"]; emotion = ctx["emotion"]
        entities = ctx["entities"]; session_id = ctx["session_id"]
        audio_path = ctx["audio_path"]; observed_at = ctx["observed_at"]
        ts = ctx["ts"]; environment = ctx["environment"]; scene_tag = ctx["scene_tag"]
        scene_raw_labels = ctx["scene_raw_labels"]; person_id = ctx["person_id"]
        tune_result = ctx["tune_result"]; abnormal_hits = ctx["abnormal_hits"]
        place_result = ctx["place_result"]; new_routine = ctx["new_routine"]
        detection = ctx["detection"]
        environment_hint = ctx.get("environment_hint", "")
        agent_reply = ctx.get("agent_reply", "")
        prior_reply = ctx.get("prior_agent_reply", "")

        import uuid
        from voicemem.utils.common.voice_input import VoiceInput, VoiceContent

        vi = VoiceInput(
            id=f"utt_{uuid.uuid4().hex[:8]}",
            time_stamp={"begin": ts, "end": ts},
            slots=[],
            contents=[VoiceContent(
                sub_id="0", time_start=ts, time_end=ts,
                sentence=text, voiceprint_id=speaker, emotion=emotion,
            )],
            environment=environment,
            agent_reply=agent_reply,       # 作为 assistant 消息一起进事实抽取
        )

        # 左脑事实抽取 + 入库（ingest_voice_input）抽进 LeftBrain.ingest_facts；
        # registry 是音频侧声纹姓名映射（跨域），由本类编排时注入。
        result = self._left.ingest_facts(
            vi,
            registry=self._get_registry(),
            session_id=session_id,
            extra_metadata={"created_at": observed_at} if observed_at else None,
        )

        # 助手刚说的那句：**原样**存一条，不抽 fact。
        #
        # 为什么不抽：抽出来的是"助手推荐了清炒芦笋和椒盐香菇"这种——它记录的是
        # 助手做了什么，不是用户其人，而且措辞跟当前对话高度重合，下一个问题一来
        # 就得高分，把真正相关的记忆挤出 top-k。
        # 为什么还要存：这样"你之前跟我说过什么"才答得上来。检索默认把 role=
        # assistant 排除在外（见 mem0_backend_store.search），只有问题在问助手
        # 自己说过什么时才放进来（asks_about_assistant）。
        if agent_reply.strip():
            try:
                self._get_repo()._vector_store.add_text(
                    self._user_id, agent_reply.strip(), attributed_to="assistant",
                    metadata={"source": "agent_reply", "turn_id": vi.id,
                              **({"time_start": observed_at} if observed_at else {})},
                )
            except Exception as e:
                print(f"[ingest] 存助手原话失败：{e}", flush=True)

        # 这一轮抽不出任何事实，但它确实是"放了段音乐给我听"：直接存一条，不走抽取。
        # 抽取判断的是「这句话里有没有关于用户的事实」，而这一轮的价值不在话里，
        # 在那段音频。没有记忆行的话，tune_id 无处可挂，之后问「刚才那首歌帮我
        # 重播」什么都找不到。
        #
        # 两种算："声学认出是音乐"，或者"**人自己说了**这是音乐"。后者不能少：
        # 识别走声学相似度，手机外放、环境吵、片段太短都会漏，而「我给你听首歌」
        # 这句话本身比任何声学特征都确凿——偏偏它也抽不出事实（是个动作，不是
        # 关于用户的事实），两头落空，那段录音就彻底进不了回放的候选池。
        from voicemem.utils.audio.perceiver import said_music as _is_said_music
        _said_music = _is_said_music(text)
        _tune_id = getattr(tune_result, "tune_id", None) if tune_result else None

        # 「说完就放」是最自然的顺序，而它恰好被切成**两轮**：一轮录音从检测到
        # 人声开始、到 VAD 判定说完为止，所以「给你听一首歌啊」结束时这轮就存盘
        # 了，音乐落在下一轮。
        #
        # 下一轮如果只有声音没有话，那它就是刚才说的那首歌——哪怕声学没认出来
        # （外放、环境吵、片段短都会漏）。不认这一条的话，池子里只剩下他说话
        # 那一轮，回放放出来是用户自己的声音，听感就是"音乐被截断了"。
        # 纯声音轮：有录音，但一个字都没说。这本身就值得存——之后问「刚才那段
        # 声音」「那首歌」时，它是最可能的答案。是不是音乐不一定（也可能是环境音），
        # 所以只标 sound_only，tune: 留给"真认出来了 / 他说了 / 从上一轮继承"。
        from voicemem.stream import SOUND_ONLY_TEXT
        _sound_only = bool(audio_path) and (
            not (text or "").strip() or (text or "").strip() == SOUND_ONLY_TEXT)

        _expect = getattr(self, "_expect_music_until", 0.0)
        _inherited = _sound_only and time.monotonic() < _expect
        if _said_music:
            self._expect_music_until = time.monotonic() + EXPECT_MUSIC_S
        elif (text or "").strip():
            self._expect_music_until = 0.0      # 又说了别的，意图作废
        if _inherited:
            print("  [music] 上一轮说了要放音乐，这一轮只有声音 → 就是那首", flush=True)

        if not result.memory_ids and (_tune_id or _said_music or _inherited or _sound_only):
            try:
                heard = getattr(tune_result, "heard_count", 0) or 0
                again = "（之前也听过）" if heard > 1 else ""
                # 认出是音乐（或他自己说了）才叫"音乐"，否则如实说"一段声音"——
                # 环境音、噪音也会走到这里，写成音乐是在编。
                what = "音乐" if (_tune_id or _said_music or _inherited) else "声音"
                _content = f"用户放了一段{what}给我听{again}。"
                mid = self._get_repo()._vector_store.add_text(
                    self._user_id, _content,
                    metadata={"source": "sound_only", "turn_id": vi.id,
                              **({"tune_id": _tune_id} if _tune_id else {}),
                              **({"time_start": observed_at} if observed_at else {})},
                )
                if mid:
                    # 上面 add_text 只写了**向量库**。memory_tags 的 memory_id 有
                    # 外键指向 sqlite 的 memories 表，缺这一行的话紧接着写
                    # tune:/scene:/speaker: 标签会撞 FOREIGN KEY constraint failed，
                    # 而那个异常被下游 catch 成一行日志——表现是"音乐明明认出来了，
                    # 库里却查不到 tune 标签"，回放只能退回按时间和文本猜。
                    # 正常轮次不会遇到：它们的记忆是 append_extracted 写的，两边都进。
                    try:
                        from voicemem.leftbrain.cognitive_graph.slot_v2 import SlotV2
                        self._get_repo()._cognitive_store.upsert_memory_record(
                            self._user_id, mid, SlotV2.DAILY_LIFE, _content,
                        )
                        # 标出"这一轮的录音里**只有声音、没有说话**"。
                        #
                        # 一轮录音是从检测到人声开始、到 VAD 判定说完为止。所以
                        # 「给你听一首歌啊」和后面那段音乐是**两轮**：前一轮存的是
                        # 他自己那句话（背景里可能已经有音乐，于是也带 tune 标签），
                        # 后一轮才是音乐本身。回放时要挑后者——挑错了放出来是
                        # 用户自己的声音，听感就是"音乐被截断了"。
                        # sound_only：这一轮的录音里只有声音、没有说话。
                        # tune:*：这一轮是"音乐"——回放的候选池按 tune:% 查，
                        #   少了它这条就进不去池子。声学认出来是哪首时
                        #   perceiver 会再打一个真的 tune_id，不冲突；认不出来
                        #   就只有 unidentified，确实不知道是哪首，别假装知道。
                        tags = [("sound_only", 0.95)]
                        if _tune_id:
                            tags.append((f"tune:{_tune_id}", 0.9))
                        elif _said_music or _inherited:
                            # 确实不知道是哪首，别假装知道；但确定是"音乐"。
                            tags.append(("tune:unidentified", 0.9))
                        self._tag_memories([mid], tags)
                    except Exception as e:
                        print(f"[ingest] 音乐轮补写 memories 失败（标签会挂不上）：{e}",
                              flush=True)
                    result.memory_ids = list(result.memory_ids or []) + [mid]
            except Exception as e:
                print(f"[ingest] 存音乐轮失败：{e}", flush=True)

        # ── audiomem：场景/声纹标签写入 + 触发提醒 + 录音归档 + 主动推送 ─────────
        audiomem = self._write_audiomem_tags(
            result, scene_tag, scene_raw_labels, detection, audio_path,
            person_id, tune_result, abnormal_hits, ts, session_id, text,
        )
        triggered_reminders = audiomem["triggered_reminders"]
        proactive_memories = audiomem["proactive_memories"]
        familiar_place_prompt = audiomem["familiar_place_prompt"]
        place_result = audiomem["place_result"]
        new_routine = audiomem["new_routine"]

        self._write_left_brain(result, text)
        heartnote_id = self._write_right_brain(
            emotion, result, text, entities, observed_at, prior_reply)
        # 情绪归因：有必要才落一条回应经验。跟 heartnote 分开调——不该被
        # "有没有识别出情绪"挡住，text_mode 下 emotion 常为空。
        self._right.learn_from_reaction(
            text, emotion, entities, prior_reply,
            memory_id=(result.memory_ids[0] if result.memory_ids else None),
            observed_at=observed_at, heartnote_id=heartnote_id,
        )

        # 异步清洁：每多 50 条 heartnote 触发一次
        threading.Thread(target=self._check_and_cleanup, daemon=True).start()
        # 异步清洁：原声定期归档，每天最多跑一次，删除超过 30 天的 WAV 文件本体
        threading.Thread(target=self._check_and_cleanup_audio, daemon=True).start()

        # ── 短期/长期归因触发（攒够一批才跑 / session 边界）──────────────────
        turn_info = self._get_session_tracker().record_turn(self._user_id, session_id)

        # 短期归因不是抽取，是**巩固**：把某个实体名下已有的记忆重读一遍、重写
        # 一句实体描述。它读的是累积状态，跟刚说的这句话没关系。原来每轮都跑，
        # 实测一次 ingest 里它独占 7.0s / 18.4s——而且刚多一条记忆就把整个实体
        # 重新概括一遍，本来也没什么新东西可总结。
        # 改成按**新增记忆条数**触发：左脑事实 + 右脑 heartnote 加起来攒够
        # SHORT_TERM_MIN_MEMORIES 条才巩固一次。按条数而不是按实体数，是因为
        # 「有多少新东西可总结」取决于新记忆的量——同一个实体被碰十次也未必
        # 多出十条内容。没到阈值就继续攒（touch 是 INSERT OR IGNORE，同一条
        # 记忆不会重复计数）。
        try:
            tracker = self._get_session_tracker()
            for mid in list(getattr(result, "memory_ids", None) or []):
                tracker.touch(self._user_id, "rb_pending_memories", str(mid))
            if heartnote_id:
                tracker.touch(self._user_id, "rb_pending_memories", str(heartnote_id))
            n = tracker.count_touched(self._user_id, "rb_pending_memories")
            if n >= SHORT_TERM_MIN_MEMORIES or turn_info["session_changed"]:
                tracker.pop_touched(self._user_id, "rb_pending_memories")   # 计数清零
                touched = tracker.pop_touched(self._user_id, "rb_entity_short")
                if touched:
                    self._get_attribution_manager().run_short_term(self._user_id, touched)
        except Exception as e:
            print(f"[Attribution] 短期归因失败: {e}")

        if turn_info["session_changed"]:
            self._run_session_boundary_batch()

        return {
            "facts_count":         result.facts_count,
            "memory_ids":          result.memory_ids,
            "affect":              result.affect,
            "triggered_reminders": triggered_reminders,
            "proactive_memories":  proactive_memories,
            "current_scene":       scene_tag or "",
            "environment_hint":    environment_hint,
            "speaker_id":          person_id or "",
            "speaker_name":        (
                self._get_registry().display_name(person_id) if person_id else speaker
            ),
            "recognized_tune":     (
                {"tune_id": tune_result.tune_id, "action": tune_result.action,
                 "heard_count": tune_result.heard_count}
                if tune_result is not None else None
            ),
            "abnormal_sounds":     [l for l, _ in abnormal_hits],
            "recognized_place":    (
                {"place_id": place_result.place_id, "action": place_result.action,
                 "visit_count": place_result.visit_count,
                 "previous_visit_at": place_result.previous_visit_at}
                if place_result is not None else None
            ),
            "familiar_place_prompt": familiar_place_prompt,
            "new_routine":         new_routine,
        }

    def _write_audiomem_tags(self, *a, **k) -> dict:
        """audiomem 写入段，转发到 AudioPerceiver._write_audiomem_tags。"""
        return self._audio._write_audiomem_tags(*a, **k)

    def _write_left_brain(self, result, text) -> None:
        """左脑写入段（LLM 打 slot 标签 + slot→entity 图层），转发到 LeftBrain.write。"""
        return self._left.write(result, text)

    def _write_right_brain(self, emotion, result, text, entities, observed_at,
                           agent_reply: str = "") -> str | None:
        """右脑写入段，转发到 RightBrain.write。``agent_reply`` 是用户这句之前
        agent 说的那句（情绪归因的上下文）。返回这轮 heartnote 的 id。"""
        return self._right.write(emotion, result, text, entities, observed_at, agent_reply)

    def _run_session_boundary_batch(self) -> None:
        """session 边界批处理：左脑子图判定 + 右脑长期归因。

        左脑这部分把 session 里攒下的检索记账拿出来判断一次
        （RunSubgraphCheckpoint）；纯 ingest（无穿插检索）时天然是空操作。

        由 Ingest() 在检测到 session_id 变化时自动调用。session_changed 靠"看到
        下一个 session 的第一条 ingest"倒推，最后一个 session 没有下一条触发，
        所以 ingest 完之后调用方必须显式调一次 Flush() 补跑最后一个 session。
        """
        try:
            self.RunSubgraphCheckpoint()
        except Exception as e:
            print(f"[Subgraph] session边界判定失败: {e}")

        # schema 描述刷新：给本 session 新增过记忆的 slot 重写一句 ≤40 词的综合
        # 描述，检索时附进 prompt，提供单条事实给不出的跨记忆聚合信息。
        try:
            self._refresh_schema_descriptions()
        except Exception as e:
            print(f"[SchemaDesc] 刷新失败: {e}")

        try:
            touched_slots = self._get_session_tracker().pop_touched(self._user_id, "rb_slot_long")
            if touched_slots:
                self._get_attribution_manager().run_long_term(self._user_id, touched_slots)
        except Exception as e:
            print(f"[Attribution] 长期归因失败: {e}")

    def _refresh_schema_descriptions(self) -> None:
        """给记忆数有变化的 slot 重写一句综合描述，转发到 LeftBrain。"""
        return self._left._refresh_schema_descriptions()

    def Flush(self) -> None:
        """对话/会话正式结束时调用一次，补跑最后一个 session 漏掉的批处理
        （子图判定 + 右脑长期归因，见 _run_session_boundary_batch 的说明）。
        幂等：没有新 touched refs 时是空操作。
        """
        self._run_session_boundary_batch()
        try:
            touched = self._get_session_tracker().pop_touched(self._user_id, "rb_entity_short")
            if touched:
                self._get_attribution_manager().run_short_term(self._user_id, touched)
        except Exception as e:
            print(f"[Attribution] 短期归因失败: {e}")

    def IngestEnv(self, *a, **k) -> dict:
        """将一段环境音事件存入记忆库，转发到 AudioPerceiver.IngestEnv。"""
        return self._audio.IngestEnv(*a, **k)

    def _check_and_cleanup(self) -> None:
        """每增加 50 条 heartnote 触发一次右脑清洁，转发到 RightBrain.check_and_cleanup。"""
        return self._right.check_and_cleanup()

    def _check_and_cleanup_audio(self, retention_days: int = 30) -> None:
        """原声定期归档：每天最多跑一次，删除超过保留期的 WAV 文件本体。
        按时间触发，用单独的状态文件节流，避免每次 Ingest 都扫一遍 DB。
        """
        try:
            import json as _json
            from datetime import datetime, timezone
            last_run = _space.kv_get(self._memory_root, "audio_cleanup_last_run", "")
            now = datetime.now(timezone.utc)
            if last_run:
                try:
                    elapsed_hours = (now - datetime.fromisoformat(last_run)).total_seconds() / 3600
                except ValueError:
                    elapsed_hours = 999
            else:
                elapsed_hours = 999

            if elapsed_hours < 24:
                return

            _space.kv_set(self._memory_root, "audio_cleanup_last_run", now.isoformat())
            self._get_audio_archive().cleanup_expired(retention_days=retention_days)
        except Exception as e:
            print(f"[Cleanup] audio check error: {e}")

    def _run_cleanup(self) -> None:
        """用 LLM 清洁右脑 heartnote，转发到 RightBrain.run_cleanup。"""
        return self._right.run_cleanup()


__all__ = ["Orchestrator", "SearchResult", "Utils"]
