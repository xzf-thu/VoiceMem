"""音频感知组件 AudioPerceiver。

从 VoiceMem 上帝类里抽出来的**音频感知那一整块**——声学场景 / 背景音乐 /
异常环境音 / 说话人声纹 / 自我介绍绑定 / 韵律情绪，以及围绕它们的场景触发提醒、
原声归档回放、环境音入库、audiomem 标签写入。

参考 mem0 的组合模式：
  * **组件自持零件**——10 个音频侧懒加载单例（env/CLAP/speaker/vp/emotion/
    music/routine/place/trigger/audio_archive）连同它们自己的缓存和锁，都在
    这个组件内部，engine 不再持有。
  * **依赖显式注入**——凡是需要用到"左脑存储 / 抽取器 / 声纹姓名映射 / 打标签 /
    事实追加 / 语义排序"这些**非音频**能力的地方，一律在 __init__ 里以 getter/
    函数引用注入（懒加载语义保持不变），组件内部通过 self._dep() 调用。

logic 一字不改：方法体原样搬运，只改"怎么拿依赖"。
"""

from __future__ import annotations

from voicemem.utils.common import space as _space

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: 短于这个秒数的一轮不做声纹识别——见 _detect_speaker 的说明。
SPEAKER_MIN_S = float(os.environ.get("VOICEMEM_SPEAKER_MIN_S", "2.5"))


# ── 结果容器 ───────────────────────────────────────────────────────────────────

@dataclass
class AudioPerception:
    """preprocess() 的结构化产物：一段音频跑完所有声学感知模块后的汇总。
    speaker/emotion 可能被感知结果覆盖（声纹 person_id、韵律情绪），其余字段
    供 Ingest 组装 ctx 与写 audiomem 标签使用。"""
    speaker: str
    emotion: str
    environment: str = ""              # AST/CLAP 背景音描述
    environment_hint: str = ""         # 即时场景提示（始终来自 AST）
    scene_tag: str | None = None       # 归类后的场景 tag，如 "transit"
    scene_raw_labels: list[str] = field(default_factory=list)
    person_id: str | None = None       # 声纹识别出的说话人 id
    tune_result: Any = None            # 背景音乐/哼唱识别结果
    abnormal_hits: list = field(default_factory=list)  # 异常环境音
    detection: dict = field(default_factory=dict)      # AST 原始一次推理结果


# ── AudioPerceiver 组件 ────────────────────────────────────────────────────────

# 场景切换时用于主动检索的查询词——选取该场景最常关联的记忆主题（audiomem）
_SCENE_RECALL_QUERY: dict[str, str] = {
    "office":  "工作任务截止日期会议",
    "transit": "通勤路上待办事项",
    "home":    "家里要做的事情购物清单",
    "café":    "灵感想法头脑风暴",
    "meeting": "会议议程项目进展",
    "outdoor": "运动健康目标",
    "quiet":   "学习计划专注任务",
}

# "回放原声"意图关键词：命中任意一个就认为用户在要求听回放，整句直接拿去语义搜索。
_PLAYBACK_PATTERNS = ["回放", "播放", "放一下", "听听", "重新放", "再放一遍"]


#: 用户嘴上说了"我要放段音乐给你听"的说法。声学识别会漏（外放、环境吵、片段短），
#: 但人明说了就没什么可怀疑的——见 _write_audiomem_tags 里 tune:unidentified 那段。
#:
#: 只认**用户是提供方**的说法。「重播第一首歌」「帮我放那首歌」是在**要**音乐，
#: 那一轮录的是他的提问，不是歌；早先的正则宽到把它也算上，结果提问轮进了回放的
#: 候选池、还是最新的一条，一问就把用户自己刚说的话放回去。
_ASKING_PLAYBACK = re.compile(
    r"重播|再放一遍|再听一遍|回放"
    r"|(你|您)(能|可以|帮|给)"
    r"|帮我(放|播|找|重|再)"
    r"|给我(放|播|听|重)"
)
_SAID_MUSIC = re.compile(
    r"给你听|放给你听|给你放|给你来(一)?(段|首)"
    r"|我(来|想|要|先)?(给你)?(放|播|哼|唱)(一)?(段|首|个)?\s*(音乐|歌|曲子|旋律|钢琴|吉他)"
    r"|(这|那)(是|首|段)?\s*(音乐|歌|曲子|旋律)\s*(好听|不错|怎么样|你听)"
    r"|哼(一)?(段|首)给你"
)


def said_music(text: str) -> bool:
    """这句话是不是"我要放段音乐给你听"。在**要**回放的那句上一律为假。"""
    t = text or ""
    return bool(_SAID_MUSIC.search(t)) and not _ASKING_PLAYBACK.search(t)


class AudioPerceiver:
    """自持零件 + 依赖显式注入的音频感知组件。

    构造参数分两类：

    组件自持的**运行参数**（音频侧行为开关与路径/身份）::

        memory_root, user_id, base_url,
        enable_scene, enable_music, enable_abnormal_sound,
        enable_voiceprint, enable_emotion

    显式**注入的非音频依赖**（全部以 getter/函数引用传入，懒加载语义不变）::

        repo               -> self._get_repo         左脑仓库（cognitive_store 等）
        extractor          -> self._get_extractor    事实抽取器
        registry           -> self._get_registry     声纹姓名映射
        tag                -> self._tag_memories      给记忆写 memory_tags
        extract_and_append -> self._extract_and_append 合成消息→抽取→入库
        rank               -> self.Rank               语义排序（回放/主动召回用）

    音频侧的 10 个懒加载单例（env/clap/speaker/vp/emotion/music/routine/place/
    trigger/audio_archive）连同缓存与锁由本组件自持，见下方 self._*_detector 等。

    说话人绑定状态（_session_person_pin / _person_origin_session）也随音频组件走。
    """

    def __init__(
        self,
        *,
        memory_root: Path,
        user_id: str,
        base_url: str | None,
        enable_scene: bool,
        enable_music: bool,
        enable_abnormal_sound: bool,
        enable_voiceprint: bool,
        enable_emotion: bool,
        repo: Callable[[], Any],
        extractor: Callable[[], Any],
        registry: Callable[[], Any],
        tag: Callable[..., Any],
        extract_and_append: Callable[..., Any],
        rank: Callable[..., Any],
        ingest_env: Callable[[], Callable[..., Any]],
        cache: dict[str, Any] | None = None,
        lock: Any = None,
    ) -> None:
        # ── 运行参数 ──
        self._memory_root = memory_root
        self._user_id = user_id
        self._base_url = base_url
        self._enable_scene = enable_scene
        self._enable_music = enable_music
        self._enable_abnormal_sound = enable_abnormal_sound
        self._enable_voiceprint = enable_voiceprint
        self._enable_emotion = enable_emotion

        # ── 注入的非音频依赖（getter/函数引用）──
        self._repo = repo
        self._extractor = extractor
        self._registry = registry
        self._tag = tag
        self._extract_and_append_fn = extract_and_append
        self._rank = rank
        # ingest_env 是"取当前 IngestEnv 入口"的 getter（call 时解析），
        # 让 _finish_clap_environment 走宿主暴露的入口，patch 宿主即可拦截。
        self._ingest_env_getter = ingest_env

        # ── 组件自持的音频零件缓存 ──
        # 允许宿主共享同一个 cache/lock（音频侧懒加载单例与宿主 _get_* 落在同一
        # 字典，既存调用点/测试对宿主 _cache 的直接读写与本组件保持一致视图）。
        self._cache: dict[str, Any] = cache if cache is not None else {}
        self._lock = lock if lock is not None else threading.Lock()

        # 说话人绑定状态（原 engine 实例状态，随音频组件走）
        # session_id → person_id：session 里一旦靠自我介绍确认过是谁，后续同
        # session 的每一句都优先信这个人，而非逐句按纯声纹分数投票。
        self._session_person_pin: dict[str, str] = {}
        # person_id → session_id：这个 id 第一次被创建时所在的 session。跟
        # _session_person_pin 结合，是 _reconcile_speaker_candidates 里除声学
        # cross_score 之外的第二个独立信号：出生 session 后来被钉死成另一个人的
        # id，大概率只是那人某句噪声话的误判碎片，不该被合并进声学匹配的第三方身份。
        self._person_origin_session: dict[str, str] = {}

    # ── audiomem：场景 + 声纹相关懒加载单例 ─────────────────────────────────────

    def _env_detector(self):
        with self._lock:
            if "env_detector" not in self._cache:
                from voicemem.utils.audio.environment.environment_detector_ast import ASTEnvironmentDetector
                self._cache["env_detector"] = ASTEnvironmentDetector()
        return self._cache["env_detector"]

    def _clap_env_detector(self):
        with self._lock:
            if "clap_env_detector" not in self._cache:
                from voicemem.utils.audio.environment.environment_detector_clap import CLAPEnvironmentDetector
                self._cache["clap_env_detector"] = CLAPEnvironmentDetector(
                    checkpoint=os.environ["VOICEMEM_CLAP_CHECKPOINT"]
                )
        return self._cache["clap_env_detector"]

    def _finish_clap_environment(self, audio_path, text, session_id, environment_hint="") -> None:
        """后台用长窗口 CLAP 复核环境音，并写入独立环境记忆。

        CLAP 的候选词表是固定集合，真实环境音类别不在表里时 pairs 会是空的；
        此时退回用 AST 的 hint 兜底（CLAP 模式已把 ctx["environment"] 清空）。
        """
        try:
            _ingest_env = self._ingest_env_getter()
            detection = self._clap_env_detector().detect_full(Path(audio_path))
            pairs = detection.get("pairs") or []
            if not pairs:
                if environment_hint:
                    _ingest_env(
                        audio_path,
                        recent_context=[{"role": "user", "content": text}] if text else None,
                        session_id=session_id,
                        environment_override=environment_hint,
                    )
                return
            env_str = "background sounds: " + ", ".join(
                f"{label}({score:.2f})" for label, score in pairs
            )
            _ingest_env(
                audio_path,
                recent_context=[{"role": "user", "content": text}] if text else None,
                session_id=session_id,
                environment_override=env_str,
            )
        except Exception as exc:
            print(f"  [clap-env] background write skipped: {exc}", flush=True)

    def _trigger_store(self):
        with self._lock:
            if "trigger_store" not in self._cache:
                from voicemem.utils.audio.environment.scene_trigger import SceneTriggerStore
                self._cache["trigger_store"] = SceneTriggerStore(
                    _space.db(self._memory_root)
                )
        return self._cache["trigger_store"]

    def _audio_archive(self):
        with self._lock:
            if "audio_archive" not in self._cache:
                from voicemem.utils.audio.audio_archive import AudioArchive
                self._cache["audio_archive"] = AudioArchive(
                    _space.db(self._memory_root)
                )
        return self._cache["audio_archive"]

    def _speaker_encoder(self):
        with self._lock:
            if "speaker_encoder" not in self._cache:
                from voicemem.utils.audio.voiceprint.speaker_encoder import SpeakerEncoder
                self._cache["speaker_encoder"] = SpeakerEncoder()
        return self._cache["speaker_encoder"]

    def _vp_store(self):
        with self._lock:
            if "vp_store" not in self._cache:
                from voicemem.utils.audio.voiceprint.voiceprint_store import VoiceprintStore
                from voicemem.utils.common.voice_config import VoiceStoreConfig
                voice_cfg = VoiceStoreConfig.from_env()
                self._cache["vp_store"] = VoiceprintStore(
                    _space.mm(self._memory_root, "voiceprints"),
                    match_threshold=voice_cfg.match_threshold,
                    candidate_threshold=voice_cfg.candidate_threshold,
                    merge_threshold=voice_cfg.merge_threshold,
                )
        return self._cache["vp_store"]

    def _claimed_by_other_identity(self, candidate_pid: str, other_pid: str) -> bool:
        """``candidate_pid`` 的出生 session 是否已被自我介绍确认成一个跟这次拟
        合并的两个 id（``candidate_pid`` 和 ``other_pid``）都不沾边的第三方身份。

        如果是，``candidate_pid`` 大概率是那个第三方一句噪声话的误判碎片，不该
        被并进 ``other_pid``。出生 session 钉死成 ``other_pid`` 自己属于正常情形，不拦。
        """
        origin_session = self._person_origin_session.get(candidate_pid)
        if origin_session is None:
            return False
        pinned = self._session_person_pin.get(origin_session)
        return pinned is not None and pinned not in (candidate_pid, other_pid)

    def _reconcile_speaker_candidates(self, person_id: str, speaker: str) -> tuple[str, str]:
        """在一次真实 match 之后，核对是否有候选声纹现在能确认并回。

        只处理声学层面：``VoiceprintStore`` 不认姓名，判断"两边攒起来的画像
        是否真的是同一个人声"；姓名冲突守卫在这里做——已经被明确报过不同姓名
        的两个 person_id，绝不因为声学分数够高就强行合并。返回可能被更新过的
        ``(person_id, speaker)``（如果当前这条正好是被合并掉的那个）。
        """
        vp_store = self._vp_store()
        registry = self._registry()
        for _cid, absorb_pid, into_pid, cross in vp_store.find_resolvable_candidates(person_id):
            absorb_name = registry.display_name(absorb_pid)
            into_name = registry.display_name(into_pid)
            absorb_named = absorb_name != absorb_pid
            into_named = into_name != into_pid
            if absorb_named and into_named and absorb_name != into_name:
                continue
            # 声学 cross_score 之外的第二独立信号：任意一侧的出生 session 若已被
            # 自我介绍钉死给第三方，说明这个 id 另有主人，不能因声学分数够高就合并。
            if self._claimed_by_other_identity(absorb_pid, into_pid) or \
                    self._claimed_by_other_identity(into_pid, absorb_pid):
                continue
            vp_store.merge_persons(absorb_pid, into_pid)
            if absorb_named and not into_named:
                registry.bind(into_pid, name=absorb_name)
            # merge_persons() 只合并声纹画像，不碰 memory_tags；把旧
            # speaker:{absorb_pid} 标签回填成 into_pid，否则检索按人过滤会漏。
            try:
                cog_store = self._repo()._cognitive_store
                if cog_store and hasattr(cog_store, "rename_tag_value"):
                    cog_store.rename_tag_value(
                        self._user_id, f"speaker:{absorb_pid}", f"speaker:{into_pid}"
                    )
            except Exception as _e:
                print(f"  [speaker_identity] 标签回填失败: {_e}", flush=True)
            if person_id == absorb_pid:
                person_id = into_pid
                if speaker == absorb_pid:
                    speaker = into_pid
        return person_id, speaker

    def _emotion_detector(self):
        with self._lock:
            if "emotion_detector" not in self._cache:
                # 韵律 VAD + 负面显著轮次 Qwen2.5-Omni 归因；接口 detect(audio_path)->str。
                from voicemem.utils.audio.emotion.paper_emotion_detector import PaperAlignedEmotionDetector
                self._cache["emotion_detector"] = PaperAlignedEmotionDetector()
        return self._cache["emotion_detector"]

    def _music_store(self):
        with self._lock:
            if "music_store" not in self._cache:
                from voicemem.utils.audio.environment.music_memory import MusicMemoryStore
                self._cache["music_store"] = MusicMemoryStore(
                    _space.mm(self._memory_root, "music_profiles")
                )
        return self._cache["music_store"]

    def _routine_store(self):
        with self._lock:
            if "routine_store" not in self._cache:
                from voicemem.utils.audio.environment.routine_memory import RoutineStore
                self._cache["routine_store"] = RoutineStore(
                    _space.db(self._memory_root)
                )
        return self._cache["routine_store"]

    def _place_store(self):
        with self._lock:
            if "place_store" not in self._cache:
                from voicemem.utils.audio.environment.place_memory import PlaceMemoryStore
                self._cache["place_store"] = PlaceMemoryStore(
                    _space.mm(self._memory_root, "place_profiles")
                )
        return self._cache["place_store"]

    # ── audiomem：场景触发提醒 ───────────────────────────────────────────────────

    def CreateSceneTrigger(self, text: str) -> dict:
        """从用户语句中解析场景触发意图并存储提醒。

        Parameters
        ----------
        text:
            用户语音转写文本，如"到公交上提醒我打电话给妈妈"。

        Returns
        -------
        dict
            ``{created: bool, scene: str, message: str}``
        """
        from voicemem.utils.audio.environment.scene_trigger import parse_trigger_intent
        scene, message, required_label = parse_trigger_intent(text)
        if scene is None:
            return {"created": False, "scene": "", "message": ""}

        store = self._trigger_store()
        trigger = store.create(self._user_id, scene.value, message, required_label=required_label)
        return {"created": True, "scene": scene.value, "message": message, "id": trigger.id}

    def GetOriginalAudio(self, memory_id: str) -> dict:
        """给定 memory_id，返回可用于"回放确认"的原始录音信息。

        原声可能已经过了保留期被清理（见 audio_archive.cleanup_expired），
        所以除了路径，还要显式校验文件是否仍然存在。

        Returns
        -------
        dict
            ``{found: bool, audio_path: str|None}``——found=False 时说明从未
            归档过，或者原声已经超过保留期被清理掉了。
        """
        path = self._audio_archive().get_audio_path(memory_id, self._user_id)
        if not path:
            return {"found": False, "audio_path": None}
        p = Path(path)
        if not p.exists():
            return {"found": False, "audio_path": None}
        return {"found": True, "audio_path": str(p)}

    def TryPlayback(self, text: str) -> dict | None:
        """检测这句话是不是在要求"回放原声"，命中就找最相关的记忆并取出原始录音。

        跟 GetOriginalAudio(memory_id) 不同——那个要求调用方已经知道具体是
        哪条记忆；这个是从"回放一下当时讨论价格的原声"这种自然语言里，先
        用语义搜索猜出用户说的是哪条记忆，再去查它的原声。

        Returns
        -------
        dict | None
            没有回放意图、搜不到相关记忆、或者原声已经过了保留期，返回 None；
            命中时返回 ``{"memory_id", "memory_text", "audio_path"}``。
        """
        if not any(p in text for p in _PLAYBACK_PATTERNS):
            return None
        hits = self._rank(text, set(), top_k=3)
        for h in hits:
            audio = self.GetOriginalAudio(h.memory_id)
            if audio["found"]:
                return {
                    "memory_id": h.memory_id,
                    "memory_text": h.text,
                    "audio_path": audio["audio_path"],
                }
        return None

    # ── 声学感知：场景 / 声纹 / 自我介绍绑定 ─────────────────────────────────────

    def _detect_scene(self, _apath):
        """声学场景检测（AST）：一次 detect_full() 推理拿到场景标签对、音乐/哼唱、
        异常环境音、原始 embedding；三个开关任一开就跑，后处理各自按开关判断。
        返回 (environment, environment_hint, scene_tag, scene_raw_labels,
        tune_result, abnormal_hits, detection)。"""
        environment = ""
        environment_hint = ""
        scene_tag: str | None = None
        scene_raw_labels: list[str] = []
        tune_result = None
        abnormal_hits: list[tuple[str, float]] = []
        detection = {"pairs": [], "music": None, "abnormal": [], "embedding": None}
        try:
            from voicemem.utils.audio.environment.scene_classifier import classify_scene
            detector = self._env_detector()
            detection = detector.detect_full(_apath)

            if self._enable_scene:
                pairs = detection["pairs"]
                if pairs:
                    parts = ", ".join(f"{l}({s:.2f})" for l, s in pairs)
                    environment = f"background sounds: {parts}"
                    environment_hint = environment
                    scene_result = classify_scene(pairs)
                    scene_tag = scene_result.tag.value
                    scene_raw_labels = [l for l, _ in scene_result.raw_matches]
        except Exception as _e:
            print(f"  [env] detection skipped: {_e}", flush=True)

        # ── 背景音乐/哼唱识别记忆（AST embedding）───────────────────
        if self._enable_music:
            try:
                music = detection.get("music")
                if music is not None:
                    tune_result = self._music_store().identify(
                        music["embedding"], labels=[l for l, _ in music["labels"]]
                    )
            except Exception as _e:
                print(f"  [music] detection skipped: {_e}", flush=True)

        # ── 异常环境音记忆（破碎声/警报/尖叫）──────────────────────
        if self._enable_abnormal_sound:
            try:
                abnormal_hits = detection.get("abnormal") or []
            except Exception as _e:
                print(f"  [abnormal] detection skipped: {_e}", flush=True)

        return (environment, environment_hint, scene_tag, scene_raw_labels,
                tune_result, abnormal_hits, detection)

    @staticmethod
    def _long_enough(_apath) -> bool:
        """这段音频够不够长到值得认人。读不出时长就放行（不因为一个探测挡住主流程）。"""
        try:
            import soundfile as sf
            return sf.info(str(_apath)).duration >= SPEAKER_MIN_S
        except Exception:
            return True

    def _detect_speaker(self, speaker, session_id, text, _apath):
        """声纹识别（3D-Speaker ERes2Net）：算声纹向量、识别 person_id、必要时回收
        候选声纹。返回 (person_id, speaker, stable_voiceprint, vec)。

        太短的音频直接不认人。1-2 秒只够算一个窗口（见 campplus_worker 的
        VOICEMEM_SPEAKER_WINDOW=3.0），向量噪声很大，分数常落在 candidate 区间
        （0.40-0.50），而 identify() 的 candidate 分支每次都 fork 一个新 person——
        同一个人于是被拆成一堆 person_*（实测一个 demo 库里 12 个，其中 7 个
        obs_count=2）。拆出来之后"说话的还是不是同一个人"就判不准了，web demo
        的陌生人门会突然翻脸说"我不认识你"。
        宁可这一轮不知道是谁，也不要往声纹库里塞噪声。门槛见
        VOICEMEM_SPEAKER_MIN_S（默认 2.5 秒）。
        """
        person_id: str | None = None
        stable_voiceprint = False
        vec = None
        if not self._long_enough(_apath):
            return person_id, speaker, stable_voiceprint, vec
        try:
            vec = self._speaker_encoder().embed(_apath)
            if vec is not None:
                pinned_pid = (
                    self._session_person_pin.get(session_id)
                    if session_id is not None else None
                )
                id_result = self._vp_store().identify(
                    vec, context=text[:80], pinned_person_id=pinned_pid,
                )
                person_id = id_result.person_id
                if id_result.action == "new" and session_id is not None:
                    self._person_origin_session.setdefault(person_id, session_id)
                # candidate 是"可能属于已有声纹"的待确认记录，不能拿来
                # 建姓名映射，否则一次误判会永久污染身份档案。
                stable_voiceprint = id_result.action in {"new", "match"}
                # caller 传入的 speaker 是默认占位符时，用识别结果覆盖
                if speaker == "Speaker 0":
                    speaker = person_id

                # ── 候选自动回收：这次 match 让画像更厚实，借机再核对
                # 一遍，收敛到同一个人的候选就并回去，避免记忆散落两个
                # person_id、speaker_filter 漏掉另一半。
                if id_result.action == "match":
                    person_id, speaker = self._reconcile_speaker_candidates(person_id, speaker)
        except Exception as _e:
            print(f"  [speaker] identification skipped: {_e}", flush=True)
        return person_id, speaker, stable_voiceprint, vec

    def _bind_self_identity(self, person_id, speaker, stable_voiceprint, vec, text, session_id):
        """自我介绍绑定（"我是annie" → 把当前声纹绑定到人名）。绑定后
        voice_input_to_messages() 会自动用真名替换 Speaker N。
        返回 (person_id, speaker, stable_voiceprint)。"""
        try:
            from voicemem.utils.audio.voiceprint.speaker_identity import parse_self_identification
            self_name = parse_self_identification(text)
            if self_name:
                registry = self._registry()
                existing_name = registry.display_name(person_id)
                # 自动抽取只补齐空名，不覆盖已确认人名（姓名更正应走显式接口）。
                # "未绑定"不等于"可占用"：声纹阈值对短句不稳，一个已攒了真实
                # 观测的声纹若被别人的自报姓名误匹配，会把之前那人的话张冠李戴。
                # 阈值定在 obs_count<=2（本条 + 最多1条紧邻历史），不用 <=1：
                # 自我介绍常不是第一句，用 <=1 会把"同一人紧接着报姓名"误判为拆分。
                obs_count = self._vp_store().get_meta(person_id).get("obs_count", 0)
                fresh_unbound = existing_name == person_id and obs_count <= 2
                if existing_name == self_name or fresh_unbound:
                    registry.bind(person_id, name=self_name)
                    if session_id is not None:
                        self._session_person_pin[session_id] = person_id
                else:
                    # 自报姓名是强证据：当前声纹已绑定成别人名、或虽未绑定
                    # 但已有历史观测（疑似误匹配）时，拆出独立声纹，再把本句
                    # 和后续记忆绑定到自报姓名。
                    if vec is not None:
                        split_result = self._vp_store().create_person(
                            vec, context=text
                        )
                        person_id = split_result.person_id
                        if session_id is not None:
                            self._person_origin_session.setdefault(person_id, session_id)
                        stable_voiceprint = True
                        if speaker == "Speaker 0":
                            speaker = person_id
                        registry.bind(person_id, name=self_name)
                        if session_id is not None:
                            self._session_person_pin[session_id] = person_id
                    else:
                        print(
                            f"  [speaker_identity] ignored conflicting auto-name "
                            f"{self_name!r}; {person_id} is already {existing_name!r} "
                            f"(obs_count={obs_count})",
                            flush=True,
                        )
        except Exception as _e:
            print(f"  [speaker_identity] bind failed: {_e}", flush=True)
        return person_id, speaker, stable_voiceprint

    def preprocess(
        self,
        text: str,
        speaker: str = "Speaker 0",
        emotion: str = "",
        session_id: int | str | None = None,
        audio_path: str | None = None,
    ) -> "AudioPerception":
        """流式预处理：Ingest() 推理路径的第 ① 步，一个可单独调用的公开接缝。

        把一轮音频跑过所有声学感知模块——声学场景 / 背景音乐 / 异常环境音 /
        说话人声纹 / 自我介绍绑定 / 韵律情绪，外加纯文本模式下的自我介绍绑定
        与情绪兜底，汇总成一个 AudioPerception。语音层只需给出 text（+ 可选
        audio_path），感知全在这里发生；返回后 Ingest() 负责组装 ctx 写记忆。

        独立公开是为了把"预处理"和"写记忆"解耦：可先拿到场景/说话人/情绪等信号
        用于实时应答，再决定是否 Ingest。不写记忆，但会就地更新声纹注册表等共享状态。
        """
        environment = ""
        environment_hint = ""
        scene_tag: str | None = None
        scene_raw_labels: list[str] = []
        person_id: str | None = None
        stable_voiceprint = False
        vec = None
        tune_result = None   # voicemem.utils.audio.environment.music_memory.TuneIdentifyResult | None
        abnormal_hits: list[tuple[str, float]] = []
        detection = {"pairs": [], "music": None, "abnormal": [], "embedding": None}
        if audio_path is not None:
            _apath = Path(audio_path)

            if self._enable_scene or self._enable_music or self._enable_abnormal_sound:
                (environment, environment_hint, scene_tag, scene_raw_labels,
                 tune_result, abnormal_hits, detection) = self._detect_scene(_apath)

            if self._enable_voiceprint:
                person_id, speaker, stable_voiceprint, vec = self._detect_speaker(
                    speaker, session_id, text, _apath,
                )
                if person_id and stable_voiceprint:
                    person_id, speaker, stable_voiceprint = self._bind_self_identity(
                        person_id, speaker, stable_voiceprint, vec, text, session_id,
                    )

            # ── 情绪识别（韵律 VAD + 负显著轮 Qwen2.5-Omni 归因）───────────
            # 只在调用方没显式传 emotion 时才自动检测填充。
            if self._enable_emotion and not emotion:
                try:
                    emotion = self._emotion_detector().detect(_apath)
                except Exception as _e:
                    print(f"  [emotion] detection skipped: {_e}", flush=True)

        # ── 自我介绍绑定，文本模式（无音频/无声纹时）──────────────────────
        # 声纹分支的绑定纯文本调用走不到。文本通道的 speaker id 是调用方给定的
        # 稳定标识，无误匹配风险，只需"只补空名、不覆盖已确认人名"，不用 obs_count 防护。
        if person_id is None and speaker and speaker != "Speaker 0":
            try:
                from voicemem.utils.audio.voiceprint.speaker_identity import parse_self_identification
                self_name = parse_self_identification(text)
                if self_name:
                    registry = self._registry()
                    if registry.display_name(speaker) == speaker:  # 只补空名
                        registry.bind(speaker, name=self_name)
            except Exception as _e:
                print(f"  [speaker_identity] text-mode bind failed: {_e}", flush=True)

        # ── 文本情绪兜底（无音频/韵律检测没给出结果时）─────────────────────
        # 情绪检测只挂在音频分支，纯文本调用 emotion 永远为空、右脑写不进。这里
        # 用 anchor_router 的中英情绪关键词表对 text 零成本匹配：有明确情绪词才算，
        # 识别不出返回 None、不写 heartnote。VOICEMEM_TEXT_EMOTION=0 关闭。
        if (
            self._enable_emotion and not emotion and text
            and os.environ.get("VOICEMEM_TEXT_EMOTION", "1") != "0"
        ):
            from voicemem.rightbrain.anchor_router import normalize_emotion_strict
            detected = normalize_emotion_strict(text)
            if detected:
                emotion = detected

        return AudioPerception(
            speaker=speaker,
            emotion=emotion,
            environment=environment,
            environment_hint=environment_hint,
            scene_tag=scene_tag,
            scene_raw_labels=scene_raw_labels,
            person_id=person_id,
            tune_result=tune_result,
            abnormal_hits=abnormal_hits,
            detection=detection,
        )

    # ── audiomem 写入段 ────────────────────────────────────────────────────────

    def _write_audiomem_tags(
        self, result, scene_tag, scene_raw_labels, detection, audio_path,
        person_id, tune_result, abnormal_hits, ts, session_id, text,
    ) -> dict:
        """audiomem 写入段：场景/声纹/音乐/异常音/地点标签 + 触发提醒 + 录音归档
        + 场景切换主动推送。返回后续 return 需要的字段。"""
        triggered_reminders: list[dict] = []
        proactive_memories: list[dict] = []
        familiar_place_prompt: dict | None = None
        place_result = None
        new_routine = None
        _fire_result = None
        if scene_tag and result.memory_ids:
            try:
                self._tag(result.memory_ids, [(f"scene:{scene_tag}", 0.95)])
            except Exception as _e:
                print(f"  [scene] tag write failed: {_e}", flush=True)

            # 场景变化 → 触发匹配的提醒（先于 update_scene，保留 scene_changed 信息）
            try:
                from voicemem.utils.audio.environment.scene_trigger import check_and_fire
                _fire_result = check_and_fire(
                    self._trigger_store(), self._user_id, scene_tag,
                    raw_labels=scene_raw_labels,
                )
                triggered_reminders = [
                    {"id": t.id, "message": t.message, "scene": t.trigger_scene}
                    for t in _fire_result.fired
                ]
            except Exception as _e:
                print(f"  [scene_trigger] check failed: {_e}", flush=True)

            # 生活声音规律：自动建立 routine 记忆——只在真正"进入"场景时记一次
            # 观测（scene_changed），积累够天数后只在刚跨过阈值那次生成记忆。
            if _fire_result is not None and _fire_result.scene_changed:
                try:
                    from datetime import datetime as _datetime
                    from voicemem.utils.audio.environment.routine_memory import bucket_label
                    routine_check = self._routine_store().observe(
                        self._user_id, scene_tag, _datetime.now()
                    )
                    if routine_check["is_new_routine"]:
                        blabel = bucket_label(routine_check["bucket"])
                        new_routine = {
                            "scene": scene_tag,
                            "bucket": routine_check["bucket"],
                            "bucket_label": blabel,
                            "distinct_days": routine_check["distinct_days"],
                        }
                        synthetic_message = [{"role": "user", "content": (
                            f"[Routine pattern detected: user is regularly in "
                            f"'{scene_tag}' scene during {blabel}, "
                            f"observed on {routine_check['distinct_days']} different days]"
                        )}]
                        custom_instructions = (
                            "This is an automatically detected behavioral routine based on "
                            "recurring acoustic scene observations over multiple days. "
                            "Extract a concise memory fact describing this habitual pattern "
                            "(e.g. 'User usually commutes around 7-9am'). "
                            "Do NOT invent specifics beyond the scene and time window given."
                        )
                        self._extract_and_append_fn(synthetic_message, custom_instructions, ts, {
                            "source": "routine",
                            "routine_scene": scene_tag,
                            "routine_bucket": routine_check["bucket"],
                            "routine_distinct_days": routine_check["distinct_days"],
                            **({"session_id": session_id} if session_id is not None else {}),
                        })
                except Exception as _e:
                    print(f"  [routine] check failed: {_e}", flush=True)

            # 熟悉地点自动聚类：同样只在"进入"场景时识别一次（scene_changed），
            # 用整段录音的原始 AST embedding 捕捉这个具体地点的声学指纹。
            if (
                _fire_result is not None and _fire_result.scene_changed
                and detection.get("embedding") is not None
            ):
                try:
                    from datetime import datetime as _datetime
                    place_result = self._place_store().identify(
                        detection["embedding"], scene=scene_tag, when=_datetime.now()
                    )
                except Exception as _e:
                    print(f"  [place] identification skipped: {_e}", flush=True)

        # WAV 存档：记录 audio_path → memory_id 映射
        if audio_path and result.memory_ids:
            try:
                self._audio_archive().record(
                    result.memory_ids, self._user_id, str(audio_path)
                )
            except Exception as _e:
                print(f"  [audio_archive] record failed: {_e}", flush=True)

        # 声纹标签：把 person_id 写入 memory_tags，供后续按说话人检索
        if person_id and result.memory_ids:
            try:
                self._tag(result.memory_ids, [(f"speaker:{person_id}", 1.0)])
            except Exception as _e:
                print(f"  [speaker] tag write failed: {_e}", flush=True)

        # 音乐/哼唱标签：把 tune_id 写入 memory_tags 供按"同一首歌/调子"检索；
        # heard_count>=2 时额外生成一条"又听到熟悉的调子"记忆事实。
        # 「这段录音里有没有音乐」和「是哪一首」是两件事，标签该按前者打。
        #
        # AST 那个环境音分类器本来就在判断前者（_MUSIC_KEYWORDS：music / singing /
        # humming / whistling / musical instrument…），detection["music"] 非空就是
        # 它说有。而 tune_result 来自 music_store.identify()——那是**同一性匹配**，
        # 回答的是"跟我存过的哪首像"，它返回 None 只说明认不出是哪首，不代表没有
        # 音乐。原来把标签绑在 tune_result 上，于是"听见了但认不出是哪首"的那些轮
        # 全都进不了回放的候选池。
        ast_music = (detection or {}).get("music") or None
        if (ast_music or tune_result is not None) and result.memory_ids:
            tune_id = getattr(tune_result, "tune_id", None) if tune_result else None
            try:
                self._tag(result.memory_ids,
                          [(f"tune:{tune_id}" if tune_id else "tune:unidentified", 0.9)])
                # detection["music"] 是 {"labels": [(标签, 分数), ...], "embedding": ...}
                _labels = (ast_music or {}).get("labels") or []
                top = f"，最强 {_labels[0][0]} {_labels[0][1]:.2f}" if _labels else ""
                which = (f"{tune_result.action}, 第 {tune_result.heard_count} 次"
                         if tune_result else "认不出是哪首")
                print(f"  [music] tag {tune_id or 'unidentified'} → "
                      f"{len(result.memory_ids)} 条记忆（{which}{top}）", flush=True)
            except Exception as _e:
                print(f"  [music] tag write failed: {_e}", flush=True)

            if tune_result.action == "match" and tune_result.heard_count >= 2:
                try:
                    tune_labels = self._music_store().get_meta(tune_result.tune_id).get("labels", [])
                    synthetic_message = [{"role": "user", "content": (
                        f"[Recognized recurring background music/humming, heard "
                        f"{tune_result.heard_count} times before: {', '.join(tune_labels) or 'unknown tune'}]"
                    )}]
                    custom_instructions = (
                        "This is a recognition event for a recurring background tune (music or humming) "
                        "that has been heard multiple times before, detected via acoustic similarity. "
                        "Extract a concise memory fact noting that this familiar tune came up again. "
                        "Do NOT invent song titles or lyrics you don't actually know."
                    )
                    self._extract_and_append_fn(synthetic_message, custom_instructions, ts, {
                        "source": "music_recognition",
                        "tune_id": tune_result.tune_id,
                        "heard_count": tune_result.heard_count,
                        **({"session_id": session_id} if session_id is not None else {}),
                    })
                except Exception as _e:
                    print(f"  [music] recognition fact skipped: {_e}", flush=True)

        # 声学没认出来，但**人明说了这是音乐**：照样打标签。
        #
        # 识别走的是声纹式的相似度匹配，手机外放、环境吵、片段太短都会漏——实测
        # 真机上放了一段歌，日志里就是一行"音乐识别没命中"。可用户嘴上说的是
        # 「我给你听首歌」，这比任何声学特征都确凿。漏了这条，那段录音就进不了
        # 回放的候选池，之后问「上周三那首歌」怎么都找不到。
        #
        # 用 tune:unidentified 而不是编一个 tune_id：确实不知道是哪首，别假装知道。
        # 候选池按 `tune:%` 查（见 web/run.py 的 _tune_memories），认得这个。
        # 变量别跟模块级的 said_music 同名——同名会把那个函数遮蔽成局部变量，
        # 在赋值之前引用就是 UnboundLocalError。
        _said = said_music(text)
        if (tune_result is None and not ast_music
                and _said and audio_path and result.memory_ids):
            try:
                self._tag(result.memory_ids, [("tune:unidentified", 0.6)])
                print(f"  [music] 声学没认出来，但他说了是音乐 → 照样标上"
                      f"（{len(result.memory_ids)} 条记忆）", flush=True)
            except Exception as _e:
                print(f"  [music] tag write failed: {_e}", flush=True)

        # 没打上标签的情况分开记。标签是「这段录音里有音乐」唯一直接的证据，
        # 回放时靠它把候选缩到真正的音乐上；缺了只能退回按时间和文本猜。
        # 实测真机上有过"音乐明明听到了却没标签"，事后光看库推不出是哪一种。
        #
        # 写成独立的 if，别接成上面那个的 elif——「听过两次就生成一条记忆」那段是
        # 嵌在 if 里面的，加 elif 会把它挤出来，tune_result 为 None 时就去读它的
        # 属性了（AttributeError: 'NoneType' object has no attribute 'action'）。
        elif audio_path and not result.memory_ids:
            why = "这一轮没有入库任何事实"
            print(f"  [music] 没打 tune 标签：{why}", flush=True)

        # 异常环境音记忆：破碎声/警报/尖叫——第一次出现就值得记一笔，每次检测到
        # 都打标签 + 生成一条记忆事实。独立警报事实的写入不依赖 result.memory_ids
        # （异常声音值不值得记，与这句话本身有没有独立事实价值无关）。
        if abnormal_hits:
            alert_memory_ids: list[str] = []
            try:
                labels_str = ", ".join(f"{l}({s:.2f})" for l, s in abnormal_hits)
                synthetic_message = [{"role": "user", "content": (
                    f"[Abnormal environmental sound event detected: {labels_str}]"
                )}]
                custom_instructions = (
                    "This is an unusual/alerting non-speech environmental sound event "
                    "(e.g. breaking glass, alarm, siren, or screaming) captured during the "
                    "conversation. Extract a concise memory fact describing this notable event. "
                    "Do NOT extract facts about the conversation topic itself."
                )
                alert_memory_ids = self._extract_and_append_fn(synthetic_message, custom_instructions, ts, {
                    "source": "abnormal_sound",
                    "abnormal_labels": [l for l, _ in abnormal_hits],
                    **({"session_id": session_id} if session_id is not None else {}),
                })
            except Exception as _e:
                print(f"  [abnormal] alert fact skipped: {_e}", flush=True)

            # 打标签：原有 turn 的记忆 + 上面新建的独立警报事实两边都打，保证至少
            # 有一条可检索记忆挂着 abnormal:<label> 标签。
            try:
                cog_store = self._repo()._cognitive_store
                tag_target_ids = list(result.memory_ids) + alert_memory_ids
                if cog_store and hasattr(cog_store, "upsert_memory_tags") and tag_target_ids:
                    tags = [
                        (f"abnormal:{label.lower().replace(' ', '_').replace(',', '')}", score)
                        for label, score in abnormal_hits
                    ]
                    for mid in tag_target_ids:
                        cog_store.upsert_memory_tags(mid, self._user_id, tags)
            except Exception as _e:
                print(f"  [abnormal] tag write failed: {_e}", flush=True)

        # 熟悉地点标签：把 place_id 写入 memory_tags，供后续按"同一个具体地点"检索。
        if place_result is not None and result.memory_ids:
            try:
                self._tag(result.memory_ids, [(f"place:{place_result.place_id}", 0.9)])
            except Exception as _e:
                print(f"  [place] tag write failed: {_e}", flush=True)

        # 熟悉环境主动提示"上次在这里"：只有识别成 match 才有"上次"可提。从这个
        # 地点之前打过 place:<id> 标签的记忆里挑跟当前话题相关的几条，附带到访信息。
        if place_result is not None and place_result.action == "match" and result.memory_ids:
            try:
                cog_store = self._repo()._cognitive_store
                place_memories: list[dict] = []
                if cog_store and hasattr(cog_store, "memory_ids_for_slots_v2"):
                    place_mem_ids = set(
                        cog_store.memory_ids_for_slots_v2(
                            self._user_id, [f"place:{place_result.place_id}"]
                        )
                    ) - set(result.memory_ids)
                    if place_mem_ids:
                        _hits = self._rank(text, place_mem_ids, top_k=3)
                        place_memories = [
                            {"memory_id": h.memory_id, "content": h.text, "score": round(h.score, 3)}
                            for h in _hits
                        ]
                familiar_place_prompt = {
                    "place_id": place_result.place_id,
                    "visit_count": place_result.visit_count,
                    "previous_visit_at": place_result.previous_visit_at,
                    "memories": place_memories,
                }
            except Exception as _e:
                print(f"  [place] proactive recall failed: {_e}", flush=True)

        # 场景切换主动推送：进入新场景时检索场景相关记忆（用 scene_changed 避免重复触发）
        if scene_tag and _fire_result is not None and _fire_result.scene_changed:
            try:
                _query = _SCENE_RECALL_QUERY.get(scene_tag, "")
                if _query:
                    _hits = self._rank(_query, set(), top_k=3)
                    proactive_memories = [
                        {"memory_id": h.memory_id, "content": h.text, "score": round(h.score, 3)}
                        for h in _hits
                    ]
            except Exception as _e:
                print(f"  [proactive] failed: {_e}", flush=True)

        return {
            "triggered_reminders": triggered_reminders,
            "proactive_memories": proactive_memories,
            "familiar_place_prompt": familiar_place_prompt,
            "place_result": place_result,
            "new_routine": new_routine,
        }

    # ── 环境音单独入库 ─────────────────────────────────────────────────────────

    def IngestEnv(
        self,
        audio_path,
        recent_context: list[dict] | None = None,
        session_id: int | str | None = None,
        environment_override: str | None = None,
    ) -> dict:
        """将一段环境音事件存入记忆库。

        Parameters
        ----------
        audio_path:
            环境音 wav 文件路径。
        recent_context:
            最近几轮对话文本，格式 [{"role": "user"/"assistant", "content": "..."}]。
            供 LLM 推断用户当时在做什么。
        session_id:
            当前会话 ID，用于时序排序。

        Returns
        -------
        dict
            ``{facts_count, memory_ids}``
        """
        import time
        from pathlib import Path as _Path

        # ── Step 1: 识别背景音；CLAP 后台复核时直接使用其结果 ────────────────
        if environment_override:
            env_str = environment_override
        else:
            try:
                pairs = self._env_detector().detect_full(_Path(audio_path)).get("pairs") or []
                env_str = (
                    "background sounds: " + ", ".join(f"{label}({score:.2f})" for label, score in pairs)
                    if pairs else ""
                )
            except Exception as e:
                print(f"  [IngestEnv] detection failed: {e}", flush=True)
                return {"facts_count": 0, "memory_ids": []}

        if not env_str:
            return {"facts_count": 0, "memory_ids": []}

        # ── Step 2: 构造 custom_instructions（环境音 + 对话上下文） ────────────
        ctx_lines = ""
        if recent_context:
            ctx_lines = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in recent_context[-6:]
            )

        custom_instructions = (
            "This is a non-speech environmental sound event captured during the conversation. "
            "Based on the detected background sounds and the recent conversation context below, "
            "extract a concise memory fact describing what the user was likely doing or experiencing at this moment. "
            "Do NOT extract facts about the conversation topic itself — focus only on the environmental activity.\n"
            + (f"Recent conversation context:\n{ctx_lines}" if ctx_lines else "")
        )

        # ── Step 3: 用 extractor 生成 fact ────────────────────────────────────
        synthetic_message = [{"role": "user", "content": f"[Environmental sound event: {env_str}]"}]
        try:
            extracted = self._extractor().extract(
                new_messages=synthetic_message,
                custom_instructions=custom_instructions,
                observation_date=time.strftime("%H:%M:%S"),
                current_date=time.strftime("%H:%M:%S"),
            )
        except Exception as e:
            print(f"  [IngestEnv] extraction failed: {e}", flush=True)
            return {"facts_count": 0, "memory_ids": []}

        if not extracted:
            return {"facts_count": 0, "memory_ids": []}

        # ── Step 4: 存入左脑 ──────────────────────────────────────────────────
        meta = {
            "source":           "environment",
            "background_sounds": env_str,
            **({"session_id": session_id} if session_id is not None else {}),
        }
        memory_ids = self._repo().append_extracted(
            extracted, user_id=self._user_id, extra_metadata=meta
        )

        return {"facts_count": len(extracted), "memory_ids": memory_ids or []}
