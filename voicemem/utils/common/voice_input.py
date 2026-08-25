"""voice_input.py — 语音模块输入适配层

将语音模块的结构化输出转换为 VoiceMem 左脑注入格式。

语音模块输出格式：
{
  "id": "str",
  "time_stamp": {"begin": "str", "end": "str"},
  "slots": ["str", ...],
  "contents": [
    {
      "sub_id":        "str",
      "time_start":    "str",
      "time_end":      "str",
      "sentence":      "str",
      "voiceprint_id": "str",
      "emotion":       "str"   # emotion2vec 输出，可为空
    }
  ]
}

voiceprint_id → 人名 / entity_id 映射通过 VoiceprintRegistry 管理。
"""
from __future__ import annotations

import os as _os

import os

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ── 数据模型 ─────────────────────────────────────────────────────────────────

@dataclass
class VoiceContent:
    sub_id: str
    time_start: str
    time_end: str
    sentence: str
    voiceprint_id: str
    emotion: str = ""          # emotion2vec 识别结果，如"中性"/"焦虑"

    @classmethod
    def from_dict(cls, d: dict) -> "VoiceContent":
        return cls(
            sub_id=str(d.get("sub_id", "")),
            time_start=str(d.get("time_start", "")),
            time_end=str(d.get("time_end", "")),
            sentence=str(d.get("sentence", "")).strip(),
            voiceprint_id=str(d.get("voiceprint_id", "")),
            emotion=str(d.get("emotion", "") or ""),
        )


@dataclass
class VoiceInput:
    id: str
    time_stamp: dict          # {"begin": str, "end": str}
    slots: list[str]
    contents: list[VoiceContent]
    environment: str = ""     # AST/CLAP background sound description, e.g. "background sounds: Washing machine(0.82)"
    #: agent 那半边（contents 是用户那半边），抽取器靠它消歧
    agent_reply: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "VoiceInput":
        ts = d.get("time_stamp") or {}
        if isinstance(ts, str):
            ts = {"begin": ts, "end": ts}
        return cls(
            id=str(d.get("id", "")),
            time_stamp=ts,
            slots=[str(s) for s in (d.get("slots") or [])],
            contents=[VoiceContent.from_dict(c) for c in (d.get("contents") or [])],
            agent_reply=str(d.get("agent_reply", "") or ""),
        )

    @property
    def begin_time(self) -> str:
        return self.time_stamp.get("begin", "")

    @property
    def end_time(self) -> str:
        return self.time_stamp.get("end", "")

    def full_transcript(self) -> str:
        return " ".join(c.sentence for c in self.contents if c.sentence)

    def dominant_emotion(self) -> str:
        """取出现次数最多的非空/非中性 emotion，都是中性则返回空串。"""
        from collections import Counter
        neutral = {"-", "中性", "neutral", "unknown", "未知", ""}
        emos = [c.emotion for c in self.contents if c.emotion and c.emotion not in neutral]
        if not emos:
            return ""
        return Counter(emos).most_common(1)[0][0]


# ── Voiceprint 注册表 ────────────────────────────────────────────────────────

@dataclass
class VoiceprintEntry:
    """单个声纹的注册信息。"""
    role: str         # "user" | "assistant"
    name: str         # 显示名，如"周总"；未设置时同 voiceprint_id
    entity_id: str    # 对应左脑 cognitive_graph entities 表的 id；可为空


class VoiceprintRegistry:
    """voiceprint_id → 人名 + entity_id 的持久化注册表。

    核心功能：
      - 所有未知声纹默认 role="user"（多人对话场景）
      - bind() 把 voiceprint_id 绑定到已知人名和/或 entity_id
        → voice_input_to_messages() 会用真实人名替换 "Speaker N"
        → LLM 抽取时看到"周总: ..."，CognitiveAnnotator 自然把记忆链接到周总实体
      - entity_id 存入记忆 metadata，供后续直接查询
    """

    ROLE_USER      = "user"
    ROLE_ASSISTANT = "assistant"

    def __init__(self, registry_path: Path, entity_resolver: Any = None) -> None:
        self._path = registry_path
        self._entries: dict[str, VoiceprintEntry] = {}
        #: 人名 -> 认知图 entity_id。由 orchestrator 注入；不注入则退化成原行为。
        self._resolve_entity = entity_resolver
        self._load()

    # ── 读写 ──────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for vpid, v in data.items():
                if isinstance(v, dict):
                    self._entries[vpid] = VoiceprintEntry(
                        role=v.get("role", self.ROLE_USER),
                        name=v.get("name", vpid),
                        entity_id=v.get("entity_id", ""),
                    )
                else:
                    # 兼容旧格式（纯字符串 role）
                    self._entries[vpid] = VoiceprintEntry(
                        role=str(v), name=vpid, entity_id=""
                    )
        except Exception:
            pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            vpid: {"role": e.role, "name": e.name, "entity_id": e.entity_id}
            for vpid, e in self._entries.items()
        }
        self._path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # ── 公开 API ──────────────────────────────────────────────────────────────

    def bind(
        self,
        voiceprint_id: str,
        *,
        name: str | None = None,
        entity_id: str | None = None,
        role: str = ROLE_USER,
    ) -> VoiceprintEntry:
        """绑定声纹到人名和/或 entity_id（增量更新，只覆盖传入的字段）。"""
        if role not in (self.ROLE_USER, self.ROLE_ASSISTANT):
            raise ValueError(f"role 必须是 'user' 或 'assistant'，got {role!r}")
        existing = self._entries.get(voiceprint_id)
        entry = VoiceprintEntry(
            role=role,
            name=name or (existing.name if existing else voiceprint_id),
            entity_id=entity_id or (existing.entity_id if existing else ""),
        )
        self._entries[voiceprint_id] = entry
        self._save()
        return entry

    # 向后兼容旧接口
    def register(self, voiceprint_id: str, role: str, name: str | None = None) -> None:
        self.bind(voiceprint_id, role=role, name=name)

    def get(self, voiceprint_id: str) -> VoiceprintEntry:
        """返回注册信息；未知声纹返回默认 entry（role=user, name=voiceprint_id）。"""
        return self._entries.get(
            voiceprint_id,
            VoiceprintEntry(role=self.ROLE_USER, name=voiceprint_id, entity_id=""),
        )

    def resolve(self, voiceprint_id: str) -> str:
        return self.get(voiceprint_id).role

    def display_name(self, voiceprint_id: str) -> str:
        return self.get(voiceprint_id).name

    def entity_id(self, voiceprint_id: str) -> str:
        """已绑定就直接返回；没有则按人名去认知图找一次，找到就回填。

        惰性解析而不是在 bind() 里做，是因为自我介绍（"我是小李"）发生在抽取
        建实体**之前**——bind 那一刻图里通常还没有这个人，当场查必然落空。
        """
        entry = self.get(voiceprint_id)
        if entry.entity_id or not self._resolve_entity:
            return entry.entity_id
        # 只查在册的：没 bind 过的声纹 get() 给的是临时 entry，名字就是 id 本身，
        # 拿它当人名去查图没有意义。查不到的（图里还没建这个人）返回 ""，下次再试。
        if voiceprint_id not in self._entries or not entry.name:
            return ""
        try:
            eid = self._resolve_entity(entry.name) or ""
        except Exception:
            return ""
        if eid:
            self.bind(voiceprint_id, name=entry.name, entity_id=eid, role=entry.role)
        return eid

    def all_display_names(self) -> list[str]:
        """所有已绑定真名的人名列表（排除还没 bind 过、name 仍等于 voiceprint_id
        本身的默认项）。用于跨人对话里判断"这条候选记忆是不是点名了别人"。"""
        return [e.name for vpid, e in self._entries.items() if e.name and e.name != vpid]

    def to_dict(self) -> dict:
        return {
            vpid: {"role": e.role, "name": e.name, "entity_id": e.entity_id}
            for vpid, e in self._entries.items()
        }


# ── Emotion 映射（情绪标签 → VoiceMem affect 词表）──────────────────────────
# 这张表最初是照着 emotion2vec 的固定九分类词表配的，只覆盖那九个词的中英
# 文写法。paper_emotion_detector.py 接入后，标签来源换成了 Qwen2.5-Omni
# 自由生成的情绪词——不再是九个词的封闭集合（比如"happiness"/"frustration"/
# "anxiety"这类同义变体都可能出现），精确字符串匹配会大量落空，affect 字段
# 悄悄变回 None。兜底：精确匹配不中时，先过 anchor_router.normalize_emotion()
# 的关键词模糊匹配（右脑锚点已经在用的同一套 8 类归一化），再从归一化结果
# 映射到 affect 词表，兼容任意措辞的自由生成标签。
_EMO_TO_AFFECT: dict[str, str] = {
    "开心": "excited",  "快乐": "excited",  "happy": "excited",
    "悲伤": "sad",      "难过": "sad",       "sad": "sad",
    "愤怒": "angry",    "生气": "angry",     "angry": "angry",
    "焦虑": "anxious",  "恐惧": "anxious",   "fearful": "anxious",
    "厌恶": "disgusted","disgusted": "disgusted",
    "惊讶": "curious",  "surprised": "curious",
    "满意": "satisfied","satisfied": "satisfied",
}

# anchor_router._CANONICAL_EMOTIONS（8 类）→ affect 词表的兜底映射。
_CANONICAL_TO_AFFECT: dict[str, str] = {
    "开心": "excited",
    "悲伤": "sad",
    "委屈": "angry",
    "孤独": "sad",
    "纠结": "anxious",
    "平静": "satisfied",
    "焦虑": "anxious",
    "疲惫": "sad",
}


def emotion_to_affect(emotion: str) -> str | None:
    """情绪标签 → VoiceMem affect 字段（中性/未知返回 None）。"""
    if not emotion:
        return None
    e = emotion.strip()
    direct = _EMO_TO_AFFECT.get(e)
    if direct:
        return direct
    from voicemem.rightbrain.anchor_router import normalize_emotion
    canonical = normalize_emotion(e)
    if canonical == "平静" and e not in ("平静", "calm", "neutral"):
        # normalize_emotion 对无法识别的输入兜底返回"平静"——区分"模型真的
        # 说是平静"和"这个词我们完全没见过、被动兜底"，避免自由生成里出现
        # 一个没见过的词就被静默当成"平静"处理，掩盖真实的映射缺口。
        return None
    return _CANONICAL_TO_AFFECT.get(canonical)


# ── 核心适配器 ───────────────────────────────────────────────────────────────

def _looks_like_voiceprint_id(vpid: str) -> bool:
    """这个 id 是系统生成的声纹编号，还是调用方传进来的人名？

    声纹编号形如 person_86fed148 / utt_9f2a / spk_3，人名没有这种前缀。
    分不清的话人名会被当成"身份不明"，记忆主语被改写成"用户"，
    按名字提问就检索不到了（"Jiaqi 搬到哪个城市了？" vs "用户搬到杭州"）。
    """
    v = (vpid or "").lower()
    return v.startswith(("person_", "utt_", "spk_", "speaker_", "voiceprint_"))


def voice_input_to_messages(
    vi: VoiceInput,
    registry: VoiceprintRegistry,
) -> list[dict[str, str]]:
    """将 VoiceInput.contents 转换为 OpenAI messages 格式。

    - 连续同一 voiceprint 的句子合并为一条 message
    - 用注册的真实人名替换 "Speaker N"（若已 bind）
      → LLM 看到"周总: ..."，CognitiveAnnotator 自然链接到周总实体
    - 未绑定姓名的声纹不能直接暴露内部 person_id 当"人名"喂给抽取模型：
      实测过（见 extract_facts_openai 测试），一段像 "person_f7b840f9: ..."
      这样的前缀，模型基本会当噪声忽略掉，退回官方 mem0 prompt 默认假设
      ——把这句话当成账号主人(User)说的，一旦 existing memories 里账号
      主人的真名已经出现过，就会直接把这段话安到主人名下（哪怕说话人其实
      是完全不同的另一个人）。改成显式的"未识别说话人"标签，同时保留
      person_id 后缀防止多个不同的未识别说话人被模型混成一个人。
    """
    if not vi.contents:
        return []

    messages: list[dict[str, str]] = []
    current_vpid: str | None = None
    current_sentences: list[str] = []

    def _flush() -> None:
        if not current_sentences or current_vpid is None:
            return
        entry = registry.get(current_vpid)
        label = entry.name
        if label == current_vpid:
            if current_vpid.lower() in ("user", "voice_demo_user"):
                # Caller-supplied account-owner channel (text-mode demos pass
                # the literal id "user" for every utterance): this IS the
                # account owner by definition, so the defensive unidentified
                # label below would be actively wrong here -- it made every
                # stored fact read "Unidentified speaker user stated..."
                # (real user complaint). That label exists for unverified
                # VOICEPRINT ids, where assuming account-owner identity
                # mis-attributes speech across real people; a fixed text
                # channel has no such ambiguity. Once the user self-
                # identifies ("我叫佳琪"), the registry binding (see
                # core.py's text-mode binding) replaces this with their
                # actual name via the normal entry.name path above.
                label = "User"
            elif _looks_like_voiceprint_id(current_vpid):
                # 真的是没绑名字的声纹。标签要满足两点：别被当成人名写进记忆
                # （"Unidentified speaker is a vegetarian" 就是这么来的），
                # 也别是一长串英文——那会把整段抽取带成英文，中文提问就检索不到。
                # 用 "Speaker N" 这种中性代号，说明单独给一条 system 提示。
                label = "Speaker 0"
            # 剩下的情况：调用方直接传了人名（评测适配器传 speaker="Jiaqi"，
            # 多人对话传对方的名字）。registry 里查不到只是因为它从来没注册过，
            # 不代表身份不明——原样当人名用。改写成"用户"会让按名字提问的检索
            # 全部落空（"Jiaqi 搬到哪个城市了？" → 记忆里写的是"用户搬到杭州"）。
        content = f"{label}: " + " ".join(current_sentences)
        messages.append({"role": entry.role, "content": content})

    for c in vi.contents:
        if not c.sentence:
            continue
        if c.voiceprint_id != current_vpid:
            _flush()
            current_vpid = c.voiceprint_id
            current_sentences = [c.sentence]
        else:
            current_sentences.append(c.sentence)

    _flush()

    # agent 那半边接在最后：用户说"就它吧"，指代只有回复里能解开
    if vi.agent_reply and vi.agent_reply.strip():
        messages.append({"role": "assistant", "content": vi.agent_reply.strip()})

    return messages


# ── Slot 映射表 ──────────────────────────────────────────────────────────────
# 映射目标是真实的 SlotV2 枚举值（work/finance/relationships/health/goals/
# daily_life/knowledge，见 cognitive_graph/slot_v2.py）——这里以前映射到的
# "task_todo"/"plan_future"/"event"/"fact"/"routine_habit"/"relationship" 等
# 值不属于任何一套真实 taxonomy，且下游 _write_slotv2_hints 还用错了 store
# 属性名/方法名，两个问题叠加导致这条路径写不出任何东西、静默失败。

_VOICE_SLOT_TO_SLOTV2: dict[str, str] = {
    "work": "work", "task": "work", "todo": "work", "deadline": "work",
    "project": "work", "meeting": "work",
    "finance": "finance", "money": "finance", "salary": "finance",
    "goal": "goals", "plan": "goals",
    "fact": "knowledge", "knowledge": "knowledge", "info": "knowledge",
    "event": "daily_life", "experience": "daily_life", "memory": "daily_life",
    "preference": "daily_life", "like": "daily_life", "dislike": "daily_life",
    "routine": "daily_life", "habit": "daily_life", "daily": "daily_life",
    "relationship": "relationships", "people": "relationships",
    "family": "relationships", "friend": "relationships",
    "health": "health", "medical": "health",
    "place": "daily_life", "location": "daily_life",
}


def map_voice_slots_to_slotv2(voice_slots: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for s in voice_slots:
        mapped = _VOICE_SLOT_TO_SLOTV2.get(s.lower().strip())
        if mapped and mapped not in seen:
            seen.add(mapped)
            result.append(mapped)
    return result


# ── Ingest ───────────────────────────────────────────────────────────────────

@dataclass
class VoiceIngestResult:
    voice_id: str
    memory_ids: list[str]
    facts_count: int
    begin_time: str
    end_time: str
    slots: list[str]
    messages_count: int
    affect: str | None = None   # 来自 emotion2vec 的情感信号
    error: str | None = None


def ingest_voice_input(
    vi: VoiceInput,
    user_id: str,
    *,
    registry: VoiceprintRegistry,
    repo: Any,
    extractor: Any,
    extra_metadata: dict | None = None,
    session_id: int | str | None = None,
) -> VoiceIngestResult:
    """将一个 VoiceInput 注入左脑记忆库。

    流程：
      1. contents → messages（用真实人名替换 Speaker N）
      2. 检索候选旧记忆（整段 turn 语义检索）供抽取阶段去重参考
      3. extractor.extract(messages, existing_memories=...) → 原子事实句（ADD-only 抽取）
      4. 对每条新 fact 逐条检索候选旧记忆（而非整 turn 一次搜）
      5. ConflictResolver.resolve(facts, existing) → ADD/UPDATE/DELETE/NONE 决策
      6. 执行决策：ADD→append_extracted，UPDATE→update_memory，DELETE→delete_memory
      7. slots 预分类写入 memory_tags
    """
    messages = voice_input_to_messages(vi, registry)
    if not messages:
        return VoiceIngestResult(
            voice_id=vi.id, memory_ids=[], facts_count=0,
            begin_time=vi.begin_time, end_time=vi.end_time,
            slots=vi.slots, messages_count=0, error="empty_contents",
        )

    # 本轮说话人真名（已绑定时才用，未绑定的话 display_name() 会退回声纹ID本身，
    # 那不是人名，传给 ConflictResolver 当"当前说话人"反而会误导它）——用来在
    # 冲突消解阶段给模型一个"这批新事实是谁说的"的硬锚点，见 extract_facts_openai
    # 里的 multi-person household rule。
    speaker_name: str | None = None
    if vi.contents:
        vpid = vi.contents[0].voiceprint_id
        resolved = registry.display_name(vpid)
        if resolved and resolved != vpid:
            speaker_name = resolved

    # 已知的"别人"姓名列表——existing_memories 的候选池是按 user_id（整个家庭）
    # 搜出来的，不按说话人过滤，所以两个不同的人说了近似模板的话时（比如都在
    # 讲"今年 donation drive 我选哪个慈善机构"），另一个人的旧记忆会被搜进候选
    # 里。ConflictResolver 的 system prompt 里虽然已经写了"不同人名不能
    # UPDATE"，但那只是软提示，模型不保证遵守（真实复现过：Jennifer 那轮把
    # Nancy 名下的记忆 UPDATE 成了 Jennifer 的答案，"Nancy...is the downtown
    # botanical society"）。这里在代码层面加一道硬过滤：候选文本里明确点了
    # 别人的名字、又没提当前说话人自己名字的，直接不给模型看，从根上断掉
    # UPDATE/DELETE 误伤的可能。
    other_person_names = (
        [n for n in registry.all_display_names() if n != speaker_name]
        if speaker_name else []
    )

    def _drop_other_named_people(cands: list[dict[str, str]]) -> list[dict[str, str]]:
        if not speaker_name or not other_person_names:
            return cands
        out = []
        for c in cands:
            text = str(c.get("text", ""))
            if speaker_name in text:
                out.append(c)
                continue
            if any(name in text for name in other_person_names):
                continue
            out.append(c)
        return out

    # ── Step 1.5: 候选旧记忆（供抽取阶段去重参考）───────────────────────────────
    # 之前抽取阶段完全没传 existing_memories，模型对库里已有什么一无所知，
    # 抽取 prompt 里设计好的"Existing Memories 仅用于去重/linked_memory_ids"
    # 这道防线形同虚设——同一件事被反复整条重抽，全指望下游 ConflictResolver
    # 兜底。这里先拿整段 turn 文本粗召回一批，喂给抽取器做去重参考。
    # 只用用户那半边召回候选：agent 的回复长得多，掺进来会把该比对的旧记忆挤出前 10
    query_text = "\n".join(str(m.get("content", "")) for m in messages
                           if m.get("role") != "assistant").strip()
    existing_for_extraction: list[dict[str, str]] = []
    if query_text and hasattr(repo, "search"):
        try:
            existing_for_extraction = [{"id": h.memory_id, "text": h.text}
                                        for h in repo.search(query_text, user_id=user_id, top_k=10)]
        except Exception:
            pass
    if not existing_for_extraction and hasattr(repo, "existing_for_extractor"):
        try:
            existing_for_extraction = repo.existing_for_extractor(user_id=user_id)
        except Exception:
            pass
    existing_for_extraction = _drop_other_named_people(existing_for_extraction)

    # ── Step 2: 抽取原子事实 ─────────────────────────────────────────────────
    # 原文兜底：抽取（走 OpenAI）失败或没抽到事实时，直接把整句原文当一条记忆存，
    # 让本地 E5 向量库照样能长脑图、能检索（0 OpenAI）。下游 ConflictResolver 无 key
    # 会自动降级成 ADD-only。
    def _raw_fallback() -> list:
        # 默认关：只存 LLM 抽取出的高质量事实。设 VOICEMEM_INGEST_RAW_FALLBACK=1 才在
        # 抽取失败/无事实时把原文兜底存下（无 key / 离线 demo 用）。
        if os.environ.get("VOICEMEM_INGEST_RAW_FALLBACK", "0") != "1":
            return []
        raw = " ".join(c.sentence for c in vi.contents if c.sentence).strip()
        if not raw:
            return []
        from voicemem.leftbrain.extract_facts_openai import ExtractedAdditiveMemory
        return [ExtractedAdditiveMemory(local_id="0", text=raw, attributed_to="user")]

    try:
        extracted = extractor.extract(
            new_messages=messages,
            existing_memories=existing_for_extraction,
            observation_date=vi.begin_time,
            current_date=vi.begin_time,
        )
    except Exception as e:
        extracted = _raw_fallback()
        print(f"[ingest] 抽取失败（{e}）→ {'原文兜底入库' if extracted else '无原文，跳过'}", flush=True)
        if not extracted:
            return VoiceIngestResult(
                voice_id=vi.id, memory_ids=[], facts_count=0,
                begin_time=vi.begin_time, end_time=vi.end_time,
                slots=vi.slots, messages_count=len(messages),
                error=f"extraction_failed: {e}",
            )

    if not extracted:
        extracted = _raw_fallback()
        if extracted:
            print("[ingest] 抽取无事实 → 原文兜底入库", flush=True)
        else:
            return VoiceIngestResult(
                voice_id=vi.id, memory_ids=[], facts_count=0,
                begin_time=vi.begin_time, end_time=vi.end_time,
                slots=vi.slots, messages_count=len(messages),
            )

    # ── Step 3-4: Conflict resolution (Mem0 V1 风格) ─────────────────────────
    from voicemem.leftbrain.extract_facts_openai import ConflictResolver

    # 候选旧记忆：逐条新 fact 分别检索（与 mem0 官方一致：main.py 是对每条
    # new_retrieved_fact 单独 embed 后 search，而不是把整个 turn 拼一句去搜）。
    # 之前是拿整段 turn 原始文本一次性搜 top-10——一个 turn 里同时聊好几件事时
    # （比如时间线+预算+合作方），真正该更新的那条旧记忆的语义信号被其它话题
    # 稀释，很容易掉出前10，模型压根看不到候选，只能判 ADD，旧版本就没人删，
    # 库里留下同一件事的好几个版本（这是"更新后旧数字仍留在库里"的直接成因）。
    # 窗口大小与第二路检索（benchmark 上 UPDATE/DELETE 漏判的直接原因）：
    #   1) top-5 太窄——库里几百条时，同一属性的旧值经常挤不进前 5，模型看不到
    #      候选只能判 ADD；放到 15，多出来的 10 条对 resolver 只是几百 token。
    #   2) "最喜欢的餐厅是 A" 和 "最喜欢的餐厅是 B" 两句话的向量并不像（值不同，
    #      而值恰恰是句子里信息量最大的词），所以再用抽取器顺带给出的属性短语
    #      "User's favorite restaurant"（不含值）搜一次：旧值那条和这个短语高度
    #      相似，必进窗口。事件类 fact 的 attribute 为空，不触发第二路。
    # VOICEMEM_CONFLICT_WIDE=0 回退到旧行为（每条 fact 仅向量 top-5，无属性检索，
    # resolver 不带单值属性规则），供消融对照。
    wide = os.environ.get("VOICEMEM_CONFLICT_WIDE", "1") != "0"
    new_fact_texts = [m.text for m in extracted if m.text]
    existing_map: dict[str, dict[str, str]] = {}
    if hasattr(repo, "search"):
        queries: list[tuple[str, int]] = [(t, 15 if wide else 5) for t in new_fact_texts]
        if wide:
            queries += [(m.attribute, 10) for m in extracted
                        if m.text and getattr(m, "attribute", "")]
        for q, k in queries:
            try:
                for h in repo.search(q, user_id=user_id, top_k=k):
                    existing_map[h.memory_id] = {"id": h.memory_id, "text": h.text}
            except Exception:
                continue
    existing = list(existing_map.values())
    if not existing and hasattr(repo, "existing_for_extractor"):
        try:
            existing = repo.existing_for_extractor(user_id=user_id)
        except Exception:
            pass
    existing = _drop_other_named_people(existing)

    # 冲突判定（新事实 vs 库里已有 → ADD/UPDATE/DELETE）是一次 LLM 调用，而且
    # prompt 里要塞进已有记忆，**库越大越慢**：实测 95 条的库上这一次就要 10.2s，
    # 占整轮 ingest 的一半。
    # VOICEMEM_ALWAYS_ADD=1 跳过它，一律新增——反正每条都带时间戳，检索时以最新
    # 为准。代价是同一属性的新旧值都留在库里（"最喜欢的餐厅是A" 和 "…是B"），
    # 靠回答模型按日期取舍。
    always_add = _os.environ.get("VOICEMEM_ALWAYS_ADD", "0") == "1"

    resolutions = []
    if existing and new_fact_texts and not always_add:
        try:
            resolver = ConflictResolver(single_valued_rule=wide)
            resolutions = resolver.resolve(new_fact_texts, existing, speaker_name=speaker_name)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("ConflictResolver failed, falling back to ADD-only: %s", e)

    # ── Step 5: 执行决策 ──────────────────────────────────────────────────────
    # 若 resolve 成功，按决策执行；否则退回原始 ADD-only 路径
    if resolutions:
        # 建立 fact_text → ExtractedAdditiveMemory 映射，供 ADD 时复用 metadata
        fact_map = {m.text: m for m in extracted if m.text}
        to_add: list = []
        for r in resolutions:
            if r.event == "ADD" and r.text:
                orig = fact_map.get(r.text) or next(iter(fact_map.values()), None)
                if orig:
                    from voicemem.leftbrain.extract_facts_openai import ExtractedAdditiveMemory
                    to_add.append(ExtractedAdditiveMemory(
                        local_id=r.memory_id,
                        text=r.text,
                        attributed_to=orig.attributed_to,
                    ))
            elif r.event == "UPDATE" and r.memory_id and r.text:
                if hasattr(repo, "update_memory"):
                    # 带上本次会话日期：被更新的那条记忆讲的已经是这次说的事了，
                    # 时间戳必须跟着走，否则新事实会挂在被并那条的旧日期上。
                    # user_id 传了才会重新跑认知图实体/关系抽取——之前 UPDATE
                    # 事实完全不会进认知图，只有 ADD 会，见 update_memory 文档。
                    repo.update_memory(r.memory_id, r.text, session_id=session_id,
                                       observed_at=vi.begin_time, user_id=user_id)
            elif r.event == "DELETE" and r.memory_id:
                if hasattr(repo, "delete_memory"):
                    repo.delete_memory(r.memory_id)
        extracted = to_add  # 只 append ADD 部分

    # entity_id 绑定：收集本批次所有有绑定的声纹
    speaker_entity_map = {
        vpid: registry.entity_id(vpid)
        for vpid in {c.voiceprint_id for c in vi.contents}
        if registry.entity_id(vpid)
    }

    # emotion2vec → affect
    affect = emotion_to_affect(vi.dominant_emotion())

    meta = {
        "turn_id":            vi.id,
        "voice_id":           vi.id,
        "time_start":         vi.begin_time,
        "time_end":           vi.end_time,
        "voice_slots":        vi.slots,
        "source":             "voice",
        "speaker_entity_map": speaker_entity_map,
        "affect":             affect,
        **({"session_id": session_id} if session_id is not None else {}),
        **({"background_sounds": vi.environment} if vi.environment else {}),
        **(extra_metadata or {}),
    }
    memory_ids = repo.append_extracted(extracted, user_id=user_id, extra_metadata=meta)
    print(f"[ingest] 入库 {len(memory_ids or [])} 条：{[m.text[:20] for m in extracted][:3]}", flush=True)

    # 预分类 slot 写入 memory_tags
    slotv2_hints = map_voice_slots_to_slotv2(vi.slots)
    if slotv2_hints and memory_ids:
        _write_slotv2_hints(repo, user_id, memory_ids, slotv2_hints)

    return VoiceIngestResult(
        voice_id=vi.id,
        memory_ids=memory_ids or [],
        facts_count=len(extracted),
        begin_time=vi.begin_time,
        end_time=vi.end_time,
        slots=vi.slots,
        messages_count=len(messages),
        affect=affect,
    )


def _write_slotv2_hints(repo: Any, user_id: str, memory_ids: list[str], slotv2_tags: list[str]) -> None:
    """语音模块自带的粗粒度 slot 提示是个中等置信度信号——跟 core.py 里
    LLM 打的标（confidence=0.95）、写入时 embedding 自动打的标（真实 cosine
    分数，见 memory_repository_v2.py._tag_memory_slots_v2）共享同一张
    memory_tags 表，取比这两者都低的固定置信度，不会覆盖更可信的判断
    （upsert_memory_tags 是 upsert，同 slot 后写入的会覆盖置信度）。"""
    try:
        store = repo._cognitive_store  # type: ignore[attr-defined]
        if store is None or not hasattr(store, "upsert_memory_tags"):
            return
        tags = [(slot, 0.5) for slot in slotv2_tags]
        for mid in memory_ids:
            store.upsert_memory_tags(mid, user_id, tags)
    except Exception:
        pass
