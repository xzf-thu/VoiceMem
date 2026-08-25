"""胼胝体 fusion：双通道检索与 reply prompt 类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from voicemem.utils.audio.emotion.types import EmotionAttribution, VAD


@dataclass(frozen=True)
class LeftMemoryHit:
    memory_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RightMemoryHit:
    turn_id: str
    analysis_text: str
    vad: VAD
    retrieval_snippet: list[str] = field(default_factory=list)
    left_context_summary: str | None = None


@dataclass
class FusionRetrievalResult:
    asr_text: str
    left_hits: list[LeftMemoryHit] = field(default_factory=list)
    right_hits: list[RightMemoryHit] = field(default_factory=list)
    left_graph_appendix: str = ""


@dataclass
class ReplyRetrievalBundle:
    """回复记忆检索结果（供 build_reply_context_prompt 使用）。"""

    retrieval: FusionRetrievalResult
    graph_emotion_context: str = ""


@dataclass
class ReplyContextPrompt:
    """分区 reply 上下文：语义区 + 情绪区 + Persona 占位等。"""

    semantic_block: str
    emotional_block: str
    left_context_summary: str
    persona_block: str = ""
    acoustic_block: str = ""
    graph_emotion_block: str = ""
    current_attribution_block: str = ""
    system_block: str = ""
    user_block: str = ""


@dataclass
class ReplyContextBundle:
    """检索 + 拼装后的回复记忆。"""

    retrieval: ReplyRetrievalBundle
    prompt: ReplyContextPrompt


@dataclass
class AnomalyTurnResult:
    retrieval: FusionRetrievalResult
    #: 归因前 reply 上下文（供 Omni）；最终回复用 ``TurnProcessResult.reply_context``。
    reply_prompt: ReplyContextPrompt
    attribution: EmotionAttribution
    pre_attribution_reply: ReplyContextPrompt | None = None

    def __post_init__(self) -> None:
        if self.pre_attribution_reply is None:
            self.pre_attribution_reply = self.reply_prompt
