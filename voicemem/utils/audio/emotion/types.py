"""右脑数据结构：VAD、异常归因与情绪图增量。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class VAD:
    """声学维度的情绪状态：Valence（效价）与 Arousal（唤醒度）。

    约定：valence ∈ [-1, 1]，负偏消极、正偏积极；arousal ∈ [0, 1]，越高唤醒越强。
    数值来源可为深度模型或启发式估计；上层逻辑与具体估计器解耦。
    """

    valence: float
    arousal: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "valence", float(max(-1.0, min(1.0, self.valence))))
        object.__setattr__(self, "arousal", float(max(0.0, min(1.0, self.arousal))))


TriggerKind = Literal["anomaly"]
EmotionNodeType = Literal["User", "EmotionEpisode", "Topic", "Event", "Entity", "Person", "Project", "Organization", "Place"]
EmotionEdgeType = Literal["EXPERIENCED", "EMOTIONAL_REACTION_TO"]
EmotionIntensity = Literal["low", "medium", "high"]


@dataclass
class TurnEmotionRecord:
    """单轮用户发言对应的情绪侧记录。"""

    turn_id: str
    session_id: str
    vad: VAD
    timestamp_s: float | None = None
    user_utterance_index: int = 0
    time_gap_from_prev_turn_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmotionSignal:
    """结构化情绪状态，来自异常轮多模态归因。"""

    label: str
    valence: float
    arousal: float
    intensity: EmotionIntensity = "medium"
    confidence: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "valence", float(max(-1.0, min(1.0, self.valence))))
        object.__setattr__(self, "arousal", float(max(0.0, min(1.0, self.arousal))))
        object.__setattr__(self, "confidence", float(max(0.0, min(1.0, self.confidence))))


@dataclass(frozen=True)
class EmotionGraphNodeInput:
    """异常归因模型建议写入情绪图的实体/主题节点。"""

    local_id: str
    name: str
    node_type: EmotionNodeType = "Entity"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmotionGraphEdgeInput:
    """异常归因模型建议写入情绪图的边。"""

    source: str
    target: str
    edge_type: EmotionEdgeType = "EMOTIONAL_REACTION_TO"
    description: str = ""
    emotion_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmotionGraphDelta:
    """一次异常归因生成的图增量。"""

    nodes: list[EmotionGraphNodeInput] = field(default_factory=list)
    edges: list[EmotionGraphEdgeInput] = field(default_factory=list)


@dataclass(frozen=True)
class TurnAttributionLLMResult:
    """异常轮多模态归因模型输出。"""

    analysis_text: str
    emotion: EmotionSignal
    acoustic_evidence: list[str] = field(default_factory=list)
    semantic_evidence: list[str] = field(default_factory=list)
    related_nodes: list[EmotionGraphNodeInput] = field(default_factory=list)
    graph_delta: EmotionGraphDelta = field(default_factory=EmotionGraphDelta)
    retrieval_snippet: list[str] = field(default_factory=list)


@dataclass
class EmotionAttribution:
    """情绪归因分析结果；仅在 VAD 负面显著时生成。"""

    turn_id: str
    session_id: str
    trigger: TriggerKind
    analysis_text: str
    vad_at_trigger: VAD
    left_context_summary: str | None = None
    emotion: EmotionSignal | None = None
    acoustic_evidence: list[str] = field(default_factory=list)
    semantic_evidence: list[str] = field(default_factory=list)
    related_nodes: list[EmotionGraphNodeInput] = field(default_factory=list)
    graph_delta: EmotionGraphDelta = field(default_factory=EmotionGraphDelta)
    user_utterance_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
