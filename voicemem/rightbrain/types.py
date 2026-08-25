"""右脑 Experience Layer 数据类型。

三大记忆类：
  user_interaction_profile   — 用户长期风格偏好
  heartnote                   — 情境情绪规律
  response_experience         — 回应成败经验（最高优先级）

检索入口：MemoryAnchor → 挂左脑实体
查询计划：MemoryQueryPlan → Preprocessor 信号 + anchors
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ── Anchor ────────────────────────────────────────────────────────────────────

AnchorType = Literal[
    "user", "person", "project", "task", "knowledge",
    "place", "routine", "asset", "event",
    "entity_edge", "slot", "current_session", "global_style",
]

AnchorRole = Literal[
    "topic", "subject", "object", "context",
    "trigger", "relationship", "evidence", "global_profile",
]

MemoryClass = Literal[
    "heartnote",
    "response_experience",
]

TTL = Literal["session", "short_term", "long_term"]


@dataclass
class MemoryAnchor:
    anchor_type: AnchorType
    anchor_id: str | None        # None = 全局（如 user_self, global_style）
    role: AnchorRole
    weight: float = 1.0
    confidence: float = 1.0


# ── Query Plan ────────────────────────────────────────────────────────────────

@dataclass
class CurrentSignals:
    """Preprocessor 检测到的当前轮信号，不存储，只用于检索优先级。"""
    affect_hint: str | None = None                          # "frustrated" / "satisfied" ...
    affect_intensity: Literal["low", "medium", "high"] | None = None
    dissatisfaction_signal: bool = False
    correction_signal: bool = False
    short_answer_needed: bool = False


@dataclass
class MemoryQueryPlan:
    user_id: str
    clean_text: str
    anchors: list[MemoryAnchor] = field(default_factory=list)
    current_signals: CurrentSignals = field(default_factory=CurrentSignals)


# ── Right Brain Memory ────────────────────────────────────────────────────────

@dataclass
class RightBrainMemory:
    id: str
    user_id: str
    memory_class: MemoryClass
    content: str
    condition: str | None           # 适用情境描述（可选）
    priority: float                 # 0~1，response_experience 通常最高
    confidence: float
    ttl: TTL
    metadata: dict[str, Any]        # 失败原因、next_time_policy 等
    evidence_turn_ids: list[str]
    evidence_memory_ids: list[str]
    created_at: str
    updated_at: str
    #: 本次检索命中的锚点得分（SUM(link.weight*link.confidence)，见
    #: store.search_by_anchors）。不落库——同一条记忆在不同查询下得分不同，
    #: 只在检索结果里携带，供最终排序把"检索相关度"混进静态 priority。
    anchor_score: float = 0.0


@dataclass
class RightBrainAnchorLink:
    id: str
    user_id: str
    right_memory_id: str
    anchor_type: AnchorType
    anchor_id: str | None
    role: AnchorRole
    weight: float
    confidence: float
    created_at: str


# ── Retrieval Result ──────────────────────────────────────────────────────────

@dataclass
class RightBrainContext:
    """右脑检索结果，供 Prompt Builder 使用。

    仅含 heartnote 和 response_experience。
    user_interaction_profile 由前刺层 (UserProfileStore) 独立管理。
    """
    response_experiences: list[RightBrainMemory] = field(default_factory=list)
    situation_patterns: list[RightBrainMemory] = field(default_factory=list)
    current_signals: CurrentSignals = field(default_factory=CurrentSignals)

    def is_empty(self) -> bool:
        return not (self.response_experiences or self.situation_patterns)

    def to_prompt_block(self) -> str:
        """生成注入 LLM prompt 的文本块。

        每条记忆前带 [YYYY-MM-DD] 日期——没有日期，下游模型对"什么时候的事"
        类问题（temporal reasoning）完全无从判断；日期一直存在 created_at 里，
        只是之前渲染时丢掉了。
        """
        lines: list[str] = []

        def _date(m: RightBrainMemory) -> str:
            d = (m.created_at or "")[:10]
            return f"[{d}] " if d else ""

        if self.response_experiences:
            lines.append("[Response experience]")
            for m in self.response_experiences:
                cond = f" (when: {m.condition})" if m.condition else ""
                lines.append(f"- {_date(m)}{m.content}{cond}")

        if self.situation_patterns:
            lines.append("[Situation pattern]")
            for m in self.situation_patterns:
                cond = f" (when: {m.condition})" if m.condition else ""
                lines.append(f"- {_date(m)}{m.content}{cond}")

        sigs = self.current_signals
        hints: list[str] = []
        if sigs.dissatisfaction_signal:
            hints.append("user shows dissatisfaction this turn")
        if sigs.correction_signal:
            hints.append("user is correcting assistant")
        if sigs.affect_hint:
            intensity = f" ({sigs.affect_intensity})" if sigs.affect_intensity else ""
            hints.append(f"affect={sigs.affect_hint}{intensity}")
        if hints:
            lines.append(f"[Current signals: {'; '.join(hints)}]")

        return "\n".join(lines)
