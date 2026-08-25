"""Fusion 默认配置。"""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class FusionConfig:
    valence_negative_threshold: float = -0.35
    min_arousal_for_anomaly: float | None = 0.25
    mem0_top_k: int = 5
    mem0_threshold: float = 0.1
    mem0_rerank: bool = False
    right_filter_model: str | None = None
    left_use_graph: bool = True
    reply_include_emotion_graph_on_normal: bool = False
    reply_include_right_json_on_normal: bool = False
    persona_enabled: bool = False
    emotion_graph_search_limit: int = 5

    def resolved_right_filter_model(self) -> str:
        return (
            self.right_filter_model
            or os.environ.get("OPENAI_MODEL", "").strip()
            or "gpt-4o-mini"
        ).strip()
