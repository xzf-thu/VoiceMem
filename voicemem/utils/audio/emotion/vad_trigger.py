"""VAD 负面显著判定（段内 fusion 触发，EMA 不参与）。"""

from __future__ import annotations

from typing import Protocol

from voicemem.utils.audio.emotion.types import VAD


class NegativeVadTriggerConfig(Protocol):
    valence_negative_threshold: float
    min_arousal_for_anomaly: float | None


def is_negative_vad_significant(vad: VAD, config: NegativeVadTriggerConfig) -> bool:
    """本轮 VAD 是否「负面显著」：valence 低于阈值，且可选 arousal 下限。"""
    if vad.valence > float(config.valence_negative_threshold):
        return False
    min_ar = config.min_arousal_for_anomaly
    if min_ar is not None and vad.arousal < float(min_ar):
        return False
    return True
