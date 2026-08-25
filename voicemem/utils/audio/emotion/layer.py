"""右脑情绪层：audio -> VAD -> 负面显著判定。

本层只负责声学状态与异常标记；异常轮的多模态归因、情绪图检索和写入
由 ``voicemem.utils.fusion.orchestrator`` / ``voicemem.utils.fusion.reply_memory`` 编排。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voicemem.utils.audio.emotion.memory_store import EmotionMemoryStore

from voicemem.utils.audio.emotion.types import (
    EmotionAttribution,
    TurnEmotionRecord,
    VAD,
)
from voicemem.utils.audio.emotion.vad_audio import HeuristicWavVADEstimator, VADEstimator
from voicemem.utils.audio.emotion.vad_trigger import is_negative_vad_significant


@dataclass
class EmotionLayerConfig:
    """情绪层策略参数（可按业务调参）。"""

    valence_negative_threshold: float = -0.4
    min_arousal_for_anomaly: float | None = 0.25


@dataclass
class EmotionLayerResult:
    turn: TurnEmotionRecord
    attributions: list[EmotionAttribution] = field(default_factory=list)
    needs_attribution: bool = False


class EmotionLayer:
    """右脑 · 情绪层：每轮 VAD 存储；负面显著轮由 pipeline 归因。"""

    def __init__(
        self,
        config: EmotionLayerConfig | None = None,
        *,
        vad_estimator: VADEstimator | None = None,
        memory_store: EmotionMemoryStore | None = None,
        user_id: str = "default",
    ) -> None:
        self._config = config or EmotionLayerConfig()
        self._vad = vad_estimator or HeuristicWavVADEstimator()
        self._memory_store: EmotionMemoryStore | None = memory_store
        self._user_id = user_id

        self._session_id: str | None = None
        self._last_user_ts: float | None = None
        self._utter_idx_session = 0

    @property
    def config(self) -> EmotionLayerConfig:
        return self._config

    @property
    def last_user_timestamp_s(self) -> float | None:
        return self._last_user_ts

    def apply_attribution(self, attr: EmotionAttribution) -> None:
        """异常轮 pipeline 完成多模态归因后写回存储。"""
        if self._memory_store is not None:
            self._memory_store.append_attribution(attr, user_id=self._user_id)

    def process_user_turn(
        self,
        *,
        turn_id: str,
        session_id: str,
        audio_path: str | None = None,
        precomputed_vad: VAD | None = None,
        timestamp_s: float | None = None,
        time_gap_from_prev_turn_s: float | None = None,
    ) -> EmotionLayerResult:
        """处理用户轮次：VAD + 存 turn；负面显著时 ``needs_attribution=True``。"""

        if precomputed_vad is None:
            if not audio_path:
                raise ValueError("需要提供 audio_path 或 precomputed_vad")
            vad = self._vad.estimate(audio_path)
        else:
            vad = precomputed_vad

        self._session_id = session_id

        record = TurnEmotionRecord(
            turn_id=turn_id,
            session_id=session_id,
            vad=vad,
            timestamp_s=timestamp_s,
            user_utterance_index=self._utter_idx_session,
            time_gap_from_prev_turn_s=time_gap_from_prev_turn_s,
        )

        needs_attribution = is_negative_vad_significant(vad, self._config)

        self._utter_idx_session += 1
        if timestamp_s is not None:
            self._last_user_ts = float(timestamp_s)

        result = EmotionLayerResult(
            turn=record,
            attributions=[],
            needs_attribution=needs_attribution,
        )
        if self._memory_store is not None:
            self._memory_store.persist_layer_result(result, user_id=self._user_id)
        return result
