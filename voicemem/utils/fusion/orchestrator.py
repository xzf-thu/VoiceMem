"""Turn 编排：EmotionLayer + 异常轮 fusion + 回复记忆构筑。"""

from __future__ import annotations

from dataclasses import dataclass

from voicemem.utils.audio.emotion.attribution_qwen_omni import OmniTurnAttributor
from voicemem.utils.audio.emotion.graph_memory import EmotionGraphMemoryStore
from voicemem.utils.audio.emotion.layer import EmotionLayer, EmotionLayerResult
from voicemem.utils.audio.emotion.memory_store import EmotionMemoryStore
from voicemem.utils.audio.emotion.types import VAD
from voicemem.utils.fusion.config import FusionConfig
from voicemem.utils.fusion.left_channel import LeftBrainSearchClient
from voicemem.utils.fusion.reply_memory import (
    PersonaProvider,
    build_normal_left_reply_memory,
    build_reply_memory,
    run_anomaly_turn,
)
from voicemem.utils.fusion.right_channel import RightChannelRelevanceFilter
from voicemem.utils.fusion.types import AnomalyTurnResult, ReplyContextPrompt


@dataclass
class TurnProcessResult:
    emotion: EmotionLayerResult
    fusion: AnomalyTurnResult | None = None
    reply_context: ReplyContextPrompt | None = None


def process_user_turn(
    *,
    emotion_layer: EmotionLayer,
    turn_id: str,
    session_id: str,
    asr_text: str,
    audio_path: str | None = None,
    precomputed_vad: VAD | None = None,
    timestamp_s: float | None = None,
    time_gap_from_prev_turn_s: float | None = None,
    user_id: str,
    left_brain: LeftBrainSearchClient | None = None,
    emotion_store: EmotionMemoryStore | None = None,
    emotion_graph: EmotionGraphMemoryStore | None = None,
    omni_attributor: OmniTurnAttributor | None = None,
    relevance_filter: RightChannelRelevanceFilter | None = None,
    persona_provider: PersonaProvider | None = None,
    fusion_config: FusionConfig | None = None,
) -> TurnProcessResult:
    """统一一轮：VAD → 异常则归因写图 → 构筑回复记忆上下文。"""
    emotion_result = emotion_layer.process_user_turn(
        turn_id=turn_id,
        session_id=session_id,
        audio_path=audio_path,
        precomputed_vad=precomputed_vad,
        timestamp_s=timestamp_s,
        time_gap_from_prev_turn_s=time_gap_from_prev_turn_s,
    )

    fusion_result: AnomalyTurnResult | None = None
    if emotion_result.needs_attribution:
        if (
            left_brain is None
            or emotion_store is None
            or emotion_graph is None
            or omni_attributor is None
        ):
            raise ValueError(
                "needs_attribution=True 时需要 left_brain、emotion_store、emotion_graph、omni_attributor"
            )
        if not audio_path:
            raise ValueError("needs_attribution=True 时需要 audio_path")

        fusion_result = run_anomaly_turn(
            asr_text=asr_text,
            audio_path=audio_path,
            turn=emotion_result.turn,
            user_id=user_id,
            left_brain=left_brain,
            emotion_store=emotion_store,
            emotion_graph=emotion_graph,
            omni_attributor=omni_attributor,
            relevance_filter=relevance_filter,
            config=fusion_config,
        )
        emotion_layer.apply_attribution(fusion_result.attribution)
        emotion_result.attributions.append(fusion_result.attribution)

    reply_context: ReplyContextPrompt | None = None
    if left_brain is not None:
        if emotion_result.needs_attribution:
            current_attr = fusion_result.attribution if fusion_result is not None else None
            bundle = build_reply_memory(
                asr_text=asr_text,
                turn=emotion_result.turn,
                user_id=user_id,
                mode="anomaly",
                left_brain=left_brain,
                emotion_store=emotion_store,
                emotion_graph=emotion_graph,
                current_attribution=current_attr,
                relevance_filter=relevance_filter,
                persona_provider=persona_provider,
                config=fusion_config,
            )
        else:
            bundle = build_normal_left_reply_memory(
                asr_text=asr_text,
                user_id=user_id,
                left_brain=left_brain,
                config=fusion_config,
            )
        reply_context = bundle.prompt

    return TurnProcessResult(
        emotion=emotion_result,
        fusion=fusion_result,
        reply_context=reply_context,
    )
