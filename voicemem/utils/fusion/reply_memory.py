"""回复记忆构筑：统一检索 + 分区 prompt + 异常轮归因前上下文。"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from voicemem.utils.audio.emotion.attribution_qwen_omni import OmniTurnAttributor
from voicemem.utils.audio.emotion.graph_memory import EmotionGraphMemoryStore, format_emotion_graph_context
from voicemem.utils.audio.emotion.memory_store import EmotionMemoryStore
from voicemem.utils.audio.emotion.query_terms import build_query_terms
from voicemem.utils.audio.emotion.types import EmotionAttribution, TurnEmotionRecord
from voicemem.utils.fusion.config import FusionConfig
from voicemem.utils.fusion.left_channel import (
    LeftBrainSearchClient,
    format_left_memory_block,
    search_left_for_reply,
)
from voicemem.utils.fusion.prompt_builder import (
    build_left_only_reply_context_prompt,
    build_reply_context_prompt,
)
from voicemem.utils.fusion.right_channel import RightChannelRelevanceFilter, search_right_channel
from voicemem.utils.fusion.types import (
    AnomalyTurnResult,
    FusionRetrievalResult,
    LeftMemoryHit,
    ReplyContextBundle,
    ReplyRetrievalBundle,
    RightMemoryHit,
)



def retrieve_for_reply(
    *,
    asr_text: str,
    turn: TurnEmotionRecord,
    user_id: str,
    left_brain: LeftBrainSearchClient | None,
    emotion_store: EmotionMemoryStore | None = None,
    emotion_graph: EmotionGraphMemoryStore | None = None,
    relevance_filter: RightChannelRelevanceFilter | None = None,
    config: FusionConfig | None = None,
    include_right_json: bool = True,
    include_emotion_graph: bool = True,
    exclude_turn_id: str | None = None,
) -> ReplyRetrievalBundle:
    """左脑（可选图）、右脑 JSON、情绪图统一检索。"""
    cfg = config or FusionConfig()
    query = (asr_text or "").strip()

    left_hits: list[LeftMemoryHit] = []
    left_graph_appendix = ""
    if left_brain is not None and query:
        left_hits, left_graph_appendix = search_left_for_reply(
            left_brain,
            asr_text=query,
            user_id=user_id,
            config=cfg,
            use_graph=cfg.left_use_graph,
        )

    right_hits: list[RightMemoryHit] = []
    if include_right_json and emotion_store is not None and query:
        right_hits = search_right_channel(
            emotion_store,
            asr_text=query,
            current_vad=turn.vad,
            user_id=user_id,
            relevance_filter=relevance_filter,
            exclude_turn_id=exclude_turn_id,
            config=cfg,
        )

    left_summary = format_left_memory_block(left_hits)
    graph_emotion_context = ""
    if (
        include_emotion_graph
        and emotion_graph is not None
        and query
        and emotion_graph.has_episodes(user_id=user_id)
    ):
        terms = build_query_terms(query, left_summary)
        if terms:
            graph_hits = emotion_graph.search(
                user_id=user_id,
                query_terms=terms,
                current_vad=turn.vad,
                limit=cfg.emotion_graph_search_limit,
            )
            graph_emotion_context = format_emotion_graph_context(graph_hits)

    retrieval = FusionRetrievalResult(
        asr_text=query,
        left_hits=left_hits,
        right_hits=right_hits,
        left_graph_appendix=left_graph_appendix,
    )
    return ReplyRetrievalBundle(
        retrieval=retrieval,
        graph_emotion_context=graph_emotion_context,
    )


@runtime_checkable
class PersonaProvider(Protocol):
    def get_persona_text(self, *, user_id: str) -> str | None:
        ...


def retrieve_left_for_normal_turn(
    *,
    asr_text: str,
    user_id: str,
    left_brain: LeftBrainSearchClient,
    config: FusionConfig | None = None,
) -> FusionRetrievalResult:
    """正常轮左脑检索（Mem0 对齐：``search(query, top_k, threshold)`` + 可选 ``search_with_graph``）。"""
    cfg = config or FusionConfig()
    query = (asr_text or "").strip()
    left_hits: list[LeftMemoryHit] = []
    left_graph_appendix = ""
    if query:
        left_hits, left_graph_appendix = search_left_for_reply(
            left_brain,
            asr_text=query,
            user_id=user_id,
            config=cfg,
            use_graph=cfg.left_use_graph,
        )
    return FusionRetrievalResult(
        asr_text=query,
        left_hits=left_hits,
        right_hits=[],
        left_graph_appendix=left_graph_appendix,
    )


def build_normal_left_reply_memory(
    *,
    asr_text: str,
    user_id: str,
    left_brain: LeftBrainSearchClient,
    config: FusionConfig | None = None,
) -> ReplyContextBundle:
    """正常轮记忆构筑：仅左脑语义向量 + 语义图 → 回复用 prompt（不检索右脑）。"""
    cfg = config or FusionConfig()
    retrieval = retrieve_left_for_normal_turn(
        asr_text=asr_text,
        user_id=user_id,
        left_brain=left_brain,
        config=cfg,
    )
    prompt = build_left_only_reply_context_prompt(retrieval, asr_text=asr_text)
    return ReplyContextBundle(
        retrieval=ReplyRetrievalBundle(retrieval=retrieval, graph_emotion_context=""),
        prompt=prompt,
    )


class StubPersonaProvider:
    """占位画像：不读 PersonaStore，返回 None 由 prompt_builder 填默认句。"""

    def get_persona_text(self, *, user_id: str) -> str | None:
        _ = user_id
        return None


def build_reply_memory(
    *,
    asr_text: str,
    turn: TurnEmotionRecord,
    user_id: str,
    mode: Literal["normal", "anomaly"],
    left_brain: LeftBrainSearchClient | None = None,
    emotion_store: EmotionMemoryStore | None = None,
    emotion_graph: EmotionGraphMemoryStore | None = None,
    current_attribution: EmotionAttribution | None = None,
    relevance_filter: RightChannelRelevanceFilter | None = None,
    persona_provider: PersonaProvider | None = None,
    config: FusionConfig | None = None,
) -> ReplyContextBundle:
    """构筑本轮回复用记忆上下文（只读检索 + prompt，不调用回复 LLM）。"""
    cfg = config or FusionConfig()

    include_right_json = True
    if mode == "normal":
        include_right_json = cfg.reply_include_right_json_on_normal

    include_emotion_graph = True
    if mode == "normal":
        include_emotion_graph = cfg.reply_include_emotion_graph_on_normal

    retrieved = retrieve_for_reply(
        asr_text=asr_text,
        turn=turn,
        user_id=user_id,
        left_brain=left_brain,
        emotion_store=emotion_store,
        emotion_graph=emotion_graph,
        relevance_filter=relevance_filter,
        config=cfg,
        include_right_json=include_right_json,
        include_emotion_graph=include_emotion_graph,
        exclude_turn_id=turn.turn_id,
    )

    provider = persona_provider or StubPersonaProvider()
    persona_text = provider.get_persona_text(user_id=user_id)

    prompt = build_reply_context_prompt(
        retrieved.retrieval,
        asr_text=asr_text,
        turn=turn,
        mode=mode,
        current_attribution=current_attribution,
        graph_context=retrieved.graph_emotion_context,
        persona_text=persona_text,
        persona_enabled=cfg.persona_enabled,
        vad_config=cfg,
    )

    return ReplyContextBundle(retrieval=retrieved, prompt=prompt)


def run_anomaly_turn(
    *,
    asr_text: str,
    audio_path: str,
    turn: TurnEmotionRecord,
    user_id: str,
    left_brain: LeftBrainSearchClient,
    emotion_store: EmotionMemoryStore,
    emotion_graph: EmotionGraphMemoryStore,
    omni_attributor: OmniTurnAttributor,
    relevance_filter: RightChannelRelevanceFilter | None = None,
    config: FusionConfig | None = None,
) -> AnomalyTurnResult:
    """异常轮：检索 → Omni 归因 → 写情绪图（归因前 reply 上下文供 Omni 使用）。"""
    cfg = config or FusionConfig()

    retrieved = retrieve_for_reply(
        asr_text=asr_text,
        turn=turn,
        user_id=user_id,
        left_brain=left_brain,
        emotion_store=emotion_store,
        emotion_graph=emotion_graph,
        relevance_filter=relevance_filter,
        config=cfg,
        include_right_json=True,
        include_emotion_graph=True,
        exclude_turn_id=turn.turn_id,
    )
    pre_reply = build_reply_context_prompt(
        retrieved.retrieval,
        asr_text=asr_text,
        turn=turn,
        mode="anomaly",
        graph_context=retrieved.graph_emotion_context,
        vad_config=cfg,
    )

    omni_result = omni_attributor.analyze_turn_with_audio(
        audio_path=audio_path,
        asr_text=asr_text,
        left_memory_block=pre_reply.left_context_summary,
        emotion_graph_context=retrieved.graph_emotion_context,
        turn=turn,
    )

    meta: dict[str, Any] = {"attributor": "qwen_omni"}
    if omni_result.retrieval_snippet:
        meta["retrieval_snippet"] = list(omni_result.retrieval_snippet)

    attribution = EmotionAttribution(
        turn_id=turn.turn_id,
        session_id=turn.session_id,
        trigger="anomaly",
        analysis_text=omni_result.analysis_text,
        vad_at_trigger=turn.vad,
        left_context_summary=pre_reply.left_context_summary,
        emotion=omni_result.emotion,
        acoustic_evidence=list(omni_result.acoustic_evidence),
        semantic_evidence=list(omni_result.semantic_evidence),
        related_nodes=list(omni_result.related_nodes),
        graph_delta=omni_result.graph_delta,
        user_utterance_index=turn.user_utterance_index,
        metadata=meta,
    )
    emotion_graph.add_attribution(user_id=user_id, turn=turn, attribution=attribution)

    return AnomalyTurnResult(
        retrieval=retrieved.retrieval,
        reply_prompt=pre_reply,
        pre_attribution_reply=pre_reply,
        attribution=attribution,
    )


def build_omni_attribution_context(
    *,
    asr_text: str,
    turn: TurnEmotionRecord,
    user_id: str,
    left_brain: LeftBrainSearchClient | None,
    emotion_graph: EmotionGraphMemoryStore | None = None,
    config: FusionConfig | None = None,
) -> tuple[str, str]:
    """为 Omni 归因准备左脑摘要与情绪图上下文（不构筑回复 prompt）。"""
    cfg = config or FusionConfig()
    query = (asr_text or "").strip()

    left_hits: list[LeftMemoryHit] = []
    left_graph_appendix = ""
    if left_brain is not None and query:
        left_hits, left_graph_appendix = search_left_for_reply(
            left_brain,
            asr_text=query,
            user_id=user_id,
            config=cfg,
            use_graph=cfg.left_use_graph,
        )

    left_summary = format_left_memory_block(left_hits)
    if left_graph_appendix.strip():
        left_summary = f"{left_summary}\n\n{left_graph_appendix.strip()}"

    graph_emotion_context = ""
    if emotion_graph is not None and query and emotion_graph.has_episodes(user_id=user_id):
        terms = build_query_terms(query, left_summary)
        if terms:
            graph_hits = emotion_graph.search(
                user_id=user_id,
                query_terms=terms,
                current_vad=turn.vad,
                limit=cfg.emotion_graph_search_limit,
            )
            graph_emotion_context = format_emotion_graph_context(graph_hits)

    return left_summary, graph_emotion_context


def run_rightbrain_memory_extract(
    *,
    asr_text: str,
    audio_path: str,
    turn: TurnEmotionRecord,
    user_id: str,
    left_brain: LeftBrainSearchClient | None,
    emotion_graph: EmotionGraphMemoryStore,
    omni_attributor: OmniTurnAttributor,
    config: FusionConfig | None = None,
) -> EmotionAttribution:
    """异常轮右脑记忆提取：左脑摘要 + Omni 多模态归因 + 情绪图写入（不含回复 prompt）。

    归因 JSON 落盘由调用方 ``EmotionLayer.apply_attribution`` 负责，与 ``run_anomaly_turn`` 一致。
    """
    cfg = config or FusionConfig()
    left_summary, graph_emotion_context = build_omni_attribution_context(
        asr_text=asr_text,
        turn=turn,
        user_id=user_id,
        left_brain=left_brain,
        emotion_graph=emotion_graph,
        config=cfg,
    )

    omni_result = omni_attributor.analyze_turn_with_audio(
        audio_path=audio_path,
        asr_text=asr_text,
        left_memory_block=left_summary,
        emotion_graph_context=graph_emotion_context or None,
        turn=turn,
    )

    meta: dict[str, Any] = {"attributor": "qwen_omni", "pipeline": "rightbrain_memory_extract"}
    if omni_result.retrieval_snippet:
        meta["retrieval_snippet"] = list(omni_result.retrieval_snippet)

    attribution = EmotionAttribution(
        turn_id=turn.turn_id,
        session_id=turn.session_id,
        trigger="anomaly",
        analysis_text=omni_result.analysis_text,
        vad_at_trigger=turn.vad,
        left_context_summary=left_summary,
        emotion=omni_result.emotion,
        acoustic_evidence=list(omni_result.acoustic_evidence),
        semantic_evidence=list(omni_result.semantic_evidence),
        related_nodes=list(omni_result.related_nodes),
        graph_delta=omni_result.graph_delta,
        user_utterance_index=turn.user_utterance_index,
        metadata=meta,
    )
    emotion_graph.add_attribution(user_id=user_id, turn=turn, attribution=attribution)
    return attribution
