"""Integrate left/right retrieval into partitioned reply prompts."""

from __future__ import annotations

from typing import Literal

from voicemem.utils.audio.emotion.types import EmotionAttribution, TurnEmotionRecord, VAD
from voicemem.utils.audio.emotion.vad_trigger import NegativeVadTriggerConfig, is_negative_vad_significant
from voicemem.utils.fusion.left_channel import format_left_memory_block
from voicemem.utils.fusion.types import FusionRetrievalResult, ReplyContextPrompt, RightMemoryHit


def format_acoustic_block(turn: TurnEmotionRecord, *, negative_significant: bool) -> str:
    v, a = turn.vad.valence, turn.vad.arousal
    if negative_significant:
        tone = (
            "Acoustic state is significantly negative; respond with empathy and reassurance, "
            "avoid minimizing the user's feelings."
        )
    elif v >= 0.15:
        tone = "Acoustic state is neutral-to-positive; respond naturally and lightly."
    else:
        tone = "Acoustic state is neutral or slightly subdued; respond calmly and clearly."
    return f"[Acoustic] Turn V/A: valence={v:.2f}, arousal={a:.2f}.\n{tone}"


def format_persona_block(persona_text: str | None, *, enabled: bool) -> str:
    body = (persona_text or "").strip()
    if not body:
        body = (
            "No user persona is available yet; use a generally polite tone and do not invent "
            "long-term traits about the user."
        )
    header = "[Persona]"
    if not enabled:
        return f"{header}\n(Persona feature disabled)\n{body}"
    return f"{header}\n{body}"


def format_current_attribution_block(attr: EmotionAttribution) -> str:
    lines = ["[Current-turn emotion attribution]"]
    lines.append(attr.analysis_text)
    if attr.emotion is not None:
        e = attr.emotion
        lines.append(
            f"Structured emotion: {e.label}; V/A=({e.valence:.2f},{e.arousal:.2f}); "
            f"intensity={e.intensity}; confidence={e.confidence:.2f}"
        )
    if attr.acoustic_evidence:
        lines.append("Acoustic evidence: " + "; ".join(attr.acoustic_evidence))
    if attr.semantic_evidence:
        lines.append("Semantic evidence: " + "; ".join(attr.semantic_evidence))
    return "\n".join(lines)


def _format_historical_right_hits(hits: list[RightMemoryHit]) -> str:
    if not hits:
        return ""
    lines = ["Historical emotion attributions relevant to this turn:"]
    for i, h in enumerate(hits, 1):
        snippet = ", ".join(h.retrieval_snippet) if h.retrieval_snippet else "—"
        lines.append(
            f"{i}. turn={h.turn_id}; V/A at time=({h.vad.valence:.2f},{h.vad.arousal:.2f}); "
            f"keywords: {snippet}; attribution: {h.analysis_text}"
        )
    return "\n".join(lines)


def _format_emotional_block(
    *,
    mode: Literal["normal", "anomaly"],
    turn: TurnEmotionRecord,
    right_hits: list[RightMemoryHit],
    negative_significant: bool,
    acoustic_block: str,
    graph_emotion_block: str,
    current_attribution_block: str,
) -> str:
    parts: list[str] = ["[Emotional context]"]

    if mode == "anomaly":
        if current_attribution_block.strip():
            parts.append(current_attribution_block.strip())
        hist = _format_historical_right_hits(right_hits)
        if hist:
            parts.append(hist)
    else:
        parts.append(acoustic_block.strip())
        if right_hits and negative_significant:
            parts.append(_format_historical_right_hits(right_hits))
        elif right_hits:
            parts.append(_format_historical_right_hits(right_hits))
        elif not negative_significant:
            parts.append(
                "No relevant historical anomaly attributions; use the acoustic block for tone."
            )
        else:
            parts.append(
                "Acoustic state is significantly negative, but no directly relevant historical "
                "attributions were found; respond calmly and with empathy."
            )

    if graph_emotion_block.strip():
        parts.append(graph_emotion_block.strip())

    return "\n\n".join(parts)


def build_left_only_reply_context_prompt(
    retrieval: FusionRetrievalResult,
    *,
    asr_text: str | None = None,
) -> ReplyContextPrompt:
    """Normal turn, left brain only: Mem0 semantic retrieval + semantic graph appendix."""
    query = (asr_text or retrieval.asr_text or "").strip()
    left_summary = format_left_memory_block(retrieval.left_hits)

    semantic_lines = [
        "[Semantic memory] Left-brain facts relevant to this turn:",
        left_summary,
    ]
    if retrieval.left_graph_appendix.strip():
        semantic_lines.append(retrieval.left_graph_appendix.strip())
    semantic = "\n".join(semantic_lines)

    system_block = (
        "You are a conversational assistant with access to long-term factual memories. "
        "Use the semantic memories and entity relationships below naturally when replying; "
        "do not recite memory IDs or list memories verbatim to the user. "
        "If two memories describe conflicting values for the same fact, trust the one with "
        "the more recent 'as of' timestamp."
    )
    user_block = f"User turn (ASR):\n{query}\n\n{semantic}"

    return ReplyContextPrompt(
        semantic_block=semantic,
        emotional_block="",
        left_context_summary=left_summary,
        system_block=system_block,
        user_block=user_block,
    )


def build_reply_context_prompt(
    retrieval: FusionRetrievalResult,
    *,
    asr_text: str | None = None,
    turn: TurnEmotionRecord | None = None,
    mode: Literal["normal", "anomaly"] = "normal",
    current_attribution: EmotionAttribution | None = None,
    graph_context: str | None = None,
    persona_text: str | None = None,
    persona_enabled: bool = False,
    vad_config: NegativeVadTriggerConfig | None = None,
) -> ReplyContextPrompt:
    query = (asr_text or retrieval.asr_text or "").strip()
    left_summary = format_left_memory_block(retrieval.left_hits)

    semantic_lines = [
        "[Semantic memory] Left-brain facts relevant to this turn:",
        format_left_memory_block(retrieval.left_hits),
    ]
    if retrieval.left_graph_appendix.strip():
        semantic_lines.append(retrieval.left_graph_appendix.strip())
    semantic = "\n".join(semantic_lines)

    graph_emotion_block = (graph_context or "").strip()
    negative_significant = False
    acoustic_block = ""
    current_attr_block = ""
    if turn is not None:
        cfg = vad_config
        if cfg is not None:
            negative_significant = is_negative_vad_significant(turn.vad, cfg)
        acoustic_block = format_acoustic_block(turn, negative_significant=negative_significant)
    if current_attribution is not None:
        current_attr_block = format_current_attribution_block(current_attribution)

    persona_block = format_persona_block(persona_text, enabled=persona_enabled)

    emotional = _format_emotional_block(
        mode=mode,
        turn=turn or TurnEmotionRecord(turn_id="", session_id="", vad=VAD(0.0, 0.0)),
        right_hits=retrieval.right_hits,
        negative_significant=negative_significant,
        acoustic_block=acoustic_block,
        graph_emotion_block=graph_emotion_block,
        current_attribution_block=current_attr_block,
    )

    system_block = (
        "You are a conversational assistant with access to long-term memories. "
        "Use the semantic, emotional, and persona sections below when replying; "
        "do not recite memory sources verbatim to the user. "
        "If two memories describe conflicting values for the same fact, trust the one with "
        "the more recent 'as of' timestamp."
    )
    user_parts = [f"User turn (ASR):\n{query}", semantic, persona_block, emotional]
    user_block = "\n\n".join(p for p in user_parts if p.strip())

    return ReplyContextPrompt(
        semantic_block=semantic,
        emotional_block=emotional,
        left_context_summary=left_summary,
        persona_block=persona_block,
        acoustic_block=acoustic_block,
        graph_emotion_block=graph_emotion_block,
        current_attribution_block=current_attr_block,
        system_block=system_block,
        user_block=user_block,
    )
