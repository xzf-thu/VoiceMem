"""异常轮多模态情绪归因 prompt 与 JSON 解析。"""

from __future__ import annotations

import json
import re
from typing import Any

from voicemem.utils.audio.emotion.types import (
    EmotionGraphDelta,
    EmotionGraphEdgeInput,
    EmotionGraphNodeInput,
    EmotionSignal,
    TurnAttributionLLMResult,
    TurnEmotionRecord,
)

TURN_ATTRIBUTION_SYSTEM_PROMPT = """You are the right-brain emotion attribution assistant. Listen to the user's audio for this turn and combine ASR semantics, acoustic VAD scores, left-brain factual context (if any), and historical emotion-graph context to explain likely causes of the user's current negative affect.

Requirements:
- Use BOTH paralinguistic cues in the audio (tone, pace, sighs, pauses, etc.) AND ASR semantics.
- Left-brain context and historical emotion graph are references only; do not mechanically reuse past causes.
- Reply with a single valid JSON object only—no markdown fences, no extra prose, no follow-up chat.
- Stop immediately after the closing `}`; never output "Human:", "Assistant:", or repeat turn_id/session_id.
- Use exact field names in emotion: `valence` and `arousal` (not valvalue/arrowal).

JSON schema:
{
  "analysis_text": "English attribution explanation",
  "emotion": {
    "label": "frustration|anxiety|sadness|anger|stress|negative|...",
    "valence": -0.7,
    "arousal": 0.6,
    "intensity": "low|medium|high",
    "confidence": 0.0
  },
  "acoustic_evidence": ["acoustic evidence 1", "acoustic evidence 2"],
  "semantic_evidence": ["semantic evidence 1", "semantic evidence 2"],
  "related_nodes": [
    {"local_id": "topic_1", "name": "main experiment", "node_type": "Topic", "metadata": {}}
  ],
  "graph_delta": {
    "nodes": [
      {"local_id": "topic_1", "name": "main experiment", "node_type": "Topic", "metadata": {}}
    ],
    "edges": [
      {"source": "user", "target": "topic_1", "edge_type": "EMOTIONAL_REACTION_TO", "emotion_label": "frustration", "description": "User shows frustration toward the blocked main experiment", "metadata": {}}
    ]
  },
  "retrieval_snippet": ["main experiment", "frustration"]
}"""


def build_turn_attribution_user_prompt(
    *,
    turn: TurnEmotionRecord,
    asr_text: str | None,
    left_context_summary: str | None,
    emotion_graph_context: str | None,
) -> str:
    lines = [
        f"turn_id={turn.turn_id}",
        f"session_id={turn.session_id}",
        f"Current-turn VAD: valence={turn.vad.valence:.2f}, arousal={turn.vad.arousal:.2f}",
    ]
    if asr_text:
        lines.append(f"ASR text:\n{asr_text.strip()}")
    if left_context_summary:
        lines.append(f"Left-brain context summary:\n{left_context_summary.strip()}")
    if emotion_graph_context:
        lines.append(f"Historical emotion-graph context:\n{emotion_graph_context.strip()}")
    return "\n".join(lines)


def _strip_code_fence(s: str) -> str:
    text = s.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```", text, re.I)
    if fence:
        return fence.group(1).strip()
    return text


_LEAK_MARKERS = re.compile(
    r"\n\s*(?:Human|Assistant|User|System)\s*:|\n\s*turn_id\s*=",
    re.I,
)


def _trim_leaked_chat_suffix(s: str) -> str:
    """模型有时在 JSON 后继续生成对话模板，截断后再解析。"""
    m = _LEAK_MARKERS.search(s)
    if m:
        return s[: m.start()].rstrip()
    return s


def _repair_llm_json_typos(blob: str) -> str:
    """修正常见键名拼写错误。"""
    out = blob
    out = re.sub(r'"valvalue"', '"valence"', out)
    out = re.sub(r'"arrowal"', '"arousal"', out)
    return out


def _sanitize_json_blob(blob: str) -> str:
    """修正常见 LLM JSON 瑕疵（尾随逗号等）。"""
    out = blob.strip()
    out = re.sub(r",\s*}", "}", out)
    out = re.sub(r",\s*]", "]", out)
    return out


def _attempt_close_json_blob(blob: str) -> str:
    """对截断或未闭合的 JSON 做最小补全（如空的 graph_delta）。"""
    t = blob.rstrip()
    t = re.sub(r",\s*$", "", t)
    if re.search(r'"graph_delta"\s*:\s*\{\s*$', t):
        t += '"nodes": [], "edges": []'
    depth = 0
    in_string = False
    escape = False
    for ch in t:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    if depth > 0:
        t += "}" * depth
    return t


def _balanced_object_slice(s: str, start: int) -> str | None:
    """从第一个 ``{`` 起做括号配对，截取完整 JSON 对象（忽略后续说明文字）。"""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _loads_json_object(blob: str) -> dict[str, Any]:
    cleaned = _sanitize_json_blob(blob)
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned)
    except json.JSONDecodeError:
        obj = json.loads(cleaned)
    if not isinstance(obj, dict):
        raise ValueError("JSON root must be object")
    return obj


def _extract_json_object(raw: str) -> dict[str, Any]:
    s = _trim_leaked_chat_suffix(_strip_code_fence(raw))
    start = s.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in response: {s[:280]!r}")

    tail = _repair_llm_json_typos(s[start:])
    candidates: list[str] = []
    balanced = _balanced_object_slice(s, start)
    if balanced:
        candidates.append(_repair_llm_json_typos(balanced))
    closed = _attempt_close_json_blob(tail)
    balanced_closed = _balanced_object_slice(closed, 0)
    if balanced_closed:
        candidates.append(_repair_llm_json_typos(balanced_closed))
    candidates.append(closed)
    candidates.append(tail)

    last_err: json.JSONDecodeError | None = None
    seen: set[str] = set()
    for blob in candidates:
        if blob in seen:
            continue
        seen.add(blob)
        try:
            return _loads_json_object(blob)
        except json.JSONDecodeError as e:
            last_err = e

    snippet = (balanced or balanced_closed or tail)[:600]
    msg = f"invalid JSON in Omni attribution response: {last_err}; snippet={snippet!r}"
    raise ValueError(msg) from last_err


def parse_turn_attribution_response(raw: str) -> TurnAttributionLLMResult:
    obj = _extract_json_object(raw)
    analysis = str(obj.get("analysis_text", "")).strip()
    if not analysis:
        raise ValueError("analysis_text is required")
    emotion_raw = obj.get("emotion")
    if not isinstance(emotion_raw, dict):
        raise ValueError("emotion object is required")
    label = str(emotion_raw.get("label", "")).strip() or "negative"
    intensity = emotion_raw.get("intensity")
    if intensity not in ("low", "medium", "high"):
        intensity = "medium"
    valence_raw = emotion_raw.get("valence", emotion_raw.get("valvalue", 0.0))
    arousal_raw = emotion_raw.get("arousal", emotion_raw.get("arrowal", 0.0))
    emotion = EmotionSignal(
        label=label,
        valence=float(valence_raw),
        arousal=float(arousal_raw),
        intensity=intensity,
        confidence=float(emotion_raw.get("confidence", 0.0)),
    )
    related_nodes = [_parse_node(x) for x in obj.get("related_nodes") or []]
    related_nodes = [x for x in related_nodes if x is not None]
    graph_raw = obj.get("graph_delta") if isinstance(obj.get("graph_delta"), dict) else {}
    graph_delta = EmotionGraphDelta(
        nodes=[x for n in graph_raw.get("nodes", []) if (x := _parse_node(n)) is not None],
        edges=[x for e in graph_raw.get("edges", []) if (x := _parse_edge(e)) is not None],
    )
    snippet_raw = obj.get("retrieval_snippet") or []
    if not isinstance(snippet_raw, list):
        raise ValueError("retrieval_snippet must be a list")
    snippet = [str(x).strip() for x in snippet_raw if str(x).strip()]
    return TurnAttributionLLMResult(
        analysis_text=analysis,
        emotion=emotion,
        acoustic_evidence=_as_str_list(obj.get("acoustic_evidence")),
        semantic_evidence=_as_str_list(obj.get("semantic_evidence")),
        related_nodes=related_nodes,
        graph_delta=graph_delta,
        retrieval_snippet=snippet,
    )


def _parse_node(raw: Any) -> EmotionGraphNodeInput | None:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name", "")).strip()
    if not name:
        return None
    return EmotionGraphNodeInput(
        local_id=str(raw.get("local_id") or raw.get("id") or name),
        name=name,
        node_type=raw.get("node_type") or raw.get("type") or "Entity",
        metadata=raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
    )


def _parse_edge(raw: Any) -> EmotionGraphEdgeInput | None:
    if not isinstance(raw, dict):
        return None
    source = str(raw.get("source", "")).strip()
    target = str(raw.get("target", "")).strip()
    if not source or not target:
        return None
    return EmotionGraphEdgeInput(
        source=source,
        target=target,
        edge_type=raw.get("edge_type") or raw.get("type") or "EMOTIONAL_REACTION_TO",
        description=str(raw.get("description", "")).strip(),
        emotion_label=raw.get("emotion_label"),
        metadata=raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
    )


def _as_str_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]
