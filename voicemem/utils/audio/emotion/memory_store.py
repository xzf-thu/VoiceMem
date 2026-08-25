"""右脑情绪记忆持久化（JSON）。

默认路径：``<memory>/emotion_memory.json``，按 ``user_id`` 分桶存储：

- ``turns``：每轮 VAD 记录
- ``attributions``：负面显著异常轮归因
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from voicemem.utils.audio.emotion.types import (
    EmotionAttribution,
    EmotionGraphDelta,
    EmotionGraphEdgeInput,
    EmotionGraphNodeInput,
    EmotionSignal,
    TurnEmotionRecord,
    VAD,
)
from voicemem.leftbrain.local_memory_store import default_memory_root

_EMOTION_JSON = "emotion_memory.json"


def emotion_memory_path(root: Path | None = None) -> Path:
    base = root if root is not None else default_memory_root()
    base.mkdir(parents=True, exist_ok=True)
    return base / _EMOTION_JSON


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def vad_to_dict(v: VAD) -> dict[str, float]:
    return {"valence": v.valence, "arousal": v.arousal}


def vad_from_dict(raw: dict[str, Any] | None) -> VAD | None:
    if not isinstance(raw, dict):
        return None
    try:
        return VAD(valence=float(raw["valence"]), arousal=float(raw["arousal"]))
    except (KeyError, TypeError, ValueError):
        return None


def turn_to_dict(record: TurnEmotionRecord) -> dict[str, Any]:
    return {
        "turn_id": record.turn_id,
        "session_id": record.session_id,
        "vad": vad_to_dict(record.vad),
        "timestamp_s": record.timestamp_s,
        "user_utterance_index": record.user_utterance_index,
        "time_gap_from_prev_turn_s": record.time_gap_from_prev_turn_s,
        "extra": dict(record.extra),
        "stored_at": _utc_iso(),
    }


def turn_from_dict(raw: dict[str, Any]) -> TurnEmotionRecord | None:
    if not isinstance(raw, dict):
        return None
    vad = vad_from_dict(raw.get("vad"))
    if vad is None:
        return None
    turn_id = str(raw.get("turn_id", "")).strip()
    session_id = str(raw.get("session_id", "")).strip()
    if not turn_id or not session_id:
        return None
    extra = raw.get("extra")
    return TurnEmotionRecord(
        turn_id=turn_id,
        session_id=session_id,
        vad=vad,
        timestamp_s=raw.get("timestamp_s"),
        user_utterance_index=int(raw.get("user_utterance_index", 0)),
        time_gap_from_prev_turn_s=raw.get("time_gap_from_prev_turn_s"),
        extra=extra if isinstance(extra, dict) else {},
    )


def attribution_to_dict(attr: EmotionAttribution) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "turn_id": attr.turn_id,
        "session_id": attr.session_id,
        "trigger": attr.trigger,
        "analysis_text": attr.analysis_text,
        "vad_at_trigger": vad_to_dict(attr.vad_at_trigger),
        "left_context_summary": attr.left_context_summary,
        "emotion": emotion_to_dict(attr.emotion) if attr.emotion is not None else None,
        "acoustic_evidence": list(attr.acoustic_evidence),
        "semantic_evidence": list(attr.semantic_evidence),
        "related_nodes": [node_to_dict(n) for n in attr.related_nodes],
        "graph_delta": graph_delta_to_dict(attr.graph_delta),
        "user_utterance_index": attr.user_utterance_index,
        "metadata": dict(attr.metadata),
        "stored_at": _utc_iso(),
    }


def attribution_from_dict(raw: dict[str, Any]) -> EmotionAttribution | None:
    if not isinstance(raw, dict):
        return None
    vad = vad_from_dict(raw.get("vad_at_trigger"))
    if vad is None:
        return None
    turn_id = str(raw.get("turn_id", "")).strip()
    session_id = str(raw.get("session_id", "")).strip()
    trigger = raw.get("trigger")
    if trigger != "anomaly":
        return None
    if not turn_id or not session_id:
        return None
    md = raw.get("metadata")
    return EmotionAttribution(
        turn_id=turn_id,
        session_id=session_id,
        trigger=trigger,
        analysis_text=str(raw.get("analysis_text", "")),
        vad_at_trigger=vad,
        left_context_summary=raw.get("left_context_summary"),
        emotion=emotion_from_dict(raw.get("emotion")),
        acoustic_evidence=_as_str_list(raw.get("acoustic_evidence")),
        semantic_evidence=_as_str_list(raw.get("semantic_evidence")),
        related_nodes=[n for x in raw.get("related_nodes") or [] if (n := node_from_dict(x)) is not None],
        graph_delta=graph_delta_from_dict(raw.get("graph_delta")),
        user_utterance_index=int(raw.get("user_utterance_index", 0)),
        metadata=md if isinstance(md, dict) else {},
    )


def emotion_to_dict(e: EmotionSignal) -> dict[str, Any]:
    return {
        "label": e.label,
        "valence": e.valence,
        "arousal": e.arousal,
        "intensity": e.intensity,
        "confidence": e.confidence,
    }


def emotion_from_dict(raw: Any) -> EmotionSignal | None:
    if not isinstance(raw, dict):
        return None
    label = str(raw.get("label", "")).strip()
    if not label:
        return None
    try:
        return EmotionSignal(
            label=label,
            valence=float(raw.get("valence", 0.0)),
            arousal=float(raw.get("arousal", 0.0)),
            intensity=raw.get("intensity") if raw.get("intensity") in ("low", "medium", "high") else "medium",
            confidence=float(raw.get("confidence", 0.0)),
        )
    except (TypeError, ValueError):
        return None


def node_to_dict(n: EmotionGraphNodeInput) -> dict[str, Any]:
    return {
        "local_id": n.local_id,
        "name": n.name,
        "node_type": n.node_type,
        "metadata": dict(n.metadata),
    }


def node_from_dict(raw: Any) -> EmotionGraphNodeInput | None:
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


def edge_to_dict(e: EmotionGraphEdgeInput) -> dict[str, Any]:
    return {
        "source": e.source,
        "target": e.target,
        "edge_type": e.edge_type,
        "description": e.description,
        "emotion_label": e.emotion_label,
        "metadata": dict(e.metadata),
    }


def edge_from_dict(raw: Any) -> EmotionGraphEdgeInput | None:
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


def graph_delta_to_dict(delta: EmotionGraphDelta) -> dict[str, Any]:
    return {
        "nodes": [node_to_dict(n) for n in delta.nodes],
        "edges": [edge_to_dict(e) for e in delta.edges],
    }


def graph_delta_from_dict(raw: Any) -> EmotionGraphDelta:
    if not isinstance(raw, dict):
        return EmotionGraphDelta()
    return EmotionGraphDelta(
        nodes=[n for x in raw.get("nodes") or [] if (n := node_from_dict(x)) is not None],
        edges=[e for x in raw.get("edges") or [] if (e := edge_from_dict(x)) is not None],
    )


def _as_str_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]


@dataclass
class EmotionUserMemory:
    """某用户已落盘的情绪记忆视图（只读拼装）。"""

    user_id: str
    turns: list[TurnEmotionRecord] = field(default_factory=list)
    attributions: list[EmotionAttribution] = field(default_factory=list)


class EmotionMemoryStore:
    """右脑情绪记忆 JSON 仓库。"""

    def __init__(self, *, root: Path | None = None) -> None:
        self._path = emotion_memory_path(root)

    @property
    def path(self) -> Path:
        return self._path

    def _load_root(self) -> dict[str, Any]:
        if not self._path.is_file():
            return {"users": {}}
        with self._path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"users": {}}
        users = data.get("users")
        if not isinstance(users, dict):
            return {"users": {}}
        return {"users": users}

    def _write_root(self, users: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"users": users}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _user_bucket(self, user_id: str, *, create: bool = True) -> dict[str, Any]:
        root = self._load_root()
        users: dict[str, Any] = root["users"]
        if user_id not in users:
            if not create:
                return {}
            users[user_id] = {
                "turns": [],
                "attributions": [],
            }
        bucket = users[user_id]
        if not isinstance(bucket, dict):
            bucket = {"turns": [], "attributions": []}
            users[user_id] = bucket
        for key in ("turns", "attributions"):
            if not isinstance(bucket.get(key), list):
                bucket[key] = []
        return bucket

    def _save_bucket(self, user_id: str, bucket: dict[str, Any]) -> None:
        root = self._load_root()
        users: dict[str, Any] = root["users"]
        users[user_id] = bucket
        self._write_root(users)

    def load(self, *, user_id: str = "default") -> EmotionUserMemory:
        bucket = self._user_bucket(user_id, create=False)
        if not bucket:
            return EmotionUserMemory(user_id=user_id)

        turns: list[TurnEmotionRecord] = []
        for raw in bucket.get("turns") or []:
            t = turn_from_dict(raw) if isinstance(raw, dict) else None
            if t is not None:
                turns.append(t)

        attrs: list[EmotionAttribution] = []
        for raw in bucket.get("attributions") or []:
            a = attribution_from_dict(raw) if isinstance(raw, dict) else None
            if a is not None:
                attrs.append(a)

        return EmotionUserMemory(
            user_id=user_id,
            turns=turns,
            attributions=attrs,
        )

    def append_turn(self, record: TurnEmotionRecord, *, user_id: str = "default") -> None:
        bucket = self._user_bucket(user_id)
        turns: list[Any] = bucket["turns"]
        for raw in turns:
            if isinstance(raw, dict) and raw.get("turn_id") == record.turn_id:
                return
        turns.append(turn_to_dict(record))
        self._save_bucket(user_id, bucket)

    def append_attribution(self, attr: EmotionAttribution, *, user_id: str = "default") -> None:
        bucket = self._user_bucket(user_id)
        bucket["attributions"].append(attribution_to_dict(attr))
        self._save_bucket(user_id, bucket)

    def persist_layer_result(
        self,
        result: Any,
        *,
        user_id: str = "default",
    ) -> None:
        """写入 ``EmotionLayerResult`` 本轮产出（turn + 归因）。"""
        self.append_turn(result.turn, user_id=user_id)
        for attr in result.attributions or []:
            self.append_attribution(attr, user_id=user_id)
