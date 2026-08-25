"""右脑情绪图记忆：异常情绪 episode、实体/主题节点与情绪反应边。"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

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

_DEFAULT_EMOTION_GRAPH_SQLITE = "voicemem_emotion_graph.sqlite"


def default_emotion_graph_db_path() -> Path:
    return default_memory_root() / _DEFAULT_EMOTION_GRAPH_SQLITE


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_emotion_node_name(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", s)
    return s


@dataclass(frozen=True)
class EmotionGraphNode:
    node_id: str
    user_id: str
    node_type: str
    name: str
    normalized_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmotionGraphEpisode:
    episode_id: str
    user_id: str
    turn_id: str
    session_id: str
    vad: VAD
    emotion: EmotionSignal
    analysis_text: str
    acoustic_evidence: list[str] = field(default_factory=list)
    semantic_evidence: list[str] = field(default_factory=list)
    left_context_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmotionGraphEdge:
    edge_id: str
    user_id: str
    source_node_id: str
    source_name: str
    target_node_id: str
    target_name: str
    edge_type: str
    turn_id: str
    session_id: str
    valence: float
    arousal: float
    emotion_label: str
    intensity: str
    analysis_text: str
    description: str = ""
    acoustic_evidence: list[str] = field(default_factory=list)
    semantic_evidence: list[str] = field(default_factory=list)
    left_context_summary: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmotionGraphSearchHit:
    node: EmotionGraphNode
    edges: list[EmotionGraphEdge] = field(default_factory=list)


@dataclass
class EmotionGraphMemoryStoreConfig:
    db_path: str | Path | None = None


class EmotionGraphMemoryStore:
    """本地 SQLite 情绪图。

    图形态：
    User -[EXPERIENCED]-> EmotionEpisode（标记异常轮次）
    User -[EMOTIONAL_REACTION_TO]-> Topic/Event/Entity（用户对某对象的情绪反应）
    """

    def __init__(
        self,
        *,
        config: EmotionGraphMemoryStoreConfig | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        if config is not None and db_path is not None:
            raise ValueError("请只传 EmotionGraphMemoryStoreConfig 或 db_path 其一")
        cfg = config or EmotionGraphMemoryStoreConfig(db_path=db_path)
        raw = cfg.db_path
        self._path = (
            Path(raw).expanduser().resolve()
            if raw is not None
            else default_emotion_graph_db_path()
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @property
    def path(self) -> Path:
        return self._path

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self._path)
        c.row_factory = sqlite3.Row
        return c

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS emotion_nodes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    metadata TEXT,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    UNIQUE(user_id, node_type, normalized_name)
                );
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS emotion_episodes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    valence REAL NOT NULL,
                    arousal REAL NOT NULL,
                    emotion_label TEXT NOT NULL,
                    intensity TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    analysis_text TEXT NOT NULL,
                    acoustic_evidence TEXT NOT NULL,
                    semantic_evidence TEXT NOT NULL,
                    left_context_summary TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(user_id, turn_id)
                );
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS emotion_edges (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    valence REAL NOT NULL,
                    arousal REAL NOT NULL,
                    emotion_label TEXT NOT NULL,
                    intensity TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    analysis_text TEXT NOT NULL,
                    description TEXT NOT NULL,
                    acoustic_evidence TEXT NOT NULL,
                    semantic_evidence TEXT NOT NULL,
                    left_context_summary TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_emotion_nodes_user ON emotion_nodes(user_id);")
            c.execute("CREATE INDEX IF NOT EXISTS idx_emotion_edges_user_target ON emotion_edges(user_id, target_node_id);")
            c.execute("CREATE INDEX IF NOT EXISTS idx_emotion_edges_turn ON emotion_edges(turn_id);")

    def add_attribution(
        self,
        *,
        user_id: str,
        turn: TurnEmotionRecord,
        attribution: EmotionAttribution,
    ) -> EmotionGraphEpisode:
        """将一次异常归因写入 episode 与情绪图边。"""
        if attribution.emotion is None:
            attribution.emotion = EmotionSignal(
                label="negative",
                valence=turn.vad.valence,
                arousal=turn.vad.arousal,
                intensity="medium",
                confidence=0.0,
            )

        now = _utc_iso()
        local_to_node: dict[str, str] = {}
        with self._conn() as c:
            user_node_id = self._upsert_node(
                c,
                user_id=user_id,
                node=EmotionGraphNodeInput(
                    local_id="user",
                    name=user_id,
                    node_type="User",
                ),
                now=now,
            )
            episode_node_id = self._upsert_node(
                c,
                user_id=user_id,
                node=EmotionGraphNodeInput(
                    local_id="episode",
                    name=f"episode:{turn.turn_id}",
                    node_type="EmotionEpisode",
                    metadata={"turn_id": turn.turn_id, "session_id": turn.session_id},
                ),
                now=now,
            )
            local_to_node.update({"user": user_node_id, "episode": episode_node_id})

            nodes = _merge_nodes(attribution.related_nodes, attribution.graph_delta.nodes)
            for node in nodes:
                nid = self._upsert_node(c, user_id=user_id, node=node, now=now)
                for key in {node.local_id, node.name, normalize_emotion_node_name(node.name)}:
                    if key:
                        local_to_node[key] = nid

            c.execute(
                """
                INSERT OR REPLACE INTO emotion_episodes
                (id, user_id, turn_id, session_id, valence, arousal, emotion_label,
                 intensity, confidence, analysis_text, acoustic_evidence,
                 semantic_evidence, left_context_summary, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    user_id,
                    turn.turn_id,
                    turn.session_id,
                    turn.vad.valence,
                    turn.vad.arousal,
                    attribution.emotion.label,
                    attribution.emotion.intensity,
                    attribution.emotion.confidence,
                    attribution.analysis_text,
                    json.dumps(attribution.acoustic_evidence, ensure_ascii=False),
                    json.dumps(attribution.semantic_evidence, ensure_ascii=False),
                    attribution.left_context_summary,
                    json.dumps(attribution.metadata, ensure_ascii=False),
                    now,
                ),
            )
            self._insert_edge(
                c,
                user_id=user_id,
                source_node_id=user_node_id,
                target_node_id=episode_node_id,
                edge_type="EXPERIENCED",
                turn=turn,
                attribution=attribution,
                description="User experienced a significantly negative emotional event",
                now=now,
            )

            target_nodes = [n for n in nodes if n.node_type not in ("User", "EmotionEpisode")]

            edges = attribution.graph_delta.edges or [
                EmotionGraphEdgeInput(
                    source="user",
                    target=n.local_id,
                    edge_type="EMOTIONAL_REACTION_TO",
                    emotion_label=attribution.emotion.label,
                    description=f"User shows {attribution.emotion.label} toward {n.name}",
                )
                for n in target_nodes
            ]
            for edge in edges:
                if edge.edge_type == "ABOUT":
                    continue
                src = _resolve_node(edge.source, local_to_node)
                tgt = _resolve_node(edge.target, local_to_node)
                if not src or not tgt or src == tgt:
                    continue
                self._insert_edge(
                    c,
                    user_id=user_id,
                    source_node_id=src,
                    target_node_id=tgt,
                    edge_type=edge.edge_type,
                    turn=turn,
                    attribution=attribution,
                    description=edge.description,
                    metadata=edge.metadata,
                    emotion_label=edge.emotion_label,
                    now=now,
                )

        return self.episode_for_turn(user_id=user_id, turn_id=turn.turn_id)

    def _upsert_node(
        self,
        c: sqlite3.Connection,
        *,
        user_id: str,
        node: EmotionGraphNodeInput,
        now: str,
    ) -> str:
        name = node.name.strip()
        if not name:
            raise ValueError("emotion graph node name cannot be empty")
        norm = normalize_emotion_node_name(name)
        cur = c.execute(
            """
            SELECT id, metadata FROM emotion_nodes
            WHERE user_id = ? AND node_type = ? AND normalized_name = ?
            """,
            (user_id, node.node_type, norm),
        )
        row = cur.fetchone()
        if row is not None:
            merged = _merge_metadata(_json_dict(row["metadata"]), node.metadata)
            c.execute(
                """
                UPDATE emotion_nodes
                SET name = ?, metadata = ?, last_seen_at = ?
                WHERE id = ?
                """,
                (name, json.dumps(merged, ensure_ascii=False), now, str(row["id"])),
            )
            return str(row["id"])

        nid = str(uuid.uuid4())
        c.execute(
            """
            INSERT INTO emotion_nodes
            (id, user_id, node_type, name, normalized_name, metadata, first_seen_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                nid,
                user_id,
                node.node_type,
                name,
                norm,
                json.dumps(node.metadata, ensure_ascii=False),
                now,
                now,
            ),
        )
        return nid

    def _insert_edge(
        self,
        c: sqlite3.Connection,
        *,
        user_id: str,
        source_node_id: str,
        target_node_id: str,
        edge_type: str,
        turn: TurnEmotionRecord,
        attribution: EmotionAttribution,
        description: str,
        now: str,
        metadata: dict[str, Any] | None = None,
        emotion_label: str | None = None,
    ) -> None:
        emotion = attribution.emotion
        assert emotion is not None
        c.execute(
            """
            INSERT INTO emotion_edges
            (id, user_id, source_node_id, target_node_id, edge_type, turn_id,
             session_id, valence, arousal, emotion_label, intensity, confidence,
             analysis_text, description, acoustic_evidence, semantic_evidence,
             left_context_summary, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                user_id,
                source_node_id,
                target_node_id,
                edge_type,
                turn.turn_id,
                turn.session_id,
                turn.vad.valence,
                turn.vad.arousal,
                emotion_label or emotion.label,
                emotion.intensity,
                emotion.confidence,
                attribution.analysis_text,
                description,
                json.dumps(attribution.acoustic_evidence, ensure_ascii=False),
                json.dumps(attribution.semantic_evidence, ensure_ascii=False),
                attribution.left_context_summary,
                json.dumps(metadata or {}, ensure_ascii=False),
                now,
            ),
        )

    def has_episodes(self, *, user_id: str) -> bool:
        """是否已有异常轮 episode（用于首轮归因前跳过空图检索）。"""
        self._ensure_schema()
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM emotion_episodes WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return int(row["n"]) > 0 if row is not None else False

    def episode_for_turn(self, *, user_id: str, turn_id: str) -> EmotionGraphEpisode:
        with self._conn() as c:
            cur = c.execute(
                """
                SELECT * FROM emotion_episodes WHERE user_id = ? AND turn_id = ?
                """,
                (user_id, turn_id),
            )
            row = cur.fetchone()
        if row is None:
            raise KeyError(f"emotion episode not found: {turn_id}")
        return _episode_from_row(row)

    def search(
        self,
        *,
        user_id: str,
        query_terms: Sequence[str],
        current_vad: VAD | None = None,
        limit: int = 5,
    ) -> list[EmotionGraphSearchHit]:
        """按实体/主题名称检索历史情绪边。"""
        self._ensure_schema()
        terms = [normalize_emotion_node_name(t) for t in query_terms if t.strip()]
        if not terms:
            return []
        nodes: list[EmotionGraphNode] = []
        with self._conn() as c:
            for term in terms:
                cur = c.execute(
                    """
                    SELECT * FROM emotion_nodes
                    WHERE user_id = ?
                      AND node_type NOT IN ('User', 'EmotionEpisode')
                      AND normalized_name LIKE ?
                    ORDER BY last_seen_at DESC
                    LIMIT ?
                    """,
                    (user_id, f"%{term}%", limit),
                )
                nodes.extend(_node_from_row(r) for r in cur.fetchall())

        seen_nodes: set[str] = set()
        out: list[EmotionGraphSearchHit] = []
        for node in nodes:
            if node.node_id in seen_nodes:
                continue
            seen_nodes.add(node.node_id)
            edges = self._edges_for_node(user_id=user_id, node_id=node.node_id, current_vad=current_vad, limit=limit)
            if edges:
                out.append(EmotionGraphSearchHit(node=node, edges=edges))
        return out[:limit]

    def _edges_for_node(
        self,
        *,
        user_id: str,
        node_id: str,
        current_vad: VAD | None,
        limit: int,
    ) -> list[EmotionGraphEdge]:
        with self._conn() as c:
            cur = c.execute(
                """
                SELECT e.*, sn.name AS source_name, tn.name AS target_name
                FROM emotion_edges e
                JOIN emotion_nodes sn ON sn.id = e.source_node_id
                JOIN emotion_nodes tn ON tn.id = e.target_node_id
                WHERE e.user_id = ?
                  AND e.edge_type = 'EMOTIONAL_REACTION_TO'
                  AND (e.source_node_id = ? OR e.target_node_id = ?)
                ORDER BY e.created_at DESC
                LIMIT ?
                """,
                (user_id, node_id, node_id, max(limit * 3, limit)),
            )
            edges = [_edge_from_row(r) for r in cur.fetchall()]
        if current_vad is not None:
            edges.sort(
                key=lambda e: (
                    abs(e.valence - current_vad.valence) + abs(e.arousal - current_vad.arousal),
                    -e.confidence,
                )
            )
        return edges[:limit]


def format_emotion_graph_context(hits: Sequence[EmotionGraphSearchHit]) -> str:
    if not hits:
        return "(No relevant historical emotion-graph memories)"
    lines = ["[Historical emotion graph context]"]
    for hit in hits:
        lines.append(f"- Node: {hit.node.name}<{hit.node.node_type}>")
        for edge in hit.edges[:3]:
            lines.append(
                f"  - {edge.emotion_label} V/A=({edge.valence:.2f},{edge.arousal:.2f}) "
                f"{edge.source_name} -[{edge.edge_type}]-> {edge.target_name}: {edge.description or edge.analysis_text}"
            )
    lines.append("Use the above as historical reference only; do not assume the same cause applies this turn.")
    return "\n".join(lines)


def _merge_nodes(
    left: Sequence[EmotionGraphNodeInput],
    right: Sequence[EmotionGraphNodeInput],
) -> list[EmotionGraphNodeInput]:
    merged: list[EmotionGraphNodeInput] = []
    seen: set[tuple[str, str]] = set()
    for node in [*left, *right]:
        key = (node.node_type, normalize_emotion_node_name(node.name))
        if not node.name.strip() or key in seen:
            continue
        seen.add(key)
        merged.append(node)
    return merged


def _resolve_node(ref: str, local_to_node: dict[str, str]) -> str | None:
    r = ref.strip()
    if not r:
        return None
    return local_to_node.get(r) or local_to_node.get(normalize_emotion_node_name(r))


def _json_dict(raw: object) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _json_list(raw: object) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(x) for x in data if str(x).strip()]


def _merge_metadata(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    merged = dict(old)
    for key, value in new.items():
        if key not in merged or merged[key] in (None, "", [], {}):
            merged[key] = value
    return merged


def _node_from_row(row: sqlite3.Row) -> EmotionGraphNode:
    return EmotionGraphNode(
        node_id=str(row["id"]),
        user_id=str(row["user_id"]),
        node_type=str(row["node_type"]),
        name=str(row["name"]),
        normalized_name=str(row["normalized_name"]),
        metadata=_json_dict(row["metadata"]),
    )


def _episode_from_row(row: sqlite3.Row) -> EmotionGraphEpisode:
    return EmotionGraphEpisode(
        episode_id=str(row["id"]),
        user_id=str(row["user_id"]),
        turn_id=str(row["turn_id"]),
        session_id=str(row["session_id"]),
        vad=VAD(valence=float(row["valence"]), arousal=float(row["arousal"])),
        emotion=EmotionSignal(
            label=str(row["emotion_label"]),
            valence=float(row["valence"]),
            arousal=float(row["arousal"]),
            intensity=str(row["intensity"]),
            confidence=float(row["confidence"]),
        ),
        analysis_text=str(row["analysis_text"]),
        acoustic_evidence=_json_list(row["acoustic_evidence"]),
        semantic_evidence=_json_list(row["semantic_evidence"]),
        left_context_summary=row["left_context_summary"],
        metadata=_json_dict(row["metadata"]),
    )


def _edge_from_row(row: sqlite3.Row) -> EmotionGraphEdge:
    return EmotionGraphEdge(
        edge_id=str(row["id"]),
        user_id=str(row["user_id"]),
        source_node_id=str(row["source_node_id"]),
        source_name=str(row["source_name"]),
        target_node_id=str(row["target_node_id"]),
        target_name=str(row["target_name"]),
        edge_type=str(row["edge_type"]),
        turn_id=str(row["turn_id"]),
        session_id=str(row["session_id"]),
        valence=float(row["valence"]),
        arousal=float(row["arousal"]),
        emotion_label=str(row["emotion_label"]),
        intensity=str(row["intensity"]),
        analysis_text=str(row["analysis_text"]),
        description=str(row["description"]),
        acoustic_evidence=_json_list(row["acoustic_evidence"]),
        semantic_evidence=_json_list(row["semantic_evidence"]),
        left_context_summary=row["left_context_summary"],
        confidence=float(row["confidence"]),
        metadata=_json_dict(row["metadata"]),
    )
