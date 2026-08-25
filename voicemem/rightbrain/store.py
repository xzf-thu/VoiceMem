"""右脑 SQLite 存储层。

表：
  right_brain_memories     — 三类经验记忆
  right_brain_anchor_links — 记忆挂左脑 anchor 的关联
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import (
    AnchorRole, AnchorType, MemoryAnchor,
    MemoryClass, RightBrainAnchorLink, RightBrainMemory, TTL,
)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


class RightBrainStore:
    """SQLite 右脑存储。线程安全：每次操作新建连接。"""

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self._path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS right_brain_memories (
                id                  TEXT PRIMARY KEY,
                user_id             TEXT NOT NULL,
                memory_class        TEXT NOT NULL,
                content             TEXT NOT NULL,
                condition_text      TEXT,
                priority            REAL NOT NULL DEFAULT 0.5,
                confidence          REAL NOT NULL DEFAULT 1.0,
                ttl                 TEXT NOT NULL DEFAULT 'long_term',
                metadata            TEXT NOT NULL DEFAULT '{}',
                evidence_turn_ids   TEXT NOT NULL DEFAULT '[]',
                evidence_memory_ids TEXT NOT NULL DEFAULT '[]',
                created_at          TEXT NOT NULL,
                updated_at          TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_rbm_user ON right_brain_memories(user_id, memory_class);

            CREATE TABLE IF NOT EXISTS right_brain_anchor_links (
                id              TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                right_memory_id TEXT NOT NULL,
                anchor_type     TEXT NOT NULL,
                anchor_id       TEXT,
                role            TEXT NOT NULL DEFAULT 'context',
                weight          REAL NOT NULL DEFAULT 1.0,
                confidence      REAL NOT NULL DEFAULT 1.0,
                created_at      TEXT NOT NULL,
                FOREIGN KEY (right_memory_id) REFERENCES right_brain_memories(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_rbal_mem  ON right_brain_anchor_links(right_memory_id);
            CREATE INDEX IF NOT EXISTS idx_rbal_anchor ON right_brain_anchor_links(user_id, anchor_type, anchor_id);
            """)

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert_memory(
        self,
        user_id: str,
        memory_class: MemoryClass,
        content: str,
        *,
        condition: str | None = None,
        priority: float = 0.5,
        confidence: float = 1.0,
        ttl: TTL = "long_term",
        metadata: dict[str, Any] | None = None,
        evidence_turn_ids: list[str] | None = None,
        evidence_memory_ids: list[str] | None = None,
        memory_id: str | None = None,
        created_at: str | None = None,
    ) -> RightBrainMemory:
        """created_at：事件真实发生时间（ISO，通常来自 Ingest 的 observed_at）。
        不传则用写入墙钟——但 benchmark/回填场景下墙钟和事件时间相差数年，渲染出的
        "[2026-08-14]" 会把 temporal 类问题带偏，所以调用方有 observed_at 时应传入。"""
        now = _utc_iso()
        mid = memory_id or _new_id()
        created = created_at or now
        with self._conn() as c:
            c.execute(
                """INSERT INTO right_brain_memories
                   (id, user_id, memory_class, content, condition_text,
                    priority, confidence, ttl, metadata,
                    evidence_turn_ids, evidence_memory_ids, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     content=excluded.content,
                     condition_text=COALESCE(excluded.condition_text, condition_text),
                     priority=MAX(excluded.priority, priority),
                     confidence=excluded.confidence,
                     metadata=excluded.metadata,
                     evidence_turn_ids=excluded.evidence_turn_ids,
                     evidence_memory_ids=excluded.evidence_memory_ids,
                     updated_at=excluded.updated_at""",
                (mid, user_id, memory_class, content, condition,
                 priority, confidence, ttl,
                 json.dumps(metadata or {}, ensure_ascii=False),
                 json.dumps(evidence_turn_ids or [], ensure_ascii=False),
                 json.dumps(evidence_memory_ids or [], ensure_ascii=False),
                 created, now),
            )
        return RightBrainMemory(
            id=mid, user_id=user_id, memory_class=memory_class,
            content=content, condition=condition,
            priority=priority, confidence=confidence, ttl=ttl,
            metadata=metadata or {},
            evidence_turn_ids=evidence_turn_ids or [],
            evidence_memory_ids=evidence_memory_ids or [],
            created_at=now, updated_at=now,
        )

    def link_anchor(
        self,
        right_memory_id: str,
        user_id: str,
        anchor: MemoryAnchor,
    ) -> None:
        now = _utc_iso()
        with self._conn() as c:
            c.execute(
                """INSERT INTO right_brain_anchor_links
                   (id, user_id, right_memory_id, anchor_type, anchor_id,
                    role, weight, confidence, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT DO NOTHING""",
                (_new_id(), user_id, right_memory_id,
                 anchor.anchor_type, anchor.anchor_id,
                 anchor.role, anchor.weight, anchor.confidence, now),
            )

    # ── Read ──────────────────────────────────────────────────────────────────

    def search_by_anchors(
        self,
        user_id: str,
        anchors: list[MemoryAnchor],
        *,
        memory_class: MemoryClass | None = None,
        limit: int = 5,
    ) -> list[RightBrainMemory]:
        """按 anchor 集合检索，按 weight*confidence*priority 降序。"""
        if not anchors:
            return self._fallback_global(user_id, memory_class=memory_class, limit=limit)

        # 构造 OR 条件：anchor_type+anchor_id 任意一条命中即可
        conditions: list[str] = []
        params: list[Any] = [user_id]
        for a in anchors:
            if a.anchor_id is not None:
                conditions.append("(anchor_type=? AND anchor_id=?)")
                params.extend([a.anchor_type, a.anchor_id])
            else:
                conditions.append("anchor_type=?")
                params.append(a.anchor_type)

        anchor_where = " OR ".join(conditions)
        mc_where = "AND m.memory_class=?" if memory_class else ""
        if memory_class:
            params.append(memory_class)
        params.append(limit)

        # 权重 = SUM(matched anchor weight*confidence) * memory.priority
        # 用 SUM 而非 MAX：同时命中 emotion+entity 的记录得分更高
        sql = f"""
            SELECT m.*, SUM(l.weight * l.confidence) AS anchor_score
            FROM right_brain_memories m
            JOIN right_brain_anchor_links l ON l.right_memory_id = m.id
            WHERE m.user_id=? AND ({anchor_where}) {mc_where}
            GROUP BY m.id
            ORDER BY anchor_score * m.priority DESC, m.confidence DESC, m.updated_at DESC
            LIMIT ?
        """
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def search_global(
        self,
        user_id: str,
        *,
        memory_class: MemoryClass | None = None,
        anchor_types: list[AnchorType] | None = None,
        limit: int = 3,
    ) -> list[RightBrainMemory]:
        """查全局 anchor（user_self / global_style）。"""
        global_types = anchor_types or ["user", "global_style"]
        ph = ",".join("?" * len(global_types))
        mc_where = "AND m.memory_class=?" if memory_class else ""
        params: list[Any] = [user_id, *global_types]
        if memory_class:
            params.append(memory_class)
        params.append(limit)
        sql = f"""
            SELECT m.*, MAX(l.weight * l.confidence) AS anchor_score
            FROM right_brain_memories m
            JOIN right_brain_anchor_links l ON l.right_memory_id = m.id
            WHERE m.user_id=? AND l.anchor_type IN ({ph}) {mc_where}
            GROUP BY m.id
            ORDER BY m.priority DESC, m.confidence DESC
            LIMIT ?
        """
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def _fallback_global(
        self,
        user_id: str,
        *,
        memory_class: MemoryClass | None = None,
        limit: int = 3,
    ) -> list[RightBrainMemory]:
        return self.search_global(user_id, memory_class=memory_class, limit=limit)

    def get_all(self, user_id: str) -> list[RightBrainMemory]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM right_brain_memories WHERE user_id=? ORDER BY priority DESC",
                (user_id,),
            ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_memory(self, memory_id: str) -> RightBrainMemory | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM right_brain_memories WHERE id=?", (memory_id,)
            ).fetchone()
        return self._row_to_memory(row) if row else None

    def update_content(self, memory_id: str, content: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE right_brain_memories SET content=?, updated_at=? WHERE id=?",
                (content, _utc_iso(), memory_id),
            )

    def merge_metadata(self, memory_id: str, updates: dict[str, Any]) -> None:
        """把 updates 合并进 metadata JSON（浅合并，同 key 覆盖）。

        用途：给已存在的记忆打标（superseded_by / refined 等），不改 content、
        不动 updated_at——打标不是内容更新，不该影响按 updated_at 的排序。
        """
        with self._conn() as c:
            row = c.execute(
                "SELECT metadata FROM right_brain_memories WHERE id=?", (memory_id,)
            ).fetchone()
            if row is None:
                return
            meta = json.loads(row["metadata"] or "{}")
            meta.update(updates)
            c.execute(
                "UPDATE right_brain_memories SET metadata=? WHERE id=?",
                (json.dumps(meta, ensure_ascii=False), memory_id),
            )

    def get_anchors_for_memory(self, right_memory_id: str) -> list[RightBrainAnchorLink]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM right_brain_anchor_links WHERE right_memory_id=?",
                (right_memory_id,),
            ).fetchall()
        return [
            RightBrainAnchorLink(
                id=r["id"], user_id=r["user_id"],
                right_memory_id=r["right_memory_id"],
                anchor_type=r["anchor_type"], anchor_id=r["anchor_id"],
                role=r["role"], weight=r["weight"], confidence=r["confidence"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def delete_user(self, user_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM right_brain_anchor_links WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM right_brain_memories WHERE user_id=?", (user_id,))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _row_to_memory(self, row: sqlite3.Row) -> RightBrainMemory:
        # search_by_anchors/search_global 的 SELECT 里带 anchor_score 列
        # （本次查询的锚点命中得分）；get_all/get_memory 没有这一列，取 0。
        # 之前这里直接丢掉了 anchor_score——SQL 里算得再精细，出了存储层
        # 就只剩静态 priority，最终 top-5 排序完全感知不到检索相关度。
        score = row["anchor_score"] if "anchor_score" in row.keys() else None
        return RightBrainMemory(
            id=row["id"], user_id=row["user_id"],
            memory_class=row["memory_class"],
            content=row["content"],
            condition=row["condition_text"],
            priority=float(row["priority"]),
            confidence=float(row["confidence"]),
            ttl=row["ttl"],
            metadata=json.loads(row["metadata"] or "{}"),
            evidence_turn_ids=json.loads(row["evidence_turn_ids"] or "[]"),
            evidence_memory_ids=json.loads(row["evidence_memory_ids"] or "[]"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            anchor_score=float(score) if score is not None else 0.0,
        )
