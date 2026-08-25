"""CognitiveGraphStoreV2 — extends CognitiveGraphStore with multi-label memory_tags.

Adds a ``memory_tags`` table that stores (memory_id, slot, confidence) tuples
using the V2 seven-slot taxonomy defined in :mod:`.slot_v2`.  All existing
CognitiveGraphStore behaviour is preserved unchanged; this subclass only
adds the new table and the four methods that operate on it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .store import CognitiveGraphStore
from .slot_v2 import ALL_SLOT_V2_VALUES


class CognitiveGraphStoreV2(CognitiveGraphStore):
    """CognitiveGraphStore extended with multi-label V2 memory tags.

    The parent class manages the core schema (entities, edges, memories, …).
    This subclass adds ``memory_tags`` and the corresponding DDL/DML helpers.

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  Passed straight through to the parent.
    embedder:
        Optional, passed straight through to the parent -- enables semantic
        entity dedup in ``upsert_entity``. See ``CognitiveGraphStore.__init__``.
    """

    def __init__(self, db_path: Path | str, embedder: Any = None) -> None:
        super().__init__(db_path, embedder=embedder)
        self._ensure_tags_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _ensure_tags_schema(self) -> None:
        """Create ``memory_tags`` and ``slot_summaries`` tables if they do not exist."""
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_tags (
                    memory_id   TEXT NOT NULL,
                    user_id     TEXT NOT NULL,
                    slot        TEXT NOT NULL,
                    confidence  REAL NOT NULL DEFAULT 1.0,
                    PRIMARY KEY (memory_id, slot),
                    FOREIGN KEY (memory_id) REFERENCES memories(id)
                );
                CREATE INDEX IF NOT EXISTS idx_tags_user_slot
                    ON memory_tags(user_id, slot);
                CREATE INDEX IF NOT EXISTS idx_tags_mid
                    ON memory_tags(memory_id);

                CREATE TABLE IF NOT EXISTS slot_summaries (
                    user_id    TEXT NOT NULL,
                    slot       TEXT NOT NULL,
                    summary    TEXT NOT NULL,
                    mem_count  INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, slot)
                );
                CREATE INDEX IF NOT EXISTS idx_slot_summaries_user
                    ON slot_summaries(user_id);

                CREATE TABLE IF NOT EXISTS slot_macro_edges (
                    user_id    TEXT NOT NULL,
                    slot_a     TEXT NOT NULL,
                    slot_b     TEXT NOT NULL,
                    weight     REAL NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, slot_a, slot_b)
                );
                CREATE INDEX IF NOT EXISTS idx_slot_macro_user
                    ON slot_macro_edges(user_id);
                """
            )

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert_memory_tags(
        self,
        memory_id: str,
        user_id: str,
        slots: list[tuple[str, float]],
    ) -> None:
        """Upsert (slot_value, confidence) pairs for a given memory.

        Parameters
        ----------
        memory_id:
            The memory's canonical identifier.
        user_id:
            Owner of the memory.
        slots:
            List of ``(slot_value, confidence)`` tuples.  ``slot_value``
            should be one of :data:`.ALL_SLOT_V2_VALUES`.  On conflict
            the row is replaced (confidence updated).
        """
        if not slots:
            return
        rows = [(memory_id, user_id, slot_val, conf) for slot_val, conf in slots]
        with self._conn() as c:
            c.executemany(
                """
                INSERT INTO memory_tags (memory_id, user_id, slot, confidence)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(memory_id, slot) DO UPDATE SET
                    confidence = excluded.confidence,
                    user_id    = excluded.user_id
                """,
                rows,
            )

    def rename_tag_value(self, user_id: str, old_slot: str, new_slot: str) -> None:
        """把某个 user 下所有 ``slot == old_slot`` 的行改成 ``slot = new_slot``。

        用于声纹画像合并（``VoiceprintStore.merge_persons()``）之后，把合并前
        已经写入的 ``speaker:<被合并的 person_id>`` 标签统一改成
        ``speaker:<合并后存活的 person_id>``——``merge_persons()`` 只合并声纹
        画像本身，不知道 ``memory_tags`` 表的存在；不做这一步的话，合并前抽取
        的记忆会在 ``speaker_filter`` 检索时永远漏掉，声纹层面"认出是同一个
        人"跟记忆层面"能查到这个人说过的话"就对不上。
        """
        with self._conn() as c:
            # UPDATE OR IGNORE：极少数情况下同一条记忆理论上可能已经有
            # new_slot（(memory_id, slot) 主键冲突），那一行会被跳过、保留
            # old_slot——总比撞主键报错丢数据安全，而且实际不会发生（一条
            # 记忆只会在创建时被打一次 speaker 标签）。
            c.execute(
                "UPDATE OR IGNORE memory_tags SET slot = ? WHERE user_id = ? AND slot = ?",
                (new_slot, user_id, old_slot),
            )

    # ── Read ──────────────────────────────────────────────────────────────────

    def memory_ids_for_slots_v2(
        self,
        user_id: str,
        slots: list[str],
    ) -> list[str]:
        """Return distinct memory_ids tagged with any of the given V2 slots.

        Parameters
        ----------
        user_id:
            Owner of the memories.
        slots:
            List of slot value strings (e.g. ``["fact", "event"]``).

        Returns
        -------
        list[str]
            Distinct memory IDs, or an empty list if the table is empty or
            no rows match.
        """
        if not slots:
            return []
        ph = ",".join("?" * len(slots))
        with self._conn() as c:
            rows = c.execute(
                f"""
                SELECT DISTINCT memory_id
                FROM memory_tags
                WHERE user_id = ? AND slot IN ({ph})
                """,
                [user_id, *slots],
            ).fetchall()
        return [r["memory_id"] for r in rows]

    def get_tags_for_memory(self, memory_id: str) -> list[tuple[str, float]]:
        """Return all V2 tags for a single memory.

        Parameters
        ----------
        memory_id:
            The memory's canonical identifier.

        Returns
        -------
        list[tuple[str, float]]
            List of ``(slot, confidence)`` pairs, ordered by confidence desc.
        """
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT slot, confidence
                FROM memory_tags
                WHERE memory_id = ?
                ORDER BY confidence DESC
                """,
                (memory_id,),
            ).fetchall()
        return [(r["slot"], float(r["confidence"])) for r in rows]

    def memory_tag_counts(self, user_id: str) -> dict[str, int]:
        """Return per-slot tag counts for a user (useful for debugging).

        Parameters
        ----------
        user_id:
            Owner of the memories.

        Returns
        -------
        dict[str, int]
            ``{slot_value: count}`` for every V2 slot.  Slots with zero
            tagged memories are still included (count = 0).
        """
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT slot, COUNT(*) AS cnt
                FROM memory_tags
                WHERE user_id = ?
                GROUP BY slot
                """,
                (user_id,),
            ).fetchall()
        counts: dict[str, int] = {s: 0 for s in ALL_SLOT_V2_VALUES}
        for r in rows:
            counts[r["slot"]] = int(r["cnt"])
        return counts

    # ── Slot summaries ────────────────────────────────────────────────────────

    def upsert_slot_summary(
        self,
        user_id: str,
        slot: str,
        summary: str,
        mem_count: int = 0,
    ) -> None:
        """Write or update the rolling summary for a slot."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO slot_summaries (user_id, slot, summary, mem_count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, slot) DO UPDATE SET
                    summary    = excluded.summary,
                    mem_count  = excluded.mem_count,
                    updated_at = excluded.updated_at
                """,
                (user_id, slot, summary, mem_count, now),
            )

    def get_slot_summary(self, user_id: str, slot: str) -> str | None:
        """Return the summary text for a slot, or None if not yet generated."""
        with self._conn() as c:
            row = c.execute(
                "SELECT summary FROM slot_summaries WHERE user_id=? AND slot=?",
                (user_id, slot),
            ).fetchone()
        return row["summary"] if row else None

    def get_slot_summaries(self, user_id: str, slots: list[str]) -> dict[str, str]:
        """Return {slot: summary} for multiple slots, omitting missing entries."""
        if not slots:
            return {}
        ph = ",".join("?" * len(slots))
        with self._conn() as c:
            rows = c.execute(
                f"SELECT slot, summary FROM slot_summaries WHERE user_id=? AND slot IN ({ph})",
                [user_id, *slots],
            ).fetchall()
        return {r["slot"]: r["summary"] for r in rows}

    # ── 语义簇宏观连接（从数据共现自动学，论文："语义簇之间宏观连接"）────────
    # 一条记忆同时打上 2 个及以上 slot 标签，就说明这几个 slot 之间存在真实的
    # 宏观关联（比如同一句话既算 work 又算 health）——每次都给涉及的 slot 两两
    # 累加一次权重，不是人工写死的关系表。

    def record_slot_cooccurrence(self, user_id: str, slots: list[str]) -> None:
        """记一条记忆里同时出现的这批 slot，两两之间累加共现权重。"""
        distinct = sorted(set(slots))
        if len(distinct) < 2:
            return
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        pairs = [
            (distinct[i], distinct[j])
            for i in range(len(distinct)) for j in range(i + 1, len(distinct))
        ]
        with self._conn() as c:
            for slot_a, slot_b in pairs:
                c.execute(
                    """INSERT INTO slot_macro_edges (user_id, slot_a, slot_b, weight, updated_at)
                       VALUES (?,?,?,1,?)
                       ON CONFLICT(user_id, slot_a, slot_b) DO UPDATE SET
                         weight=weight+1, updated_at=excluded.updated_at""",
                    (user_id, slot_a, slot_b, now),
                )

    def get_macro_related_slots(
        self, user_id: str, slot: str, *, limit: int = 3, min_weight: float = 2.0,
    ) -> list[str]:
        """按共现权重取跟 slot 关联最强的几个宏观关联 slot（权重需过
        min_weight——只出现过一次的共现太偶然，不该被当成稳定的宏观连接）。"""
        with self._conn() as c:
            rows = c.execute(
                """SELECT CASE WHEN slot_a=? THEN slot_b ELSE slot_a END AS related, weight
                   FROM slot_macro_edges
                   WHERE user_id=? AND (slot_a=? OR slot_b=?) AND weight>=?
                   ORDER BY weight DESC LIMIT ?""",
                (slot, user_id, slot, slot, min_weight, limit),
            ).fetchall()
        return [r["related"] for r in rows]

    def get_slot_summary_mem_count(self, user_id: str, slot: str) -> int:
        """Return the mem_count recorded at last summary update (0 if not yet set)."""
        with self._conn() as c:
            row = c.execute(
                "SELECT mem_count FROM slot_summaries WHERE user_id=? AND slot=?",
                (user_id, slot),
            ).fetchone()
        return int(row["mem_count"]) if row else 0

    def count_memories_in_slot(self, user_id: str, slot: str) -> int:
        """Count memories tagged with this slot for this user."""
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS cnt FROM memory_tags WHERE user_id=? AND slot=?",
                (user_id, slot),
            ).fetchone()
        return int(row["cnt"]) if row else 0

    # ── Cleanup override ──────────────────────────────────────────────────────

    def delete_user(self, user_id: str) -> None:
        """Delete all data for *user_id*, including V2 tags and summaries."""
        with self._conn() as c:
            c.execute("DELETE FROM memory_tags WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM slot_summaries WHERE user_id=?", (user_id,))
        super().delete_user(user_id)


__all__ = ["CognitiveGraphStoreV2"]
