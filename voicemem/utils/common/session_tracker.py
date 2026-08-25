"""跟踪每个用户的 ingest 轮次 + session 边界，供左右脑批处理任务触发用。

两种触发信号：
  - 每 3 轮 ingest（is_third_turn）：右脑短期归因（更新 entity.description）
  - session_id 变化（session_changed）：右脑长期归因（更新 slot.description）+
    左脑子图判定（把这个 session 里检索攒下的账做一次判断，见
    core.py::RunSubgraphCheckpoint / PrimeSubgraphFromQuery）

同时提供一个通用的"touched"记录簿：谁在这一轮/这个session里被碰过，
批处理时只需要处理这些，不用全量扫描。namespace 是调用方自己起的字符串
（比如 "rb_entity_short"/"rb_slot_long"/"subgraph_pool"），互不干扰。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionTracker:
    """SQLite 存取轮次/session状态 + touched refs。线程安全：每次操作新建连接。"""

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
            CREATE TABLE IF NOT EXISTS session_state (
                user_id         TEXT PRIMARY KEY,
                last_session_id TEXT,
                turn_count      INTEGER NOT NULL DEFAULT 0,
                updated_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS touched_refs (
                user_id    TEXT NOT NULL,
                namespace  TEXT NOT NULL,
                ref        TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (user_id, namespace, ref)
            );
            """)

    def record_turn(self, user_id: str, session_id) -> dict:
        """每次 Ingest() 调用一次。

        Returns
        -------
        dict: {"turn_count": int, "session_changed": bool, "is_third_turn": bool}
        session_changed 只有在新旧 session_id 都非空且不同时才为 True——
        调用方从不传 session_id 的话，这套触发机制就一直不启动，不会误触发。
        """
        new_sid = str(session_id) if session_id is not None else None
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM session_state WHERE user_id=?", (user_id,)
            ).fetchone()
            if row is None:
                c.execute(
                    """INSERT INTO session_state (user_id, last_session_id, turn_count, updated_at)
                       VALUES (?,?,1,?)""",
                    (user_id, new_sid, _utc_iso()),
                )
                return {"turn_count": 1, "session_changed": False, "is_third_turn": False}

            last_sid = row["last_session_id"]
            session_changed = last_sid is not None and new_sid is not None and last_sid != new_sid
            turn_count = row["turn_count"] + 1
            c.execute(
                "UPDATE session_state SET last_session_id=?, turn_count=?, updated_at=? WHERE user_id=?",
                (new_sid, turn_count, _utc_iso(), user_id),
            )
            return {
                "turn_count": turn_count,
                "session_changed": session_changed,
                "is_third_turn": turn_count % 3 == 0,
            }

    def get_current_session(self, user_id: str) -> str | None:
        """最近一次 record_turn() 记录的 session_id——供 Algorithm 1 的 ρ(H)
        公式做 session 范围限定用（论文里 Q 是"当前会话"的查询集合，不是
        终身历史）。没记录过 turn，或从没传过 session_id 时返回 None。"""
        with self._conn() as c:
            row = c.execute(
                "SELECT last_session_id FROM session_state WHERE user_id=?", (user_id,)
            ).fetchone()
        return row["last_session_id"] if row else None

    # ── Touched refs ──────────────────────────────────────────────────────────

    def touch(self, user_id: str, namespace: str, ref: str) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR IGNORE INTO touched_refs (user_id, namespace, ref, created_at)
                   VALUES (?,?,?,?)""",
                (user_id, namespace, ref, _utc_iso()),
            )

    def count_touched(self, user_id: str, namespace: str) -> int:
        """只看积了多少，不清空。给「攒够 N 个再跑」的批处理用。"""
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM touched_refs WHERE user_id=? AND namespace=?",
                (user_id, namespace),
            ).fetchone()
        return int(row["n"]) if row else 0

    def pop_touched(self, user_id: str, namespace: str) -> list[str]:
        """取出并清空该 namespace 累积的 touched refs。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT ref FROM touched_refs WHERE user_id=? AND namespace=?",
                (user_id, namespace),
            ).fetchall()
            c.execute(
                "DELETE FROM touched_refs WHERE user_id=? AND namespace=?",
                (user_id, namespace),
            )
        return [r["ref"] for r in rows]
