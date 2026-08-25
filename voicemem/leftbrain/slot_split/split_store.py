"""Dynamic slot store for VoiceMem.

存储子图机制（SubgraphManager）涌现出来的新 slot：[(name, description, parent_slots), ...]。
parent_slots 是个列表——run_for_retrieved_pool（priming阶段跨slot_ref检索）晋升的
slot，参与entity原本可能分散在好几个不同slot_ref下，全部记下来，不只留一个。

旧版 sub_slot 机制（parent_slot → sub_slot 两层结构，配合 route_sub_slot 查询路由 /
assign_to_sub_slot 写入分配）已完全删除——新 slot 的路由现在靠 memory_tags 里的
slot 标签本身（含 base-7 和这里的动态 slot 名），不再需要单独的 sub_slot 层级。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Dynamic slot store ────────────────────────────────────────────────────────

@dataclass
class DynamicSlot:
    name: str
    user_id: str
    description: str
    embedding: list[float] | None
    parent_slots: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_iso)


class DynamicSlotStore:
    """SQLite 存储自动涌现的新 slot。线程安全：每次操作新建连接。"""

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
            CREATE TABLE IF NOT EXISTS dynamic_slots (
                name        TEXT NOT NULL,
                user_id     TEXT NOT NULL,
                description TEXT NOT NULL,
                embedding   TEXT,
                created_at  TEXT NOT NULL,
                PRIMARY KEY (user_id, name)
            );
            """)
            # parent_slots_json：老库没有这一列，ALTER TABLE 补上（新库 CREATE
            # TABLE 不含它，靠这一步统一补齐，两种情况都能拿到这一列）
            try:
                c.execute("ALTER TABLE dynamic_slots ADD COLUMN parent_slots_json TEXT NOT NULL DEFAULT '[]'")
            except sqlite3.OperationalError:
                pass  # 列已存在

    def get_dynamic_slots(self, user_id: str) -> list[DynamicSlot]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM dynamic_slots WHERE user_id=? ORDER BY name",
                (user_id,),
            ).fetchall()
        return [DynamicSlot(
            name=r["name"], user_id=r["user_id"], description=r["description"],
            embedding=json.loads(r["embedding"]) if r["embedding"] else None,
            parent_slots=json.loads(r["parent_slots_json"]) if r["parent_slots_json"] else [],
            created_at=r["created_at"],
        ) for r in rows]

    def create_dynamic_slot(
        self,
        user_id: str,
        name: str,
        description: str,
        embedding: list[float] | None = None,
        parent_slots: list[str] | None = None,
    ) -> DynamicSlot:
        parent_slots = parent_slots or []
        ds = DynamicSlot(
            name=name, user_id=user_id, description=description,
            embedding=embedding, parent_slots=parent_slots, created_at=_utc_iso(),
        )
        with self._conn() as c:
            c.execute(
                """INSERT OR IGNORE INTO dynamic_slots
                   (name, user_id, description, embedding, created_at, parent_slots_json)
                   VALUES (?,?,?,?,?,?)""",
                (name, user_id, description,
                 json.dumps(embedding) if embedding else None, ds.created_at,
                 json.dumps(parent_slots)),
            )
        return ds

    def get_parent_slots(self, user_id: str, name: str) -> list[str]:
        """动态 slot 的全部父 slot（子图判定时参与晋升的 entity 各自原本所在
        的 slot_ref，可能不止一个）。非动态 slot 或没记录到父 slot 时返回 []。"""
        with self._conn() as c:
            row = c.execute(
                "SELECT parent_slots_json FROM dynamic_slots WHERE user_id=? AND name=?",
                (user_id, name),
            ).fetchone()
        if not row or not row["parent_slots_json"]:
            return []
        try:
            return json.loads(row["parent_slots_json"])
        except Exception:
            return []

    def get_children(self, user_id: str, parent_name: str) -> list[DynamicSlot]:
        """返回 parent_slots 包含 parent_name 的全部动态 slot（直接子节点）。
        供分层分类"往下钻"用——parent_name 可以是 base-7 也可以是另一个
        动态 slot（子图可能多级嵌套），不区分对待。"""
        return [s for s in self.get_dynamic_slots(user_id) if parent_name in s.parent_slots]

    def exists(self, user_id: str, name: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM dynamic_slots WHERE user_id=? AND name=?",
                (user_id, name),
            ).fetchone()
        return row is not None
