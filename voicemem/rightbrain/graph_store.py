"""右脑 slot→entity→memory 图层。

跟左脑同样的三层结构，但内容是感性/主观的，不是左脑那种事实性分类：

  slot（5个初始类目：情绪 / 喜好与厌恶 / 表达风格 / 思维模式 / 应对方式）
    └── entity（具体的感性节点，如"情绪"下的"开心"、"喜好与厌恶"下的"被打断"）
          └── memory（挂在某个 entity 下的具体记忆 id，指向 right_brain_memories.id）

slot 是这里真正的图节点（有自己的 id + description），不是左脑那种字符串属性。

entity 去重靠语义相似度（不是精确字符串匹配）：新提炼出的感性标签先跟同一
slot 下已有 entity 的 embedding 比相似度，够像就复用，不够像才新建。
"""

from __future__ import annotations

import os

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from voicemem.utils.common._graph_common import cosine as _cosine, new_id as _new_id, utc_iso as _utc_iso

#: 同一个 slot 下，新标签跟已有实体像到什么程度就算同一个。
#:
#: 原来是 0.65——对 E5 的中文短语来说等于「全都合并」：实测基线本来就很高，
#: 「讨厌吃饭吧唧嘴」跟「讨厌被打断」有 0.934，「讨厌开长会」跟它 0.922，全被
#: 归成同一个实体。后果是右脑跑一阵之后再也不长新节点，用户说什么新东西图上
#: 都没反应，看着像没记住。
#: 实测真正该合并的那种（「喜欢手冲咖啡」↔「偏好手冲咖啡」）是 0.964，
#: 0.95 正好把两类分开。
DEFAULT_MATCH_THRESHOLD = float(os.environ.get("VOICEMEM_RB_ENTITY_MERGE", "0.95"))


# 初始5个感性slot；description先留空，后续再补。
# emotion 的初始 entity 复用原有8个情绪标签。
SEED_SLOTS: list[tuple[str, str, list[str]]] = [
    ("情绪", "", ["焦虑", "悲伤", "委屈", "孤独", "纠结", "平静", "开心", "疲惫"]),
    ("喜好与厌恶", "", []),
    ("表达风格", "", []),
    ("思维模式", "", []),
    ("应对方式", "", []),
]


@dataclass
class RBSlot:
    id: str
    user_id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=_utc_iso)


@dataclass
class RBEntity:
    id: str
    user_id: str
    slot_id: str
    name: str
    description: str = ""
    embedding: list[float] | None = None
    source_entity_id: str | None = None  # 左脑 cognitive_graph Entity.id，仅"关系节点"类 entity 会填
    created_at: str = field(default_factory=_utc_iso)
    updated_at: str = field(default_factory=_utc_iso)


class RightBrainGraphStore:
    """SQLite 存取右脑 slot→entity→memory 三层图。线程安全：每次操作新建连接。"""

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
            CREATE TABLE IF NOT EXISTS rb_slots (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                name        TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL,
                UNIQUE (user_id, name)
            );

            CREATE TABLE IF NOT EXISTS rb_entities (
                id                TEXT PRIMARY KEY,
                user_id           TEXT NOT NULL,
                slot_id           TEXT NOT NULL,
                name              TEXT NOT NULL,
                description       TEXT NOT NULL DEFAULT '',
                embedding         TEXT,
                source_entity_id  TEXT,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL,
                UNIQUE (user_id, slot_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_rbe_slot ON rb_entities(user_id, slot_id);

            CREATE TABLE IF NOT EXISTS rb_entity_memories (
                id          TEXT PRIMARY KEY,
                entity_id   TEXT NOT NULL,
                user_id     TEXT NOT NULL,
                memory_id   TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                UNIQUE (entity_id, memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_rbem_entity ON rb_entity_memories(entity_id);
            CREATE INDEX IF NOT EXISTS idx_rbem_memory ON rb_entity_memories(user_id, memory_id);
            """)
            # 迁移：老库（CREATE TABLE IF NOT EXISTS 对已存在的表不生效）补上
            # source_entity_id 列——关系节点功能上线前建的 rb_graph.sqlite 都缺这列。
            cols = {row["name"] for row in c.execute("PRAGMA table_info(rb_entities)")}
            if "source_entity_id" not in cols:
                c.execute("ALTER TABLE rb_entities ADD COLUMN source_entity_id TEXT")
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_rbe_source "
                "ON rb_entities(user_id, slot_id, source_entity_id)"
            )

    # ── 种子初始化 ────────────────────────────────────────────────────────────

    def ensure_seed_slots(self, user_id: str) -> None:
        """确保该用户已有5个初始slot（+情绪下面的8个初始entity）。幂等，可重复调用。"""
        for slot_name, slot_desc, seed_entities in SEED_SLOTS:
            slot = self.get_or_create_slot(user_id, slot_name, description=slot_desc)
            for ent_name in seed_entities:
                self.get_or_create_entity(user_id, slot.id, ent_name)

    # ── Slot ──────────────────────────────────────────────────────────────────

    def get_or_create_slot(self, user_id: str, name: str, *, description: str = "") -> RBSlot:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM rb_slots WHERE user_id=? AND name=?", (user_id, name)
            ).fetchone()
            if row:
                return _row_to_slot(row)
            slot = RBSlot(id=_new_id(), user_id=user_id, name=name, description=description)
            c.execute(
                "INSERT INTO rb_slots (id, user_id, name, description, created_at) VALUES (?,?,?,?,?)",
                (slot.id, slot.user_id, slot.name, slot.description, slot.created_at),
            )
            return slot

    def get_slot_by_name(self, user_id: str, name: str) -> RBSlot | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM rb_slots WHERE user_id=? AND name=?", (user_id, name)
            ).fetchone()
        return _row_to_slot(row) if row else None

    def list_slots(self, user_id: str) -> list[RBSlot]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM rb_slots WHERE user_id=? ORDER BY name", (user_id,)
            ).fetchall()
        return [_row_to_slot(r) for r in rows]

    def set_slot_description(self, slot_id: str, description: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE rb_slots SET description=? WHERE id=?", (description, slot_id))

    # ── Entity（精确名字匹配，适合种子/固定词表场景） ────────────────────────

    def get_or_create_entity(
        self,
        user_id: str,
        slot_id: str,
        name: str,
        *,
        description: str = "",
        embedding: list[float] | None = None,
    ) -> RBEntity:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM rb_entities WHERE user_id=? AND slot_id=? AND name=?",
                (user_id, slot_id, name),
            ).fetchone()
            if row:
                return _row_to_entity(row)
            return self._insert_entity(c, user_id, slot_id, name, description, embedding)

    # ── Entity（语义相似度匹配，适合自由生成的感性 entity） ───────────────────

    def find_similar_entity(
        self,
        user_id: str,
        slot_id: str,
        embedding: list[float],
        *,
        threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> RBEntity | None:
        """在同一 slot 下找语义最相似的 entity，相似度不够则返回 None。"""
        best: RBEntity | None = None
        best_sim = -1.0
        for ent in self.get_entities_for_slot(user_id, slot_id):
            if ent.embedding is None:
                continue
            sim = _cosine(embedding, ent.embedding)
            if sim > best_sim:
                best_sim, best = sim, ent
        return best if best is not None and best_sim >= threshold else None

    def get_or_create_entity_semantic(
        self,
        user_id: str,
        slot_id: str,
        name: str,
        embedding: list[float],
        *,
        description: str = "",
        threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> tuple[RBEntity, bool]:
        """先按语义相似度找现有 entity，找到就复用；没有就新建。

        Returns
        -------
        (entity, created) — created=True 表示这次新建了一个 entity。
        """
        existing = self.find_similar_entity(user_id, slot_id, embedding, threshold=threshold)
        if existing is not None:
            return existing, False
        with self._conn() as c:
            ent = self._insert_entity(c, user_id, slot_id, name, description, embedding)
        return ent, True

    def _insert_entity(
        self,
        c: sqlite3.Connection,
        user_id: str,
        slot_id: str,
        name: str,
        description: str,
        embedding: list[float] | None,
        source_entity_id: str | None = None,
    ) -> RBEntity:
        now = _utc_iso()
        ent = RBEntity(
            id=_new_id(), user_id=user_id, slot_id=slot_id, name=name,
            description=description, embedding=embedding,
            source_entity_id=source_entity_id, created_at=now, updated_at=now,
        )
        c.execute(
            """INSERT INTO rb_entities
               (id, user_id, slot_id, name, description, embedding, source_entity_id,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (ent.id, ent.user_id, ent.slot_id, ent.name, ent.description,
             json.dumps(embedding) if embedding is not None else None,
             ent.source_entity_id, ent.created_at, ent.updated_at),
        )
        return ent

    # ── Entity（按左脑 entity.id 精确关联，适合"关系节点"场景） ────────────────
    # 一个左脑实体（人/地点/项目...）永远对应同一个右脑关系节点，靠 source_entity_id
    # 精确匹配，不走语义相似度——语义相似度是给自由生成的抽象标签去重用的，这里
    # 反而不需要：左脑实体改名/合并不影响这里的匹配（ID 不变），跟今天修的
    # 锚点用真实 entity.id 而不是名字字符串是同一个思路。

    def get_entity_by_source_id(
        self, user_id: str, slot_id: str, source_entity_id: str,
    ) -> RBEntity | None:
        with self._conn() as c:
            row = c.execute(
                """SELECT * FROM rb_entities
                   WHERE user_id=? AND slot_id=? AND source_entity_id=?""",
                (user_id, slot_id, source_entity_id),
            ).fetchone()
        return _row_to_entity(row) if row else None

    def get_or_create_entity_by_source_id(
        self, user_id: str, slot_id: str, source_entity_id: str, name: str,
    ) -> tuple[RBEntity, bool]:
        """按左脑 entity.id 查找/创建关系节点。

        Returns
        -------
        (entity, created) — created=True 表示这次新建了一个 entity。
        """
        existing = self.get_entity_by_source_id(user_id, slot_id, source_entity_id)
        if existing is not None:
            return existing, False
        with self._conn() as c:
            # 同名节点已存在但 source_entity_id 对不上：左脑对同一个现实事物
            # 可能给出不止一个 entity.id（同名实体在后续 utterance 里被重新
            # 抽取/换了 entity_type），而 rb_entities 有 UNIQUE(user_id,
            # slot_id, name)——直接 INSERT 会撞唯一约束抛异常，调用方
            # (core.py::_finish_ingest 的关系节点循环) 整段中断，这条
            # utterance 剩下实体的锚点和 link_memory 全部丢失。同名即同物，
            # 复用已有节点；已有节点还没认领 source_entity_id 时顺手认领，
            # 让后续查找能走更快的 source_id 路径。
            row = c.execute(
                "SELECT * FROM rb_entities WHERE user_id=? AND slot_id=? AND name=?",
                (user_id, slot_id, name),
            ).fetchone()
            if row is not None:
                ent = _row_to_entity(row)
                if not ent.source_entity_id:
                    now = _utc_iso()
                    c.execute(
                        "UPDATE rb_entities SET source_entity_id=?, updated_at=? WHERE id=?",
                        (source_entity_id, now, ent.id),
                    )
                    ent.source_entity_id = source_entity_id
                    ent.updated_at = now
                return ent, False
            ent = self._insert_entity(
                c, user_id, slot_id, name, "", None, source_entity_id=source_entity_id,
            )
        return ent, True

    def get_entities_for_slot(self, user_id: str, slot_id: str,
                              *, newest_first: bool = False) -> list[RBEntity]:
        """默认按名字排（稳定、便于展示）。``newest_first`` 按插入顺序倒排——
        脑图要留几个席位给刚长出来的实体，按名字排的话新的排在哪全看它叫什么。"""
        order = "rowid DESC" if newest_first else "name"
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM rb_entities WHERE user_id=? AND slot_id=? ORDER BY {order}",
                (user_id, slot_id),
            ).fetchall()
        return [_row_to_entity(r) for r in rows]

    def get_entity_by_name(self, user_id: str, slot_id: str, name: str) -> RBEntity | None:
        """精确按名字查（只读，不创建）——给检索用，不像 get_or_create_entity
        那样在没查到时顺手建一个空的。"""
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM rb_entities WHERE user_id=? AND slot_id=? AND name=?",
                (user_id, slot_id, name),
            ).fetchone()
        return _row_to_entity(row) if row else None

    def get_entity(self, entity_id: str) -> RBEntity | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM rb_entities WHERE id=?", (entity_id,)).fetchone()
        return _row_to_entity(row) if row else None

    def get_slot(self, slot_id: str) -> RBSlot | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM rb_slots WHERE id=?", (slot_id,)).fetchone()
        return _row_to_slot(row) if row else None

    def set_entity_description(self, entity_id: str, description: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE rb_entities SET description=?, updated_at=? WHERE id=?",
                (description, _utc_iso(), entity_id),
            )

    # ── Entity ↔ Memory ─────────────────────────────────────────────────────

    def link_memory(self, entity_id: str, user_id: str, memory_id: str) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR IGNORE INTO rb_entity_memories
                   (id, entity_id, user_id, memory_id, created_at)
                   VALUES (?,?,?,?,?)""",
                (_new_id(), entity_id, user_id, memory_id, _utc_iso()),
            )

    def get_memories_for_entity(self, entity_id: str) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT memory_id FROM rb_entity_memories WHERE entity_id=?", (entity_id,)
            ).fetchall()
        return [r["memory_id"] for r in rows]

    def get_entities_for_memory(self, user_id: str, memory_id: str) -> list[RBEntity]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT rbe.* FROM rb_entities rbe
                   JOIN rb_entity_memories rbem ON rbem.entity_id = rbe.id
                   WHERE rbem.user_id=? AND rbem.memory_id=?""",
                (user_id, memory_id),
            ).fetchall()
        return [_row_to_entity(r) for r in rows]


def _row_to_slot(row: sqlite3.Row) -> RBSlot:
    return RBSlot(
        id=row["id"], user_id=row["user_id"], name=row["name"],
        description=row["description"], created_at=row["created_at"],
    )


def _row_to_entity(row: sqlite3.Row) -> RBEntity:
    return RBEntity(
        id=row["id"], user_id=row["user_id"], slot_id=row["slot_id"],
        name=row["name"], description=row["description"],
        embedding=json.loads(row["embedding"]) if row["embedding"] else None,
        source_entity_id=row["source_entity_id"],
        created_at=row["created_at"], updated_at=row["updated_at"],
    )
