"""左脑 slot→entity 图层。

层级：slot（base SlotV2 / 子图机制涌现的 dynamic_slot）
        └── entity（挂在某个 slot 下的具体节点，如 "work" 下的 "XX项目"）
              └── memory（挂在某个 entity 下的具体记忆 id）

slot_ref 是个不透明的字符串标识，兼容两种 slot 来源：
  - base slot   : SlotV2 的字符串值，如 "work"
  - dynamic_slot: DynamicSlotStore.create_dynamic_slot() 用的 name（SubgraphManager 涌现）

不在乎 slot_ref 具体来自哪一种——调用方自己决定传哪个。

entity 去重靠语义相似度（不是精确字符串匹配）：新事实先跟同一 slot 下已有
entity 的 embedding 比相似度，够像就复用，不够像才新建。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

#: 维度不符只提醒一次，别每轮刷屏。
_WARNED_DIM: set = set()

from voicemem.utils.common._graph_common import cosine as _cosine, new_id as _new_id, utc_iso as _utc_iso

DEFAULT_MATCH_THRESHOLD = 0.65


@dataclass
class GraphEntity:
    id: str
    user_id: str
    slot_ref: str
    name: str
    description: str = ""
    embedding: list[float] | None = None
    can_split: bool = True
    created_at: str = field(default_factory=_utc_iso)
    updated_at: str = field(default_factory=_utc_iso)


class GraphEntityStore:
    """SQLite 存取左脑 slot→entity→memory 三层图。线程安全：每次操作新建连接。"""

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
            CREATE TABLE IF NOT EXISTS graph_entities (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                slot_ref    TEXT NOT NULL,
                name        TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                embedding   TEXT,
                can_split   INTEGER NOT NULL DEFAULT 1,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                UNIQUE (user_id, slot_ref, name)
            );
            CREATE INDEX IF NOT EXISTS idx_ge_slot ON graph_entities(user_id, slot_ref);

            CREATE TABLE IF NOT EXISTS graph_entity_memories (
                id          TEXT PRIMARY KEY,
                entity_id   TEXT NOT NULL,
                user_id     TEXT NOT NULL,
                memory_id   TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                UNIQUE (entity_id, memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_gem_entity ON graph_entity_memories(entity_id);
            CREATE INDEX IF NOT EXISTS idx_gem_memory ON graph_entity_memories(user_id, memory_id);

            CREATE TABLE IF NOT EXISTS graph_query_activations (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                query_id    TEXT NOT NULL,
                entity_id   TEXT NOT NULL,
                session_id  TEXT,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_gqa_user_query ON graph_query_activations(user_id, query_id);
            """)
            # session_id 列的 ALTER TABLE 必须先跑完，index 才能建在这一列上
            # ——之前把两者反过来（先在 executescript 里建引用 session_id 的
            # index，ALTER TABLE 放在 executescript 后面）在一个 Phase 4 之前
            # 就已存在的旧库上会直接炸：CREATE TABLE IF NOT EXISTS 对已存在的
            # 表是空操作，不会带出新列，紧跟着的 CREATE INDEX 找不到 session_id
            # 列直接报错，ALTER TABLE 那一步永远轮不到执行（真实复现：
            # openai_voice_demo 从 Phase 0 就开始用的 graph_entities.sqlite，
            # 跑到 Phase 8 全链路真实回归测试时才暴露"no such column:
            # session_id"）。新库（CREATE TABLE 自带 session_id 列）不受影响，
            # 只有跨版本升级的旧库会踩这个顺序错误。
            try:
                c.execute("ALTER TABLE graph_query_activations ADD COLUMN session_id TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists — skip
            c.execute("CREATE INDEX IF NOT EXISTS idx_gqa_user_session ON graph_query_activations(user_id, session_id)")

    # ── Query Activation（簇涌现 ρ 公式用）───────────────────────────────────
    # 记录"每一次检索激活了哪些 graph_entity"，供 SubgraphManager 算论文的
    # ρ(H) = (1/|Q|)·Σ_{q∈Q} |A_q∩H|/|A_q∪H|——这里的实体空间必须跟
    # SubgraphManager 判定候选子集用的是同一套（graph_entities，不是
    # cognitive_graph 那套 NER 实体），否则算出来的交并集毫无意义。

    def record_query_activation(
        self, user_id: str, query_id: str, entity_ids: list[str], *, session_id: str | None = None
    ) -> None:
        """记一次查询激活的 graph_entity 集合（A_q）。为空则不记。

        session_id：论文里 Q 是"当前会话"的查询集合，不是终身历史——这里存
        下触发这次检索时所在的 session，供 compute_rho() 按 session 过滤。
        调用方不传 session_id（没有 session 概念）时存 NULL，compute_rho()
        同样不传时会退回旧行为（用全部历史），不强制要求所有调用方都升级。
        """
        if not entity_ids:
            return
        now = _utc_iso()
        with self._conn() as c:
            c.executemany(
                "INSERT INTO graph_query_activations (id, user_id, query_id, entity_id, session_id, created_at)"
                " VALUES (?,?,?,?,?,?)",
                [(_new_id(), user_id, query_id, eid, session_id, now) for eid in set(entity_ids)],
            )

    def compute_rho(
        self, user_id: str, candidate_entity_ids: set[str], *, session_id: str | None = None
    ) -> float:
        """ρ(H) = (1/|Q|)·Σ_{q∈Q} |A_q∩H|/|A_q∪H|——H 是候选 entity 子集。

        session_id 传了时，Q 限定为该 session 内记录的查询激活（论文字面：
        Q 是当前会话的查询集合，不是终身历史）；不传时退回旧行为，Q 是该
        用户全部历史查询激活（没有 session 概念的调用方，比如没接
        SessionTracker 的测试/开发脚本，行为不受影响）。没有匹配的历史查询
        激活记录时返回 0.0（没有查询行为信号，无法评估——由调用方决定这种
        情况下怎么兜底）。"""
        if not candidate_entity_ids:
            return 0.0
        with self._conn() as c:
            if session_id is not None:
                rows = c.execute(
                    "SELECT query_id, entity_id FROM graph_query_activations WHERE user_id=? AND session_id=?",
                    (user_id, session_id),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT query_id, entity_id FROM graph_query_activations WHERE user_id=?",
                    (user_id,),
                ).fetchall()
        if not rows:
            return 0.0
        by_query: dict[str, set[str]] = {}
        for r in rows:
            by_query.setdefault(r["query_id"], set()).add(r["entity_id"])
        total = 0.0
        for a_q in by_query.values():
            union = a_q | candidate_entity_ids
            if not union:
                continue
            total += len(a_q & candidate_entity_ids) / len(union)
        return total / len(by_query)

    # ── Entity（精确名字匹配，适合种子/固定词表场景） ────────────────────────

    def get_or_create_entity(
        self,
        user_id: str,
        slot_ref: str,
        name: str,
        *,
        description: str = "",
        embedding: list[float] | None = None,
    ) -> GraphEntity:
        """同一 (user_id, slot_ref, name) 已存在就直接返回，否则新建。"""
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM graph_entities WHERE user_id=? AND slot_ref=? AND name=?",
                (user_id, slot_ref, name),
            ).fetchone()
            if row:
                return _row_to_entity(row)
            return self._insert_entity(c, user_id, slot_ref, name, description, embedding)

    # ── Entity（语义相似度匹配，适合自由生成的感性/事实类entity） ─────────────

    def find_similar_entity(
        self,
        user_id: str,
        slot_ref: str,
        embedding: list[float],
        *,
        threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> GraphEntity | None:
        """在同一 slot 下找语义最相似的 entity，相似度不够则返回 None。"""
        best: GraphEntity | None = None
        best_sim = -1.0
        mismatched = 0
        for ent in self.get_entities_for_slot(user_id, slot_ref):
            if ent.embedding is None:
                continue
            if len(ent.embedding) != len(embedding):
                mismatched += 1          # 换过 embedder，老向量维度对不上
                continue
            sim = _cosine(embedding, ent.embedding)
            if sim > best_sim:
                best_sim, best = sim, ent
        if mismatched and not _WARNED_DIM:
            _WARNED_DIM.add(1)
            print(f"[GraphEntity] ⚠ {mismatched} 个实体向量的维度跟当前 embedder 不符，"
                  "已跳过——换过 embedder 之后老向量作废，语义去重会失效。"
                  "重新 embed 一遍或清掉这些向量。", flush=True)
        return best if best is not None and best_sim >= threshold else None

    def get_or_create_entity_semantic(
        self,
        user_id: str,
        slot_ref: str,
        name: str,
        embedding: list[float],
        *,
        description: str = "",
        threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> tuple[GraphEntity, bool]:
        """先按语义相似度找现有 entity，找到就复用；没有就新建。

        Returns
        -------
        (entity, created) — created=True 表示这次新建了一个 entity。
        """
        existing = self.find_similar_entity(user_id, slot_ref, embedding, threshold=threshold)
        if existing is not None:
            return existing, False
        with self._conn() as c:
            ent = self._insert_entity(c, user_id, slot_ref, name, description, embedding)
        return ent, True

    def _insert_entity(
        self,
        c: sqlite3.Connection,
        user_id: str,
        slot_ref: str,
        name: str,
        description: str,
        embedding: list[float] | None,
    ) -> GraphEntity:
        now = _utc_iso()
        ent = GraphEntity(
            id=_new_id(), user_id=user_id, slot_ref=slot_ref, name=name,
            description=description, embedding=embedding, can_split=True,
            created_at=now, updated_at=now,
        )
        c.execute(
            """INSERT INTO graph_entities
               (id, user_id, slot_ref, name, description, embedding, can_split, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (ent.id, ent.user_id, ent.slot_ref, ent.name, ent.description,
             json.dumps(embedding) if embedding is not None else None,
             1, ent.created_at, ent.updated_at),
        )
        return ent

    def get_entities_for_slot(self, user_id: str, slot_ref: str) -> list[GraphEntity]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM graph_entities WHERE user_id=? AND slot_ref=? ORDER BY name",
                (user_id, slot_ref),
            ).fetchall()
        return [_row_to_entity(r) for r in rows]

    def update_slot_ref(self, entity_id: str, new_slot_ref: str) -> None:
        """把 entity 正式迁移到新 slot 下（子图判定通过、新建 slot 后调用）。

        目标 slot 下如果已经有同名 entity——这会发生，因为一个 entity 被挪走后，
        write 时的语义查重是 scoped 在 slot_ref 内的，同名概念之后再被提到，会在
        原 slot 下重新造一个同名 entity（找不到已经挪走的那个）；这个新造的如果
        之后也被子图判定挪去同一个目标 slot，就会跟先挪过去的那个撞
        UNIQUE(user_id, slot_ref, name)——这种情况按同一个概念处理，合并成一个：
        把这条的 memory 链接转移过去，再删掉这条，而不是让 UPDATE 崩掉。
        """
        with self._conn() as c:
            row = c.execute("SELECT * FROM graph_entities WHERE id=?", (entity_id,)).fetchone()
            if row is None:
                return
            name, user_id = row["name"], row["user_id"]
            existing = c.execute(
                "SELECT id FROM graph_entities WHERE user_id=? AND slot_ref=? AND name=? AND id!=?",
                (user_id, new_slot_ref, name, entity_id),
            ).fetchone()

            if existing is None:
                c.execute(
                    "UPDATE graph_entities SET slot_ref=?, updated_at=? WHERE id=?",
                    (new_slot_ref, _utc_iso(), entity_id),
                )
                return

            target_id = existing["id"]
            mem_rows = c.execute(
                "SELECT memory_id, created_at FROM graph_entity_memories WHERE entity_id=?",
                (entity_id,),
            ).fetchall()
            for mrow in mem_rows:
                c.execute(
                    """INSERT OR IGNORE INTO graph_entity_memories
                       (id, entity_id, user_id, memory_id, created_at)
                       VALUES (?,?,?,?,?)""",
                    (_new_id(), target_id, user_id, mrow["memory_id"], mrow["created_at"]),
                )
            c.execute("DELETE FROM graph_entity_memories WHERE entity_id=?", (entity_id,))
            c.execute("DELETE FROM graph_entities WHERE id=?", (entity_id,))

    def get_entity(self, entity_id: str) -> GraphEntity | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM graph_entities WHERE id=?", (entity_id,)
            ).fetchone()
        return _row_to_entity(row) if row else None

    def set_description(self, entity_id: str, description: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE graph_entities SET description=?, updated_at=? WHERE id=?",
                (description, _utc_iso(), entity_id),
            )

    def mark_cannot_split(self, entity_id: str) -> None:
        """把 can_split 永久置 False。单向操作——不提供反向的 set True 接口，
        避免误用；新entity默认就是True，需要重新激活时应该新建/关联新entity，
        而不是把旧entity的can_split翻回去。"""
        with self._conn() as c:
            c.execute(
                "UPDATE graph_entities SET can_split=0, updated_at=? WHERE id=?",
                (_utc_iso(), entity_id),
            )

    # ── Entity ↔ Memory ─────────────────────────────────────────────────────

    def link_memory(self, entity_id: str, user_id: str, memory_id: str) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR IGNORE INTO graph_entity_memories
                   (id, entity_id, user_id, memory_id, created_at)
                   VALUES (?,?,?,?,?)""",
                (_new_id(), entity_id, user_id, memory_id, _utc_iso()),
            )

    def get_entities_for_memory(self, user_id: str, memory_id: str) -> list[GraphEntity]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT ge.* FROM graph_entities ge
                   JOIN graph_entity_memories gem ON gem.entity_id = ge.id
                   WHERE gem.user_id=? AND gem.memory_id=?""",
                (user_id, memory_id),
            ).fetchall()
        return [_row_to_entity(r) for r in rows]


def _row_to_entity(row: sqlite3.Row) -> GraphEntity:
    return GraphEntity(
        id=row["id"], user_id=row["user_id"], slot_ref=row["slot_ref"],
        name=row["name"], description=row["description"],
        embedding=json.loads(row["embedding"]) if row["embedding"] else None,
        can_split=bool(row["can_split"]),
        created_at=row["created_at"], updated_at=row["updated_at"],
    )
