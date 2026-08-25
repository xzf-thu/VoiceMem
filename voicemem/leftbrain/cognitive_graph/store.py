"""认知图 SQLite 存储层。

表结构：
  entities            — 认知图节点（人/项目/知识/任务等）
  entity_edges        — 节点间有向边（typed relation）
  entity_memory_links — 记忆挂回节点的链接（含角色）
  memories            — 记忆元数据 wrapper（槽位/敏感度/TTL）
  slot_profiles       — 每个槽位的摘要快照
  affective_edges     — 右脑情感边（预留）
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from voicemem.utils.common._graph_common import cosine as _cosine, new_id as _new_id, utc_iso as _utc_iso

from .slot_v2 import SlotV2
from .types import (
    AffectiveEdge,
    AnnotatedFact,
    Entity,
    EntityEdge,
    EntityMemoryLink,
    EntityMemoryRole,
    EntityType,
    MemoryRecord,
    SlotProfile,
)


def normalize_name(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(u"^[\u201c\u201d\u2018\u2019\"']+|[\u201c\u201d\u2018\u2019\"']+$", "", s)
    return s


# Real bug found via live testing: SlotV2(row["slot"]) crashes with
# ValueError on rows written under the OLDER, pre-unification SlotType
# taxonomy (types.py's 7-category "people/projects/knowledge/tasks/places/
# routines/assets" set, not SlotV2's "work/finance/relationships/health/
# goals/daily_life/knowledge") -- confirmed real rows in a live memory store
# with slot='people' (person entities from before the Phase 2 taxonomy
# unification, or some other still-live write path never fully migrated).
# Every caller of find_entities()/get_entity() has NO try/except around it,
# so one bad row raises out of a plain list comprehension and kills the
# whole query -- and because AnchorRouter._build_anchors() (the right
# brain's entity-anchor resolution, called on EVERY Search()) calls
# find_entities() unconditionally, this was silently zeroing out ALL right-
# brain results for the entire user on every single search, not just
# queries touching the bad entity. Mirrors voice_input.py's
# _VOICE_SLOT_TO_SLOTV2 mapping for the same old->new taxonomy migration;
# anything still unrecognized falls back to KNOWLEDGE rather than crashing.
_LEGACY_SLOT_TO_SLOTV2 = {
    "people": "relationships", "person": "relationships",
    "projects": "work", "project": "work", "tasks": "work", "task": "work",
    "places": "daily_life", "place": "daily_life",
    "routines": "daily_life", "routine": "daily_life",
    "assets": "daily_life", "asset": "daily_life",
}


def _coerce_slot_v2(raw: str) -> SlotV2:
    try:
        return SlotV2(raw)
    except ValueError:
        return SlotV2(_LEGACY_SLOT_TO_SLOTV2.get(raw, "knowledge"))


# 跟 slot_split/graph_entity_store.py 用同一个阈值——两边都是"同一个实体的
# 语义去重"这件事，标准不该不一样。
SEMANTIC_MATCH_THRESHOLD = 0.65

# entity_edges 的 weight 是"这条关系被观察/复述过几次"的累加计数（首次创建
# 为 1，之后每次同一条边再被抽取到就 +1——不是像 confidence 那样取 max 封顶）。
# strong/weak 是这个累加权重的阈值化分类：只出现过一次的关系还只是弱证据
# （可能是误抽取或一次性提及），复现 ≥2 次才升级成 strong（真实、稳定的关系）。
EDGE_STRONG_THRESHOLD = 2.0

# 读取时按 updated_at 做指数衰减（不需要单独的后台衰减任务）：距上次被强化
# 越久，边在当次读取里返回的有效权重越低，反映"很久没被提起的关系没那么
# 活跃了"，但不改写库里存的原始 weight——衰减只影响这次读取的返回值。
EDGE_DECAY_HALFLIFE_DAYS = 30.0

# 记忆热度：同样是"读取时按 updated_at/last_hit_at 指数衰减，不改库里存的
# 原始值"这套模式（跟 entity_edges.weight 复用同一个 _decayed_weight()
# 实现），语义是"多久没被检索命中过"。命中一次 heat +1（累加，不是取
# max）；半衰期比 entity_edges 短——记忆的"还有用"判断应该比关系强弱更
# 敏感地方，用户几周不问起的事应该比几周没被提到过的人物关系更快冷却。
MEMORY_HEAT_DECAY_HALFLIFE_DAYS = 14.0

# 归档阈值：读取时衰减后的 heat 低于这个值，且这条记忆存在时间够久（见
# list_archivable_memories 的 min_age_days），才认为是"长期不用"，交给
# 调用方决定要不要真的归档（这里只负责判定，不负责执行——执行需要调用
# mem0 那边的 expiration_date，见 mem0_backend_store.py::archive_memory）。
ARCHIVE_HEAT_THRESHOLD = 0.3


def _decayed_weight(weight: float, updated_at: str, *, halflife_days: float = EDGE_DECAY_HALFLIFE_DAYS) -> float:
    """\u6307\u6570\u8870\u51cf\uff1aelapsed=0 \u65f6\u539f\u6837\u8fd4\u56de weight\uff0c\u6bcf\u8fc7\u4e00\u4e2a\u534a\u8870\u671f\u6253\u5bf9\u6298\u3002"""
    try:
        updated = datetime.fromisoformat(updated_at)
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        elapsed_days = (datetime.now(timezone.utc) - updated).total_seconds() / 86400.0
    except (TypeError, ValueError):
        return weight
    if elapsed_days <= 0 or halflife_days <= 0:
        return weight
    return weight * (0.5 ** (elapsed_days / halflife_days))


def make_entity_id(entity_type: EntityType, name_norm: str) -> str:
    slug = re.sub(r"[^a-z0-9一-鿿]", "_", name_norm)[:32].strip("_")
    return f"{entity_type.value}_{slug}_{uuid.uuid4().hex[:6]}"


def normalize_relation(rel: str) -> str:
    s = rel.strip().lower()
    s = re.sub(r"[^a-z0-9一-鿿]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_") or "related_to"


def _normalize_role(raw: str | None) -> str:
    """把 LLM 给的角色描述规范化为 EntityMemoryRole 的四个值之一。"""
    if not raw:
        return EntityMemoryRole.CONTEXT.value
    r = raw.strip().lower()
    if r in ("subject", "主语", "主体", "speaker"):
        return EntityMemoryRole.SUBJECT.value
    if r in ("object", "宾语", "对象", "target"):
        return EntityMemoryRole.OBJECT.value
    if r in ("owner", "所有者", "owner of"):
        return EntityMemoryRole.OWNER.value
    return EntityMemoryRole.CONTEXT.value


class CognitiveGraphStore:
    """SQLite 认知图存储。线程安全：每次操作新建连接。"""

    def __init__(self, db_path: Path | str, embedder: Any = None) -> None:
        """embedder: 可选，有 `embed_texts(list[str]) -> list[list[float]]`
        方法的对象（跟 local_memory_store.TextEmbedder 同一个 protocol）。传了
        才会在 upsert_entity 里做语义去重（同名不同措辞的实体合并成一个），
        不传就退回旧行为（精确字符串匹配），不会报错——保持向后兼容任何还
        没升级调用方式的调用方。
        """
        self._path = Path(db_path)
        self._embedder = embedder
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        self._migrate_schema()

    # ── Connection ────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self._path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        return c

    # ── Schema ────────────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                name        TEXT NOT NULL,
                name_norm   TEXT NOT NULL,
                slot        TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                confidence  REAL NOT NULL DEFAULT 1.0,
                importance  REAL NOT NULL DEFAULT 0.5,
                aliases     TEXT NOT NULL DEFAULT '[]',
                properties  TEXT NOT NULL DEFAULT '{}',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ent_user ON entities(user_id);
            CREATE INDEX IF NOT EXISTS idx_ent_norm ON entities(user_id, name_norm, entity_type);

            CREATE TABLE IF NOT EXISTS entity_edges (
                id                  TEXT PRIMARY KEY,
                user_id             TEXT NOT NULL,
                from_entity_id      TEXT NOT NULL,
                to_entity_id        TEXT NOT NULL,
                relation_type       TEXT NOT NULL,
                role_label          TEXT,
                confidence          REAL NOT NULL DEFAULT 1.0,
                weight              REAL NOT NULL DEFAULT 1.0,
                edge_type           TEXT NOT NULL DEFAULT 'weak',
                status              TEXT NOT NULL DEFAULT 'active',
                evidence_memory_ids TEXT NOT NULL DEFAULT '[]',
                created_at          TEXT NOT NULL,
                updated_at          TEXT NOT NULL,
                UNIQUE (user_id, from_entity_id, to_entity_id, relation_type)
            );
            CREATE INDEX IF NOT EXISTS idx_edge_from ON entity_edges(user_id, from_entity_id);
            CREATE INDEX IF NOT EXISTS idx_edge_to   ON entity_edges(user_id, to_entity_id);

            CREATE TABLE IF NOT EXISTS entity_memory_links (
                id             TEXT PRIMARY KEY,
                memory_id      TEXT NOT NULL,
                entity_id      TEXT NOT NULL,
                user_id        TEXT NOT NULL,
                role           TEXT NOT NULL DEFAULT 'context',
                relation_hint  TEXT,
                created_at     TEXT NOT NULL,
                UNIQUE (memory_id, entity_id)
            );
            CREATE INDEX IF NOT EXISTS idx_eml_mem ON entity_memory_links(memory_id);
            CREATE INDEX IF NOT EXISTS idx_eml_ent ON entity_memory_links(entity_id);

            CREATE TABLE IF NOT EXISTS memories (
                id           TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL,
                slot         TEXT NOT NULL,
                memory_type  TEXT NOT NULL DEFAULT 'fact',
                content      TEXT NOT NULL,
                confidence   REAL NOT NULL DEFAULT 1.0,
                sensitivity  REAL NOT NULL DEFAULT 0.0,
                ttl          INTEGER,
                heat         REAL NOT NULL DEFAULT 1.0,
                last_hit_at  TEXT,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id, slot);

            CREATE TABLE IF NOT EXISTS slot_profiles (
                user_id      TEXT NOT NULL,
                slot         TEXT NOT NULL,
                summary      TEXT NOT NULL DEFAULT '',
                entity_ids   TEXT NOT NULL DEFAULT '[]',
                entity_count INTEGER NOT NULL DEFAULT 0,
                memory_count INTEGER NOT NULL DEFAULT 0,
                last_updated TEXT NOT NULL,
                PRIMARY KEY (user_id, slot)
            );

            CREATE TABLE IF NOT EXISTS affective_edges (
                id                  TEXT PRIMARY KEY,
                user_id             TEXT NOT NULL,
                from_entity_id      TEXT NOT NULL,
                to_entity_id        TEXT,
                trigger_frame       TEXT,
                emotion             TEXT,
                appraisal           TEXT,
                response_policy     TEXT,
                confidence          REAL NOT NULL DEFAULT 1.0,
                evidence_memory_ids TEXT NOT NULL DEFAULT '[]',
                created_at          TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ae_user ON affective_edges(user_id, from_entity_id);

            CREATE TABLE IF NOT EXISTS query_activations (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                query_id    TEXT NOT NULL,
                entity_id   TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_qa_user_query ON query_activations(user_id, query_id);
            CREATE INDEX IF NOT EXISTS idx_qa_user_entity ON query_activations(user_id, entity_id);
            """)

    def _migrate_schema(self) -> None:
        """对已有数据库做 ALTER TABLE 补列，安全幂等。"""
        migrations = [
            "ALTER TABLE entities ADD COLUMN importance REAL NOT NULL DEFAULT 0.5",
            "ALTER TABLE entities ADD COLUMN aliases TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE entities ADD COLUMN embedding TEXT",
            "ALTER TABLE entities ADD COLUMN description TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE entity_edges ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
            "ALTER TABLE entity_edges ADD COLUMN weight REAL NOT NULL DEFAULT 1.0",
            "ALTER TABLE entity_edges ADD COLUMN edge_type TEXT NOT NULL DEFAULT 'weak'",
            "ALTER TABLE entity_memory_links ADD COLUMN role TEXT NOT NULL DEFAULT 'context'",
            "ALTER TABLE entity_memory_links ADD COLUMN relation_hint TEXT",
            "ALTER TABLE slot_profiles ADD COLUMN summary TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE slot_profiles ADD COLUMN entity_ids TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE affective_edges ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0",
            "ALTER TABLE affective_edges ADD COLUMN evidence_memory_ids TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE memories ADD COLUMN heat REAL NOT NULL DEFAULT 1.0",
            "ALTER TABLE memories ADD COLUMN last_hit_at TEXT",
        ]
        with self._conn() as c:
            for sql in migrations:
                try:
                    c.execute(sql)
                except sqlite3.OperationalError:
                    pass  # column already exists — skip

    # ── Entity CRUD ───────────────────────────────────────────────────────────

    def upsert_entity(
        self,
        user_id: str,
        name: str,
        entity_type: EntityType,
        slot: SlotV2 | None = None,
        description: str | None = None,
        confidence: float = 1.0,
        importance: float = 0.5,
        aliases: list[str] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> Entity:
        """找到已有同名或语义相近的同类实体则更新，否则新建。返回 Entity。

        去重分两层：先精确字符串匹配（最常见情况，免一次全表扫描）；没命中
        且构造时传了 embedder，才退一步做语义相似度匹配（cosine≥
        SEMANTIC_MATCH_THRESHOLD）——之前这里只有精确匹配一条路，LLM 抽取
        同一个实体在不同轮次措辞不稳定（"项目A" vs "A项目"、别名、大小写/
        标点差异之外的真实语义改写）时会各自建一个新节点，图里全是重复实体，
        一跳邻接扩散也会因为实体分裂而漏边。没传 embedder 时行为跟以前完全
        一样，不影响没升级调用方式的调用方。

        description（d_v）：首次创建实体时用调用方传入的原句填充；实体已
        存在且已有 description 时不覆盖（第一次出现的原句最贴近"这是什么"，
        后续每次提及都重写会来回抖动）——description 为空的老实体（迁移前
        数据）遇到新描述会补上，不算覆盖。
        """
        name_norm = normalize_name(name)
        effective_slot = slot or SlotV2.KNOWLEDGE
        now = _utc_iso()

        embedding: list[float] | None = None
        if self._embedder is not None:
            try:
                embedding = self._embedder.embed_texts([name])[0]
            except Exception:
                embedding = None

        with self._conn() as c:
            # person/user 只按名字精确去重，不做语义匹配——两个真人名字听起来
            # 相近（比如"张伟"和"张卫"）语义上很可能不是同一个人，精确匹配更安全。
            if entity_type in (EntityType.PERSON, EntityType.USER):
                row = c.execute(
                    "SELECT * FROM entities WHERE user_id=? AND name_norm=?"
                    " AND entity_type IN ('person','user')",
                    (user_id, name_norm),
                ).fetchone()
            else:
                row = c.execute(
                    "SELECT * FROM entities WHERE user_id=? AND name_norm=? AND entity_type=?",
                    (user_id, name_norm, entity_type.value),
                ).fetchone()
                if row is None and embedding is not None:
                    candidates = c.execute(
                        "SELECT * FROM entities WHERE user_id=? AND entity_type=? AND embedding IS NOT NULL",
                        (user_id, entity_type.value),
                    ).fetchall()
                    best_row, best_sim = None, -1.0
                    for cand in candidates:
                        try:
                            cand_emb = json.loads(cand["embedding"])
                        except (TypeError, ValueError):
                            continue
                        sim = _cosine(embedding, cand_emb)
                        if sim > best_sim:
                            best_sim, best_row = sim, cand
                    if best_row is not None and best_sim >= SEMANTIC_MATCH_THRESHOLD:
                        row = best_row

            if row:
                new_conf = max(float(row["confidence"]), confidence)
                new_imp  = max(float(row["importance"]), importance)
                props = json.loads(row["properties"] or "{}")
                if properties:
                    props.update(properties)
                existing_aliases = json.loads(row["aliases"] or "[]")
                for a in (aliases or []):
                    if a not in existing_aliases:
                        existing_aliases.append(a)
                # 语义匹配命中但字面名字不同——把这次的说法记成别名，往后同一个
                # 说法可以直接精确匹配命中，不用每次都重新算相似度。
                if name.strip() and name.strip() not in existing_aliases and normalize_name(name) != row["name_norm"]:
                    existing_aliases.append(name.strip())
                existing_desc = row["description"] if "description" in row.keys() else ""
                new_desc = existing_desc or (description or "").strip()
                c.execute(
                    """UPDATE entities
                       SET confidence=?, importance=?, aliases=?, properties=?, description=?, updated_at=?
                       WHERE id=?""",
                    (new_conf, new_imp,
                     json.dumps(existing_aliases, ensure_ascii=False),
                     json.dumps(props, ensure_ascii=False), new_desc, now, row["id"]),
                )
                return Entity(
                    id=row["id"], user_id=user_id,
                    entity_type=entity_type, name=row["name"],
                    name_norm=row["name_norm"], slot=effective_slot,
                    description=new_desc,
                    confidence=new_conf, importance=new_imp,
                    aliases=existing_aliases, properties=props,
                    created_at=row["created_at"], updated_at=now,
                )
            else:
                eid = make_entity_id(entity_type, name_norm)
                props = properties or {}
                alias_list = aliases or []
                desc = (description or "").strip()
                c.execute(
                    """INSERT INTO entities
                       (id,user_id,entity_type,name,name_norm,slot,description,confidence,importance,aliases,properties,embedding,created_at,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (eid, user_id, entity_type.value, name.strip(), name_norm,
                     effective_slot.value, desc, confidence, importance,
                     json.dumps(alias_list, ensure_ascii=False),
                     json.dumps(props, ensure_ascii=False),
                     json.dumps(embedding) if embedding is not None else None,
                     now, now),
                )
                return Entity(
                    id=eid, user_id=user_id,
                    entity_type=entity_type, name=name.strip(),
                    name_norm=name_norm, slot=effective_slot,
                    description=desc,
                    confidence=confidence, importance=importance,
                    aliases=alias_list, properties=props,
                    created_at=now, updated_at=now,
                )

    def get_entity(self, entity_id: str) -> Entity | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM entities WHERE id=?", (entity_id,)).fetchone()
        return self._row_to_entity(row) if row else None

    def find_entities(
        self, user_id: str, *,
        slots: list[SlotV2] | None = None,
        entity_ids: list[str] | None = None,
        name_norm: str | None = None,
    ) -> list[Entity]:
        wheres, params = ["user_id=?"], [user_id]
        if slots:
            ph = ",".join("?" * len(slots))
            wheres.append(f"slot IN ({ph})")
            params.extend(s.value for s in slots)
        if entity_ids:
            ph = ",".join("?" * len(entity_ids))
            wheres.append(f"id IN ({ph})")
            params.extend(entity_ids)
        if name_norm:
            wheres.append("name_norm=?")
            params.append(name_norm)
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM entities WHERE " + " AND ".join(wheres), params
            ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def find_entities_by_name_fuzzy(
        self, user_id: str, name_norm: str, *, slots: list[SlotV2] | None = None
    ) -> list[Entity]:
        """先精确匹配，再做 LIKE '%name%' 兜底。"""
        exact = self.find_entities(user_id, name_norm=name_norm, slots=slots)
        if exact:
            return exact
        wheres = ["user_id=?", "name_norm LIKE ?"]
        params: list = [user_id, f"%{name_norm}%"]
        if slots:
            ph = ",".join("?" * len(slots))
            wheres.append(f"slot IN ({ph})")
            params.extend(s.value for s in slots)
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM entities WHERE " + " AND ".join(wheres), params
            ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        props = json.loads(row["properties"] or "{}")
        # importance/aliases/description may not exist in migrated rows (handled by DEFAULT)
        importance = float(row["importance"]) if "importance" in row.keys() else 0.5
        aliases = json.loads(row["aliases"] or "[]") if "aliases" in row.keys() else []
        description = row["description"] if "description" in row.keys() else ""
        return Entity(
            id=row["id"], user_id=row["user_id"],
            entity_type=EntityType(row["entity_type"]),
            name=row["name"], name_norm=row["name_norm"],
            slot=_coerce_slot_v2(row["slot"]),
            description=description or "",
            confidence=float(row["confidence"]),
            importance=importance,
            aliases=aliases,
            properties=props if isinstance(props, dict) else {},
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    # ── Entity Edge CRUD ──────────────────────────────────────────────────────

    def upsert_edge(
        self,
        user_id: str,
        from_entity_id: str,
        to_entity_id: str,
        relation_type: str,
        *,
        role_label: str | None = None,
        confidence: float = 1.0,
        status: str = "active",
        evidence_memory_id: str | None = None,
    ) -> EntityEdge:
        rel = normalize_relation(relation_type)
        now = _utc_iso()
        with self._conn() as c:
            row = c.execute(
                """SELECT * FROM entity_edges
                   WHERE user_id=? AND from_entity_id=? AND to_entity_id=? AND relation_type=?""",
                (user_id, from_entity_id, to_entity_id, rel),
            ).fetchone()
            if row:
                eids = json.loads(row["evidence_memory_ids"] or "[]")
                # weight 只在这是条"新证据"时才 +1（同一个 memory_id 被重复
                # 传进来——比如同一条 fact 被重新标注——不该让权重虚高）；没传
                # evidence_memory_id 时没法去重，按调用一次算一次新证据。
                is_new_evidence = (not evidence_memory_id) or evidence_memory_id not in eids
                if evidence_memory_id and evidence_memory_id not in eids:
                    eids.append(evidence_memory_id)
                new_conf = max(float(row["confidence"]), confidence)
                old_weight = float(row["weight"]) if "weight" in row.keys() else 1.0
                new_weight = old_weight + 1.0 if is_new_evidence else old_weight
                new_edge_type = "strong" if new_weight >= EDGE_STRONG_THRESHOLD else "weak"
                c.execute(
                    """UPDATE entity_edges
                       SET confidence=?, weight=?, edge_type=?, evidence_memory_ids=?,
                           role_label=COALESCE(?,role_label), updated_at=?
                       WHERE id=?""",
                    (new_conf, new_weight, new_edge_type, json.dumps(eids), role_label, now, row["id"]),
                )
                return EntityEdge(
                    id=row["id"], user_id=user_id,
                    from_entity_id=from_entity_id, to_entity_id=to_entity_id,
                    relation_type=rel, role_label=role_label or row["role_label"],
                    confidence=new_conf, weight=new_weight, edge_type=new_edge_type,
                    status=row["status"] if "status" in row.keys() else "active",
                    evidence_memory_ids=eids,
                    created_at=row["created_at"], updated_at=now,
                )
            else:
                eid = _new_id()
                eids = [evidence_memory_id] if evidence_memory_id else []
                c.execute(
                    """INSERT INTO entity_edges
                       (id,user_id,from_entity_id,to_entity_id,relation_type,
                        role_label,confidence,weight,edge_type,status,evidence_memory_ids,created_at,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (eid, user_id, from_entity_id, to_entity_id, rel,
                     role_label, confidence, 1.0, "weak", status, json.dumps(eids), now, now),
                )
                return EntityEdge(
                    id=eid, user_id=user_id,
                    from_entity_id=from_entity_id, to_entity_id=to_entity_id,
                    relation_type=rel, role_label=role_label,
                    confidence=confidence, weight=1.0, edge_type="weak", status=status,
                    evidence_memory_ids=eids,
                    created_at=now, updated_at=now,
                )

    def edges_for_entity(self, entity_id: str, user_id: str) -> list[EntityEdge]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT * FROM entity_edges
                   WHERE user_id=? AND (from_entity_id=? OR to_entity_id=?)
                     AND status='active'""",
                (user_id, entity_id, entity_id),
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def _row_to_edge(self, row: sqlite3.Row) -> EntityEdge:
        # weight/edge_type may not exist in migrated rows (handled by DEFAULT).
        # 衰减只算在返回值里，不改库里存的原始 weight/updated_at——下次这条边
        # 再被强化时，累加权重接着从原始值算，不会因为读取过而被打断。
        raw_weight = float(row["weight"]) if "weight" in row.keys() else 1.0
        decayed = _decayed_weight(raw_weight, row["updated_at"])
        edge_type = row["edge_type"] if "edge_type" in row.keys() else (
            "strong" if raw_weight >= EDGE_STRONG_THRESHOLD else "weak"
        )
        return EntityEdge(
            id=row["id"], user_id=row["user_id"],
            from_entity_id=row["from_entity_id"],
            to_entity_id=row["to_entity_id"],
            relation_type=row["relation_type"],
            role_label=row["role_label"],
            confidence=float(row["confidence"]),
            weight=decayed, edge_type=edge_type or "weak",
            status=row["status"] if "status" in row.keys() else "active",
            evidence_memory_ids=json.loads(row["evidence_memory_ids"] or "[]"),
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    # ── Entity-Memory Link ────────────────────────────────────────────────────

    def link_memory(
        self,
        memory_id: str,
        entity_id: str,
        user_id: str,
        *,
        role: EntityMemoryRole = EntityMemoryRole.CONTEXT,
        relation_hint: str | None = None,
    ) -> None:
        now = _utc_iso()
        with self._conn() as c:
            c.execute(
                """INSERT INTO entity_memory_links
                   (id, memory_id, entity_id, user_id, role, relation_hint, created_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(memory_id, entity_id) DO UPDATE SET
                     role=CASE WHEN excluded.role != 'context' THEN excluded.role ELSE role END,
                     relation_hint=COALESCE(excluded.relation_hint, relation_hint)""",
                (_new_id(), memory_id, entity_id, user_id,
                 role.value, relation_hint, now),
            )

    def memory_ids_for_entities(self, entity_ids: list[str]) -> list[str]:
        if not entity_ids:
            return []
        ph = ",".join("?" * len(entity_ids))
        with self._conn() as c:
            rows = c.execute(
                f"SELECT DISTINCT memory_id FROM entity_memory_links WHERE entity_id IN ({ph})",
                entity_ids,
            ).fetchall()
        return [r["memory_id"] for r in rows]

    def memory_ids_for_slots(self, user_id: str, slots: list[SlotV2]) -> list[str]:
        """返回直接标注为给定 slot 的记忆 id（查 memories.slot，非实体类型）。"""
        if not slots:
            return []
        ph = ",".join("?" * len(slots))
        with self._conn() as c:
            rows = c.execute(
                f"SELECT id FROM memories WHERE user_id=? AND slot IN ({ph})",
                [user_id, *[s.value for s in slots]],
            ).fetchall()
        return [r["id"] for r in rows]

    def entity_ids_for_memory(self, memory_id: str) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT entity_id FROM entity_memory_links WHERE memory_id=?",
                (memory_id,),
            ).fetchall()
        return [r["entity_id"] for r in rows]

    def all_linked_memory_ids(self, user_id: str) -> list[str]:
        """返回该用户所有在 entity_memory_links 中出现的 memory_id。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT memory_id FROM entity_memory_links WHERE user_id=?",
                (user_id,),
            ).fetchall()
        return [r["memory_id"] for r in rows]

    def all_memory_record_ids(self, user_id: str) -> list[str]:
        """返回该用户所有在 memories 表中已有记录的 memory_id。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT id FROM memories WHERE user_id=?",
                (user_id,),
            ).fetchall()
        return [r["id"] for r in rows]

    def neighbor_entity_ids(self, user_id: str, entity_ids: list[str]) -> list[str]:
        """返回给定实体集合的 1-hop 邻居实体 id（不含自身）。"""
        if not entity_ids:
            return []
        ph = ",".join("?" * len(entity_ids))
        with self._conn() as c:
            rows = c.execute(
                f"""SELECT DISTINCT
                        CASE WHEN from_entity_id IN ({ph}) THEN to_entity_id
                             ELSE from_entity_id END AS neighbor_id
                    FROM entity_edges
                    WHERE user_id=? AND status='active'
                      AND (from_entity_id IN ({ph}) OR to_entity_id IN ({ph}))""",
                # 占位符顺序：CASE 里的 IN({ph}) 在 user_id=? 之前——之前写成
                # [user_id] + ids*3，user_id 被绑到 CASE 的第一个占位、真正的
                # user_id=? 拿到的是一个 entity id，永远查不到行：一跳邻居扩散
                # 从来没生效过（实测 821 题 f_nbr 全为 0）。
                entity_ids + [user_id] + entity_ids * 2,
            ).fetchall()
        seed_set = set(entity_ids)
        return [r["neighbor_id"] for r in rows if r["neighbor_id"] not in seed_set]

    # ── Memory Record (认知图自己的记忆元数据) ─────────────────────────────────

    def upsert_memory_record(
        self,
        user_id: str,
        memory_id: str,
        slot: SlotV2,
        content: str,
        *,
        memory_type: str = "fact",
        confidence: float = 1.0,
        sensitivity: float = 0.0,
        ttl: int | None = None,
    ) -> MemoryRecord:
        now = _utc_iso()
        with self._conn() as c:
            c.execute(
                """INSERT INTO memories
                   (id, user_id, slot, memory_type, content, confidence, sensitivity, ttl, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     slot=excluded.slot, confidence=excluded.confidence, updated_at=excluded.updated_at""",
                (memory_id, user_id, slot.value, memory_type, content,
                 confidence, sensitivity, ttl, now, now),
            )
        return MemoryRecord(
            id=memory_id, user_id=user_id, slot=slot,
            memory_type=memory_type, content=content,
            confidence=confidence, sensitivity=sensitivity, ttl=ttl,
            created_at=now, updated_at=now,
        )

    def get_memory_record(self, memory_id: str) -> MemoryRecord | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        if not row:
            return None
        return MemoryRecord(
            id=row["id"], user_id=row["user_id"],
            slot=_coerce_slot_v2(row["slot"]), memory_type=row["memory_type"],
            content=row["content"], confidence=float(row["confidence"]),
            sensitivity=float(row["sensitivity"]), ttl=row["ttl"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    # ── 记忆热度（检索命中→热度累加，读取时按 last_hit_at 指数衰减）────────────

    def record_memory_hits(self, memory_ids: list[str]) -> None:
        """一批真实检索命中（通常是 Rank() 一次返回的 top-k）：每条 heat
        累加 +1（不是取 max），刷新 last_hit_at。只对 ``memories`` 表里已经
        有记录的 id 生效——静默跳过没有记录的（比如还没跑过
        ingest_annotated_fact 的 id），不因为找不到行报错，调用方不需要先
        判断这些记忆是否已被认知图记录过。一次连接里 executemany，不是
        每条命中各开一次连接。"""
        if not memory_ids:
            return
        now = _utc_iso()
        with self._conn() as c:
            c.executemany(
                "UPDATE memories SET heat = heat + 1.0, last_hit_at = ? WHERE id = ?",
                [(now, mid) for mid in memory_ids],
            )

    def get_memory_heat(self, memory_id: str) -> float | None:
        """返回按 last_hit_at（没命中过就用 created_at）指数衰减后的热度；
        没有这条记忆返回 None（不是 0——0 是"存在但很冷"，None 是"根本
        没有这条记忆的记账"，两者语义不同）。"""
        with self._conn() as c:
            row = c.execute(
                "SELECT heat, last_hit_at, created_at FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
        if not row:
            return None
        anchor = row["last_hit_at"] or row["created_at"]
        return _decayed_weight(float(row["heat"]), anchor, halflife_days=MEMORY_HEAT_DECAY_HALFLIFE_DAYS)

    def list_archivable_memories(
        self, user_id: str, *, min_age_days: float = 30.0, heat_threshold: float = ARCHIVE_HEAT_THRESHOLD,
    ) -> list[str]:
        """返回该用户衰减后热度低于阈值、且存在时间够久的记忆 id 列表——
        "够久"是为了避免刚 ingest、还没被检索过一次的新记忆被误判成冷门
        （新记忆的 heat 默认 1.0，衰减锚点是 created_at，如果不设最小年龄，
        一条几分钟前才写入的记忆会被当成"从没命中过、已经很冷"直接归档）。
        只负责判定，不负责真的执行归档（执行需要调用方拿着这批 id 去调
        Mem0BackendStore.archive_memory，见其文档字符串）。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, heat, last_hit_at, created_at FROM memories WHERE user_id=?",
                (user_id,),
            ).fetchall()
        now = datetime.now(timezone.utc)
        archivable: list[str] = []
        for r in rows:
            anchor = r["last_hit_at"] or r["created_at"]
            decayed = _decayed_weight(float(r["heat"]), anchor, halflife_days=MEMORY_HEAT_DECAY_HALFLIFE_DAYS)
            if decayed >= heat_threshold:
                continue
            try:
                created = datetime.fromisoformat(r["created_at"])
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = (now - created).total_seconds() / 86400.0
            except (TypeError, ValueError):
                continue
            if age_days >= min_age_days:
                archivable.append(r["id"])
        return archivable

    # ── Slot Profile ──────────────────────────────────────────────────────────

    def refresh_slot_profile(
        self, user_id: str, slot: SlotV2, *, summary: str = ""
    ) -> SlotProfile:
        now = _utc_iso()
        with self._conn() as c:
            ec = c.execute(
                "SELECT COUNT(*) FROM entities WHERE user_id=? AND slot=?",
                (user_id, slot.value),
            ).fetchone()[0]
            mc = c.execute(
                """SELECT COUNT(DISTINCT eml.memory_id)
                   FROM entity_memory_links eml
                   JOIN entities e ON eml.entity_id=e.id
                   WHERE e.user_id=? AND e.slot=?""",
                (user_id, slot.value),
            ).fetchone()[0]
            # collect entity_ids for this slot
            eid_rows = c.execute(
                "SELECT id FROM entities WHERE user_id=? AND slot=?",
                (user_id, slot.value),
            ).fetchall()
            entity_ids = [r["id"] for r in eid_rows]

            c.execute(
                """INSERT INTO slot_profiles
                   (user_id, slot, summary, entity_ids, entity_count, memory_count, last_updated)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(user_id,slot) DO UPDATE SET
                     summary=CASE WHEN excluded.summary!='' THEN excluded.summary ELSE summary END,
                     entity_ids=excluded.entity_ids,
                     entity_count=excluded.entity_count,
                     memory_count=excluded.memory_count,
                     last_updated=excluded.last_updated""",
                (user_id, slot.value, summary,
                 json.dumps(entity_ids, ensure_ascii=False), ec, mc, now),
            )
        return SlotProfile(
            user_id=user_id, slot=slot, summary=summary,
            entity_ids=entity_ids, entity_count=ec,
            memory_count=mc, last_updated=now,
        )

    def slot_profiles(self, user_id: str) -> list[SlotProfile]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM slot_profiles WHERE user_id=?", (user_id,)
            ).fetchall()
        result = []
        for r in rows:
            entity_ids = json.loads(r["entity_ids"] or "[]") if "entity_ids" in r.keys() else []
            summary = r["summary"] if "summary" in r.keys() else ""
            result.append(SlotProfile(
                user_id=r["user_id"], slot=_coerce_slot_v2(r["slot"]),
                summary=summary, entity_ids=entity_ids,
                entity_count=r["entity_count"], memory_count=r["memory_count"],
                last_updated=r["last_updated"],
            ))
        return result

    # ── Affective Edge (right brain stub) ─────────────────────────────────────

    def upsert_affective_edge(
        self,
        user_id: str,
        from_entity_id: str,
        *,
        to_entity_id: str | None = None,
        trigger_frame: str | None = None,
        emotion: str | None = None,
        appraisal: str | None = None,
        response_policy: str | None = None,
        confidence: float = 1.0,
        evidence_memory_ids: list[str] | None = None,
    ) -> AffectiveEdge:
        now = _utc_iso()
        eid = _new_id()
        eids = evidence_memory_ids or []
        with self._conn() as c:
            c.execute(
                """INSERT INTO affective_edges
                   (id,user_id,from_entity_id,to_entity_id,trigger_frame,
                    emotion,appraisal,response_policy,confidence,evidence_memory_ids,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (eid, user_id, from_entity_id, to_entity_id,
                 trigger_frame, emotion, appraisal, response_policy,
                 confidence, json.dumps(eids), now),
            )
        return AffectiveEdge(
            id=eid, user_id=user_id, from_entity_id=from_entity_id,
            to_entity_id=to_entity_id, trigger_frame=trigger_frame,
            emotion=emotion, appraisal=appraisal,
            response_policy=response_policy, confidence=confidence,
            evidence_memory_ids=eids, created_at=now,
        )

    def affective_edges_for_entity(self, entity_id: str, user_id: str) -> list[AffectiveEdge]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM affective_edges WHERE user_id=? AND from_entity_id=?",
                (user_id, entity_id),
            ).fetchall()
        result = []
        for r in rows:
            conf = float(r["confidence"]) if "confidence" in r.keys() else 1.0
            eids = json.loads(r["evidence_memory_ids"] or "[]") if "evidence_memory_ids" in r.keys() else []
            result.append(AffectiveEdge(
                id=r["id"], user_id=r["user_id"],
                from_entity_id=r["from_entity_id"],
                to_entity_id=r["to_entity_id"],
                trigger_frame=r["trigger_frame"],
                emotion=r["emotion"], appraisal=r["appraisal"],
                response_policy=r["response_policy"],
                confidence=conf, evidence_memory_ids=eids,
                created_at=r["created_at"],
            ))
        return result

    # ── Core ingest ───────────────────────────────────────────────────────────

    def ingest_annotated_fact(
        self,
        user_id: str,
        annotated: "AnnotatedFact",
        memory_ids: list[str],
    ) -> list[Entity]:
        """把一条 AnnotatedFact 的实体/关系/链接全部落库，返回 Entity 列表。"""
        from .types import EntityAnnotation
        entities: list[Entity] = []
        name_to_entity: dict[str, Entity] = {}
        name_to_role: dict[str, str] = {}   # 记录每个实体在此 fact 里的角色

        for ann in annotated.entities:
            if not ann.name.strip():
                continue
            ent = self.upsert_entity(
                user_id, ann.name, ann.entity_type,
                slot=annotated.slot,
                description=annotated.fact_text,
                confidence=annotated.confidence,
            )
            entities.append(ent)
            name_to_entity[normalize_name(ann.name)] = ent
            name_to_role[normalize_name(ann.name)] = _normalize_role(ann.role)

        # 写入记忆元数据 wrapper
        for mid in memory_ids:
            self.upsert_memory_record(
                user_id, mid, annotated.slot, annotated.fact_text,
                confidence=annotated.confidence,
            )

        # 链接实体 ↔ 记忆，带 role 和 relation_hint
        for mid in memory_ids:
            for norm_name, ent in name_to_entity.items():
                role_str = name_to_role.get(norm_name, EntityMemoryRole.CONTEXT.value)
                # relation_hint: 用 "entity_name --slot--> memory" 简短描述
                hint = f"{ent.name} ({ent.entity_type.value})"
                from .types import EntityMemoryRole as _R
                try:
                    role_enum = _R(role_str)
                except ValueError:
                    role_enum = _R.CONTEXT
                self.link_memory(mid, ent.id, user_id,
                                 role=role_enum, relation_hint=hint)

        # 写入实体间关系边
        for rel in annotated.relations:
            from_ent = name_to_entity.get(normalize_name(rel.from_name))
            to_ent = name_to_entity.get(normalize_name(rel.to_name))
            if from_ent and to_ent and from_ent.id != to_ent.id:
                self.upsert_edge(
                    user_id, from_ent.id, to_ent.id,
                    rel.relation_type,
                    role_label=rel.role_label,
                    confidence=rel.confidence,
                    evidence_memory_id=memory_ids[0] if memory_ids else None,
                )

        # 刷新涉及槽位的 profile（只更新计数，不生成 LLM 摘要）
        for slot in {e.slot for e in entities}:
            self.refresh_slot_profile(user_id, slot)

        return entities

    # ── Graph context ─────────────────────────────────────────────────────────

    def entity_context(
        self, entity_id: str, user_id: str, *, depth: int = 1
    ) -> dict[str, Any]:
        """返回一个节点的完整上下文：节点资料 + 关联边 + 关联 memory_ids + 情感边。"""
        entity = self.get_entity(entity_id)
        if not entity:
            return {}
        edges = self.edges_for_entity(entity_id, user_id)
        memory_ids = self.memory_ids_for_entities([entity_id])
        affective = self.affective_edges_for_entity(entity_id, user_id)

        neighbor_ids: set[str] = set()
        for e in edges:
            neighbor_ids.add(
                e.from_entity_id if e.to_entity_id == entity_id else e.to_entity_id
            )

        neighbors: list[dict] = []
        if depth > 0 and neighbor_ids:
            for nid in neighbor_ids:
                ne = self.get_entity(nid)
                if ne:
                    neighbors.append({
                        "id": ne.id, "name": ne.name,
                        "entity_type": ne.entity_type.value,
                        "slot": ne.slot.value,
                    })

        return {
            "entity": {
                "id": entity.id, "name": entity.name,
                "entity_type": entity.entity_type.value,
                "slot": entity.slot.value,
                "description": entity.description,
                "confidence": entity.confidence,
                "importance": entity.importance,
                "aliases": entity.aliases,
                "properties": entity.properties,
                "created_at": entity.created_at, "updated_at": entity.updated_at,
            },
            "edges": [
                {
                    "from": e.from_entity_id, "to": e.to_entity_id,
                    "relation": e.relation_type, "role": e.role_label,
                    "confidence": e.confidence, "weight": e.weight,
                    "edge_type": e.edge_type, "status": e.status,
                }
                for e in edges
            ],
            "memory_ids": memory_ids,
            "neighbors": neighbors,
            "affective_edges": [
                {
                    "trigger": a.trigger_frame, "emotion": a.emotion,
                    "appraisal": a.appraisal, "policy": a.response_policy,
                    "confidence": a.confidence,
                }
                for a in affective
            ],
        }

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def delete_user(self, user_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM entity_memory_links WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM entity_edges WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM entities WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM slot_profiles WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM affective_edges WHERE user_id=?", (user_id,))
