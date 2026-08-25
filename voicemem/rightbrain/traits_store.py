"""右脑 v2：一个节点 = 一条关于这个人的判断，证据挂在下面。

    rb_traits                          节点
      claim      压力大时想被安抚        写入时就定好，5-15 字
      slot       五选一（情绪/应对方式/表达风格/思维模式/喜好与厌恶）
      embedding  claim 的向量 —— 右脑终于能按语义检索
    rb_evidence                        证据
      quote      你先别给方案，让我说完   用户原话
      emotion    烦躁                  情绪是证据的属性，不是节点
      cause_id   ← 左脑那条 fact

**为什么要换掉 slot → entity → heartnote 那套**：``entity`` 那一层身兼三职——
有时是关于人的判断（"讨厌被打断"），有时是话题（"手冲咖啡""NUS"），有时是情绪词
（"焦虑"）。三种东西混在一层，后果是实测到的这些：

  · 所有悲伤的事堆进「悲伤」一个节点（61 条），所有话都链到「佳琪」（52 条）
  · 标题格式无法统一——三类东西本来就没有统一写法
  · 描述靠事后的巩固批处理补，跑得少，于是大量节点没描述

这里只放**关于人的判断**。话题实体归左脑的认知图；情绪降级成证据的属性；
助手的自我复盘（response_experience）不进这张表。
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: 两条 claim 像到这个程度就算同一条，证据并进去。
#: 0.95 是实测出来的分界：本地 E5 对中文短语的基线相似度就有 0.9+，
#: 「喜欢手冲咖啡」↔「偏好手冲咖啡」是 0.964（该合并），
#: 「讨厌吃饭吧唧嘴」↔「讨厌被打断」是 0.934（不该合并）。
MERGE_THRESHOLD = 0.95

#: 五个 slot。去掉了原来的「人物地点态度」——它存的是话题（手冲咖啡/NUS/佳琪），
#: 本来就该在左脑，也正是「佳琪 ×52」那个大杂烩的来源。
SLOTS = ("情绪", "应对方式", "表达风格", "思维模式", "喜好与厌恶")

#: 五个 slot → UI 的三类
SLOT_TO_CLUSTER = {
    "情绪":       "emotion",
    "应对方式":    "personality",
    "表达风格":    "personality",
    "思维模式":    "personality",
    "喜好与厌恶":  "preference",
}


@dataclass
class Evidence:
    quote: str
    emotion: str = ""
    cause: str = ""            # 左脑 fact 的原文（渲染时当"为什么"）
    cause_id: str = ""
    at: str = ""


@dataclass
class Trait:
    id: str
    slot: str
    claim: str
    confidence: float = 0.9
    evidence: list[Evidence] = field(default_factory=list)
    updated_at: str = ""

    @property
    def cluster(self) -> str:
        return SLOT_TO_CLUSTER.get(self.slot, "personality")


#: claim 前面常见的主语。节点标题是「讨厌被打断」而不是「用户讨厌被打断」——
#: 整张图讲的都是同一个人，每个标题都顶着「用户」两个字纯属噪音。
_SUBJECTS = ("用户可能", "用户似乎", "用户倾向于", "用户", "他/她", "对方", "我")


def normalize_claim(claim: str) -> str:
    """把 claim 收拾成节点标题该有的样子：无主语、无句号、一句短话。

    几条写入路径的产出质量不一样——合并抽取那条有明确格式要求，助手复盘那条
    （response_experience 的 user_trait）没有，实测吐出过
    「用户喜欢分享自己的经历，可能不太关注助手的问候。」这种带主语的整句。
    与其在每条路径上各写一遍要求，不如在入口统一收口。
    """
    c = (claim or "").strip().strip("「」\"'").rstrip("。.！!；;，,")
    for s in _SUBJECTS:
        if c.startswith(s) and len(c) > len(s) + 2:
            c = c[len(s):].lstrip("，,、 ")
            break
    # 「A，可能B」这种双句只留前半句——后半句几乎都是模型加的推测
    if "，" in c and len(c) > 15:
        head = c.split("，")[0].strip()
        if len(head) >= 5:
            c = head
    return c.strip()


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TraitStore:
    """rb_traits / rb_evidence 两张表，跟其余结构化存储共用 space 那个 sqlite。"""

    def __init__(self, db_path, embed) -> None:
        self._db = str(db_path)
        self._embed = embed                 # fn(text) -> list[float]
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS rb_traits (
                id             TEXT PRIMARY KEY,
                user_id        TEXT NOT NULL,
                slot           TEXT NOT NULL,
                claim          TEXT NOT NULL,
                embedding      BLOB,
                confidence     REAL NOT NULL DEFAULT 0.9,
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS rb_evidence (
                id         TEXT PRIMARY KEY,
                trait_id   TEXT NOT NULL,
                user_id    TEXT NOT NULL,
                quote      TEXT NOT NULL,
                emotion    TEXT NOT NULL DEFAULT '',
                cause      TEXT NOT NULL DEFAULT '',
                cause_id   TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_ev_trait ON rb_evidence(trait_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_tr_user ON rb_traits(user_id, slot)")

    def _conn(self):
        c = sqlite3.connect(self._db, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    # ── 写 ────────────────────────────────────────────────────────────────────

    def add(self, user_id: str, slot: str, claim: str, ev: Evidence) -> str:
        """加一条判断 + 它的证据。已经有意思相同的 claim 就并进去，不新建节点。"""
        claim = normalize_claim(claim)
        if not claim or slot not in SLOTS:
            return ""

        vec = self._vec(claim)
        tid = self._find_similar(user_id, slot, vec)
        now = _now()
        with self._conn() as c:
            if tid is None:
                tid = uuid.uuid4().hex
                c.execute("INSERT INTO rb_traits "
                          "(id,user_id,slot,claim,embedding,confidence,created_at,updated_at) "
                          "VALUES (?,?,?,?,?,?,?,?)",
                          (tid, user_id, slot, claim,
                           vec.astype(np.float32).tobytes() if vec is not None else None,
                           0.9, now, now))
            else:
                c.execute("UPDATE rb_traits SET updated_at=? WHERE id=?", (now, tid))
            # 每个字段都过一遍 str()：证据常常来自旧数据或 LLM 输出，
            # 缺字段时是 None，而这几列都是 NOT NULL，直接插会整轮写入失败。
            c.execute("INSERT INTO rb_evidence "
                      "(id,trait_id,user_id,quote,emotion,cause,cause_id,created_at) "
                      "VALUES (?,?,?,?,?,?,?,?)",
                      (uuid.uuid4().hex, tid, user_id, str(ev.quote or ""),
                       str(ev.emotion or ""), str(ev.cause or ""),
                       str(ev.cause_id or ""), str(ev.at or now)))
        return tid

    def _vec(self, text: str):
        try:
            v = np.asarray(self._embed(text), dtype=np.float32)
            n = float(np.linalg.norm(v))
            return v / n if n else v
        except Exception:
            return None

    def _find_similar(self, user_id: str, slot: str, vec) -> str | None:
        if vec is None:
            return None
        with self._conn() as c:
            rows = c.execute("SELECT id, embedding FROM rb_traits "
                             "WHERE user_id=? AND slot=? AND embedding IS NOT NULL",
                             (user_id, slot)).fetchall()
        best, best_sim = None, 0.0
        for r in rows:
            v = np.frombuffer(r["embedding"], dtype=np.float32)
            if v.shape != vec.shape:
                continue
            sim = float(v @ vec)
            if sim > best_sim:
                best, best_sim = r["id"], sim
        return best if best_sim >= MERGE_THRESHOLD else None

    # ── 读 ────────────────────────────────────────────────────────────────────

    def all(self, user_id: str, *, per_slot: int = 8) -> list[Trait]:
        """给脑图用：每个 slot 取证据最多的前几条 + 最近新增的几条。"""
        out: list[Trait] = []
        with self._conn() as c:
            for slot in SLOTS:
                rows = c.execute(
                    """SELECT t.*, COUNT(e.id) n FROM rb_traits t
                       LEFT JOIN rb_evidence e ON e.trait_id = t.id
                       WHERE t.user_id=? AND t.slot=? GROUP BY t.id
                       HAVING n > 0""", (user_id, slot)).fetchall()
                by_ev = sorted(rows, key=lambda r: -r["n"])
                by_new = sorted(rows, key=lambda r: r["updated_at"], reverse=True)
                fresh = max(1, per_slot // 2)
                picked, seen = [], set()
                # 一半给最近新增的（刚说的那句要能立刻看见），一半给证据最多的
                for r in by_new[:fresh] + by_ev:
                    if r["id"] in seen:
                        continue
                    seen.add(r["id"])
                    picked.append(r)
                    if len(picked) >= per_slot:
                        break
                for r in picked:
                    out.append(self._to_trait(c, r))
        return out

    def search(self, user_id: str, query: str, *, top_k: int = 5) -> list[Trait]:
        """按语义查判断。

        原来的右脑只能按情绪锚点匹配，所以每轮返回的总是同样那几条静态画像。
        claim 有了向量之后这里才是真正的检索。
        """
        return [t for t, _ in self.search_scored(user_id, query, top_k=top_k)]

    def search_scored(self, user_id: str, query: str, *, top_k: int = 5
                      ) -> list[tuple[Trait, float]]:
        """同 :meth:`search`，但带上余弦相似度。

        检索侧要用它当 priority——判断跟这句话有多相关，直接决定它该不该占
        top-N 的位置，固定 priority 会让不相关的判断挤掉真正相关的。
        """
        q = self._vec(query)
        if q is None:
            return []
        with self._conn() as c:
            rows = c.execute("SELECT * FROM rb_traits WHERE user_id=? AND embedding IS NOT NULL",
                             (user_id,)).fetchall()
            scored = []
            for r in rows:
                v = np.frombuffer(r["embedding"], dtype=np.float32)
                if v.shape != q.shape:
                    continue
                scored.append((float(v @ q), r))
            scored.sort(key=lambda t: -t[0])
            return [(self._to_trait(c, r), s) for s, r in scored[:top_k]]

    def _to_trait(self, c, r) -> Trait:
        evs = c.execute("SELECT * FROM rb_evidence WHERE trait_id=? ORDER BY created_at DESC",
                        (r["id"],)).fetchall()
        return Trait(
            id=r["id"], slot=r["slot"], claim=r["claim"],
            confidence=r["confidence"], updated_at=r["updated_at"],
            evidence=[Evidence(quote=e["quote"], emotion=e["emotion"],
                               cause=e["cause"], cause_id=e["cause_id"],
                               at=e["created_at"]) for e in evs],
        )

    def counts(self, user_id: str) -> tuple[int, int]:
        with self._conn() as c:
            t = c.execute("SELECT COUNT(*) FROM rb_traits WHERE user_id=?", (user_id,)).fetchone()[0]
            e = c.execute("SELECT COUNT(*) FROM rb_evidence WHERE user_id=?", (user_id,)).fetchone()[0]
        return t, e
