"""Small utilities shared by the graph-store layers (``leftbrain/cognitive_graph/store.py``,
``leftbrain/slot_split/graph_entity_store.py``, ``leftbrain/memory_repository_v2.py``,
``rightbrain/graph_store.py``). These four modules each independently redefined the
same ``_utc_iso``/``_new_id``/``_cosine`` helpers -- pulled out here since they were
byte-for-byte identical, not just similar.
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors. 0.0 if either is a zero vector.

    长度不等返回 0.0。原来直接 ``zip`` 会**静默截断**到较短的那个——换 embedder
    之后库里新旧维度并存（384 / 1536），比出来的是前 384 维的无意义数值，
    不崩也不报错，实体去重就照着这个数乱合并。宁可判成"不相似"。
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0
