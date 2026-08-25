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
    """Cosine similarity between two equal-length vectors. 0.0 if either is a zero vector."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0
