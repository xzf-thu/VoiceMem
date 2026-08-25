"""向量工具：cosine_sim，供 core.py 的 slot 标签语义归一化等场景复用。

旧版 sub_slot 机制（分裂 run_split_check / 涌现 emerge_slots 创建新 sub_slot，
以及查询路由 route_sub_slot / 写入分配 assign_to_sub_slot）已完全删除——新 slot
的创建和路由现在统一由 SubgraphManager（entity 共现子图判定）负责。
"""

from __future__ import annotations

import math


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_sim(a: list[float], b: list[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)
