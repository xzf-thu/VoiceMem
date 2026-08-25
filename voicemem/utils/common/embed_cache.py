"""embedding 结果的进程内缓存。

一次 ingest 实测发了 15 次 embedding，其中同一句 fact 发了 3 次（字符串一模一样）、
同样 3 个实体发了 2 轮（左脑一次、orchestrator 又一次）、7 条 slot 描述每轮重算
（slot 定义是静态的，不随输入变）。去重之后剩 4 次左右。

按 (model, text) 做 key。同一段文本的向量是确定的，缓存不改变任何语义。
"""
from __future__ import annotations

import threading

_MAX = 4096                      # 够一次会话用；超了就整体清掉，不做 LRU 记账
_lock = threading.Lock()
_cache: dict[tuple[str, str], list[float]] = {}


def get(model: str, text: str) -> list[float] | None:
    with _lock:
        return _cache.get((model, text))


def put(model: str, text: str, vec: list[float]) -> None:
    if not vec:
        return
    with _lock:
        if len(_cache) >= _MAX:
            _cache.clear()
        _cache[(model, text)] = vec


def resolve(model: str, texts: list[str], compute) -> list[list[float]]:
    """命中的直接取，没命中的**合成一次批量调用**交给 ``compute(list) -> list``。

    合批同样重要：OpenAI 的 embeddings 接口本来就吃数组，发 5 条和发 1 条几乎同价，
    而现在的调用方是一条一次——15 次里只有 slot 描述那次是批量的。
    """
    out: list[list[float] | None] = [get(model, t) for t in texts]
    missing = [i for i, v in enumerate(out) if v is None]
    if missing:
        fresh = compute([texts[i] for i in missing])
        for i, vec in zip(missing, fresh):
            out[i] = vec
            put(model, texts[i], vec)
    return [v or [] for v in out]
