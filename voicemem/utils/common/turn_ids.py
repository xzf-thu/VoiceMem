"""可读 turn_id 生成（左右脑对齐用）。"""

from __future__ import annotations

import re


def sanitize_session_id(session_id: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", (session_id or "").strip())
    return s or "session"


def format_turn_id(*, session_id: str, turn_index: int) -> str:
    """生成可读 turn_id，例如 ``integrated_memory_chat_turn0002``。

    ``turn_index`` 为会话内从 1 开始的轮次编号。
    """
    return f"{sanitize_session_id(session_id)}_turn{int(turn_index):04d}"


def _memory_suffix(*, extractor_local_id: str | None, seq: int) -> str:
    if extractor_local_id is not None and str(extractor_local_id).strip():
        raw = str(extractor_local_id).strip()
        if raw.isdigit():
            return f"mem{int(raw):02d}"
        return f"mem_{sanitize_session_id(raw)}"
    return f"mem{int(seq):02d}"


def format_memory_id(
    *,
    turn_id: str | None = None,
    session_id: str | None = None,
    turn_index: int | None = None,
    extractor_local_id: str | None = None,
    seq: int = 0,
) -> str:
    """生成可读语义记忆 id，例如 ``integrated_memory_chat_turn0002_mem00``。"""
    base = (turn_id or "").strip()
    if not base and session_id and turn_index is not None:
        base = format_turn_id(session_id=session_id, turn_index=turn_index)
    if not base:
        base = "memory"
    return f"{base}_{_memory_suffix(extractor_local_id=extractor_local_id, seq=seq)}"
