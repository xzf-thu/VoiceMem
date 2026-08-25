"""轻量级 token 用量记录，供评测脚本核算真实花费用。

通过环境变量配置：
  COST_LOG_PATH — 写到哪个 jsonl 文件（不设就不记录，生产环境零开销）
  COST_TAG      — 这次跑批的标签（比如 "locomo-final"），方便汇总时分组
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_COST_LOG_PATH = os.environ.get("COST_LOG_PATH")
_COST_TAG = os.environ.get("COST_TAG", "?")


def log_usage(step: str, model: str, usage) -> None:
    if not _COST_LOG_PATH or usage is None:
        return
    line = json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "tag": _COST_TAG,
        "step": step,
        "model": model,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
    }, ensure_ascii=False)
    with open(_COST_LOG_PATH, "a") as f:
        f.write(line + "\n")
