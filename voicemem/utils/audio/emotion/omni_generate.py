"""Qwen Omni 文本生成辅助：收敛 generate 参数并屏蔽已知无害告警。"""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from typing import Any


def build_omni_text_gen_kwargs(
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    tokenizer: Any,
) -> dict[str, Any]:
    """文本 JSON 输出用；temperature=0 时不传采样参数，避免 transformers UserWarning。"""
    do_sample = float(temperature) > 0.0
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    return kwargs


class _OmniInferenceLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "System prompt modified" in msg:
            return False
        if "audio output may not work" in msg:
            return False
        return True


@contextmanager
def suppress_omni_text_inference_noise():
    """屏蔽自定义 system prompt / 贪心解码时的已知 Omni+transformers 噪音。"""
    log_filter = _OmniInferenceLogFilter()
    root = logging.getLogger()
    root.addFilter(log_filter)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*do_sample.*", category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message=".*System prompt modified.*",
                category=UserWarning,
            )
            yield
    finally:
        root.removeFilter(log_filter)
