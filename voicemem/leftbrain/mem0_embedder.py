"""把 mem0 的 embedding provider 适配成 VoiceMem 的 embedder。

mem0 自带十来个 provider（ollama / huggingface / gemini / bedrock / azure /
vertexai / together / lmstudio / fastembed / langchain / openai），而 mem0 本来
就是 VoiceMem 的依赖——没必要各写一遍，包一层接口差异就都能用。

两边的接口差在三处：
  · mem0 是 ``embed(text, memory_action)`` 一次一条；VoiceMem 要 ``embed_texts(list)``
    批量（slot 锚点那种一次七条的场景，逐条发等于七次往返）。
  · mem0 用 ``memory_action`` 区分入库/检索；VoiceMem 用两个方法名区分。
    对称模型（大多数）两者同向量，非对称的（E5 那类要 query:/passage: 前缀）
    靠这个参数才对得上，所以要老实传下去。
  · 维度 VoiceMem 要能问出来（换 embedder 之后靠它判断老向量作废没有），
    mem0 不一定给，问不到就现算一条探针。
"""
from __future__ import annotations

from typing import Any


class Mem0Embedder:
    """mem0 provider → VoiceMem embedder。"""

    def __init__(self, provider: str, config: dict[str, Any] | None = None) -> None:
        from mem0.utils.factory import EmbedderFactory
        self._provider = provider
        self._inner = EmbedderFactory.create(provider, dict(config or {}), None)
        self._dims: int | None = None

    # ── VoiceMem 侧的接口 ────────────────────────────────────────────────────
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """入库侧：一批文本 → 一批向量。"""
        return [self._one(t, "add") for t in texts]

    def embed_query_text(self, text: str) -> list[float]:
        """检索侧。非对称模型靠 memory_action 走 query 那一路。"""
        return self._one(text, "search")

    @property
    def dimensions(self) -> int:
        if self._dims is None:
            cfg = getattr(self._inner, "config", None)
            self._dims = int(getattr(cfg, "embedding_dims", 0) or 0) or len(
                self._one("dimension probe", "search"))
        return self._dims

    @property
    def model_name(self) -> str:
        cfg = getattr(self._inner, "config", None)
        return str(getattr(cfg, "model", "") or self._provider)

    # ── 内部 ─────────────────────────────────────────────────────────────────
    def _one(self, text: str, action: str) -> list[float]:
        try:
            return list(self._inner.embed(text, action))
        except TypeError:      # 少数 provider 的 embed() 不收 memory_action
            return list(self._inner.embed(text))


def mem0_providers() -> set[str]:
    """mem0 认得的 provider 名字。装不上 mem0 就返回空集——这条路不可用而已，
    不该让 import voicemem.config 失败。"""
    try:
        from mem0.utils.factory import EmbedderFactory
        return set(getattr(EmbedderFactory, "provider_to_class", {}) or {})
    except Exception:
        return set()
