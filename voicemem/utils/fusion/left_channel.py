"""左脑通道：Mem0 / 本地 repository 语义检索。"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from voicemem.utils.fusion.config import FusionConfig
from voicemem.utils.fusion.types import LeftMemoryHit


@runtime_checkable
class LeftBrainSearchClient(Protocol):
    def search(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int = 5,
        threshold: float | None = None,
        rerank: bool | None = None,
        filters: dict[str, Any] | None = None,
    ) -> Any:
        ...


@runtime_checkable
class LeftBrainGraphSearchClient(LeftBrainSearchClient, Protocol):
    def search_with_graph(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int = 5,
        threshold: float | None = None,
        relation_depth: int = 1,
    ) -> Any:
        ...


def _normalize_mem0_results(raw: Any) -> list[LeftMemoryHit]:
    hits: list[LeftMemoryHit] = []
    if raw is None:
        return hits
    rows = raw
    if isinstance(raw, dict):
        rows = raw.get("results") or raw.get("memories") or []
    if not isinstance(rows, list):
        return hits
    for item in rows:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or item.get("memory_id") or "").strip()
        text = str(item.get("memory") or item.get("text") or "").strip()
        if not text:
            continue
        score = float(item.get("score") or item.get("similarity") or 0.0)
        md = item.get("metadata")
        meta = md if isinstance(md, dict) else {}
        if not mid:
            mid = text[:32]
        hits.append(LeftMemoryHit(memory_id=mid, text=text, score=score, metadata=meta))
    return hits


def _normalize_local_hits(raw: Any) -> list[LeftMemoryHit]:
    hits: list[LeftMemoryHit] = []
    if not raw:
        return hits
    for item in raw:
        if hasattr(item, "memory_id") and hasattr(item, "text"):
            hits.append(
                LeftMemoryHit(
                    memory_id=str(item.memory_id),
                    text=str(item.text),
                    score=float(getattr(item, "score", 0.0)),
                    metadata=dict(getattr(item, "metadata", {}) or {}),
                )
            )
    return hits


def search_left_channel(
    client: LeftBrainSearchClient,
    *,
    asr_text: str,
    user_id: str,
    config: FusionConfig | None = None,
) -> list[LeftMemoryHit]:
    """左脑检索：query=ASR，filters 含 user_id。"""
    cfg = config or FusionConfig()
    query = (asr_text or "").strip()
    if not query:
        return []

    try:
        raw = client.search(
            query,
            user_id=user_id,
            top_k=cfg.mem0_top_k,
            threshold=cfg.mem0_threshold,
            rerank=cfg.mem0_rerank,
            filters={"user_id": user_id},
        )
    except TypeError:
        raw = client.search(
            query,
            user_id=user_id,
            top_k=cfg.mem0_top_k,
            threshold=cfg.mem0_threshold,
        )

    hits = _normalize_mem0_results(raw)
    if not hits:
        hits = _normalize_local_hits(raw)
    return hits


def _hit_timestamp(h: LeftMemoryHit) -> str:
    """挑一个能代表"这条记忆最后确认于何时"的时间戳。

    从未被冲突消解更新过的记忆，时间戳在写入时存的 ``metadata["time_start"]``；
    被 ``ConflictResolver`` UPDATE 过的记忆，时间戳被覆盖进
    ``metadata["created_at"]``（见 ``local_memory_store.update_memory``），
    所以两个 key 都要试，``created_at`` 优先——它代表更晚一次的确认。
    """
    md = h.metadata or {}
    return str(md.get("created_at") or md.get("time_start") or "").strip()


def format_left_memory_block(hits: list[LeftMemoryHit]) -> str:
    if not hits:
        return "(No relevant left-brain memories)"
    lines = []
    for i, h in enumerate(hits, 1):
        ts = _hit_timestamp(h)
        ts_part = f" (as of {ts})" if ts else ""
        lines.append(f"{i}. [{h.memory_id}]{ts_part} {h.text}")
    return "\n".join(lines)


def _normalize_graph_search_hits(raw: Any) -> tuple[list[LeftMemoryHit], str]:
    """从 ``search_with_graph`` 结果提取向量 hits 与图附录文本。"""
    hits: list[LeftMemoryHit] = []
    appendix_parts: list[str] = []
    if not raw:
        return hits, ""

    rows = raw if isinstance(raw, list) else []
    seen_graph_fingerprints: set[tuple[tuple[str, str, str], ...]] = set()
    for item in rows:
        memory = getattr(item, "memory", None)
        graph = getattr(item, "graph", None)
        if memory is not None and hasattr(memory, "text"):
            hits.append(
                LeftMemoryHit(
                    memory_id=str(memory.memory_id),
                    text=str(memory.text),
                    score=float(getattr(memory, "score", 0.0)),
                    metadata=dict(getattr(memory, "metadata", {}) or {}),
                )
            )
        if graph is None:
            continue
        relations = getattr(graph, "relations", None) or []
        mid = str(getattr(graph, "memory_id", "") or "")
        if not relations:
            continue
        rel_fp = tuple(
            sorted(
                (
                    str(getattr(rel, "source_name", "")),
                    str(getattr(rel, "relation_type", "")),
                    str(getattr(rel, "target_name", "")),
                )
                for rel in relations[:16]
            )
        )
        if rel_fp and rel_fp in seen_graph_fingerprints:
            continue
        if rel_fp:
            seen_graph_fingerprints.add(rel_fp)
        block_lines = [f"Memory [{mid}] linked relations:"] if mid else ["Linked relations:"]
        for rel in relations[:8]:
            block_lines.append(
                f"  - {getattr(rel, 'source_name', '')} -[{getattr(rel, 'relation_type', '')}]-> "
                f"{getattr(rel, 'target_name', '')}"
                + (f" ({rel.description})" if getattr(rel, "description", "") else "")
            )
        appendix_parts.append("\n".join(block_lines))

    appendix = "\n\n".join(appendix_parts)
    if appendix:
        appendix = "[Semantic graph appendix]\n" + appendix
    return hits, appendix


def search_left_for_reply(
    client: LeftBrainSearchClient,
    *,
    asr_text: str,
    user_id: str,
    config: FusionConfig | None = None,
    use_graph: bool = True,
) -> tuple[list[LeftMemoryHit], str]:
    """左脑检索；若 client 支持且 ``use_graph`` 则附带语义图附录。"""
    cfg = config or FusionConfig()
    query = (asr_text or "").strip()
    if not query:
        return [], ""

    if use_graph and cfg.left_use_graph and isinstance(client, LeftBrainGraphSearchClient):
        try:
            raw = client.search_with_graph(
                query,
                user_id=user_id,
                top_k=cfg.mem0_top_k,
                threshold=cfg.mem0_threshold,
            )
        except TypeError:
            raw = client.search_with_graph(
                query,
                user_id=user_id,
                top_k=cfg.mem0_top_k,
            )
        hits, appendix = _normalize_graph_search_hits(raw)
        if hits:
            return hits, appendix

    return search_left_channel(client, asr_text=asr_text, user_id=user_id, config=cfg), ""
