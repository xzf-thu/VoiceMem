"""LeftBrainMemoryRepositoryV2 — semantic-slot V2 + multi-label memory_tags retrieval.

Extends :class:`LeftBrainMemoryRepository` with:

* **Embedding-based slot matching** — cosine similarity between the query
  embedding and per-slot description embeddings replaces brittle keyword
  matching.
* **Multi-label tags** — each memory can belong to several V2 slots
  (stored in the ``memory_tags`` table via :class:`CognitiveGraphStoreV2`).
* **Weighted scope** — direct entity matches, 1-hop neighbours, and slot
  hits are combined into a continuous-weight dict before the vector search.
* **Backfill helper** — :meth:`backfill_memory_tags_from_text` embeds all
  stored memories and writes V2 tags in bulk.

Only *new* public methods are added; every method from the parent class
continues to work unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from voicemem.utils.common._graph_common import cosine as _cosine

from voicemem.leftbrain.memory_repository import (
    LeftBrainMemoryRepository,
    LeftBrainMemoryRepositoryConfig,
)
from voicemem.leftbrain.cognitive_graph.store import CognitiveGraphStore
from voicemem.leftbrain.cognitive_graph.store_v2 import CognitiveGraphStoreV2
from voicemem.leftbrain.cognitive_graph.slot_v2 import (
    SLOT_V2_DESCRIPTIONS,
    ALL_SLOT_V2_VALUES,
    SLOT_RELATIONS,
)
from voicemem.leftbrain.cognitive_graph.query_slot_classifier import (
    QueryClassification,
    QuerySlotClassifier,
    SlotClassifierConfig,
)
from voicemem.leftbrain.local_memory_store import MemorySearchHit
from voicemem.llm_config import resolve_api_key, resolve_model

# ---------------------------------------------------------------------------
# Module-level embedding cache — shared across all instances in the process.
# ---------------------------------------------------------------------------
_SLOT_EMBED_CACHE: dict[str, list[float]] = {}


class LeftBrainMemoryRepositoryV2(LeftBrainMemoryRepository):
    """LeftBrainMemoryRepository with semantic-slot V2 retrieval.

    The constructor accepts the same arguments as the parent.  If a
    :class:`CognitiveGraphStore` (non-V2) is instantiated by the parent's
    ``__init__``, it is automatically upgraded to a
    :class:`CognitiveGraphStoreV2` using the same database file.

    Parameters
    ----------
    embedder:
        A :class:`~.local_memory_store.TextEmbedder` instance.
    config:
        Repository configuration (same as parent).
    **kwargs:
        Additional keyword arguments forwarded to the parent constructor.
    """

    def __init__(self, embedder: Any, *, config: LeftBrainMemoryRepositoryConfig | None = None,
                 **kwargs: Any) -> None:
        super().__init__(embedder, config=config, **kwargs)

        # Upgrade the cognitive store to V2 if the parent created a plain one.
        # embedder=self._embedder carried through -- without this the upgraded
        # store silently loses the embedder the parent wired in, and
        # upsert_entity()'s semantic dedup never activates (fell back to
        # exact-string matching with no error, i.e. a fix that looked wired
        # but wasn't -- caught this via a real duplicate-entity test, not by
        # inspection alone).
        if (
            self._cognitive_store is not None
            and isinstance(self._cognitive_store, CognitiveGraphStore)
            and not isinstance(self._cognitive_store, CognitiveGraphStoreV2)
        ):
            self._cognitive_store = CognitiveGraphStoreV2(self._cognitive_store._path, embedder=self._embedder)

    # ── Slot embedding helpers ────────────────────────────────────────────────

    def _ensure_slot_embeddings(self) -> None:
        """Lazily populate the module-level slot embedding cache.

        Uses :meth:`embed_texts` on the underlying embedder (batch call) so
        all 7 slot descriptions are embedded in a single API round-trip.
        """
        global _SLOT_EMBED_CACHE
        missing = [s for s in ALL_SLOT_V2_VALUES if s not in _SLOT_EMBED_CACHE]
        if not missing:
            return
        texts = [SLOT_V2_DESCRIPTIONS[s] for s in missing]
        vecs = self._embedder.embed_texts(texts)
        for slot_val, vec in zip(missing, vecs):
            _SLOT_EMBED_CACHE[slot_val] = vec

    def _detect_slots_semantic(
        self,
        query_embedding: list[float],
        *,
        threshold: float,
        top_k: int,
    ) -> list[str]:
        """Return up to *top_k* V2 slot names whose embedding cosine similarity
        to *query_embedding* exceeds *threshold*, sorted by similarity desc.
        """
        self._ensure_slot_embeddings()
        scored: list[tuple[float, str]] = []
        for slot_val in ALL_SLOT_V2_VALUES:
            sim = _cosine(query_embedding, _SLOT_EMBED_CACHE[slot_val])
            if sim >= threshold:
                scored.append((sim, slot_val))
        scored.sort(reverse=True)
        return [sv for _, sv in scored[:top_k]]

    # ── Memory text helper for backfill ──────────────────────────────────────

    def _all_memory_texts(self, user_id: str) -> list[tuple[str, str]]:
        """Return all ``(id, content)`` pairs from the cognitive store's
        ``memories`` table for *user_id*.
        """
        with self._cognitive_store._conn() as c:  # type: ignore[union-attr]
            rows = c.execute(
                "SELECT id, content FROM memories WHERE user_id=?",
                (user_id,),
            ).fetchall()
        return [(r["id"], r["content"]) for r in rows]

    # ── Auto-tag on ingest ────────────────────────────────────────────────────

    def append_extracted(self, memories, *, user_id: str, **kwargs) -> list[str]:  # type: ignore[override]
        """Store memories (parent) then auto-tag them with domain slots."""
        saved_ids = super().append_extracted(memories, user_id=user_id, **kwargs)
        if not saved_ids or self._cognitive_store is None:
            return saved_ids

        # Collect memory texts that were actually saved (parent skips empty text).
        # Build the same filtered list the parent uses, then pair with saved_ids.
        non_empty_texts = [(m.text or "").strip() for m in memories if (m.text or "").strip()]
        texts_to_tag: list[tuple[str, str]] = list(zip(saved_ids, non_empty_texts))
        self._tag_memory_slots_v2(texts_to_tag, user_id=user_id)
        return saved_ids

    def update_memory(self, memory_id: str, new_text: str, **kwargs) -> bool:  # type: ignore[override]
        """Update memory (parent, incl. cognitive-graph re-annotation) then
        re-tag domain slots -- an updated fact's slot can change (e.g. "还没
        定日期" 改成 "机票订好了，周五出发" moves from a vague planning slot
        toward daily_life/goals), and the old tag set otherwise never gets
        refreshed for the lifetime of that memory_id.
        """
        user_id = kwargs.get("user_id")
        updated = super().update_memory(memory_id, new_text, **kwargs)
        if updated and user_id and self._cognitive_store is not None:
            self._tag_memory_slots_v2([(memory_id, new_text.strip())], user_id=user_id)
        return updated

    def _tag_memory_slots_v2(self, texts_to_tag: list[tuple[str, str]], *, user_id: str) -> None:
        texts_to_tag = [(mid, t) for mid, t in texts_to_tag if t]
        if not texts_to_tag or self._cognitive_store is None:
            return
        try:
            self._ensure_slot_embeddings()
            vecs = self._embedder.embed_texts([t for _, t in texts_to_tag])
            for (mid, _), vec in zip(texts_to_tag, vecs):
                slots_above: list[tuple[str, float]] = [
                    (slot_val, round(_cosine(vec, _SLOT_EMBED_CACHE[slot_val]), 6))
                    for slot_val in ALL_SLOT_V2_VALUES
                    if _cosine(vec, _SLOT_EMBED_CACHE[slot_val]) >= 0.20
                ]
                if slots_above:
                    self._cognitive_store.upsert_memory_tags(mid, user_id, slots_above)  # type: ignore[union-attr]
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("auto-tag slot failed: %s", exc)

    # ── Slot-based retrieval (new primary method) ─────────────────────────────

    def search_slot_based(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int = 5,
        classifier: QuerySlotClassifier | None = None,
        summary_update_every: int = 10,
    ) -> tuple[list[MemorySearchHit], dict[str, str], QueryClassification, dict]:
        """Slot-based retrieval: LLM classifies query → slot + entities → search.

        Pipeline:
          1. LLM call → QueryClassification(slots, entities)
          2. Primary slot memory IDs from memory_tags
          3. Entity-filtered intersection (if entities given)
          4. Fallback to full-slot IDs if entity intersection empty
          5. Vector search within candidate IDs
          6. Retrieve related-slot summaries
          7. Optionally trigger async summary update for primary slot

        Returns
        -------
        (hits, related_summaries, classification, trace)
        """
        trace: dict = {
            "classification": {},
            "primary_slot": None,
            "slot_mem_count": 0,
            "entity_mem_count": 0,
            "candidate_count": 0,
            "search_mode": "fallback",
        }

        if self._cognitive_store is None:
            hits = self.search(query, user_id=user_id, top_k=top_k)
            return hits, {}, QueryClassification(), trace

        # ── Step 1: LLM classify ─────────────────────────────────────────────
        clf = classifier or QuerySlotClassifier(
            SlotClassifierConfig(base_url=self._base_url)
        )
        classification = clf.classify(query)
        print(f"[slot_classifier] query={query!r}  →  slots={classification.slots}  entities={classification.entities}")
        trace["classification"] = {
            "slots": classification.slots,
            "entities": classification.entities,
        }
        primary_slot = classification.primary_slot()
        trace["primary_slot"] = primary_slot

        # ── Step 2: slot memory IDs ──────────────────────────────────────────
        slot_mids: set[str] = set()
        if primary_slot and hasattr(self._cognitive_store, "memory_ids_for_slots_v2"):
            slot_mids = set(
                self._cognitive_store.memory_ids_for_slots_v2(user_id, [primary_slot])
            )
        trace["slot_mem_count"] = len(slot_mids)

        # ── Step 3: entity filter ────────────────────────────────────────────
        entity_mids: set[str] = set()
        if classification.entities and hasattr(self._cognitive_store, "find_entities_by_name_fuzzy"):
            for ent_name in classification.entities:
                ents = self._cognitive_store.find_entities_by_name_fuzzy(user_id, ent_name)
                if ents:
                    ent_ids = [e.id for e in ents]
                    mids = self._cognitive_store.memory_ids_for_entities(ent_ids)
                    entity_mids.update(mids)
        trace["entity_mem_count"] = len(entity_mids)

        # ── Step 4: candidate IDs ────────────────────────────────────────────
        if entity_mids and slot_mids:
            candidate_ids = entity_mids & slot_mids
            if not candidate_ids:
                # No overlap: widen to entity union slot
                candidate_ids = entity_mids | slot_mids
                trace["search_mode"] = "entity+slot-union"
            else:
                trace["search_mode"] = "entity+slot-intersection"
        elif entity_mids:
            candidate_ids = entity_mids
            trace["search_mode"] = "entity-only"
        elif slot_mids:
            candidate_ids = slot_mids
            trace["search_mode"] = "slot-only"
        else:
            candidate_ids = set()
            trace["search_mode"] = "fallback"

        trace["candidate_count"] = len(candidate_ids)

        # ── Step 5: vector search ────────────────────────────────────────────
        if candidate_ids:
            raw_hits = self._vector_store.search(
                query,
                user_id=user_id,
                top_k=top_k,
                memory_id_filter=candidate_ids,
            )
            hits: list[MemorySearchHit] = raw_hits[:top_k]
            # Supplement if fewer than top_k
            if len(hits) < top_k:
                seen = {h.memory_id for h in hits}
                for h in self.search(query, user_id=user_id, top_k=top_k):
                    if h.memory_id not in seen:
                        hits.append(h)
                        seen.add(h.memory_id)
                        if len(hits) >= top_k:
                            break
                if trace["search_mode"] != "fallback":
                    trace["search_mode"] += "+supplement"
        else:
            hits = self.search(query, user_id=user_id, top_k=top_k)
            trace["search_mode"] = "fallback"

        # ── Step 6: related slot summaries ───────────────────────────────────
        related_slots = SLOT_RELATIONS.get(primary_slot, []) if primary_slot else []
        related_summaries: dict[str, str] = {}
        if related_slots and hasattr(self._cognitive_store, "get_slot_summaries"):
            related_summaries = self._cognitive_store.get_slot_summaries(user_id, related_slots)

        # ── Step 7: maybe trigger summary update ─────────────────────────────
        if primary_slot and hasattr(self._cognitive_store, "count_memories_in_slot"):
            self._maybe_update_slot_summary(
                user_id, primary_slot, summary_update_every
            )

        return hits, related_summaries, classification, trace

    def _maybe_update_slot_summary(
        self, user_id: str, slot: str, update_every: int = 10
    ) -> None:
        """Trigger async summary update if enough new memories accumulated."""
        import threading

        try:
            current_count = self._cognitive_store.count_memories_in_slot(user_id, slot)  # type: ignore[union-attr]
            last_count = self._cognitive_store.get_slot_summary_mem_count(user_id, slot)  # type: ignore[union-attr]
            if current_count - last_count >= update_every:
                threading.Thread(
                    target=self._update_slot_summary,
                    args=(user_id, slot, current_count),
                    daemon=True,
                ).start()
        except Exception:
            pass

    def _update_slot_summary(self, user_id: str, slot: str, mem_count: int) -> None:
        """Generate a rolling summary for a slot and store it (background thread)."""
        import logging

        try:
            # Fetch the top-20 most recent memories in this slot
            slot_mids = set(
                self._cognitive_store.memory_ids_for_slots_v2(user_id, [slot])  # type: ignore[union-attr]
            )
            if not slot_mids:
                return
            hits = self._vector_store.search(
                SLOT_V2_DESCRIPTIONS[slot],
                user_id=user_id,
                top_k=20,
                memory_id_filter=slot_mids,
            )
            if not hits:
                return
            snippets = "\n".join(f"- {h.text}" for h in hits)
            summary_prompt = (
                f"以下是用户关于「{slot}」领域的记忆片段，请用2-4句话综合概括这个人在这方面的主要情况：\n\n"
                f"{snippets}\n\n请直接给出综合描述，不要重复列举原句。"
            )
            from openai import OpenAI
            client = OpenAI(
                api_key=resolve_api_key(),
                base_url=self._base_url,
            )
            resp = client.chat.completions.create(
                model=resolve_model(),
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=200,
                temperature=0,
            )
            summary_text = (resp.choices[0].message.content or "").strip()
            if summary_text:
                self._cognitive_store.upsert_slot_summary(  # type: ignore[union-attr]
                    user_id, slot, summary_text, mem_count
                )
                logging.getLogger(__name__).info(
                    "Updated slot summary for %s/%s (count=%d)", user_id, slot, mem_count
                )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("_update_slot_summary failed: %s", exc)

    # ── Backfill ──────────────────────────────────────────────────────────────

    def backfill_memory_tags_from_text(
        self,
        *,
        user_id: str,
        slot_threshold: float = 0.20,
    ) -> dict[str, int]:
        """Embed all stored memories and write V2 slot tags.

        Reads every memory from the cognitive store's ``memories`` table,
        embeds the content text, computes cosine similarity against the slot
        description embeddings, and writes tags above *slot_threshold* to the
        ``memory_tags`` table via
        :meth:`CognitiveGraphStoreV2.upsert_memory_tags`.

        Parameters
        ----------
        user_id:
            Owner of the memories to tag.
        slot_threshold:
            Minimum cosine similarity for a slot tag to be written.

        Returns
        -------
        dict[str, int]
            ``{"tagged": N, "total": M}`` where *N* is the number of memories
            that received at least one tag and *M* is the total number of
            memories processed.

        Raises
        ------
        RuntimeError
            If ``self._cognitive_store`` does not support ``upsert_memory_tags``
            (i.e. is not a :class:`CognitiveGraphStoreV2`).
        """
        if not hasattr(self._cognitive_store, "upsert_memory_tags"):
            raise RuntimeError(
                "cognitive_store does not support upsert_memory_tags; "
                "ensure CognitiveGraphStoreV2 is used."
            )

        # Ensure slot embeddings are ready.
        self._ensure_slot_embeddings()

        memory_texts = self._all_memory_texts(user_id)
        total = len(memory_texts)
        tagged = 0

        if not memory_texts:
            return {"tagged": 0, "total": 0}

        # Embed in one batch for efficiency.
        texts = [content for _, content in memory_texts]
        vecs = self._embedder.embed_texts(texts)

        for (memory_id, _content), mem_vec in zip(memory_texts, vecs):
            slots_above: list[tuple[str, float]] = []
            for slot_val in ALL_SLOT_V2_VALUES:
                sim = _cosine(mem_vec, _SLOT_EMBED_CACHE[slot_val])
                if sim >= slot_threshold:
                    slots_above.append((slot_val, round(sim, 6)))

            if slots_above:
                self._cognitive_store.upsert_memory_tags(  # type: ignore[union-attr]
                    memory_id, user_id, slots_above
                )
                tagged += 1

        return {"tagged": tagged, "total": total}


__all__ = ["LeftBrainMemoryRepositoryV2"]
