"""左脑双写存储：Mem0 Platform 风格 JSON + SQLite 向量库 + 认知图。

语义记忆：``memories.json`` + ``voicemem_leftbrain.sqlite``（向量）。
认知图：``cognitive_graph.sqlite``（entities / edges / slot_profiles / right-brain stub）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from voicemem.leftbrain.extract_facts_openai import ExtractedAdditiveMemory
from voicemem.leftbrain.local_memory_store import (
    MemorySearchHit,
    OpenAILocalEmbedder,
    OpenAILocalEmbedderConfig,
    TextEmbedder,
    default_memory_root,
)
from voicemem.leftbrain.mem0_backend_store import Mem0BackendStore
from voicemem.leftbrain.cognitive_graph import (
    CognitiveAnnotator,
    CognitiveAnnotatorConfig,
    CognitiveGraphStore,
    NullAnnotator,
)

_DEFAULT_JSON_NAME = "memories.json"
_DEFAULT_COGNITIVE_DB_NAME = "cognitive_graph.sqlite"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LeftBrainMemoryRepositoryConfig:
    """``json_path`` / ``db_path`` 为 ``None`` 时使用 ``default_memory_root()`` 下默认文件名。"""

    json_path: Path | None = None
    db_path: Path | None = None
    existing_limit: int = 50
    # 认知图
    cognitive_db_path: Path | None = None
    enable_cognitive_graph: bool = False


class LeftBrainMemoryRepository:
    """Mem0 JSON 镜像、向量库、认知图（Cognitive Graph）。

    ``search`` 纯向量检索；
    ``search_with_cognitive_scope`` 先从认知图缩小范围再向量检索。
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        *,
        config: LeftBrainMemoryRepositoryConfig | None = None,
        cognitive_annotator: CognitiveAnnotator | NullAnnotator | None = None,
        experience_repo: Any | None = None,
        vector_store: Any | None = None,
    ) -> None:
        self._embedder = embedder
        cfg = config or LeftBrainMemoryRepositoryConfig()
        # cfg.db_path 是调用方（core.py）传入的 "<memory_root>/voicemem_leftbrain.sqlite"
        # —— mem0 迁移前这个文件本身就是向量库，迁移后 mem0 需要的是一整个目录
        # （qdrant collection + history db），取其父目录延续原来"每个 VoiceMem
        # 实例自己的 memory_root 相互隔离"的语义；未显式传 db_path 时才退回全局默认目录。
        root = cfg.db_path.parent if cfg.db_path is not None else default_memory_root()
        root.mkdir(parents=True, exist_ok=True)
        self._json_path = (
            Path(cfg.json_path).expanduser().resolve()
            if cfg.json_path is not None
            else root / _DEFAULT_JSON_NAME
        )
        self._existing_limit = cfg.existing_limit
        # 原始事实存储真的走 mem0（论文：实体节点指向"底层 Mem0 后端原始记忆条目
        # 索引"）。见 mem0_backend_store.py 顶部注释，里面说清楚了跟以前那套
        # 自建 SQLite 向量库的行为差异（尤其是 id 生成方式变了——mem0 自己生成
        # id，不再用 cfg.db_path 指向的文件）。
        # 默认走 mem0；传入 vector_store（如 zep 等实现同接口的对象）即替换 memory engine。
        self._vector_store = vector_store or Mem0BackendStore(embedder, memory_root=root)

        # 认知图
        self._cognitive_store: CognitiveGraphStore | None = None
        if cfg.enable_cognitive_graph:
            cog_db = (
                Path(cfg.cognitive_db_path).expanduser().resolve()
                if cfg.cognitive_db_path is not None
                else root / _DEFAULT_COGNITIVE_DB_NAME
            )
            self._cognitive_store = CognitiveGraphStore(cog_db, embedder=self._embedder)
        self._cognitive_annotator: CognitiveAnnotator | NullAnnotator | None = cognitive_annotator

        # 右脑（可选注入，不影响任何现有方法）
        self._experience_repo = experience_repo

    @property
    def json_path(self) -> Path:
        return self._json_path

    @property
    def db_path(self) -> Path:
        return self._vector_store._path  # noqa: SLF001 — 开发脚本需展示路径

    @property
    def vector_store(self) -> Mem0BackendStore:
        return self._vector_store

    # 这份左脑记忆镜像原来是 memory_root 下单独一个 memories.json。一个 space 只留
    # 一个 json（那是空间描述文件），所以镜像挪进 sqlite 的 kv 表——它本来就是内部
    # 状态，不是给人看的。旧的 memories.json 还在就自动读进来一次。
    _KV_KEY = "leftbrain_store"

    def load_json_store(self) -> dict[str, Any]:
        from voicemem.utils.common import space as _space
        data = _space.kv_get(self._json_path.parent, self._KV_KEY)
        if data is None and self._json_path.is_file():
            try:
                with self._json_path.open(encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = None
        if not isinstance(data, dict):
            return {"count": 0, "results": []}
        results = data.get("results")
        if not isinstance(results, list):
            return {"count": 0, "results": []}
        return {"count": len(results), "results": results}

    def _write_json_store(self, results: list[dict[str, Any]]) -> None:
        from voicemem.utils.common import space as _space
        _space.kv_set(self._json_path.parent, self._KV_KEY,
                      {"count": len(results), "results": results})

    def existing_for_extractor(self, *, user_id: str, limit: int | None = None) -> list[dict[str, str]]:
        """供 additive 抽取：``[{"id", "text"}, ...]``（优先 JSON，与 Mem0 字段对齐）。"""
        cap = limit if limit is not None else self._existing_limit
        store = self.load_json_store()
        rows: list[dict[str, str]] = []
        for obj in store["results"]:
            if not isinstance(obj, dict):
                continue
            if obj.get("user_id", user_id) != user_id:
                continue
            mid = str(obj.get("id", "")).strip()
            memory = str(obj.get("memory", "")).strip()
            if mid and memory:
                rows.append({"id": mid, "text": memory})
        return rows[-cap:]

    def update_memory(self, memory_id: str, new_text: str,
                      session_id: int | str | None = None,
                      observed_at: str | None = None,
                      user_id: str | None = None) -> bool:
        """原地更新记忆文本（含重新 embed）。同步更新 JSON 镜像+认知图。

        session_id / observed_at 非 None 时刷新 metadata（observed_at 即 created_at，
        新事实来自哪次会话就该标哪次会话的日期）。

        user_id 传入时会重新跑一遍认知图实体/关系抽取（跟 append_extracted 对
        ADD 事实做的是同一步）——之前这里只更新了文本，UPDATE 决策的事实永远
        不会进认知图，实体/关系停留在这条记忆第一次被写入（可能是完全不同措辞）
        时候的样子，新版本说的实体变了也不会同步。user_id 为 None 时（旧调用方
        没传）保持原来的行为，只更新文本，不动认知图。
        """
        updated = self._vector_store.update_memory(
            memory_id, new_text, session_id=session_id, observed_at=observed_at)
        if updated:
            store = self.load_json_store()
            for obj in store["results"]:
                if isinstance(obj, dict) and str(obj.get("id", "")) == memory_id:
                    obj["memory"] = new_text
                    if observed_at is not None:      # 镜像里的日期同步刷，别和库里不一致
                        obj["created_at"] = observed_at
                    break
            self._write_json_store(store["results"])

            if user_id is not None and self._cognitive_store is not None and self._cognitive_annotator is not None:
                try:
                    annotated = self._cognitive_annotator.annotate([new_text])
                    if annotated:
                        self._cognitive_store.ingest_annotated_fact(user_id, annotated[0], [memory_id])
                except Exception as _cog_err:
                    import logging
                    logging.getLogger(__name__).warning("UPDATE 认知图重新写入失败: %s", _cog_err)
        return updated

    def delete_memory(self, memory_id: str) -> bool:
        """删除单条记忆。同步更新 JSON 镜像。"""
        deleted = self._vector_store.delete_memory(memory_id)
        if deleted:
            store = self.load_json_store()
            results = [obj for obj in store["results"]
                       if not (isinstance(obj, dict) and str(obj.get("id", "")) == memory_id)]
            self._write_json_store(results)
        return deleted

    def append_extracted(
        self,
        memories: Sequence[ExtractedAdditiveMemory],
        *,
        user_id: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """追加记忆：向量入库（mem0，真正的 id 来源）+ JSON 镜像 + 认知图写入。

        id 现在由 mem0 生成，不再是本地 format_memory_id() 算出来的——mem0
        的 add() 没有"你指定 id 我照用"这个选项。之前这里的写法是先算好一个
        本地 id、把它同时喂给 JSON 镜像和 vector_store，但完全没看
        add_records_with_ids() 的返回值，等于 JSON 镜像/认知图链接用的 id
        和 mem0 库里真正存的 id 是两套不同的东西，全对不上（真实跑过 Search()
        才发现的：认知图里链接的 memory_id 在 mem0 里根本不存在，右脑记账
        Algorithm 1 用的 GraphEntityStore 链接也全部失效）。现在改成：先把
        文本+metadata 拼好交给 mem0，用它真实返回的 id 再去写 JSON 镜像和
        认知图，全程只有一套 id。
        """
        base_md = dict(extra_metadata or {})

        vector_items: list[tuple[str, str, str, dict[str, Any]]] = []
        pending: list[tuple[str, str, dict[str, Any]]] = []  # (body, attributed_to, metadata) 按序对应
        for m in memories:
            body = (m.text or "").strip()
            if not body:
                continue
            md: dict[str, Any] = dict(base_md)
            if m.local_id:
                md["extractor_local_id"] = m.local_id
            if m.linked_memory_ids:
                md["linked_memory_ids"] = list(m.linked_memory_ids)
            md["mem0_attributed_to"] = m.attributed_to or "user"
            attributed_to = m.attributed_to or "user"
            vector_items.append(("", body, attributed_to, md))
            pending.append((body, attributed_to, md))

        if not vector_items:
            return []

        saved_ids = self._vector_store.add_records_with_ids(user_id, vector_items)
        if len(saved_ids) != len(pending):
            # add_records_with_ids() 内部逐条调 mem0，个别条目失败时会比输入
            # 少——按位置截到实际成功写入的这些，不能整体错位关联到别的事实。
            import logging
            logging.getLogger(__name__).warning(
                "append_extracted: mem0 写入 %d/%d 条成功，按成功的部分继续",
                len(saved_ids), len(pending),
            )
            pending = pending[: len(saved_ids)]

        store = self.load_json_store()
        results: list[dict[str, Any]] = list(store["results"])
        now = _utc_iso()
        for mid, (body, attributed_to, md) in zip(saved_ids, pending):
            results.append(
                {
                    "id": mid,
                    "memory": body,
                    "user_id": user_id,
                    "metadata": md,
                    "categories": [],
                    "created_at": now,
                    "updated_at": None,
                }
            )
        self._write_json_store(results)

        # 认知图写入：annotate facts → entities + slots + edges
        if self._cognitive_store is not None and self._cognitive_annotator is not None:
            fact_texts = [body for body, _, _ in pending]
            try:
                annotated_facts = self._cognitive_annotator.annotate(fact_texts)
                for annotated, mid in zip(annotated_facts, saved_ids):
                    self._cognitive_store.ingest_annotated_fact(user_id, annotated, [mid])
            except Exception as _cog_err:
                import logging
                logging.getLogger(__name__).warning("认知图写入失败: %s", _cog_err)

        return saved_ids

    def search(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int = 5,
        threshold: float | None = None,
        include_assistant: bool = False,
    ) -> list[MemorySearchHit]:
        return self._vector_store.search(
            query,
            user_id=user_id,
            top_k=top_k,
            threshold=threshold,
            include_assistant=include_assistant,
        )

    def search_with_graph(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int = 5,
        threshold: float | None = None,
        relation_depth: int = 1,
    ) -> list[GraphSearchHit]:
        """语义检索后附加本地图关系上下文。"""
        hits = self.search(query, user_id=user_id, top_k=top_k, threshold=threshold)
        if self._graph_store is None:
            return [
                GraphSearchHit(
                    memory=h,
                    graph=GraphMemoryContext(memory_id=h.memory_id),
                )
                for h in hits
            ]
        return self._graph_store.enrich_hits(
            hits,
            user_id=user_id,
            relation_depth=relation_depth,
        )

    @property
    def cognitive_store(self) -> "CognitiveGraphStore | None":
        return self._cognitive_store

    @property
    def experience_repo(self):
        """右脑 ExperienceRepository（可能为 None）。"""
        return self._experience_repo

    def search_combined(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int = 5,
        scope_min: int = 3,
        scope_ratio_max: float = 0.60,
        use_slot_filtering: bool = True,
        signals=None,   # CurrentSignals | None
    ) -> tuple[list, Any, dict]:
        """左右脑并行检索，返回 (left_hits, right_context, trace)。

        - left_hits: 与 search_with_cognitive_scope 完全相同
        - right_context: RightBrainContext（experience_repo=None 时为空）
        - trace: 左脑 trace + right_brain_empty 标记

        所有现有调用方只用左脑时不需要改任何代码，
        只有需要右脑时才调用此方法。
        """
        from concurrent.futures import ThreadPoolExecutor

        # ── 左脑（已有逻辑，不做任何修改）────────────────────────────────────
        def _left():
            return self.search_with_cognitive_scope(
                query,
                user_id=user_id,
                top_k=top_k,
                scope_min=scope_min,
                scope_ratio_max=scope_ratio_max,
                use_slot_filtering=use_slot_filtering,
            )

        # ── 右脑（无 experience_repo 时快速返回空）────────────────────────────
        def _right():
            if self._experience_repo is None:
                from voicemem.rightbrain.types import RightBrainContext, CurrentSignals
                return RightBrainContext(current_signals=signals or CurrentSignals())
            plan = self._experience_repo.build_query_plan(
                query, user_id, signals=signals,
            )
            return self._experience_repo.retrieve(plan)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_left  = pool.submit(_left)
            fut_right = pool.submit(_right)
            left_hits, trace = fut_left.result()
            right_context    = fut_right.result()

        trace["right_brain_active"] = self._experience_repo is not None
        trace["right_brain_empty"]  = right_context.is_empty()
        return left_hits, right_context, trace

    def backfill_cognitive_graph_from_json(
        self,
        *,
        user_id: str,
        batch_size: int = 10,
        force: bool = False,
    ) -> dict[str, int]:
        """对 memories.json 中已有记忆批量补打认知图标注。

        逐条检查 entity_memory_links 是否已存在，跳过已处理的 memory_id。
        ``force=True`` 时先清空该用户的认知图数据再重建。

        Returns:
            {"processed": n, "skipped": n, "entities_created": n}
        """
        if self._cognitive_store is None:
            raise ValueError("未配置 cognitive_store（enable_cognitive_graph=False）")
        if self._cognitive_annotator is None:
            raise ValueError("未配置 cognitive_annotator")

        if force:
            self._cognitive_store.delete_user(user_id)

        store = self.load_json_store()
        processed = skipped = entities_created = 0

        # 收集未处理的 memory_id
        pending: list[tuple[str, str]] = []  # (memory_id, text)
        for obj in store["results"]:
            if not isinstance(obj, dict) or obj.get("user_id") != user_id:
                continue
            mid = str(obj.get("id", "")).strip()
            text = str(obj.get("memory", "")).strip()
            if not mid or not text:
                continue
            # 跳过条件：entity links 已存在 AND memory record 也已存在
            # 若只有 entity links 但没有 memory record，仍需重新处理以补写 slot
            has_links = bool(self._cognitive_store.entity_ids_for_memory(mid))
            has_record = bool(self._cognitive_store.get_memory_record(mid))
            if not force and has_links and has_record:
                skipped += 1
                continue
            pending.append((mid, text))

        # 分批标注
        for i in range(0, len(pending), batch_size):
            batch = pending[i: i + batch_size]
            texts = [t for _, t in batch]
            try:
                annotated_facts = self._cognitive_annotator.annotate(texts)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("backfill 批次 %d 标注失败: %s", i, e)
                skipped += len(batch)
                continue

            for (mid, _), ann in zip(batch, annotated_facts):
                entities = self._cognitive_store.ingest_annotated_fact(user_id, ann, [mid])
                entities_created += len(entities)
                processed += 1

        return {"processed": processed, "skipped": skipped, "entities_created": entities_created}

    def backfill_memory_slots_from_entities(self, *, user_id: str) -> dict[str, int]:
        """为已有 entity_memory_links 但缺少 memories 记录的记忆补写 slot。

        不调 LLM。从该记忆链接的实体中取出现次数最多的 slot 作为该记忆的 slot。
        用于修复"认知图已回填实体但 memories 表为空"的历史数据。
        """
        if self._cognitive_store is None:
            raise ValueError("未配置 cognitive_store")

        linked_mids = set(self._cognitive_store.all_linked_memory_ids(user_id))
        existing_mids = set(self._cognitive_store.all_memory_record_ids(user_id))
        missing_mids = linked_mids - existing_mids

        # 从 JSON store 读文本
        store = self.load_json_store()
        mid_to_text: dict[str, str] = {}
        for obj in store["results"]:
            if isinstance(obj, dict) and obj.get("user_id") == user_id:
                mid = str(obj.get("id", "")).strip()
                text = str(obj.get("memory", "")).strip()
                if mid and text:
                    mid_to_text[mid] = text

        written = skipped = 0
        for mid in missing_mids:
            text = mid_to_text.get(mid, "")
            if not text:
                skipped += 1
                continue
            entity_ids = self._cognitive_store.entity_ids_for_memory(mid)
            if not entity_ids:
                skipped += 1
                continue
            entities = self._cognitive_store.find_entities(user_id, entity_ids=entity_ids)
            if not entities:
                skipped += 1
                continue
            # 取链接实体中出现最多的 slot
            from collections import Counter
            dominant_slot = Counter(e.slot for e in entities).most_common(1)[0][0]
            self._cognitive_store.upsert_memory_record(user_id, mid, dominant_slot, text)
            written += 1

        return {"written": written, "skipped": skipped, "total_missing": len(missing_mids)}

    def sync_vectors_from_json(self, *, user_id: str) -> int:
        """将 JSON 中尚未写入 SQLite 的条目补全向量（用于迁移或修复）。"""
        store = self.load_json_store()
        existing_ids = set(self._vector_store.list_ids(user_id=user_id))
        items: list[tuple[str, str, str, dict[str, Any]]] = []
        for obj in store["results"]:
            if not isinstance(obj, dict) or obj.get("user_id") != user_id:
                continue
            mid = str(obj.get("id", "")).strip()
            memory = str(obj.get("memory", "")).strip()
            if not mid or not memory or mid in existing_ids:
                continue
            md = obj.get("metadata")
            meta = md if isinstance(md, dict) else {}
            attributed = str(meta.get("mem0_attributed_to", "user"))
            items.append((mid, memory, attributed, meta))
        if not items:
            return 0
        self._vector_store.add_records_with_ids(user_id, items)
        return len(items)


def create_openai_memory_repository(
    *,
    config: LeftBrainMemoryRepositoryConfig | None = None,
    embedder_config: OpenAILocalEmbedderConfig | None = None,
    cognitive_annotator_config: "CognitiveAnnotatorConfig | None" = None,
) -> LeftBrainMemoryRepository:
    """使用 OpenAI Embeddings 的默认仓库（需 ``OPENAI_API_KEY``）。

    若 config.enable_cognitive_graph=True，自动创建认知图标注器。
    """
    from voicemem.leftbrain.cognitive_graph import CognitiveAnnotatorConfig as _CogCfg
    embedder = OpenAILocalEmbedder(embedder_config)
    annotator: CognitiveAnnotator | NullAnnotator | None = None
    cfg = config or LeftBrainMemoryRepositoryConfig()
    if cfg.enable_cognitive_graph:
        annotator = CognitiveAnnotator(cognitive_annotator_config or _CogCfg())
    return LeftBrainMemoryRepository(embedder, config=cfg, cognitive_annotator=annotator)
