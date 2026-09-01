"""Experience Repository：右脑检索 + 写入的高层接口。

检索顺序（spec 第9节）：
  1. response_experience    — 最高优先级，回应成败教训
  2. heartnote                   — 情境情绪规律
  3. user_interaction_profile    — 全局用户风格

写入：upsert_experience + link_anchors
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .anchor_router import AnchorRouter
from .store import RightBrainStore
from .types import (
    CurrentSignals, MemoryAnchor, MemoryClass,
    MemoryQueryPlan, RightBrainContext, RightBrainMemory, TTL,
)

_DEFAULT_DB_NAME = "right_brain.sqlite"


class ExperienceRepository:
    """右脑 Experience Layer 的完整入口。

    用法（最小）::

        repo = ExperienceRepository.create(db_path="memory/right_brain.sqlite")
        plan = repo.build_query_plan("Lang 上次的方案", user_id="u1")
        ctx  = repo.retrieve(plan)
        print(ctx.to_prompt_block())
    """

    def __init__(
        self,
        store: RightBrainStore,
        anchor_router: AnchorRouter,
    ) -> None:
        self._store = store
        self._router = anchor_router

    @classmethod
    def create(
        cls,
        db_path: Path | str,
        *,
        cognitive_store=None,     # CognitiveGraphStore | None
    ) -> "ExperienceRepository":
        store  = RightBrainStore(db_path)
        router = AnchorRouter(cognitive_store)
        return cls(store, router)

    # ── Query Plan ────────────────────────────────────────────────────────────

    def build_query_plan(
        self,
        query: str,
        user_id: str,
        *,
        signals: CurrentSignals | None = None,
        entities: list[str] | None = None,
        emotion: str | None = None,
        context: str | None = None,
    ) -> MemoryQueryPlan:
        """``context``：agent 上一句（用户正在回应它），只参与锚点、不进 clean_text。"""
        return self._router.build_query_plan(
            query, user_id, signals=signals, entities=entities, emotion=emotion,
            context=context,
        )

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        plan: MemoryQueryPlan,
        *,
        experience_limit: int = 3,
        pattern_limit: int = 3,
    ) -> RightBrainContext:
        """三类记忆并行检索，按优先级组合返回。"""
        anchors = plan.anchors
        uid     = plan.user_id
        sigs    = plan.current_signals

        # 1. response_experience 这一类**不再写入、也不再检索**（见 brain.py 的
        #    learn_from_reaction）：它只有写没有读——真正有用的 next_time 存进了
        #    metadata、没有任何读取方；进 prompt 的 assistant_did 长成"助手用轻松的
        #    语气引导用户"，每轮占一个名额却给不出信息。助手该怎么做本来就能从用户
        #    侧特征推出来（「他低落时想要理解和认同」已经说明该给什么了）。
        #    这里返回空列表而不是删掉整条通路：老库里已经存下的那些不该突然变成
        #    没人处理的孤儿，类型和前端兼容都还在。
        experiences: list = []

        # 2. heartnote
        patterns = self._store.search_by_anchors(
            uid, anchors, memory_class="heartnote", limit=pattern_limit,
        )

        # user_interaction_profile 由前刺层 UserProfileStore 独立管理，不在此检索
        return RightBrainContext(
            response_experiences=experiences,
            situation_patterns=patterns,
            current_signals=sigs,
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def write_experience(
        self,
        user_id: str,
        memory_class: MemoryClass,
        content: str,
        anchors: list[MemoryAnchor],
        *,
        condition: str | None = None,
        priority: float = 0.5,
        confidence: float = 1.0,
        ttl: TTL = "long_term",
        metadata: dict[str, Any] | None = None,
        evidence_turn_ids: list[str] | None = None,
        evidence_memory_ids: list[str] | None = None,
        memory_id: str | None = None,
        created_at: str | None = None,
    ) -> RightBrainMemory:
        """写入一条右脑记忆并挂 anchors。

        ``created_at``：事件真实发生时间（回填/benchmark 场景必传，否则渲染出的
        日期是写入墙钟，temporal 类问题会被带偏；见 store.upsert_memory）。
        """
        mem = self._store.upsert_memory(
            user_id, memory_class, content,
            condition=condition,
            priority=priority, confidence=confidence, ttl=ttl,
            metadata=metadata,
            evidence_turn_ids=evidence_turn_ids,
            evidence_memory_ids=evidence_memory_ids,
            memory_id=memory_id,
            created_at=created_at,
        )
        for anchor in anchors:
            self._store.link_anchor(mem.id, user_id, anchor)
        return mem

    # ── Convenience helpers ───────────────────────────────────────────────────

    def write_response_experience(
        self,
        user_id: str,
        content: str,
        anchors: list[MemoryAnchor],
        *,
        condition: str | None = None,
        failed: bool = False,
        metadata: dict[str, Any] | None = None,
        evidence_turn_ids: list[str] | None = None,
        **kwargs,
    ) -> RightBrainMemory:
        """快捷写入 response_experience，失败经验 priority 自动拉高。"""
        priority = kwargs.pop("priority", 0.9 if failed else 0.6)
        meta = dict(metadata or {})
        if failed:
            meta.setdefault("previous_failure", True)
        return self.write_experience(
            user_id, "response_experience", content, anchors,
            condition=condition, priority=priority, metadata=meta,
            evidence_turn_ids=evidence_turn_ids, **kwargs,
        )

    def write_situation_pattern(
        self,
        user_id: str,
        content: str,
        anchors: list[MemoryAnchor],
        *,
        condition: str | None = None,
        confidence: float = 0.8,
        **kwargs,
    ) -> RightBrainMemory:
        """快捷写入情感情境模式。"""
        return self.write_experience(
            user_id, "heartnote", content, anchors,
            condition=condition, confidence=confidence, **kwargs,
        )

    # ── Inspect ───────────────────────────────────────────────────────────────

    def list_all(self, user_id: str) -> list[RightBrainMemory]:
        return self._store.get_all(user_id)

    @property
    def db_path(self) -> Path:
        return self._store._path  # noqa: SLF001
