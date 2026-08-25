"""LLM 标注器：对已抽取的 fact 文本打 slot + 实体 + 关系标签。

设计原则：
- 独立于 mem0 抽取 extractor，作为写入流程的第二步
- 单次 LLM 调用处理一批 facts（最多 BATCH_SIZE 条，控制 token）
- 解析失败则 fallback 到 slot=knowledge、entities=[]
- 支持同步调用（无 async 依赖）
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from .slot_v2 import ALL_SLOT_V2_VALUES, SLOT_V2_DESCRIPTIONS, SlotV2
from .types import (
    AnnotatedFact,
    EntityAnnotation,
    EntityType,
    RelationAnnotation,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 10  # 每次 LLM 调用最多处理的 fact 数

# ── System prompt ──────────────────────────────────────────────────────────────

# Slot 定义复用 slot_v2.py 的 SLOT_V2_DESCRIPTIONS——跟 Classify()/查询分类
# （query_slot_classifier.py）、写入时 embedding 打标（memory_repository_v2.py）
# 用的是同一套 7 类 life-domain taxonomy，不再是这里单独一套 entity-category
# 分类（people/projects/tasks/places/routines/assets），避免"一条 fact 在
# ingest 时被标成一类、在 Classify() 检索时又按另一套完全不同的类目找"。
_SLOT_DEFS = "\n".join(f"- {s.value}: {SLOT_V2_DESCRIPTIONS[s]}" for s in SlotV2)

# 剩下的 prompt 主体（JSON 示例部分）字面含大量 { }，用普通字符串拼接而不是
# f-string 一整块，省得每个 JSON 字面量花括号都要转义成 {{ }}。
_SYSTEM_PROMPT = (
    "You are a cognitive graph annotator. Given a list of memory facts extracted from conversations, annotate each fact with:\n"
    f"1. **slot**: which life-domain slot it belongs to (one of: {', '.join(ALL_SLOT_V2_VALUES)})\n"
    "2. **entities**: named entities in the fact (person, project, task, knowledge, preference, place, routine, asset, organization, or event)\n"
    "3. **relations**: typed edges between entities in the fact\n"
    "\n"
    "Slot definitions:\n"
    f"{_SLOT_DEFS}\n"
    "\n"
    "Note: the user's own profile/identity, their preferences, and their boundaries/constraints\n"
    "are handled elsewhere (not a slot here) — do not force-fit those into one of the slots above;\n"
    "pick the closest life-domain slot only if the fact also carries factual/structural content.\n"
    "\n"
    "Entity types: user, person, project, task, knowledge, preference, place, routine, asset, organization, event"
) + """

Common relation types:
- collaborates_on (person → project)
- owns_knowledge_about (person → knowledge)
- assigned_task_in (person → project)
- belongs_to (task → project)
- used_in (knowledge → project)
- located_in (person/project → place)
- works_at (person → organization)
- participated_in (person → event)
- manages (person → project/person)
- prefers (person → preference)

Return a JSON object with key "annotations", which is a list of objects (one per input fact), each with:
{
  "slot": "<slot_type>",
  "entities": [{"name": "...", "entity_type": "...", "role": "<subject|object|context|owner>"}],
  "relations": [{"from": "...", "to": "...", "relation_type": "...", "role_label": "...", "confidence": 0.9}],
  "confidence": 0.9
}

Entity role values:
- subject: the entity performs the action ("Lang said X")
- object: the entity receives the action ("we talked about Lang")
- owner: the fact directly describes this entity ("Caroline's preference is...")
- context: the entity is mentioned as background

If a fact has no clear entities or relations, return empty arrays. Never invent entities not mentioned in the fact."""

_USER_TEMPLATE = """Annotate each of the following memory facts. Return results in order.

Facts:
{facts_block}

Return JSON: {{"annotations": [...]}}"""


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class CognitiveAnnotatorConfig:
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    batch_size: int = BATCH_SIZE
    timeout: float = 60.0

    def resolved_model(self) -> str:
        return (self.model or os.environ.get("OPENAI_MODEL", "").strip() or "gpt-4o-mini").strip()


# ── Annotator ──────────────────────────────────────────────────────────────────

class CognitiveAnnotator:
    """LLM-based cognitive graph annotator.

    使用方法：
        annotator = CognitiveAnnotator(config)
        annotated_facts = annotator.annotate(["fact1 text", "fact2 text"])
    """

    def __init__(self, config: CognitiveAnnotatorConfig | None = None) -> None:
        self._cfg = config or CognitiveAnnotatorConfig()
        self._model = self._cfg.resolved_model()
        self._batch_size = self._cfg.batch_size
        self._client = self._build_client()

    def _build_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("需要 openai>=1.0: pip install openai") from e
        kw: dict[str, Any] = {
            "api_key": self._cfg.api_key or os.environ.get("OPENAI_API_KEY"),
            "timeout": self._cfg.timeout,
        }
        if self._cfg.base_url:
            kw["base_url"] = self._cfg.base_url
        return OpenAI(**kw)

    def annotate(self, facts: list[str]) -> list[AnnotatedFact]:
        """批量标注 fact 列表，返回 AnnotatedFact 列表（与输入等长）。

        抽取那一步已经顺便把标注带回来了（见 leftbrain/merged_extraction.py），
        命中的直接用，剩下的才发一次 LLM——全命中时这里 0 次网络往返。
        """
        if not facts:
            return []
        from voicemem.leftbrain import merged_extraction

        out: list[AnnotatedFact | None] = []
        todo: list[str] = []
        for f in facts:
            ann = merged_extraction.take_annotation(f) if merged_extraction.enabled() else None
            if ann is None:
                todo.append(f)
                out.append(None)
            else:
                out.append(_build_annotated_fact(f, ann))

        if todo:
            fresh: list[AnnotatedFact] = []
            for i in range(0, len(todo), self._batch_size):
                fresh.extend(self._annotate_batch(todo[i: i + self._batch_size]))
            it = iter(fresh)
            out = [a if a is not None else next(it, None) for a in out]

        # 兜底：某条既没命中暂存、LLM 又没给回来时不能返回 None（下游 zip 会错位）
        return [a if a is not None else _build_annotated_fact(f, {})
                for f, a in zip(facts, out)]

    def _annotate_batch(self, facts: list[str]) -> list[AnnotatedFact]:
        facts_block = "\n".join(f"{j+1}. {f}" for j, f in enumerate(facts))
        user_prompt = _USER_TEMPLATE.format(facts_block=facts_block)

        raw = self._call_llm(user_prompt)
        annotations = self._parse_response(raw, len(facts))

        result: list[AnnotatedFact] = []
        for fact_text, ann in zip(facts, annotations):
            result.append(_build_annotated_fact(fact_text, ann))
        return result

    def _temp_kw(self) -> dict:
        return {} if self._model in {"gpt-5", "gpt-5-mini"} else {"temperature": 0}

    def _call_llm(self, user_prompt: str) -> str:
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    **self._temp_kw(),
                    response_format={"type": "json_object"},
                    max_tokens=2048,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt == 2:
                    logger.warning("CognitiveAnnotator LLM 失败: %s", e)
                    return "{}"
                time.sleep(5 * (attempt + 1))
        return "{}"

    def _parse_response(self, raw: str, expected: int) -> list[dict]:
        try:
            data = json.loads(raw)
            anns = data.get("annotations", [])
            if isinstance(anns, list) and len(anns) == expected:
                return anns
            # 长度不对时 pad 或 truncate
            while len(anns) < expected:
                anns.append({})
            return anns[:expected]
        except Exception:
            return [{} for _ in range(expected)]


# ── Parse helpers ──────────────────────────────────────────────────────────────

_VALID_SLOTS = set(ALL_SLOT_V2_VALUES)
_VALID_ENTITY_TYPES = {e.value for e in EntityType}


def _parse_slot(raw: Any) -> SlotV2:
    if isinstance(raw, str) and raw in _VALID_SLOTS:
        return SlotV2(raw)
    return SlotV2.KNOWLEDGE


def _parse_entity_type(raw: Any) -> EntityType:
    if isinstance(raw, str) and raw in _VALID_ENTITY_TYPES:
        return EntityType(raw)
    return EntityType.KNOWLEDGE


def _build_annotated_fact(fact_text: str, ann: dict) -> AnnotatedFact:
    slot = _parse_slot(ann.get("slot"))
    confidence = float(ann.get("confidence", 1.0))

    entities: list[EntityAnnotation] = []
    for e in ann.get("entities", []):
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        if not name:
            continue
        entities.append(EntityAnnotation(
            name=name,
            entity_type=_parse_entity_type(e.get("entity_type")),
            role=str(e.get("role", "")) or None,
        ))

    relations: list[RelationAnnotation] = []
    for r in ann.get("relations", []):
        if not isinstance(r, dict):
            continue
        from_name = str(r.get("from", "")).strip()
        to_name = str(r.get("to", "")).strip()
        rel_type = str(r.get("relation_type", "")).strip()
        if not from_name or not to_name or not rel_type:
            continue
        relations.append(RelationAnnotation(
            from_name=from_name, to_name=to_name,
            relation_type=rel_type,
            role_label=str(r.get("role_label", "")) or None,
            confidence=float(r.get("confidence", confidence)),
        ))

    return AnnotatedFact(
        fact_text=fact_text,
        slot=slot,
        entities=entities,
        relations=relations,
        confidence=confidence,
    )


# ── Null annotator (no-op, for testing/disabling) ─────────────────────────────

class NullAnnotator:
    """不调用 LLM，全部返回 slot=knowledge、entities=[]。"""

    def annotate(self, facts: list[str]) -> list[AnnotatedFact]:
        return [AnnotatedFact(fact_text=f, slot=SlotV2.KNOWLEDGE) for f in facts]
