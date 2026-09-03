"""LLM-based query slot classifier.

classify(): 单次 LLM 调用，只在 base-7 里选 slot + 抽 entities，不摊平动态 slot。
classify_child(): 分层分类"往下钻"的那一步——给定已选中的父 slot 和它的子 slot
列表，判断有没有子 slot 比父 slot 本身更精确；由调用方（core.py::Classify()）
递归调用，实现"先选大类，再层层往更精确的子类钻，钻不动就停在当前这层"的分类流程。

Returns: QueryClassification(slots=["work"], entities=["阿里"])
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from .slot_v2 import ALL_SLOT_V2_VALUES
from voicemem.llm_config import resolve_api_key, resolve_model

logger = logging.getLogger(__name__)

_BASE_SLOTS = """- work: career, job, company, projects, colleagues, workplace, 工作, 职业, 辞职, 升职
- finance: money, salary, income, expenses, investments, savings, 财务, 薪资, 投资
- relationships: friends, family, romantic, social connections, 朋友, 家人, 感情
- health: physical health, exercise, diet, sleep, medical, 健康, 运动, 生病
- goals: future plans, dreams, aspirations, self-improvement, 目标, 计划, 梦想
- daily_life: daily routines, hobbies, leisure, lifestyle, 日常, 爱好, 习惯
- knowledge: learning, concepts, skills, facts, technology, 知识, 技能, 学习"""

_SYSTEM_PROMPT_TEMPLATE = """You are a memory router. Given a user's utterance, identify:
1. slots: which life domains the query is about (pick 1-2 from the list below)
2. entities: specific named things mentioned — people, organizations, places, AND
   specific objects/assets/projects/collections the user owns or refers to
   (e.g. "my fish tank", "the coin collection", "my car", "the report").
   Do NOT include generic activities or topics with no specific referent
   (e.g. "运动" alone is not an entity, but "我的鱼缸" or "硬币收藏" is).

Available slots:
{slot_list}

ALWAYS return at least one slot — an empty list is never a valid answer.
If the utterance asks about a specific event or moment ("When did X go to the
museum?", "Where did Y camp?"), do not stop at "it's asking about a date/place".
Classify by what kind of life activity that event is:
- attending a support group / meeting friends → relationships
- running a race / a medical visit → health
- giving a talk at school / a work trip → work
- a pottery class / a museum trip / painting → daily_life
- reading a book / learning a skill → knowledge
Only when nothing fits at all, pick the single closest slot anyway.

Return JSON only: {{"slots": [...], "entities": [...]}}

Examples:
- "我想辞职" → {{"slots": ["work"], "entities": []}}
- "When did she go to the support group?" → {{"slots": ["relationships"], "entities": ["support group"]}}
- "When did he run the charity race?" → {{"slots": ["health"], "entities": ["charity race"]}}
- "最近跑步很累" → {{"slots": ["health", "daily_life"], "entities": []}}
- "和阿里的项目进展" → {{"slots": ["work"], "entities": ["阿里"]}}
- "想存钱买房" → {{"slots": ["finance", "goals"], "entities": []}}
- "妈妈身体怎么样" → {{"slots": ["health", "relationships"], "entities": ["妈妈"]}}
- "Python异步编程怎么学" → {{"slots": ["knowledge"], "entities": []}}
- "我的鱼缸里养了什么" → {{"slots": ["daily_life"], "entities": ["鱼缸"]}}
- "我硬币收藏现在有多少枚" → {{"slots": ["daily_life"], "entities": ["硬币", "硬币收藏"]}}
- "帮我看看那份报告改完了没" → {{"slots": ["work"], "entities": ["报告"]}}"""


def _build_system_prompt(extra_slots: list[tuple[str, str]] | None) -> str:
    slot_list = _BASE_SLOTS
    if extra_slots:
        extras = "\n".join(f"- {name}: {desc}" for name, desc in extra_slots)
        slot_list = slot_list + "\n" + extras
    return _SYSTEM_PROMPT_TEMPLATE.format(slot_list=slot_list)


@dataclass
class SlotClassifierConfig:
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 15.0

    def resolved_model(self) -> str:
        return resolve_model(self.model)


@dataclass
class QueryClassification:
    slots: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)

    def primary_slot(self) -> str | None:
        return self.slots[0] if self.slots else None


class QuerySlotClassifier:
    """Single-call LLM classifier: query → domain slots + named entities."""

    def __init__(self, config: SlotClassifierConfig | None = None) -> None:
        self._cfg = config or SlotClassifierConfig()
        self._model = self._cfg.resolved_model()
        self._client = self._build_client()

    def _build_client(self) -> Any:
        from openai import OpenAI
        kw: dict[str, Any] = {
            "api_key": resolve_api_key(self._cfg.api_key),
            "timeout": self._cfg.timeout,
        }
        if self._cfg.base_url:
            kw["base_url"] = self._cfg.base_url
        return OpenAI(**kw)

    def classify(
        self,
        query: str,
        extra_slots: list[tuple[str, str]] | None = None,
    ) -> QueryClassification:
        """Classify query into life-domain slots and named entities.

        extra_slots: [(name, description), ...] — 动态涌现的新 slot，追加到 prompt。
        Returns empty QueryClassification on any failure — callers must handle
        the no-slot fallback (full corpus search).
        """
        known = set(ALL_SLOT_V2_VALUES)
        if extra_slots:
            known.update(name for name, _ in extra_slots)
        system_prompt = _build_system_prompt(extra_slots)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
                max_tokens=512,
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(raw)
            slots = [s for s in (data.get("slots") or []) if s in known]
            entities = [
                str(e).strip()
                for e in (data.get("entities") or [])
                if str(e).strip()
            ]
            return QueryClassification(slots=slots[:2], entities=entities)
        except Exception as exc:
            print(f"[slot_classifier] ERROR: {exc}")
            logger.warning("QuerySlotClassifier.classify failed: %s", exc)
            return QueryClassification()

    def classify_child(
        self,
        query: str,
        parent_name: str,
        children: list[tuple[str, str]],
    ) -> str | None:
        """已经选中 parent_name 这个 slot 后，在它的子 slot 里再选一次——分层
        分类的"往下钻"这一步。children 为空、或者都不如父 slot 本身精确时，
        返回 None（调用方据此回退到 parent_name，不勉强钻到不合适的子层）。
        """
        if not children:
            return None
        child_list = "\n".join(f"- {name}: {desc}" for name, desc in children)
        prompt = (
            f"用户的问题已经被归类到「{parent_name}」这个大类下。这个大类下面还细分了"
            f"这些更具体的子话题：\n{child_list}\n\n"
            f"用户问题：{query}\n\n"
            f"这几个子话题里，有没有哪个比「{parent_name}」这个大类本身更精确地对应这个问题？"
            f"如果有，选1个最贴切的；如果都不如直接用「{parent_name}」这个大类准确"
            "（比如问题比较笼统、不特别指向任何一个细分子话题），就说没有更精确的匹配。\n"
            '只输出 JSON：{"child": "子话题名"} 或 {"child": null}'
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=256,
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(raw)
            child = data.get("child")
            valid_names = {name for name, _ in children}
            return child if child in valid_names else None
        except Exception as exc:
            print(f"[slot_classifier] classify_child ERROR: {exc}")
            logger.warning("QuerySlotClassifier.classify_child failed: %s", exc)
            return None


__all__ = [
    "QueryClassification",
    "QuerySlotClassifier",
    "SlotClassifierConfig",
]
