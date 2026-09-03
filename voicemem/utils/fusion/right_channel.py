"""右脑通道：历史 anomaly 归因 + LLM 相关性筛选。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from voicemem.utils.audio.emotion.memory_store import EmotionMemoryStore
from voicemem.utils.audio.emotion.types import EmotionAttribution, VAD
from voicemem.utils.fusion.config import FusionConfig
from voicemem.utils.fusion.types import RightMemoryHit
from voicemem.llm_config import resolve_api_key, resolve_model


@runtime_checkable
class RightChannelRelevanceFilter(Protocol):
    def filter_relevant(
        self,
        *,
        asr_text: str,
        current_vad: VAD,
        candidates: list[RightMemoryHit],
    ) -> list[RightMemoryHit]:
        ...


def load_anomaly_attributions(
    store: EmotionMemoryStore,
    *,
    user_id: str,
    exclude_turn_id: str | None = None,
) -> list[RightMemoryHit]:
    mem = store.load(user_id=user_id)
    hits: list[RightMemoryHit] = []
    for attr in mem.attributions:
        if attr.trigger != "anomaly":
            continue
        if exclude_turn_id and attr.turn_id == exclude_turn_id:
            continue
        snippet = []
        if isinstance(attr.metadata, dict):
            raw = attr.metadata.get("retrieval_snippet")
            if isinstance(raw, list):
                snippet = [str(x) for x in raw if str(x).strip()]
        hits.append(
            RightMemoryHit(
                turn_id=attr.turn_id,
                analysis_text=attr.analysis_text,
                vad=attr.vad_at_trigger,
                retrieval_snippet=snippet,
                left_context_summary=attr.left_context_summary,
            )
        )
    return hits


def _format_candidates(candidates: list[RightMemoryHit]) -> str:
    lines = []
    for c in candidates:
        lines.append(
            f"- turn_id={c.turn_id}; V/A=({c.vad.valence:.2f},{c.vad.arousal:.2f}); "
            f"attribution: {c.analysis_text[:180]}"
        )
    return "\n".join(lines) if lines else "(no candidates)"


class OpenAIRightChannelRelevanceFilter:
    """用 cheap 文本模型判断历史 anomaly 归因是否与本轮 ASR/VAD 相关。"""

    def __init__(self, *, model: str | None = None, temperature: float = 0.0) -> None:
        self._model = model
        self._temperature = temperature

    def _resolved_model(self) -> str:
        return resolve_model(self._model)

    def filter_relevant(
        self,
        *,
        asr_text: str,
        current_vad: VAD,
        candidates: list[RightMemoryHit],
    ) -> list[RightMemoryHit]:
        if not candidates:
            return []
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("请安装: pip install openai>=1.0") from e

        api_key = resolve_api_key()
        if not api_key:
            raise ValueError("缺少 OPENAI_API_KEY")

        system = (
            "You are a memory-relevance judge. Given the current user ASR and VAD, plus a list of "
            "historical emotion attributions, decide which past attributions are relevant to this turn. "
            'Output JSON only: {"relevant_turn_ids": ["id1", "id2"]}'
        )
        user = (
            f"Current ASR:\n{(asr_text or '').strip()}\n\n"
            f"Current VAD: valence={current_vad.valence:.2f}, arousal={current_vad.arousal:.2f}\n\n"
            f"Historical anomaly attributions:\n{_format_candidates(candidates)}"
        )
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=self._resolved_model(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=self._temperature,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        ids = data.get("relevant_turn_ids") or []
        if not isinstance(ids, list):
            return []
        id_set = {str(x) for x in ids}
        return [c for c in candidates if c.turn_id in id_set]


@dataclass
class FixtureRightChannelRelevanceFilter:
    """单测：按 turn_id 前缀或关键字匹配。"""

    relevant_turn_ids: set[str] = field(default_factory=set)
    keyword: str | None = None

    def filter_relevant(
        self,
        *,
        asr_text: str,
        current_vad: VAD,
        candidates: list[RightMemoryHit],
    ) -> list[RightMemoryHit]:
        _ = current_vad
        if self.relevant_turn_ids:
            return [c for c in candidates if c.turn_id in self.relevant_turn_ids]
        if self.keyword:
            kw = self.keyword.lower()
            text = (asr_text or "").lower()
            if kw in text:
                return list(candidates)
            return []
        return list(candidates)


def search_right_channel(
    store: EmotionMemoryStore,
    *,
    asr_text: str,
    current_vad: VAD,
    user_id: str,
    relevance_filter: RightChannelRelevanceFilter | None = None,
    exclude_turn_id: str | None = None,
    config: FusionConfig | None = None,
) -> list[RightMemoryHit]:
    _ = config
    candidates = load_anomaly_attributions(
        store,
        user_id=user_id,
        exclude_turn_id=exclude_turn_id,
    )
    if not candidates:
        return []
    filt = relevance_filter or FixtureRightChannelRelevanceFilter()
    return filt.filter_relevant(
        asr_text=asr_text,
        current_vad=current_vad,
        candidates=candidates,
    )
