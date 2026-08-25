"""本地 query 分类器：query → slots(+entities)，全程不碰 LLM/网络。

和内置 ``QuerySlotClassifier``（单次 LLM）接口对齐，可通过 ``VoiceMem(schema=...)``
注入替换它，把 ``Classify`` 这一步（抽 slot + entity）从 OpenAI API 切成本地模型——
和 ``embedding`` 注入完全对称：

    from voicemem import VoiceMem
    from voicemem.leftbrain.cognitive_graph.local_query_classifier import LocalQueryClassifier
    vm = VoiceMem(schema=lambda: LocalQueryClassifier())   # slots 走本地 E5，0 LLM
    vm.search("我在哪工作")                                  # Classify 不再打 LLM

设计取舍（都如实说明，不藏）：
- **slots**：E5 余弦 vs 7 个 base-7 槽描述取 top-k（实测 ~93% 和 LLM 一致）。
- **entities**：默认空 → 走 voicemem 既有的 slot-only 缩窄（不是新的坏状态）。开放域
  实体识别**不是必须 LLM**：传 ``ner=<callable: query -> list[str]>`` 即可接本地
  NER（gliner / spaCy 等）把实体也本地化。
- **不实现 classify_child**：涌现子 slot 的“下钻”交给 LLM 版；子 slot 的诞生本就在
  写入侧维护，不在 search 热路径。``engine.Classify`` 检测到没有 ``classify_child``
  会自动跳过下钻，只走 base-7。

E5 的 ``"query: "`` / ``"passage: "`` 前缀是必须的（不是装饰）。模型首次用自动下载；
传 ``model=`` 可复用一个已加载的 SentenceTransformer（如和本地 embedder 共享，省一份内存）。
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QueryClassification

# 跟记忆向量共用同一份 E5（models/embedding/），省一份权重
def _model_name() -> str:
    from voicemem.utils.common.paths import hf_model
    return hf_model("embedding", "intfloat/multilingual-e5-small", "VOICEMEM_E5_MODEL")


_MODEL_NAME = _model_name()

# 和内置 LLM 分类器同一套 base-7 槽描述（故意保持一致，是同一个分类决策的本地近似）。
_SLOT_DESCRIPTIONS = {
    "work": "career, job, company, projects, colleagues, workplace, 工作, 职业, 辞职, 升职",
    "finance": "money, salary, income, expenses, investments, savings, 财务, 薪资, 投资",
    "relationships": "friends, family, romantic, social connections, 朋友, 家人, 感情",
    "health": "physical health, exercise, diet, sleep, medical, 健康, 运动, 生病",
    "goals": "future plans, dreams, aspirations, self-improvement, 目标, 计划, 梦想",
    "daily_life": "daily routines, hobbies, leisure, lifestyle, 日常, 爱好, 习惯",
    "knowledge": "learning, concepts, skills, facts, technology, 知识, 技能, 学习",
}


class LocalQueryClassifier:
    """query → QueryClassification(slots, entities)，本地 E5，无 LLM/网络。"""

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        model=None,
        ner: Callable[[str], Sequence[str]] | None = None,
        top_k: int = 2,
    ) -> None:
        self._model_name = model_name
        self._model = model                 # 可复用已加载的 SentenceTransformer
        self._ner = ner                     # 可选本地实体识别（默认无 → slot-only）
        self._top_k = top_k
        self._slot_names = list(_SLOT_DESCRIPTIONS)
        self._slot_embs = None              # 懒算一次的槽描述向量

    def _m(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _slots_matrix(self):
        if self._slot_embs is None:
            texts = [f"passage: {k}: {v}" for k, v in _SLOT_DESCRIPTIONS.items()]
            self._slot_embs = np.asarray(self._m().encode(texts, normalize_embeddings=True))
        return self._slot_embs

    def classify(self, query: str, extra_slots=None) -> QueryClassification:
        """base-7 里挑 top-k 个 slot（E5 余弦）+ 可选本地实体。extra_slots（动态子
        slot 候选）本地版忽略——子 slot 下钻交给 LLM 版/写入侧维护。"""
        q = np.asarray(self._m().encode([f"query: {query}"], normalize_embeddings=True)[0])
        order = np.argsort(-(self._slots_matrix() @ q))[: self._top_k]
        slots = [self._slot_names[i] for i in order]
        entities = list(self._ner(query)) if self._ner else []
        return QueryClassification(slots=slots, entities=entities)
