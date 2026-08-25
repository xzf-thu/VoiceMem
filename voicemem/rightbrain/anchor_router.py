"""Anchor Router：把用户输入和 Preprocessor 信号转换为 MemoryQueryPlan。

核心逻辑：
  1. 复用左脑 CognitiveGraphStore 的 entity 匹配，不额外调 LLM
  2. entity.entity_type → anchor_type（人 → person，项目 → project …）
  3. entity_edges → entity_edge anchor（Boss给任务 比 Boss本人 更精准）
  4. 无命中时加 user_self + global_style fallback

不依赖右脑表，只读左脑 SQLite。
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .types import CurrentSignals, MemoryAnchor, MemoryQueryPlan

if TYPE_CHECKING:
    from voicemem.leftbrain.cognitive_graph.store import CognitiveGraphStore

# 左脑 EntityType.value → MemoryAnchor anchor_type
# 原来这里按左脑的 SlotType（一套独立于 SlotV2 的 7 类 entity-category
# taxonomy）映射；SlotType 已随 slot taxonomy 统一（改用 SlotV2 表示"生活
# 领域"）被删除，而这里真正需要的一直是"这个实体是什么类型"——EntityType
# 本来就直接表达这个，不需要再经过 slot 转译一层。
# organization 并入 person（原 ENTITY_TYPE_TO_SLOT 里 ORGANIZATION 也是并入
# people 槽位）；event 并入 knowledge（原映射同样把 EVENT 归到 knowledge）。
# user/preference 类实体不在表里，走下面 .get(..., "knowledge") 的默认值，
# 跟原来 SlotType 路径的兜底行为一致。
_ENTITY_TYPE_TO_ANCHOR: dict[str, str] = {
    "person":       "person",
    "organization": "person",
    "project":      "project",
    "task":         "task",
    "knowledge":    "knowledge",
    "event":        "knowledge",
    "place":        "place",
    "routine":      "routine",
    "asset":        "asset",
}

# anchor_type → role 默认值
_ANCHOR_ROLE: dict[str, str] = {
    "person":       "subject",
    "user":         "global_profile",
    "project":      "context",
    "task":         "topic",
    "knowledge":    "context",
    "place":        "context",
    "routine":      "context",
    "asset":        "context",
    "entity_edge":  "trigger",
    "global_style": "global_profile",
}

# anchor_type → 检索权重
_ANCHOR_WEIGHT: dict[str, float] = {
    "task":         1.0,
    "entity_edge":  0.9,
    "project":      0.9,
    "person":       0.7,
    "knowledge":    0.6,
    "place":        0.5,
    "routine":      0.5,
    "asset":        0.4,
    "user":         0.3,
    "global_style": 0.3,
}

_CANONICAL_EMOTIONS = {"焦虑", "悲伤", "委屈", "孤独", "纠结", "平静", "开心", "疲惫"}

_EMOTION_KEYWORDS: list[tuple[str, str]] = [
    # 焦虑系
    ("焦虑", "焦虑"), ("压力", "焦虑"), ("紧张", "焦虑"), ("担忧", "焦虑"),
    ("恐惧", "焦虑"), ("害怕", "焦虑"), ("不安", "焦虑"), ("慌", "焦虑"),
    # 悲伤系
    ("悲伤", "悲伤"), ("难过", "悲伤"), ("失落", "悲伤"), ("沮丧", "悲伤"),
    ("伤心", "悲伤"), ("绝望", "悲伤"), ("崩溃", "悲伤"), ("难受", "悲伤"),
    # 委屈系
    # "生气/火大/烦躁" 以前一个都不在表里，而"压力"在焦虑系——于是"我好生气啊，
    # 我的老板老压力我"被判成【焦虑】，明说了生气却认不出。表是按顺序匹配的，
    # 所以这几个要排在"压力"之类的泛词之前才盖得住（见下面 _EMOTION_KEYWORDS
    # 的用法）。
    ("委屈", "委屈"), ("愤怒", "委屈"), ("气愤", "委屈"), ("不满", "委屈"),
    ("生气", "委屈"), ("火大", "委屈"), ("恼火", "委屈"), ("烦躁", "委屈"),
    ("憋屈", "委屈"), ("不公平", "委屈"),
    # 孤独系
    ("孤独", "孤独"), ("空虚", "孤独"), ("寂寞", "孤独"),
    # 纠结系
    ("纠结", "纠结"), ("矛盾", "纠结"), ("迷茫", "纠结"), ("犹豫", "纠结"),
    # 平静系
    ("平静", "平静"), ("淡然", "平静"), ("冷静", "平静"), ("释然", "平静"),
    ("坦然", "平静"),
    # 开心系
    ("开心", "开心"), ("高兴", "开心"), ("兴奋", "开心"), ("期待", "开心"),
    ("愉快", "开心"), ("满足", "开心"), ("自豪", "开心"), ("憧憬", "开心"),
    ("感激", "开心"), ("轻松", "开心"), ("坚定", "开心"),
    # 疲惫系
    ("疲惫", "疲惫"), ("疲倦", "疲惫"), ("困倦", "疲惫"), ("无力", "疲惫"),
    ("累", "疲惫"),
]

# 英文版——VoiceMem 中英文都要支持，emotion 标签不能只认中文关键词，
# 否则纯英文场景（如 ASR/TTS 生成的英文 emotion_tag）全部误判成"平静"。
_EMOTION_KEYWORDS_EN: list[tuple[str, str]] = [
    # anxious
    ("anxious", "焦虑"), ("anxiety", "焦虑"), ("nervous", "焦虑"), ("worried", "焦虑"),
    ("worry", "焦虑"), ("stressed", "焦虑"), ("stress", "焦虑"), ("tense", "焦虑"),
    ("fearful", "焦虑"), ("afraid", "焦虑"), ("scared", "焦虑"), ("panicked", "焦虑"),
    ("panic", "焦虑"), ("uneasy", "焦虑"), ("apprehensive", "焦虑"),
    # sad
    ("sad", "悲伤"), ("sadness", "悲伤"), ("upset", "悲伤"), ("depressed", "悲伤"),
    ("disappointed", "悲伤"), ("heartbroken", "悲伤"), ("miserable", "悲伤"),
    ("dejected", "悲伤"), ("despair", "悲伤"), ("sorrowful", "悲伤"), ("grief", "悲伤"),
    # wronged / angry
    ("wronged", "委屈"), ("angry", "委屈"), ("anger", "委屈"), ("mad", "委屈"),
    ("furious", "委屈"), ("irritated", "委屈"), ("annoyed", "委屈"), ("frustrated", "委屈"),
    ("resentful", "委屈"), ("indignant", "委屈"), ("unfair", "委屈"), ("bitter", "委屈"),
    # lonely
    ("lonely", "孤独"), ("loneliness", "孤独"), ("isolated", "孤独"), ("empty", "孤独"),
    ("alone", "孤独"),
    # conflicted
    ("conflicted", "纠结"), ("torn", "纠结"), ("confused", "纠结"), ("uncertain", "纠结"),
    ("hesitant", "纠结"), ("ambivalent", "纠结"), ("indecisive", "纠结"), ("perplexed", "纠结"),
    ("lost", "纠结"),
    # calm
    ("calm", "平静"), ("relaxed", "平静"), ("peaceful", "平静"), ("composed", "平静"),
    ("serene", "平静"), ("settled", "平静"), ("neutral", "平静"),
    # happy
    ("happy", "开心"), ("happiness", "开心"), ("joy", "开心"), ("joyful", "开心"),
    ("excited", "开心"), ("excitement", "开心"), ("glad", "开心"), ("pleased", "开心"),
    ("delighted", "开心"), ("proud", "开心"), ("grateful", "开心"), ("thankful", "开心"),
    ("relieved", "开心"), ("hopeful", "开心"), ("cheerful", "开心"), ("satisfied", "开心"),
    ("content", "开心"), ("amused", "开心"),
    # tired
    ("tired", "疲惫"), ("exhausted", "疲惫"), ("fatigue", "疲惫"), ("fatigued", "疲惫"),
    ("weary", "疲惫"), ("drained", "疲惫"), ("sleepy", "疲惫"), ("worn out", "疲惫"),
]

_EN_KEYWORD_RE: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE), canonical)
    for kw, canonical in _EMOTION_KEYWORDS_EN
]


def normalize_emotion_strict(emotion: str) -> str | None:
    """Map a free-form emotion string (Chinese or English) to a canonical
    label; return None when nothing matches.

    锚点相关的调用方应该用这个版本：未识别的情绪词（guilty / jealous /
    nostalgic…）以前一律被兜底成"平静"，再以最高权重(1.2)写进检索锚点——
    等于往查询里注入一个高权重的错误信号。识别不出就不加 emotion 锚点，
    比加一个错的强。"""
    e = emotion.strip()
    if not e:
        return None
    if e in _CANONICAL_EMOTIONS:
        return e
    # 按**词在句子里出现的位置**取最早的那个，不按表的顺序。
    # 表的顺序是分组写的（焦虑系在前、委屈系在后），照表序匹配的话
    # 「我好生气啊，我的老板老压力我」会先撞上"压力"→【焦虑】，而人开口第一个词
    # 就是"生气"。人说话时最先说出来的情绪词通常就是主情绪。
    best = None
    for keyword, canonical in _EMOTION_KEYWORDS:
        i = e.find(keyword)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, canonical)
    if best is not None:
        return best[1]
    for pattern, canonical in _EN_KEYWORD_RE:
        if pattern.search(e):
            return canonical
    return None


def normalize_emotion(emotion: str) -> str:
    """Like normalize_emotion_strict, but falls back to 平静 for callers that
    need a guaranteed canonical label (e.g. graph node naming)."""
    return normalize_emotion_strict(emotion) or "平静"


_STOP = {
    "who", "is", "are", "was", "were", "what", "when", "where", "why",
    "how", "the", "a", "an", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "my", "your", "his", "her", "their", "our", "tell",
    "me", "about", "did", "do", "does", "has", "have", "had", "can",
    "could", "would", "should", "with", "from", "that", "this", "it",
    "be", "been", "being", "not", "no", "any", "some", "which",
}


class AnchorRouter:
    """根据当前输入生成 MemoryQueryPlan。

    cognitive_store 可以为 None（此时只返回 fallback anchors）。
    """

    def __init__(
        self,
        cognitive_store: "CognitiveGraphStore | None" = None,
    ) -> None:
        self._store = cognitive_store

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
        """``context``：agent 上一句。用户这句往往要放回它里面才完整（"那算了"），
        所以它提到的实体也进锚点，但降为 context 角色、权重减半——它是背景，
        不是用户这轮的主语。clean_text 仍只是用户原话。"""
        anchors = self._build_anchors(
            query, user_id, hint_entities=entities, hint_emotion=emotion,
            context_text=context,
        )
        return MemoryQueryPlan(
            user_id=user_id,
            clean_text=query.strip(),
            anchors=anchors,
            current_signals=signals or CurrentSignals(),
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    # agent 那句话里匹配到的实体，锚点权重打这个折（背景 < 用户这轮的主语）
    _CONTEXT_WEIGHT_SCALE = 0.5

    def _build_anchors(self, query: str, user_id: str, hint_entities: list[str] | None = None,
                       hint_emotion: str | None = None,
                       context_text: str | None = None) -> list[MemoryAnchor]:
        anchors: list[MemoryAnchor] = []
        seen_ids: set[str] = set()

        if self._store is not None:
            from voicemem.leftbrain.cognitive_graph.store import normalize_name

            # 先扫用户那句：两边都提到的实体先被满权重占住，不会被背景那遍降下去
            sources: list[tuple[str, float]] = [(query, 1.0)]
            if context_text and context_text.strip():
                sources.append((context_text, self._CONTEXT_WEIGHT_SCALE))

            matched_entities: list[tuple[Any, float]] = []
            all_ents = self._store.find_entities(user_id)   # 中文反查用，两遍共用一次查询
            for text, scale in sources:
                # 候选词：英文单词 + bigram（原有逻辑）
                raw_words = re.findall(r"\b\w{2,}\b", text.lower())
                candidates = [w for w in raw_words if w not in _STOP]
                for i in range(len(raw_words) - 1):
                    a, b = raw_words[i], raw_words[i + 1]
                    if a not in _STOP and b not in _STOP:
                        candidates.append(f"{a} {b}")

                for cand in candidates:
                    ents = self._store.find_entities_by_name_fuzzy(user_id, cand)
                    for e in ents:
                        if e.id not in seen_ids:
                            seen_ids.add(e.id)
                            matched_entities.append((e, scale))

                # 中文反查：遍历所有实体名，检查是否出现在输入中
                # （中文无空格分词，英文词边界正则对中文无效）
                text_lower = text.lower()
                for e in all_ents:
                    if e.id in seen_ids:
                        continue
                    name_l = e.name.lower()
                    name_n = (e.name_norm or "").lower()
                    if len(name_l) >= 2 and (name_l in text_lower or name_n in text_lower):
                        seen_ids.add(e.id)
                        matched_entities.append((e, scale))

            # voice module 提供的 entities：直接用名字做 anchor（和 Ingest 写入一致）
            if hint_entities:
                for name in hint_entities:
                    key = name.lower().strip()
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                    anchors.append(MemoryAnchor(
                        anchor_type="entity",
                        anchor_id=key,
                        role="subject",
                        weight=1.0,
                        confidence=1.0,
                    ))

            for ent, scale in matched_entities:
                anchor_type = _ENTITY_TYPE_TO_ANCHOR.get(ent.entity_type.value, "knowledge")
                anchors.append(MemoryAnchor(
                    anchor_type=anchor_type,
                    anchor_id=ent.id,
                    # 只在 agent 那句里出现的：记 context，权重减半
                    role=("context" if scale < 1.0
                          else _ANCHOR_ROLE.get(anchor_type, "context")),
                    weight=_ANCHOR_WEIGHT.get(anchor_type, 0.5) * scale,
                    confidence=ent.confidence,
                ))
                # 右脑写入时实体锚点用的是 name.lower().strip()（core.py::Ingest 里的
                # entity anchor_id 写法），不是左脑的 entity.id——两套ID体系不通。
                # 这里额外补一个用同样规则归一化的 "entity" 锚点，
                # 让模糊匹配到的实体也能查到当初写入时挂的锚点。
                name_key = ent.name.lower().strip()
                if name_key not in seen_ids:
                    seen_ids.add(name_key)
                    anchors.append(MemoryAnchor(
                        anchor_type="entity",
                        anchor_id=name_key,
                        role="subject",
                        weight=1.0 * scale,
                        confidence=ent.confidence,
                    ))

            # entity_edge anchors：命中 ≥2 个实体时，把它们之间的边也加进来
            if len(matched_entities) >= 2:
                entity_ids = [e.id for e, _ in matched_entities]
                for e, _scale in matched_entities:
                    edges = self._store.edges_for_entity(e.id, user_id)
                    for edge in edges:
                        if (edge.from_entity_id in entity_ids
                                and edge.to_entity_id in entity_ids
                                and edge.id not in seen_ids):
                            seen_ids.add(edge.id)
                            anchors.append(MemoryAnchor(
                                anchor_type="entity_edge",
                                anchor_id=edge.id,
                                role="trigger",
                                weight=_ANCHOR_WEIGHT["entity_edge"],
                                confidence=edge.confidence,
                            ))

        # emotion anchor：按当前情感检索过去的情感事件（最高权重）。
        # strict 版：识别不出的情绪词不加锚点（而不是兜底成"平静"）。
        if hint_emotion:
            canonical = normalize_emotion_strict(hint_emotion)
            if canonical is not None:
                anchors.append(MemoryAnchor(
                    anchor_type="emotion",
                    anchor_id=canonical,
                    role="trigger",
                    weight=1.2,
                    confidence=1.0,
                ))

        # Fallback：user_self + global_style 始终加入（权重低）
        anchors.append(MemoryAnchor(
            anchor_type="user",
            anchor_id="user_self",
            role="global_profile",
            weight=_ANCHOR_WEIGHT["user"],
            confidence=1.0,
        ))
        anchors.append(MemoryAnchor(
            anchor_type="global_style",
            anchor_id="global_style",
            role="global_profile",
            weight=_ANCHOR_WEIGHT["global_style"],
            confidence=1.0,
        ))

        return anchors
