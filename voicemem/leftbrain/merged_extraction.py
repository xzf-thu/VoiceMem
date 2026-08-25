"""把「抽事实 / 打标签 / 右脑标签」三次 LLM 调用并成一次。

实测一次 ingest 打 12 次 chat，其中这三样吃的是**同一句话**、输出互不依赖：

    extract_facts_openai   抽出 fact 文本
    CognitiveAnnotator     给 fact 打 slot / entity / relation
    _extract_rb_traits     从原话看出「这人什么样」

合成一次调用要解决一个次序问题：annotator 的输入是 extractor 的输出，看起来必须
串行。做法是让 extractor 一次把标注也吐出来，暂存在这里；annotator 和
_extract_rb_traits 先查暂存，查得到就直接用，查不到才走自己那次 LLM 调用。

这样不用改任何函数签名，而且**坏了会自动退回旧路径**——合并输出缺字段、解析失败、
或者模型没按格式来，下游照旧各自调用一次，行为跟合并前一致。

``VOICEMEM_MERGED_EXTRACTION=0`` 关掉合并。
"""
from __future__ import annotations

import os
import threading

_lock = threading.Lock()
_annotations: dict[str, dict] = {}      # fact 文本 -> {slot, entities, relations}
_traits: dict[str, list] = {}           # 原话      -> [(slot, label), ...]
_MAX = 512


def enabled() -> bool:
    return os.environ.get("VOICEMEM_MERGED_EXTRACTION", "1") != "0"


def _put(store: dict, key: str, value) -> None:
    if not key:
        return
    with _lock:
        if len(store) >= _MAX:
            store.clear()
        store[key] = value


def _take(store: dict, key: str):
    """取出即删：一条 fact 只会被 annotate 一次，留着只会占内存。"""
    with _lock:
        return store.pop(key, None)


def put_annotation(fact_text: str, ann: dict) -> None:
    _put(_annotations, (fact_text or "").strip(), ann)


def take_annotation(fact_text: str) -> dict | None:
    return _take(_annotations, (fact_text or "").strip())


def put_traits(utterance: str, traits: list) -> None:
    _put(_traits, (utterance or "").strip(), traits)


def take_traits(utterance: str) -> list | None:
    return _take(_traits, (utterance or "").strip())


# 追加到抽取 prompt 后面的那一段。刻意写得短——这段每次 ingest 都要发一遍，
# 而且原来的抽取 prompt 已经很长了。字段名跟 annotator 的输出格式保持一致，
# 下游解析代码一行都不用改。
PROMPT_ADDENDUM = """

Additionally, for EACH item in "memory", include these three fields:
- "slot": one of [{slots}]
- "entities": [{{"name": "...", "entity_type": "user|person|project|task|knowledge|preference|place|routine|asset|organization|event", "role": "subject|object|context|owner"}}]
- "relations": [{{"from": "...", "to": "...", "relation_type": "...", "confidence": 0.9}}]
  Relation direction must match reality: a boss manages the user, not the reverse.

And add ONE top-level field "traits": subjective things this utterance reveals about
the speaker. Each item {{"slot": "...", "label": "<5-15 chars>"}}, slot is one of:
喜好与厌恶 (likes/dislikes), 表达风格 (how they communicate),
思维模式 (how they think/decide), 应对方式 (how they cope with stress).
Only include a category the utterance clearly shows. "traits": [] is fine.
Never invent entities or traits that are not in the text."""


def prompt_addendum() -> str:
    from voicemem.leftbrain.cognitive_graph.slot_v2 import ALL_SLOT_V2_VALUES
    return PROMPT_ADDENDUM.format(slots=", ".join(ALL_SLOT_V2_VALUES))
