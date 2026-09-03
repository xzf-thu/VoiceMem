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


# 情绪和特质是**整段**的，不挂在某条 fact 上，而且抽取和右脑写入是同一轮、同一
# 个线程里前后脚发生的。原来按原话做 key，结果取不到——抽取拿到的原话带着
# "Speaker 0: " 前缀，跟右脑那边收到的 text 对不上。改成线程局部的「上一次」：
# 同一线程内前后脚匹配，多线程（评测并发跑）之间互不干扰。
_local = threading.local()


def put_emotion(utterance: str, emo: str) -> None:
    _local.emotion = emo


def take_emotion(utterance: str = "") -> str | None:
    v = getattr(_local, "emotion", None)
    _local.emotion = None          # 取出即清，别让上一轮的漏到下一轮
    return v


def put_traits(utterance: str, traits: list) -> None:
    _local.traits = traits


def take_traits(utterance: str = "") -> list | None:
    v = getattr(_local, "traits", None)
    _local.traits = None
    return v


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

{traits_body}

Outside that case, only include a category the utterance clearly shows —
for a one-off event or a plain question, "traits": [] is the right answer.

Each label becomes the TITLE of a node on a graph, so write it as a short
pattern — roughly 5-15 characters (or 3-8 words), no subject, no full stop.

{label_rule}

Copy the SHAPE of these, never the wording:
{label_examples}

Also add ONE top-level field "emotion": how the speaker feels, as a SINGLE
word, in the speaker's own language (中文：愉悦/开心/平静/焦虑/难过/委屈/愤怒/
惊讶/疲惫/失望…; English: glad/calm/anxious/sad/wronged/angry/tired/…).
**Judge from what they actually say.** "我喜欢草莓" is 愉悦, not 悲伤;
"我好生气啊" is 愤怒, not 焦虑. If the utterance carries no clear feeling
(a plain fact, a question), return "" — an empty string is the right answer
far more often than a guess. A wrong label is worse than none: it gets shown
to the user as 【label】 next to their own words.

Never invent entities, traits or feelings that are not in the text.

Keep one-off requests OUT of "memory": asking for a recommendation, asking what
you remember about them, asking you to do something right now. When the same
sentence ALSO states a lasting fact, write only the lasting half — never both in
one item. "我下周要考GRE，你有什么书推荐吗" gives exactly one memory,
「用户下周要参加GRE考试」, and nothing about the book request.

OUTPUT SHAPE — your JSON object must have EXACTLY these three top-level keys:
{{"memory": [...], "emotion": "...", "traits": [...]}}
The prompt above describes only the "memory" key. "emotion" and "traits" are
REQUIRED as well; omitting them is an error. Use "" and [] when there is nothing.

════════════════════════════════════════════════════════════════════════
LANGUAGE — this overrides every example above.
{label_rule}
It applies to EVERY string you output: the "memory" texts, every trait
"label", and "emotion". The only exception is "slot", which is an internal
key and must stay exactly as listed (情绪 / 应对方式 / 表达风格 / 思维模式 /
喜好与厌恶). If the speaker wrote in English, every one of those strings must
be English — copying the Chinese wording from the examples is a mistake.
════════════════════════════════════════════════════════════════════════"""


#: 示例只给**一种**语言，而且要跟这一轮说的话同语言。
#:
#: 曾经把中英示例并列写进去，结果比只有中文示例更糟：模型连左脑的事实都跟着写
#: 成中文了。示例对输出的牵引力比规则强得多——所以规则要留，但示例必须先选对。
#: traits 说明块的中英两份。整块换，不只换末尾那组好/差示例——slot 说明里
#: 自带的例子（"评审前会紧张""被打断就烦"）才是模型照抄的来源，实测只改末尾
#: 那组没用，英文输入照样存中文。
_TRAITS_BODY = {"zh": """  情绪        WHEN they feel WHAT — the situation plus the feeling it triggers.
              "评审前会紧张", "被打断就烦", "项目延期会焦虑"
  应对方式     what they DO about a feeling, or how they want to be treated.
              "压力大时想被安抚", "难受时想一个人待着"
  表达风格     habits of speaking and communicating
  思维模式     how they think, weigh things, decide
  喜好与厌恶   what they like or dislike

情绪 vs 应对方式 is the one people get wrong: "被打断就烦" is 情绪 (a feeling
appearing), "被打断了就先走开" is 应对方式 (an action taken). If the label has no
verb of doing or wanting in it, it is 情绪.

For "情绪" the label must read as **a pattern, not a bare feeling word**:
"评审前会紧张", "被打断就烦", "一个人待着会踏实" — NOT "焦虑" / "开心".
It becomes the title of a node on a graph; a bare word tells the user nothing.

When the utterance states a RECURRING tendency about the speaker — "一…就…",
"每次…都…", "总是", "从来不", "我这人…", or any habit/reaction that clearly
holds beyond this one moment — a trait is REQUIRED. "我一开长会就走神" is
喜好与厌恶「不喜欢开长会」; "我每次汇报前都睡不着" is 情绪「汇报前会睡不着」.
Do not skip it just because the same content also went into "memory": "memory"
records WHAT HAPPENED, "traits" records WHAT THIS PERSON IS LIKE, and one
sentence very often carries both.

""", "en": """  情绪        WHEN they feel WHAT — the situation plus the feeling it triggers.
              "tense before design reviews", "annoyed when interrupted",
              "anxious when a project slips"
  应对方式     what they DO about a feeling, or how they want to be treated.
              "wants comfort under stress", "needs to be alone when upset"
  表达风格     habits of speaking and communicating
  思维模式     how they think, weigh things, decide
  喜好与厌恶   what they like or dislike

情绪 vs 应对方式 is the one people get wrong: "annoyed when interrupted" is 情绪
(a feeling appearing), "walks away when interrupted" is 应对方式 (an action
taken). If the label has no verb of doing or wanting in it, it is 情绪.

For "情绪" the label must read as **a pattern, not a bare feeling word**:
"tense before design reviews", "annoyed when interrupted", "calm when alone"
— NOT "anxious" / "happy". It becomes the title of a node on a graph; a bare
word tells the user nothing.

When the utterance states a RECURRING tendency about the speaker — "every time
…", "always", "never", "I'm the kind of person who…", or any habit/reaction
that clearly holds beyond this one moment — a trait is REQUIRED. "I zone out in
long meetings" is 喜好与厌恶 "dislikes long meetings"; "I can't sleep before a
demo" is 情绪 "can't sleep before a demo".
Do not skip it just because the same content also went into "memory": "memory"
records WHAT HAPPENED, "traits" records WHAT THIS PERSON IS LIKE, and one
sentence very often carries both.

"""}

_EXAMPLES = {
    "zh": ("  好：讨厌被打断 / 压力大时想被安抚 / 先要结论再要解释\n"
           "  差：用户倾向于详细规划和结构化思考。（带主语的整句）\n"
           "  差：我是计算机专业（照抄原话/事实）"),
    "en": ("  good: hates being interrupted / wants comfort under stress / "
           "conclusion first\n"
           "  bad: The user tends to plan in detail. (a full sentence with a subject)\n"
           "  bad: I major in computer science (copying the utterance / a plain fact)"),
}


def prompt_addendum() -> str:
    """追加到**用户消息**末尾的那一段。

    注意是用户消息，不是 system。放 system 末尾时模型只认里面的 per-item 字段
    （slot/entities 出得来），顶层的 emotion/traits 一律丢掉——用户消息里那份输出
    格式说明写死了顶层只有 "memory"，模型严格照做。于是右脑每轮拿到的都是
    emotion="" + traits=[]，write() 直接早退，脑图一个节点都不长。
    最后那段 OUTPUT SHAPE 就是为此显式重申顶层结构，别删。
    """
    from voicemem.leftbrain.cognitive_graph.slot_v2 import ALL_SLOT_V2_VALUES
    from voicemem.lang import label_rule, memory_language
    key = memory_language()
    return PROMPT_ADDENDUM.format(slots=", ".join(ALL_SLOT_V2_VALUES),
                                  label_rule=label_rule(),
                                  label_examples=_EXAMPLES[key],
                                  traits_body=_TRAITS_BODY[key])
