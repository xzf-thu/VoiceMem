"""SpeakerIdentity — 自我介绍解析。

用户说"我是annie"/"我叫小明"，从文本里把这个自称的名字抽出来，
调用方（core.py）负责把当前声纹的 person_id 绑定到这个名字
（VoiceprintRegistry.bind），下次同一声纹出现时无需再报名字，
ingest_voice_input() 会自动用绑定好的名字替换 "Speaker N"。
"""
from __future__ import annotations

import re

# 拉丁字母名字（如 "我是annie" / "my name is Annie"）
#
# "my name is X" / "i'm X" 这两条英文模式之前捕获组没有约束，"I'm having a
# great time" 会把 "having" 当成名字——英文里 "I'm <动词/介词>..." 这种句式
# 太常见了，跟中文不一样：中文人名后面紧跟标点/空白就是自然断句，但英文里
# 几乎每个词后面都有空白，"跟着空白"这个信号根本没有区分度。
# 改成要求：捕获的词首字母大写（真名字通常是专有名词，ASR转写一般会保留
# 大写），且后面紧跟标点/"and"/结尾（不能后面还接别的词），排除掉
# "I'm OK with that" 这种大写但明显不是名字、后面还接着说的情况。
_LATIN_PATTERNS = [
    # 收尾必须是标点/空白/结尾。少了这条，"我是Jiaqi的同学"会捕获 "Jiaqi"——
    # 说话的人明明**不是** Jiaqi，却把 Jiaqi 这个名字绑到了他自己的声纹上。
    # （中文名字那两条模式本来就有这个约束，拉丁这两条漏了。）
    re.compile(r"我(?:的名字)?是\s*([A-Za-z][A-Za-z0-9_-]{0,19})(?=[，,。.!！？?、\s]|$)"),
    re.compile(r"我叫\s*([A-Za-z][A-Za-z0-9_-]{0,19})(?=[，,。.!！？?、\s]|$)"),
    re.compile(r"\b(?i:my name is)\s+([A-Z][A-Za-z0-9_-]{0,19})(?=[,.!?]|\s+and\b|$)"),
    re.compile(r"\b(?i:i'?m)\s+([A-Z][A-Za-z0-9_-]{0,19})(?=[,.!?]|\s+and\b|$)"),
    re.compile(r"\b(?i:i am)\s+([A-Z][A-Za-z0-9_-]{0,19})(?=[,.!?]|\s+and\b|$)"),
]

# 中文名字（如 "我是小明"）——限定 2-4 字且后面紧跟标点/空白/结尾，
# 避免把"我是不是应该..."之类误判成名字
_CJK_PATTERNS = [
    re.compile(r"我(?:的名字)?是([一-鿿]{2,4})(?=[，,。.!！？?\s]|$)"),
    re.compile(r"我叫([一-鿿]{2,4})(?=[，,。.!！？?\s]|$)"),
]

# 常见误触发——捕获到的不是名字而是状态/疑问词
_STOPWORDS = {"不是", "谁啊", "什么", "怎么", "真的", "觉得", "认为", "这样", "那个"}

# 疑问词。"我叫什么名字？"问的是"你还记得我吗"，却被当成自我介绍，把名字绑成
# "什么名字"——真在库里发生过（voiceprint_registry.json 里那条 name="什么名字"）。
# 逐个列举穷举不完（什么名字/啥名字/什么来着…），直接看捕获到的名字里含不含疑问词。
_QUESTION_CHARS = "什谁哪啥吗呢么"

# ``我是`` 既可用于自我介绍，也常被用来表达状态（例如“我是担心”）。中文
# ASR 不一定提供可靠的词性信息，因此宁可不自动绑定，也不能把情绪/动作永久
# 写成声纹姓名；用户仍可通过明确的“我叫…”完成绑定。
_CJK_NON_NAMES = {
    "担心", "焦虑", "害怕", "难过", "开心", "高兴", "生气", "疲惫",
    "紧张", "着急", "困惑", "烦躁", "出门", "回家", "工作", "上班",
}

# 英文短句 ``I'm X.`` 只在 X 是句末的专有名词形式时才会进入这里，但 ASR
# 可能把普通词首字母大写。把已知的状态、动作和对话词排除，防止一次误识别
# 污染之后所有该声纹的记忆归属。
_LATIN_NON_NAMES = {
    "actually", "afraid", "alone", "angry", "anxious", "busy", "confused",
    "excited", "fine", "going", "happy", "heading", "here", "hungry",
    "leaving", "lost", "nervous", "okay", "ok", "ready", "sad", "sorry",
    "tired", "walking", "worried",
}


def parse_self_identification(text: str) -> str | None:
    """从一句话里抽取自我介绍的名字，识别不到返回 None。

    Examples::
        "我是annie，最近工作很累" → "annie"
        "我叫小明"                → "小明"
        "我是不是应该走了"        → None
        "我叫什么名字？"          → None（问句，不是自我介绍）
        "我是Jiaqi的同学"        → None（说话的不是 Jiaqi）
    """
    for pattern in _LATIN_PATTERNS + _CJK_PATTERNS:
        m = pattern.search(text)
        if m:
            name = m.group(1).strip()
            if (
                name
                and name not in _STOPWORDS
                and name not in _CJK_NON_NAMES
                and name.casefold() not in _LATIN_NON_NAMES
                and not any(c in _QUESTION_CHARS for c in name)
            ):
                return name
    return None
