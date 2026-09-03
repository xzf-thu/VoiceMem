"""存进记忆的文本用什么语言。

只有两个值：``en``（默认）和 ``zh``。

左脑早就有 ``use_input_language``（``extract_facts_openai``），抽出来的事实跟着
用户说话的语言走。右脑没有对应的东西：特质标签和情绪词的 prompt 里写死了"中文，
5-15字"，于是英文用户的记忆库里事实是英文、画像是中文——这份画像每轮都会拼进
system prompt 给回复模型，中文就这样漏进了英文对话。

**为什么不做"跟随用户语言"**：一个记忆库应该只有一种语言。跟随输入意味着同一
个库里中英混存，检索是按向量做的，中文问句和英文记忆在向量空间里离得很远，
混存等于让一半记忆检索不到。语言是**库的属性**，不是单句的属性。

**不受这个开关影响的两样东西**，它们是内部枚举、不是给人读的文本：

  · slot 名（情绪 / 表达风格 / 思维模式 / 应对方式 / 喜好与厌恶）——检索、配额、
    脑图聚类都按它做键，翻译了会全线对不上。
  · 8 个规范情绪（焦虑/悲伤/委屈/孤独/纠结/平静/开心/疲惫）——``anchor_router``
    里有中英两张关键词表往它们上归一，所以模型**输出**英文情绪词没问题，
    归一之后落到的仍是这 8 个内部值。
"""

from __future__ import annotations

import os

#: 环境变量名。也可以 VoiceMem(memory_language="zh")、demo 的 --lang。
ENV = "VOICEMEM_MEMORY_LANGUAGE"
SUPPORTED = ("en", "zh")
DEFAULT = "en"

_override: str | None = None


def _check(value: str) -> str:
    v = (value or "").strip().lower()
    if v not in SUPPORTED:
        raise ValueError(f"memory_language 只能是 {' / '.join(SUPPORTED)}，收到 {value!r}")
    return v


def set_memory_language(value: str | None) -> None:
    """进程级设置。None / "" 表示不覆盖，继续按 env。"""
    global _override
    _override = _check(value) if (value or "").strip() else None


def memory_language() -> str:
    if _override:
        return _override
    env = (os.environ.get(ENV, "") or "").strip()
    return _check(env) if env else DEFAULT


def is_zh() -> bool:
    return memory_language() == "zh"


def label_rule() -> str:
    """拼进 prompt 的一句语言要求。所有存进记忆的自由文本都该带上它。"""
    lang = "Chinese" if is_zh() else "English"
    return (f"Write every label in {lang}, whatever language the speaker used. "
            f"Do not mix in any other language.")
