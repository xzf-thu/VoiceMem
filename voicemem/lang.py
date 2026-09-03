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


def _set(lang: str) -> None:
    global _override
    _override = lang


def set_memory_language(value: str | None) -> None:
    """进程级设置。None / "" 表示清掉覆盖，回到 env / 默认。"""
    global _override
    _override = _check(value) if (value or "").strip() else None


def memory_language() -> str:
    if _override:
        return _override
    env = (os.environ.get(ENV, "") or "").strip()
    return _check(env) if env else DEFAULT


def resolve_for_space(memory_root, explicit: str | None = None) -> str:
    """把这个实例的语言定下来，并落到它对应的**空间**上。

    ``VoiceMem(memory_language=...)`` 以前只写进程级 override，于是：先建一个
    zh 实例、再建一个不传参数的实例，后者会继承 zh——文档写的默认 en 变成了
    取决于构造顺序（issue #9）。根子是同一个概念存了两个地方：demo 那边从空间
    json 读，库这边只改全局。

    这里统一到空间上：

        显式给了 → 写进这个空间的 json，并生效
        没给     → 读这个空间自己的记录；空间没记录再回落 env / 默认，
                   并把结果写回去（建的时候定一次，之后不再变）

    两个实例各读各的空间，构造顺序不再影响任何东西。
    """
    import json as _json
    from voicemem.utils.common import space as _space
    try:
        path = _space.json_path(memory_root)
    except Exception:
        # 拿不到空间目录（极少数纯内存用法）：退回原来的全局行为
        set_memory_language(explicit)
        return memory_language()

    stored = ""
    try:
        if path.exists():
            stored = (_json.loads(path.read_text(encoding="utf-8"))
                      .get("space", {}).get("language", "") or "")
    except Exception:
        stored = ""

    if explicit:
        lang = _check(explicit)
    elif stored:
        lang = _check(stored)
    else:
        env = (os.environ.get(ENV, "") or "").strip()
        lang = _check(env) if env else DEFAULT

    if lang != stored:
        try:
            doc = (_json.loads(path.read_text(encoding="utf-8"))
                   if path.exists() else {})
            doc.setdefault("space", {})["language"] = lang
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        except Exception as e:
            print(f"[lang] 写空间语言失败（不影响使用）：{e}", flush=True)

    _set(lang)
    return lang


def is_zh() -> bool:
    return memory_language() == "zh"


def label_rule() -> str:
    """拼进 prompt 的一句语言要求。所有存进记忆的自由文本都该带上它。"""
    lang = "Chinese" if is_zh() else "English"
    return (f"Write every label in {lang}, whatever language the speaker used. "
            f"Do not mix in any other language.")


#: 8 个规范情绪（内部值一律中文，见 anchor_router._CANONICAL_EMOTIONS）→ 展示词。
#:
#: 内部值不能翻译：右脑的锚点匹配、配额、脑图聚类都按它做键。但**存进记忆、给
#: 用户看的那一份**要跟库语言一致，否则英文库里会冒出「开心」「平静」。
#: 这里挑的英文词都在 anchor_router._EMOTION_KEYWORDS_EN 里，所以英文标签再被
#: 读回来时能正确归一回同一个内部值，不会丢。
_EMOTION_EN = {
    "焦虑": "anxious", "悲伤": "sad", "委屈": "wronged", "孤独": "lonely",
    "纠结": "conflicted", "平静": "calm", "开心": "happy", "疲惫": "tired",
}


def display_emotion(canonical: str) -> str:
    """规范情绪 → 当前库语言下的写法。不认识的原样返回。"""
    if is_zh():
        return canonical
    return _EMOTION_EN.get(canonical, canonical)
