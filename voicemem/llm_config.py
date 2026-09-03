"""模型与端点的唯一解析口。

原来每个组件各写各的：有的 ``构造参数 → env → 默认``，有的只读 env，有的直接
写死模型名；env 变量名也散着（``OPENAI_MODEL`` / ``OPENAI_CHAT_MODEL`` /
``OPENAI_EMBEDDING_MODEL`` / ``OPENAI_TTS_MODEL`` / ``OPENAI_REALTIME_MODEL``），
其中两个还是 **import 时**读的——设完 env 再 import 才生效，顺序反了就静默失效。
结果是"换个模型"这件事要改好几个地方，漏一个就成了配置只生效一半。

这里把它收成五个**角色**（role），每个角色都可选、可分别配：

    role        默认                      谁在用
    ─────────────────────────────────────────────────────────────────────────
    chat        gpt-4o-mini               后台整理记忆：抽取 / 标注 / 槽位分类 /
                                          摘要 / 内心 OS / 右脑清洁
    reply       跟随 chat                 用户直接听得见的回复
    embedding   text-embedding-3-small
    tts         gpt-4o-mini-tts           （本地后端这里填的是 voice 文件路径）
    realtime    gpt-realtime

角色名就是**唯一的词汇**，三个入口说的都是它，env 变量名由角色名派生（见
``env_name()``）：``chat`` → ``VOICEMEM_CHAT_MODEL``、``reply`` →
``VOICEMEM_REPLY_MODEL``，以此类推。

旧的 ``OPENAI_MODEL`` / ``OPENAI_CHAT_MODEL`` / ``OPENAI_EMBEDDING_MODEL`` /
``OPENAI_TTS_MODEL`` / ``OPENAI_REALTIME_MODEL`` 仍然认（新名字优先）。改名是因为
这五个变量**只有本项目自己读**，openai SDK 和 mem0 一个都不读——挂着 OPENAI_ 的
名字，却在用户拿 DeepSeek / Qwen / vLLM 时逼人去设一个跟 OpenAI 无关的变量。

``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` **不改名**：那两个是 openai SDK 和 mem0
自己直接读的（``openai/_client.py`` 里 ``os.environ.get("OPENAI_API_KEY")``），
我们改了它们读不到，就成了"配了一半"——本项目支持任何 OpenAI 兼容端点，把
base_url 指过去即可，那两个变量此时表示的是"协议"，不是"厂商"。

一条优先级链，就近的赢——同一件事，写在哪一层由场景决定，不是"要配好几遍"：

    组件显式参数  >  models={...}  >  VOICEMEM_*_MODEL  >  旧的 OPENAI_*  >  默认值

    OpenAIAdditiveExtractorConfig(model="o4-mini")     # 单个组件，最就近
    VoiceMem(models={"chat": "gpt-4.1-mini"})          # 进程级（落在 MODELS 上）
    export VOICEMEM_CHAT_MODEL=gpt-4.1-mini            # 部署时配，代码不用动

每次取值时现算，所以 import 顺序、env 什么时候设都不影响。

``reply`` 默认跟随 ``chat``：回复是听得见的一路，允许单独配一个更强的模型，但
不配的时候不该出现"设了 OPENAI_MODEL 而回复还在用默认模型"这种一半生效。

换**实现**（换成本地模型、自建服务）是另一件事，走 VoiceMem 的可注入位
（embedder / tts / classifier ...）；这里只管**模型名和端点**。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: 角色默认值为 None 表示"跟随 chat"。
_FOLLOW_CHAT = None
#: 区分"没传 default"和"传了 default=None"。
_UNSET = object()


def env_name(role: str) -> str:
    """env 变量名从角色名**机械派生**：``chat`` → ``VOICEMEM_CHAT_MODEL``。

    手写一张角色→变量名的对照表，迟早会有一行写错，加新角色时也总会忘了补——
    这种错误是静默的（读了一个谁都没设的变量，安静地掉回默认值）。派生出来就
    不可能对不上。
    """
    return f"VOICEMEM_{role.upper()}_MODEL"


@dataclass(frozen=True)
class Role:
    default: str | None
    #: 改名前的变量名。这个没法派生（旧名字本来就不成规律），只能一条条列；
    #: 仍然认，但新名字优先。
    legacy: str = ""


ROLES: dict[str, Role] = {
    "chat":      Role("gpt-4o-mini",            legacy="OPENAI_MODEL"),
    "reply":     Role(_FOLLOW_CHAT,             legacy="OPENAI_CHAT_MODEL"),
    "embedding": Role("text-embedding-3-small", legacy="OPENAI_EMBEDDING_MODEL"),
    "tts":       Role("gpt-4o-mini-tts",        legacy="OPENAI_TTS_MODEL"),
    "realtime":  Role("gpt-realtime",           legacy="OPENAI_REALTIME_MODEL"),
}


class Models:
    """进程级模型表。惰性取值——每次 get 现读，不在构造时定死。"""

    def __init__(self) -> None:
        self._over: dict[str, str] = {}

    def update(self, mapping: dict[str, str] | None = None, **kw: str) -> "Models":
        """设置 override。未知 role 直接报错——写错名字静默不生效比报错难查得多。

        这张表是**进程级**的，不是每个 VoiceMem 实例一份。同一进程里第二个实例
        传了不同的模型，会把第一个的也改掉——这一点容易被 ``VoiceMem(models=...)``
        的写法误导，所以真发生时打一行出来，不让它静默。
        """
        for role, name in {**(mapping or {}), **kw}.items():
            if role not in ROLES:
                raise ValueError(f"未知的模型角色 {role!r}，可选：{', '.join(ROLES)}")
            if not name:
                continue
            name = str(name).strip()
            prev = self._over.get(role)
            if prev and prev != name:
                print(f"[models] {role}: {prev} → {name}。模型表是进程级的，"
                      f"这次覆盖对已经建好的 VoiceMem 同样生效。", flush=True)
            self._over[role] = name
        return self

    def get(self, role: str = "chat", explicit: str | None = None,
            default: str | None = _UNSET) -> str | None:
        """``default`` 显式给了就顶掉角色自带的默认值——本地 TTS 后端用得上：
        它的"模型"是一个 voice 文件路径，没有通用默认，给 None 让它自己报错。"""
        if role not in ROLES:
            raise ValueError(f"未知的模型角色 {role!r}，可选：{', '.join(ROLES)}")
        spec = ROLES[role]
        name = (explicit or self._over.get(role)
                or os.environ.get(env_name(role), "").strip()
                or (os.environ.get(spec.legacy, "").strip() if spec.legacy else ""))
        if name:
            return name.strip()
        if default is not _UNSET:
            return default
        return spec.default if spec.default is not _FOLLOW_CHAT else self.get("chat")

    def as_dict(self) -> dict[str, str]:
        """当前生效的全表，打日志/排查用。"""
        return {role: self.get(role) for role in ROLES}

    def explain(self) -> str:
        """一行一个角色：现在用的是谁、env 变量叫什么。排查"到底走的哪个模型"。"""
        return "\n".join(f"{role:<10} {self.get(role):<24} {env_name(role)}"
                          for role in ROLES)

    # 便利属性：MODELS.chat / MODELS.reply / ...
    chat = property(lambda self: self.get("chat"))
    reply = property(lambda self: self.get("reply"))
    embedding = property(lambda self: self.get("embedding"))
    tts = property(lambda self: self.get("tts"))
    realtime = property(lambda self: self.get("realtime"))


#: 进程级单例。VoiceMem(models=...) 和 from_config({"models": ...}) 都落在它上面。
MODELS = Models()


def resolve_model(explicit: str | None = None, role: str = "chat",
                  default: str | None = _UNSET) -> str | None:
    """显式参数 → override → env（新名字，再旧名字）→ 默认。"""
    return MODELS.get(role, explicit, default)


def resolve_api_key(explicit: str | None = None) -> str | None:
    """显式参数 → ``OPENAI_API_KEY`` → None（让调用方自己决定报什么错）。

    名字不改成 VOICEMEM_*：openai SDK 和 mem0 会绕过本项目直接读它，改名会变成
    一半组件有 key、一半没有。用别家模型时把 ``OPENAI_BASE_URL`` 指过去即可，
    这两个变量此时表示的是**协议**，不是厂商。
    """
    return explicit or os.environ.get("OPENAI_API_KEY")


def resolve_base_url(explicit: str | None = None) -> str | None:
    """显式参数 → ``OPENAI_BASE_URL`` → None（让 SDK 走官方端点）。

    这里刻意不加 ``VOICEMEM_BASE_URL`` 别名：mem0 和 openai SDK 会绕过本项目
    直接读 ``OPENAI_BASE_URL``，只设新名字会变成一半的组件指向自建端点、
    另一半仍打真 OpenAI——比名字不好看糟糕得多。
    """
    return explicit or os.environ.get("OPENAI_BASE_URL") or None


#: 旧名字，保持 import 不炸。
CHAT_MODEL_DEFAULT = ROLES["chat"].default
