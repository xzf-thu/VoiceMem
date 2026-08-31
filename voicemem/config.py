"""统一配置入口：一个 dict 配齐所有本地/api 模型（仿 mem0 的 from_config 模式）。

每个组件写成 ``{"provider": ..., "config": {...}}``，打开一个 dict 就知道每个模型
走本地还是 api。``build_kwargs(config)`` 把这份声明式 dict 解析成现有
``VoiceMem(**kwargs)`` 能吃的注入参数——它是在现有 ``VoiceMem(embedding=fn, schema=fn,
…)`` 注入机制**之上**的一层糖，不改任何现有行为。

一份完整 config 长这样（每段的 config 都可省，省了就用内置默认）::

    CONFIG = {
        "api_key": "sk-...",              # 顶层，透传给 VoiceMem（也写进 OPENAI_API_KEY）
        "base_url": None,                 # 顶层，透传给 VoiceMem
        "mode": "multi_modal",            # 顶层，透传给 VoiceMem

        "embedding": {"provider": "local"},                 # 记忆向量走本地 E5
        "slots":     {"provider": "local"},                 # slot 分类走本地 E5（0 LLM）
        "vad":       {"provider": "silero"},                # 判「说完了」；custom 换自己的
        "memory_engine": {"provider": "mem0"},              # 向量库后端（默认 mem0）
        "tts": {"provider": "openai",                       # 出声（可选的一层）
                "config": {"model": "gpt-4o-mini-tts", "voice": "coral"}},
        "llm": {"provider": "openai",                       # 左右脑内部 LLM（打标签/归因…）
                "config": {"model": "gpt-4o-mini", "api_key": "sk-...", "base_url": None}},

        # reply 段：回复模型。两种写法都认——
        "reply": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
        # 或者 demo 那份嵌套写法（web/run.py 用）：llm 段解析成 reply，tts 段
        # 解析成第九个可替换位，realtime 仍归 web demo 自己读：
        # "reply": {
        #     "llm":      {"provider": "openai", "config": {"model": "gpt-4o"}},
        #     "tts":      {"provider": "openai", "config": {"model": "gpt-4o-mini-tts"}},
        #     "realtime": {"provider": "openai", "config": {"model": "gpt-realtime"}},
        # },
    }

provider → 内置实现 的映射（傻瓜清晰，一眼看懂）：

    embedding.provider     local  -> LocalE5Embedder（本地 E5，0 网络）
                           openai -> OpenAILocalEmbedder（OpenAI Embeddings API）
    slots.provider         local  -> LocalQueryClassifier（本地 E5，0 LLM）
                           openai -> QuerySlotClassifier（单次 LLM）
    vad.provider           silero -> make_vad（内置，config 可给 model / threshold）
                           custom -> config.obj 那个对象（要有 is_speech(frame)->bool）
    memory_engine.provider mem0   -> Mem0BackendStore（默认，也可不传走内置默认）
    tts.provider           openai -> OpenAITTS（OpenAI TTS api，可配 voice/instructions）
                           local  -> PiperTTS（离线 piper，别名 piper）
                           voxcpm -> VoxCPMTTS（离线 VoxCPM2）
                           breeze -> BreezeTTS（Breeze TTS 2 流式服务，可用自然语言
                                     指挥语气；权重非商用许可，故不做默认）
    llm.provider           openai -> 落 OPENAI_MODEL / OPENAI_API_KEY / OPENAI_BASE_URL
    reply.provider         openai -> voicemem.reply.openai_reply（内置，流式）
                           custom -> config.fn 里那个可调用对象（等价于 VoiceMem(reply=fn)）

不认识的 provider 会报清晰错误。
"""
from __future__ import annotations

import os


def _split(component: dict | None) -> tuple[str, dict]:
    """把 ``{"provider": ..., "config": {...}}`` 拆成 (provider, config)；config 可省。"""
    component = component or {}
    provider = component.get("provider")
    cfg = component.get("config") or {}
    return provider, cfg


def _bad(component: str, provider, known) -> None:
    raise ValueError(
        f"未知的 {component}.provider={provider!r}；可选：{' / '.join(known)}"
    )


def _embedding_factory(provider, cfg):
    """embedding：local -> LocalE5Embedder；openai -> OpenAILocalEmbedder。"""
    if provider == "local":
        def make():
            from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder
            return LocalE5Embedder()
        return make
    if provider == "openai":
        def make():
            from voicemem.leftbrain.local_memory_store import (
                OpenAILocalEmbedder, OpenAILocalEmbedderConfig,
            )
            return OpenAILocalEmbedder(OpenAILocalEmbedderConfig(
                model=cfg.get("model"),
                api_key=cfg.get("api_key"),
                base_url=cfg.get("base_url"),
                dimensions=cfg.get("dimensions"),
            ))
        return make
    _bad("embedding", provider, ["local", "openai"])


def _slots_factory(provider, cfg):
    """slots：local -> LocalQueryClassifier；openai -> QuerySlotClassifier。"""
    if provider == "local":
        def make():
            from voicemem.leftbrain.cognitive_graph.local_query_classifier import LocalQueryClassifier
            # 和本地 embedder 共享一份 E5（省一份内存），除非调用方显式传了 model。
            kw = dict(cfg)
            if "model" not in kw:
                from voicemem.leftbrain.local_e5_embedder import shared_e5
                kw["model"] = shared_e5()
            return LocalQueryClassifier(**kw)
        return make
    if provider == "openai":
        def make():
            from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QuerySlotClassifier
            return QuerySlotClassifier()
        return make
    _bad("slots", provider, ["local", "openai"])


def _vad_factory(provider, cfg):
    """vad：silero -> 内置 make_vad（可配 model/threshold）；custom -> config.obj 那个对象。"""
    if provider in (None, "silero"):
        def make():
            from voicemem.utils.audio.stream_io import make_vad
            return make_vad(model=cfg.get("model"), threshold=cfg.get("threshold", 0.5))
        return make
    if provider == "custom":
        obj = cfg.get("obj")
        if obj is None or not hasattr(obj, "is_speech"):
            raise ValueError(
                'vad.provider="custom" 需要 config.obj 给一个有 is_speech(frame)->bool '
                "的对象；直接注入更省事：VoiceMem(vad=lambda: MyVad())"
            )
        return lambda: obj
    _bad("vad", provider, ["silero", "custom"])


def _memory_engine_factory(provider, cfg):
    """memory_engine：mem0 -> Mem0BackendStore（默认，也可不传走内置默认）。"""
    if provider == "mem0":
        # None → 让 VoiceMem 走内置默认 memory_engine（即 Mem0BackendStore）。
        # 不需要在这里显式构造：内置默认已经是 mem0，直接不覆盖最省事、语义一致。
        return None
    _bad("memory_engine", provider, ["mem0"])


def _tts_factory(provider, cfg):
    """tts：openai -> OpenAI TTS api；local/piper -> 离线 piper；voxcpm -> VoxCPM2。

    provider 省了就跟 TTS_BACKEND 环境变量。provider 名在这儿就校验掉——留到第一次
    出声才报错的话，那已经是在对话中间了。
    """
    from voicemem.tts import TTS_PROVIDERS
    if provider is not None and str(provider).lower() not in TTS_PROVIDERS:
        _bad("tts", provider, sorted(set(TTS_PROVIDERS)))

    def make():
        from voicemem.tts import make_tts
        return make_tts(provider, **cfg)
    return make


# reply 段的 demo 嵌套写法（web/run.py 的 CONFIG["reply"]）里，这三个是子段名而不是
# provider/config。llm 解析成 reply、tts 解析成 tts 可替换位，realtime 归 web 自己读。
_REPLY_DEMO_KEYS = ("llm", "tts", "realtime")


def _reply_factory(provider, cfg):
    """reply：openai -> 内置流式 provider；custom -> 直接用 config.fn 那个函数。"""
    if provider in (None, "openai"):
        from voicemem.reply import openai_reply
        return openai_reply(model=cfg.get("model"), api_key=cfg.get("api_key"),
                            base_url=cfg.get("base_url"), system=cfg.get("system"))
    if provider == "custom":
        fn = cfg.get("fn")
        if not callable(fn):
            raise ValueError(
                'reply.provider="custom" 需要 config.fn 给一个可调用对象；'
                "直接传函数更省事：VoiceMem(reply=fn)"
            )
        return fn
    _bad("reply", provider, ["openai", "custom"])


def build_kwargs(config: dict) -> dict:
    """把统一 config dict 解析成 ``VoiceMem(**kwargs)`` 能吃的注入参数 dict。

    返回的 dict 只含实际给出的项（缺省的组件不放 key，交给 VoiceMem 用内置默认）：
    ``api_key`` / ``base_url`` / ``mode`` + ``embedding`` / ``schema`` /
    ``memory_engine`` 等注入函数（无参工厂，语义同 ``VoiceMem(embedding=lambda: ...)``）。

    ``reply`` 段解析成 ``VoiceMem(reply=fn)``；demo 那份嵌套写法里 ``llm`` 解析成
    reply、``tts`` 解析成 ``VoiceMem(tts=…)``，``realtime`` 仍由 web demo 自己读。
    """
    config = config or {}
    kwargs: dict = {}

    # ── 顶层：api_key / base_url / mode 直接透传 ──
    if config.get("api_key") is not None:
        kwargs["api_key"] = config["api_key"]
    if config.get("base_url") is not None:
        kwargs["base_url"] = config["base_url"]
    if config.get("mode") is not None:
        kwargs["mode"] = config["mode"]
    if config.get("memory_root") is not None:
        kwargs["memory_root"] = config["memory_root"]
    if config.get("user_id") is not None:
        kwargs["user_id"] = config["user_id"]
    if config.get("space") is not None:
        kwargs["space"] = config["space"]

    # ── embedding：VoiceMem 的注入键名是 embedding ──
    if "embedding" in config:
        provider, cfg = _split(config["embedding"])
        kwargs["embedding"] = _embedding_factory(provider, cfg)

    # ── slots：映射到 VoiceMem 的注入键名 schema（Classify 用的分类器）──
    if "slots" in config:
        provider, cfg = _split(config["slots"])
        kwargs["schema"] = _slots_factory(provider, cfg)

    # ── vad：判「说完了」的 VAD（VoiceStream 用）──
    if "vad" in config:
        provider, cfg = _split(config["vad"])
        kwargs["vad"] = _vad_factory(provider, cfg)

    # ── memory_engine：mem0 是内置默认，返回 None 就不覆盖 ──
    if "memory_engine" in config:
        provider, cfg = _split(config["memory_engine"])
        factory = _memory_engine_factory(provider, cfg)
        if factory is not None:
            kwargs["memory_engine"] = factory

    # ── llm：左右脑内部 LLM。现有代码读 OPENAI_MODEL / OPENAI_API_KEY /
    #    OPENAI_BASE_URL 这些 env，这里把 config 落到这些 env（api_key/base_url
    #    也透传给 VoiceMem 参数，保持和顶层一致）。──
    if "llm" in config:
        provider, cfg = _split(config["llm"])
        if provider not in (None, "openai"):
            _bad("llm", provider, ["openai"])
        if cfg.get("model"):
            os.environ["OPENAI_MODEL"] = cfg["model"]
        if cfg.get("api_key"):
            os.environ["OPENAI_API_KEY"] = cfg["api_key"]
            kwargs.setdefault("api_key", cfg["api_key"])
        if cfg.get("base_url"):
            os.environ["OPENAI_BASE_URL"] = cfg["base_url"]
            kwargs.setdefault("base_url", cfg["base_url"])

    # ── tts：第九个可替换位。两处都认——顶层 "tts"，或 demo 那份 reply.tts
    #    （web/run.py 一直这么写，以前核心不读、白写了，现在读）。顶层优先。──
    tts_seg = config.get("tts")
    if tts_seg is None:
        _r = config.get("reply") or {}
        if any(k in _r for k in _REPLY_DEMO_KEYS):
            tts_seg = _r.get("tts")
    if tts_seg is not None:
        provider, cfg = _split(tts_seg)
        kwargs["tts"] = _tts_factory(provider, cfg)

    # ── reply：回复模型。两种写法——扁平 {"provider","config"}，或 demo 那份
    #    {"llm","tts","realtime"} 嵌套（核心只取 llm，tts/realtime 仍由 web 自己读）。──
    if "reply" in config:
        seg = config["reply"] or {}
        if any(k in seg for k in _REPLY_DEMO_KEYS):
            seg = seg.get("llm") or {}
        provider, cfg = _split(seg)
        kwargs["reply"] = _reply_factory(provider, cfg)

    return kwargs
