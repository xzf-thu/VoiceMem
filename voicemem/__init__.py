"""Voicemem：左右脑分离的记忆框架，带音频原生感知层。

本文件就是一张「组件目录」——voicemem 的每一个组件都在这里对外暴露，
`from voicemem import X` 即可拿到。音频类组件（SpeakerEncoder /
ASTEnvironmentDetector 等）依赖 torch / sherpa-onnx，所以采用惰性加载
（PEP 562 `__getattr__`）：只有真正访问到某个名字时才 import 它对应的模块，
`import voicemem` 本身永远不会拉起这些重依赖——纯文本安装照样能用核心。

分组一览：
  · 核心          VoiceMem / SearchResult / RightBrainHit / AudioPerception
  · 回复（输出侧） openai_reply / normalize_reply（见 voicemem/reply.py）
  · 语音接入适配   VoiceInput / VoiceContent / VoiceprintRegistry / ingest_voice_input …
  · 音频原生感知   SpeakerEncoder / *EnvironmentDetector / Scene* / Music/Place/Routine …
  · 右脑情绪层     EmotionLayer / EmotionLayerConfig / EmotionLayerResult
  · 便捷记忆 API    Memory / inject / recall / remember
  · 子包           leftbrain / rightbrain / utils（audio·common·fusion 都在 utils 下）
"""

from __future__ import annotations

# ── 第三方库的 INFO 噪音 ────────────────────────────────────────────────────
# 跑一次基础用法，openai SDK 会为每次请求打一行 "HTTP Request: POST ... 200 OK"
# （二三十行），funasr 每转写一块刷一条 rtf 进度条，mem0 每存一条打一行
# "Updating memory with data=..."。真正的结果被埋在中间，第一次用的人会以为出错。
#
# 用 filter 而不是 setLevel：调用方之后再 basicConfig 也盖不掉这里的设置。
# 想看全部：VOICEMEM_VERBOSE=1。
def _quiet_third_party_logs() -> None:
    import logging
    import os

    if os.environ.get("VOICEMEM_VERBOSE", "0") != "0":
        return

    os.environ.setdefault("TQDM_DISABLE", "1")          # funasr / transformers 的进度条

    # 用 setLevel 而不是 addFilter：filter 只作用于挂它的那个 logger，不会传给
    # 子 logger（openai._base_client、funasr.xxx 都是子 logger），实测挡不住。
    # 等级则是继承的：给 "openai" 设了 WARNING，"openai._base_client" 也照办。
    # httpx2 / httpcore2 是 mem0 那条依赖链带进来的分叉版本，logger 名字也带 2。
    # 只写 "httpx" 挡不住它——每次 embedding / chat 调用都会刷一行
    # "HTTP Request: POST https://api.openai.com/v1/embeddings ..."，
    # 一次 ingest 十几行。
    for name in ("openai", "httpx", "httpx2", "httpcore", "httpcore2",
                 "mem0", "funasr", "modelscope",
                 "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # transformers 有一类提示走的是它自己的 warning 系统，logging 等级管不到
    # （"Using a slow image processor as use_fast is unset..."）。用它的开关。
    # tokenizers 在 fork 之后会刷一段 "The current process just got forked..."
    # 的告警。我们本来就不靠它的并行（重活在 ASR/embedding 那边），关掉。
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


_quiet_third_party_logs()



try:                       # 装出来的包才有元数据；从源码目录直接跑时没有
    from importlib.metadata import PackageNotFoundError, version as _v
    __version__ = _v("voicemem")
except Exception:
    __version__ = "0.0.0.dev"

import importlib
from typing import TYPE_CHECKING

# ── 名字 → 来源模块 的惰性映射（"模块路径:属性名"）─────────────────────────────
# 每个组件都登记在这里；__getattr__ 在首次访问时按需 import，并缓存进本模块
# 命名空间，之后就是普通属性访问，没有额外开销。
_LAZY: dict[str, str] = {
    # ── 核心（门面 VoiceMem 在 core.py；编排实现/数据结构在 orchestrator.py；
    #    LeftBrain/RightBrain 为真组件）──
    "VoiceMem":                 "voicemem.core:VoiceMem",
    "Utils":                    "voicemem.orchestrator:Utils",
    "LeftBrain":                "voicemem.leftbrain.brain:LeftBrain",
    "RightBrain":               "voicemem.rightbrain.brain:RightBrain",
    "SearchResult":             "voicemem.orchestrator:SearchResult",
    "RightBrainHit":            "voicemem.rightbrain.brain:RightBrainHit",
    "AudioPerception":          "voicemem.utils.audio.perceiver:AudioPerception",
    "VoiceStream":              "voicemem.stream:VoiceStream",
    "openai_reply":             "voicemem.reply:openai_reply",
    "normalize_reply":          "voicemem.reply:normalize",
    "Turn":                     "voicemem.stream:Turn",
    "StreamState":              "voicemem.stream:StreamState",

    # ── 语音接入适配层（上游语音模块结构化输出 → 左脑注入）──
    "VoiceInput":               "voicemem.utils.common.voice_input:VoiceInput",
    "VoiceContent":             "voicemem.utils.common.voice_input:VoiceContent",
    "VoiceIngestResult":        "voicemem.utils.common.voice_input:VoiceIngestResult",
    "VoiceprintRegistry":       "voicemem.utils.common.voice_input:VoiceprintRegistry",
    "VoiceprintEntry":          "voicemem.utils.common.voice_input:VoiceprintEntry",
    "ingest_voice_input":       "voicemem.utils.common.voice_input:ingest_voice_input",
    "voice_input_to_messages":  "voicemem.utils.common.voice_input:voice_input_to_messages",
    "emotion_to_affect":        "voicemem.utils.common.voice_input:emotion_to_affect",
    "map_voice_slots_to_slotv2":"voicemem.utils.common.voice_input:map_voice_slots_to_slotv2",

    # ── 音频原生感知：说话人声纹 ──
    "SpeakerEncoder":           "voicemem.utils.audio.voiceprint.speaker_encoder:SpeakerEncoder",
    "VoiceprintStore":          "voicemem.utils.audio.voiceprint.voiceprint_store:VoiceprintStore",
    "IdentifyResult":           "voicemem.utils.audio.voiceprint.voiceprint_store:IdentifyResult",
    "parse_self_identification":"voicemem.utils.audio.voiceprint.speaker_identity:parse_self_identification",

    # ── 音频原生感知：声学环境 / 场景 ──
    "ASTEnvironmentDetector":   "voicemem.utils.audio.environment.environment_detector_ast:ASTEnvironmentDetector",
    "CLAPEnvironmentDetector":  "voicemem.utils.audio.environment.environment_detector_clap:CLAPEnvironmentDetector",
    "SceneTag":                 "voicemem.utils.audio.environment.scene_classifier:SceneTag",
    "SceneResult":              "voicemem.utils.audio.environment.scene_classifier:SceneResult",
    "classify_scene":           "voicemem.utils.audio.environment.scene_classifier:classify_scene",
    "scene_to_response_directive":"voicemem.utils.audio.environment.scene_classifier:scene_to_response_directive",
    "infer_scene_from_text":    "voicemem.utils.audio.environment.scene_classifier:infer_scene_from_text",

    # ── 音频原生感知：场景提醒触发 ──
    "SceneTrigger":             "voicemem.utils.audio.environment.scene_trigger:SceneTrigger",
    "SceneTriggerStore":        "voicemem.utils.audio.environment.scene_trigger:SceneTriggerStore",
    "TriggerFireResult":        "voicemem.utils.audio.environment.scene_trigger:TriggerFireResult",
    "parse_trigger_intent":     "voicemem.utils.audio.environment.scene_trigger:parse_trigger_intent",
    "check_and_fire":           "voicemem.utils.audio.environment.scene_trigger:check_and_fire",

    # ── 音频原生感知：音乐 / 地点 / 生活规律记忆 ──
    "MusicMemoryStore":         "voicemem.utils.audio.environment.music_memory:MusicMemoryStore",
    "TuneIdentifyResult":       "voicemem.utils.audio.environment.music_memory:TuneIdentifyResult",
    "PlaceMemoryStore":         "voicemem.utils.audio.environment.place_memory:PlaceMemoryStore",
    "PlaceIdentifyResult":      "voicemem.utils.audio.environment.place_memory:PlaceIdentifyResult",
    "RoutineStore":             "voicemem.utils.audio.environment.routine_memory:RoutineStore",
    "bucket_label":             "voicemem.utils.audio.environment.routine_memory:bucket_label",

    # ── 录音归档 / 会话 / 配置 ──
    "AudioArchive":             "voicemem.utils.audio.audio_archive:AudioArchive",
    "SessionTracker":           "voicemem.utils.common.session_tracker:SessionTracker",
    "VoiceStoreConfig":         "voicemem.utils.common.voice_config:VoiceStoreConfig",

    # ── 右脑情绪层 ──
    "EmotionLayer":             "voicemem.utils.audio.emotion:EmotionLayer",
    "EmotionLayerConfig":       "voicemem.utils.audio.emotion:EmotionLayerConfig",
    "EmotionLayerResult":       "voicemem.utils.audio.emotion:EmotionLayerResult",

    # ── 启动自检（逐组件测速 + 门控启动）──
    "run_startup_check":        "voicemem.startup_check:run_startup_check",
    "check_and_gate":           "voicemem.startup_check:check_and_gate",
    "StartupReport":            "voicemem.startup_check:StartupReport",

    # ── 便捷记忆 API ──
    "Memory":                   "voicemem.memory_api:Memory",
    "build_memory_context":     "voicemem.memory_api:build_memory_context",
    "inject":                   "voicemem.memory_api:inject",
    "recall":                   "voicemem.memory_api:recall",
    "remember":                 "voicemem.memory_api:remember",
}

# 子包也作为组件对外暴露：`from voicemem import emotion` / `voicemem.leftbrain` …
# 各子包内部的组件由其自身 __init__.__all__ 负责，这里只把包本身列进目录。
_SUBPACKAGES: tuple[str, ...] = (
    "leftbrain", "rightbrain", "utils",
)

__all__ = sorted([*_LAZY, *_SUBPACKAGES])


def __getattr__(name: str):
    """PEP 562 惰性属性访问：按需 import 组件，避免 `import voicemem` 触发重依赖。"""
    target = _LAZY.get(name)
    if target is not None:
        module_path, _, attr = target.partition(":")
        value = getattr(importlib.import_module(module_path), attr)
        globals()[name] = value          # 缓存，后续走普通属性访问
        return value
    if name in _SUBPACKAGES:
        module = importlib.import_module(f"voicemem.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])


# 让静态类型检查器 / IDE 也能看到这些名字（运行期不执行，不触发重依赖）。
if TYPE_CHECKING:  # pragma: no cover
    from voicemem.core import VoiceMem
    from voicemem.orchestrator import SearchResult
    from voicemem.rightbrain.brain import RightBrainHit
    from voicemem.utils.audio.perceiver import AudioPerception
    from voicemem.utils.audio.emotion import (
        EmotionLayer, EmotionLayerConfig, EmotionLayerResult,
    )

def sample_audio(path):
    """把示例音频的路径解析成真的能打开的路径。

    README 和 examples 里写的是 ``assets/input.wav``——那是**仓库里**的相对路径。
    ``pip install voicemem`` 的人没有那个目录，照着 README 跑第一个例子就是
    LibsndfileError，而那正是别人对这个项目的第一印象。

    所以：给的路径存在就原样用（clone 仓库的人不受任何影响）；不存在、而包里带了
    同名的示例音频，就回落到包内那份。传别的路径时行为不变——找不到就还是找不到，
    让原来的错误照常抛出来，不会把"用户写错路径"悄悄变成"放了段示例音频"。
    """
    import os
    from pathlib import Path

    if path is None:
        return None
    p = Path(str(path))
    if p.exists():
        return str(path)
    packaged = Path(__file__).resolve().parent / "assets" / p.name
    if packaged.is_file():
        return str(packaged)
    return str(path)
