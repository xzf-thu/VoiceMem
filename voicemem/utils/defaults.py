"""voicemem 各能力的内置默认实现工厂（util 名 -> 无参工厂）。

core.py 的 Utils 用它建默认；传函数给 VoiceMem(embedding=..., schema=...) 即覆盖对应项。
九个位子：embedding / schema / entity / emotion / voiceprint / asr / vad /
memory_engine / tts。前八个在核心链路上（按 mode 由 _NEED 决定加载哪些），
tts 不在——记忆系统只到文本为止，出声是可选的一层。
放这里而不是 core.py，是让顶层门面只讲「系统骨架」，不被这些具体默认实现的 import 撑大。
"""
from __future__ import annotations

import os


def default_utils(base_url, memory_root):
    def embedding():
        from voicemem.leftbrain.local_memory_store import OpenAILocalEmbedder, OpenAILocalEmbedderConfig
        return OpenAILocalEmbedder(OpenAILocalEmbedderConfig(base_url=base_url))
    def schema():
        # 默认本地 E5 分类器：0 LLM、0 网络——投机预取那 0–300ms 预算里不能走网络，
        # 而 Classify 就在那条路上（voicemem/stream.py 的 _speculate）。
        # sentence-transformers 不在基础依赖里（随 [demo] extra 装），缺了就回落到
        # LLM 版并打一行说明——静默回落等于悄悄开始花钱。
        # VOICEMEM_SLOTS=openai 可强制用 LLM 版（要实体抽取 / 子 slot 下钻时）。
        if os.environ.get("VOICEMEM_SLOTS", "local").lower() != "openai":
            try:
                from voicemem.leftbrain.cognitive_graph.local_query_classifier import LocalQueryClassifier
                from voicemem.leftbrain.local_e5_embedder import shared_e5
                return LocalQueryClassifier(model=shared_e5())   # 和本地 embedder 共享一份 E5
            except ImportError as e:
                print(f"[slots] 本地分类器不可用（{e}）→ 回落 LLM 版 QuerySlotClassifier。"
                      "装 sentence-transformers（或 pip install -e '.[demo]'）可用本地版。",
                      flush=True)
        from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QuerySlotClassifier
        return QuerySlotClassifier()
    def entity():
        from voicemem.leftbrain.cognitive_graph.annotator import CognitiveAnnotator, CognitiveAnnotatorConfig
        return CognitiveAnnotator(CognitiveAnnotatorConfig(base_url=base_url))
    def emotion():
        from voicemem.utils.audio.emotion.paper_emotion_detector import PaperAlignedEmotionDetector
        return PaperAlignedEmotionDetector()
    def voiceprint():
        from voicemem.utils.audio.voiceprint.speaker_encoder import SpeakerEncoder
        return SpeakerEncoder(device="cpu")
    def asr():
        # 默认 FunASR paraformer-zh-streaming（中文更准）；VOICEMEM_ASR=sherpa 回退到
        # sherpa-onnx 流式 zipformer（中英双语、纯 onnx 不依赖 torch）。
        if os.environ.get("VOICEMEM_ASR", "funasr").lower() == "sherpa":
            from voicemem.utils.audio.asr import StreamingASR
            from voicemem.utils.common.paths import model_path
            return StreamingASR(str(model_path(
                "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20", kind="asr")))
        from voicemem.utils.audio.asr import FunASRStreamingASR
        return FunASRStreamingASR()
    def vad():
        # 判「说完了」的 VAD。默认内置 silero；换自己的传一个有 is_speech(frame)->bool
        # 的对象即可（VoiceMem(vad=lambda: MyVad()) 或 config 的 vad 段）。
        from voicemem.utils.audio.stream_io import make_vad
        return make_vad()
    def tts():
        # 第九个可替换位。核心链路不用它——记忆系统只到文本为止，出声是调用方的事，
        # 所以 tts 不进 _NEED（warmup 不会拉起来），谁要出声谁 utils.get("tts")。
        # 不把 base_url 传下去：那个通常指向自建 LLM/embedding 服务，多半没有
        # /audio/speech，跟过去只会在出声时才炸。要换端点用 OPENAI_TTS_BASE_URL。
        from voicemem.tts import make_tts
        return make_tts()
    def memory_engine():
        from pathlib import Path
        from voicemem.leftbrain.mem0_backend_store import Mem0BackendStore
        # memory_root 由 Orchestrator 传下来（已解析过默认值）；这里的兜底只在
        # 直接构造 default_utils 时用得上，跟上面保持同一个默认。
        return Mem0BackendStore(embedding(),
                                memory_root=Path(memory_root or Path.cwd() / "voicemem_memory"))
    return {"embedding": embedding, "schema": schema, "entity": entity, "emotion": emotion,
            "voiceprint": voiceprint, "asr": asr, "vad": vad, "memory_engine": memory_engine,
            "tts": tts}
