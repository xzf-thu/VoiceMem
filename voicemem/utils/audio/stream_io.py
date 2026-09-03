"""流式输入的音频小工具：重采样 + silero VAD。

原样搬自 web/utils.py（脑图 html 发 24k、流式 ASR 要 16k；silero VAD 判说完），
提升为核心能力，供 voicemem/stream.py 的 VoiceStream 与 web demo 共用（消除重复）。

VAD 现在是可注入能力（``VoiceMem(vad=...)`` / config 的 ``vad`` 段），``make_vad`` 只是
内置的那个 silero 实现；换成自己的只要给个有 ``is_speech(frame) -> bool`` 的对象。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from voicemem.utils.common.paths import model_path, require


def resample(f32, src=24000, dst=16000):        # 脑图 html 发 24k，流式 ASR 要 16k
    n = int(len(f32) * dst / src)
    return np.interp(np.arange(n) * src / dst, np.arange(len(f32)), f32).astype(np.float32)


def read_wav(path) -> tuple[np.ndarray, int]:
    """读音频文件 → (float32 单声道, 采样率)。有 soundfile 就用它（格式全），
    没有就退回标准库 wave（只认 PCM wav，但不多一个依赖）。"""
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype="float32")
        return (audio[:, 0] if audio.ndim > 1 else audio), sr
    except ImportError:
        import wave
        with wave.open(str(path), "rb") as w:
            sr, n_ch = w.getframerate(), w.getnchannels()
            pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        return (pcm[::n_ch] if n_ch > 1 else pcm), sr


def transcribe_file(asr, path, chunk_s: float = 0.6) -> str:
    """用流式 ASR 把整个文件转写成一段文本：分块喂完再 flush 收尾。

    只是把「流式接口」包成「整段接口」，不引入第二套 ASR——ingest(audio=...) 这类
    一次性调用没必要为此再拉一个非流式模型。
    """
    audio, sr = read_wav(path)
    if sr != 16000:
        audio = resample(audio, src=sr)
    asr.reset()
    step, text = max(1, int(16000 * chunk_s)), ""
    for i in range(0, len(audio), step):
        text = asr.feed(audio[i:i + step]) or text
    flush = getattr(asr, "flush", None)          # 块式 ASR 把不足一块的尾巴补零吐出来
    if flush is not None:
        text = flush() or text
    return (text or "").strip()


def make_vad(model: str | None = None, threshold: float = 0.5):
    """内置 VAD：silero（sherpa-onnx 包的）。返回一个只有 ``is_speech(frame)`` 的小对象。

    ``model`` 不给就走 ``VOICEMEM_SILERO_VAD`` / ``VOICEMEM_MODELS_DIR/silero_vad.onnx``。
    这个 .onnx 没有自动下载兜底，缺了就明确报出来（而不是让 sherpa 抛个看不懂的错）。
    """
    import sherpa_onnx
    path = require(
        Path(model) if model else model_path("silero_vad.onnx", "vad", kind="vad"),
        "silero VAD 模型 silero_vad.onnx",
    )
    v = sherpa_onnx.VoiceActivityDetector(sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(model=str(path), threshold=threshold),
        sample_rate=16000), buffer_size_in_seconds=30)

    class _V:
        def is_speech(self, frame): v.accept_waveform(frame); return v.is_speech_detected()
    return _V()
