"""文本 → 语音。回复层只产出文本（见 ``voicemem/reply.py``），出声是这里、可选的一层。

两个后端，都吐 **24kHz PCM16**：默认 OpenAI api，``TTS_BACKEND=local``（或
``reply.tts.provider == "local"``）走离线 piper。

``speak_stream()`` 是「边生成边合成」：吐满一句就送去合成，不等全文生成完——
等全文再合成的话，字早打完了音频还没起头（实测 TTS 首帧就要 ~1.2s）。
"""
from __future__ import annotations

import asyncio
import os
from functools import lru_cache

import numpy as np

from voicemem.utils.audio.stream_io import resample

TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
TTS_BACKEND = os.environ.get("TTS_BACKEND", "openai")   # openai(api) | local(离线小模型)
SAMPLE_RATE = 24000

# 怎么切给 TTS 的段，决定了多久能出第一声。实测 gpt-4o-mini-tts 的首帧延迟随文本
# 长度涨：8字 615ms / 25字 902ms / 100字 1318ms——所以**第一段要尽量短**（早出声），
# 后面的段可以长（少调几次、语气连贯）。
_SENT_END  = "。！？!?…\n"          # 句末：正常的切段点
_SOFT_END  = "，,、；;：: "          # 句中停顿：只有第一段用，为了抢出第一声
_FIRST_MIN = 6                      # 第一段攒够这么多字，遇到任何停顿就发
_FIRST_MAX = 20                     # 一个停顿都没有时，第一段也不能再等了
_SENT_MIN  = 12                     # 后续段的最短长度
_SENT_MAX  = 60                     # 后续段的兜底：LLM 一口气不换气也得切

_client = None


def _openai():
    """client 首次用到时才建，``import voicemem.tts`` 不会因此要求有 key。"""
    global _client
    if _client is None:
        from openai import AsyncOpenAI
        _client = AsyncOpenAI()
    return _client


def _tts_cfg(reply):
    seg = (reply or {}).get("tts") or {}
    return seg.get("provider"), (seg.get("config") or {})


def cut_point(buf: str, first: bool) -> bool:
    """这段够不够发去合成了。"""
    s = buf.strip()
    if not s:
        return False
    if first:                                  # 抢第一声：逗号也算，实在没有就按长度切
        return (len(s) >= _FIRST_MIN and s[-1] in _SENT_END + _SOFT_END) or len(s) >= _FIRST_MAX
    return (len(s) >= _SENT_MIN and s[-1] in _SENT_END) or len(s) >= _SENT_MAX


async def tts_stream(text, reply=None):
    """可切换 TTS：默认走 OpenAI api；reply.tts.provider==local（或 TTS_BACKEND=local）
    走离线本地小模型。两条都吐 24kHz PCM16 流，调用方一视同仁。

    **按样本边界切块**：http 流是按网络包切的，实测 69 块里 62 块是奇数字节，而
    PCM16 一个样本占 2 字节——消费方 ``Int16Array``/``np.frombuffer`` 撞上奇数长度
    直接报错，那一整块音频就没了。这里把跨块的半个样本留到下一块，保证吐出去的
    每块都是完整样本。
    """
    provider, cfg = _tts_cfg(reply)
    backend_name = provider or TTS_BACKEND          # 对齐现有 TTS_BACKEND 语义
    backend = {"local": _local_tts_stream, "voxcpm": _voxcpm_tts_stream}.get(
        backend_name, _openai_tts_stream)
    tail = b""
    async for chunk in backend(text, cfg.get("model")):
        buf = tail + chunk
        cut = len(buf) & ~1                         # 向下取到偶数
        tail = buf[cut:]
        if cut:
            yield buf[:cut]
    if tail:
        yield tail + b"\x00"                        # 收尾那半个样本补齐


async def _openai_tts_stream(text, model=None):
    """在线 api：OpenAI TTS（gpt-4o-mini-tts），response_format=pcm 就是 24k PCM16。"""
    async with _openai().audio.speech.with_streaming_response.create(
            model=model or TTS_MODEL, voice="alloy", input=text, response_format="pcm") as resp:
        async for chunk in resp.iter_bytes():
            yield chunk


@lru_cache(maxsize=1)
def _piper_voice():
    """离线小模型：默认 piper（纯离线 onnx，中英皆可）。装：pip install piper-tts；
    VOICEMEM_TTS_MODEL 指向 voice 的 .onnx。想换 kokoro / edge-tts 等，只改这个函数
    和下面 _local_tts_stream 的取样即可。piper api 随版本，对照其文档。"""
    from piper import PiperVoice
    return PiperVoice.load(os.environ["VOICEMEM_TTS_MODEL"])


async def _local_tts_stream(text, model=None):
    """离线本地 TTS：合成 → 重采样到 24k → 分块 yield，接口和在线版完全一致。
    （离线 voice 由 VOICEMEM_TTS_MODEL 指定；model 形参仅为和在线版对齐签名。）"""
    v = _piper_voice()
    sr = getattr(getattr(v, "config", None), "sample_rate", 22050)
    for raw in v.synthesize_stream_raw(text):          # 同步生成器，int16 bytes @ sr
        f = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
        out = resample(f, src=sr, dst=SAMPLE_RATE)     # 统一到 24k
        yield (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


@lru_cache(maxsize=1)
def _voxcpm_model():
    """离线大模型：VoxCPM2（2B，48k 输出，中英+多语）。装：pip install voxcpm。
    VOICEMEM_TTS_MODEL 可指向本地目录，缺省用 HF 上的 openbmb/VoxCPM2（走本地缓存）。"""
    from voxcpm import VoxCPM
    return VoxCPM.from_pretrained(
        os.environ.get("VOICEMEM_TTS_MODEL") or "openbmb/VoxCPM2", load_denoiser=False)


async def _voxcpm_tts_stream(text, model=None):
    """和 _local_tts_stream 同形：合成 → 重采样到 24k → 分块 yield。"""
    m = _voxcpm_model()
    sr = m.tts_model.sample_rate
    for f in m.generate_streaming(text=text):
        out = resample(np.asarray(f, np.float32).reshape(-1), src=sr, dst=SAMPLE_RATE)
        yield (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


async def speak_stream(deltas, reply=None, on_delta=None):
    """文本增量流 → 语音流。合成跟生成**并行**：吐满一句就丢进队列，另一条协程
    取出来合成，边生成边出声。

    ``deltas``：异步迭代器（``vm.reply_stream(turn)`` 就是）。
    ``on_delta``：每收到一个文本增量回调一次（想边说边打字就传它）。
    """
    queue: asyncio.Queue = asyncio.Queue()
    out: asyncio.Queue = asyncio.Queue()

    async def synth():
        while (seg := await queue.get()) is not None:
            async for pcm in tts_stream(seg, reply):
                await out.put(pcm)
        await out.put(None)

    worker = asyncio.create_task(synth())

    async def feed():
        buf, sent = "", 0
        try:
            async for d in deltas:
                if on_delta:
                    on_delta(d)
                buf += d
                if cut_point(buf, first=sent == 0):
                    await queue.put(buf.strip())
                    buf, sent = "", sent + 1
            if buf.strip():
                await queue.put(buf.strip())
        finally:
            await queue.put(None)               # 生成出错也要让 synth() 收工

    feeder = asyncio.create_task(feed())
    try:
        while (pcm := await out.get()) is not None:
            yield pcm
    finally:
        for t in (feeder, worker):
            if not t.done():
                t.cancel()
        await asyncio.gather(feeder, worker, return_exceptions=True)
