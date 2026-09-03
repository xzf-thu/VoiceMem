"""语音转文字：流式识别（实时 partial）+ 非流式精转写（最终文本）。

流式两个实现，接口一致（``feed(samples) -> 累积文本`` / ``flush()`` / ``reset()``），
由 ``utils/defaults.py`` 的 ``asr`` 工厂按 ``VOICEMEM_ASR`` 选：

  · ``FunASRStreamingASR``  FunASR paraformer-zh-streaming（**默认**，中文更准）
  · ``StreamingASR``        sherpa-onnx 流式 zipformer（中英双语、纯 onnx 无 torch）
"""
from __future__ import annotations

import re

import numpy as np

SAMPLE_RATE = 16000

SENSEVOICE_EMOTION_MAP = {
    "NEUTRAL": "中性",
    "HAPPY": "开心",
    "ANGRY": "愤怒",
    "SAD": "悲伤",
    "FEARFUL": "恐惧",
    "FEAR": "恐惧",
    "DISGUSTED": "厌恶",
    "SURPRISED": "惊讶",
}


def pick_device() -> str:
    """自动选最佳设备: cuda > mps(Apple M) > cpu。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class StreamingASR:
    """sherpa-onnx 流式 zipformer，出实时 partial 文本。``VOICEMEM_ASR=sherpa`` 时启用。"""

    def __init__(self, asr_dir: str) -> None:
        import sherpa_onnx          # 惰性：默认走 FunASR 时不拉 sherpa
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f"{asr_dir}/tokens.txt",
            encoder=f"{asr_dir}/encoder-epoch-99-avg-1.onnx",
            decoder=f"{asr_dir}/decoder-epoch-99-avg-1.onnx",
            joiner=f"{asr_dir}/joiner-epoch-99-avg-1.onnx",
            num_threads=2, sample_rate=SAMPLE_RATE, feature_dim=80,
            decoding_method="greedy_search",
        )
        self.stream = self.rec.create_stream()

    def feed(self, samples):
        self.stream.accept_waveform(SAMPLE_RATE, samples)
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        return self.rec.get_result(self.stream)

    def flush(self) -> str:
        """接口对齐 FunASRStreamingASR；sherpa 逐帧就把结果吐完了，没有尾巴要补。"""
        return self.rec.get_result(self.stream)

    def reset(self) -> None:
        self.stream = self.rec.create_stream()


# ── 默认流式 ASR：FunASR paraformer-zh-streaming ────────────────────────────────

class FunASRStreamingASR:
    """FunASR ``paraformer-zh-streaming``，出实时 partial 文本（核心默认流式 ASR）。

    paraformer 是**按块**推理的（``chunk_size=[0,10,5]`` → 600ms/块），而
    ``VoiceStream.feed()`` 喂进来的帧长由调用方决定（web 前端发 20ms 一帧），所以这里
    内部攒够 600ms 才推一次；``feed()`` 与 sherpa 版语义一致，返回**累积**文本
    （``VoiceStream`` 是 ``self._text = asr.feed(frame)`` 赋值语义，不能返回增量）。

    ``flush()``：VAD 判说完时由 ``VoiceStream`` 调一次——把不足一块的尾巴补零、跑一次
    ``is_final=True``，取出解码器 look-ahead 里还没吐的最后几个字。flush 之后若又来
    音频（说到一半停顿又续上），自动起一条新的子流继续累积，不会把这一轮的文本丢掉。
    """

    CHUNK_SIZE = [0, 10, 5]                    # paraformer-streaming 标配
    STRIDE     = CHUNK_SIZE[1] * 960           # 9600 samples @16k = 600ms
    LOOK_BACK  = dict(encoder_chunk_look_back=4, decoder_chunk_look_back=1)

    #: 离线包里的位置。跟回退那套并列放在 asr/ 下——两个都是流式 ASR，区别只是
    #: 默认(FunASR，中文更准) / 回退(sherpa，纯 onnx 不依赖 torch)。
    LOCAL_DIR = "funasr-paraformer-zh-streaming"

    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        import logging as _logging
        import os as _os

        # `import funasr` 这一行本身就会把 **root logger** 从 WARNING 拉到 INFO 并挂
        # 一个 handler（实测：import 前 WARNING/0 handlers，import 后 INFO/1）。
        # 后果不只是它自己刷屏——之后 openai/httpx 的 INFO 也全冒出来，一次基础用法
        # 能刷几十行 "HTTP Request: POST ... 200 OK"，真正的结果被埋在中间。
        # 记下 import 前的状态，import 完原样恢复。VOICEMEM_VERBOSE=1 保留原样。
        _quiet = _os.environ.get("VOICEMEM_VERBOSE", "0") == "0"
        _root = _logging.getLogger()
        _lvl, _handlers = _root.level, list(_root.handlers)

        from funasr import AutoModel          # 惰性：只有真用流式 ASR 才拉 funasr

        if _quiet:
            _os.environ.setdefault("TQDM_DISABLE", "1")   # 每转写一块刷一条 rtf 进度条
            _root.setLevel(_lvl)
            for _h in list(_root.handlers):
                if _h not in _handlers:
                    _root.removeHandler(_h)
        # funasr 的 AutoModel 里有一句 logging.basicConfig(level=log_level)，默认
        # INFO —— 它设的是 **root**，于是 openai/httpx 那些库的 INFO 也跟着全冒出来
        # （"HTTP Request: POST ... 200 OK" 刷几十行）。在 voicemem/__init__ 里给
        # 各个 logger 设等级挡不住这个，因为它改的是 root。直接把参数传进去。
        if model is None:
            # 有离线包就用本地，没有就交给 funasr 按模型名自己下（848M，会卡在
            # 用户说的第一句上——所以离线包里带着它，别人拉下来开箱即用）。
            from voicemem.utils.common.paths import models_dir
            local = models_dir() / "asr" / self.LOCAL_DIR
            model = str(local) if (local / "config.yaml").exists() else "paraformer-zh-streaming"
        self.model = AutoModel(model=model, device=device or pick_device(),
                               disable_update=True,
                               log_level="ERROR" if _quiet else "INFO")
        self.reset()

    def _run(self, samples, is_final: bool) -> str:
        res = self.model.generate(input=samples, cache=self._cache, is_final=is_final,
                                  chunk_size=self.CHUNK_SIZE, **self.LOOK_BACK)
        if res and res[0].get("text"):
            self._text += res[0]["text"]
        return self._text

    def feed(self, samples) -> str:
        """喂任意长度的 16k float32 帧；攒够 600ms 推一块。返回累积文本。"""
        if self._final:                        # 上一轮 flush 过又来音频 → 起新子流续着攒
            self._cache, self._final = {}, False
        self._buf = np.concatenate([self._buf, np.asarray(samples, dtype=np.float32)])
        while len(self._buf) >= self.STRIDE:
            self._run(self._buf[:self.STRIDE], False)
            self._buf = self._buf[self.STRIDE:]
        return self._text

    def flush(self) -> str:
        """VAD 判说完时调：尾巴补零跑 is_final=True，别丢最后几个字。幂等。"""
        if self._final:
            return self._text
        tail = self._buf
        self._buf = np.zeros(0, dtype=np.float32)
        tail = (np.pad(tail, (0, self.STRIDE - len(tail))) if len(tail)
                else np.zeros(self.STRIDE, dtype=np.float32))
        self._final = True
        return self._run(tail, True)

    def reset(self) -> None:
        self._cache: dict = {}
        self._buf = np.zeros(0, dtype=np.float32)
        self._text = ""
        self._final = False


class Transcriber:
    """SenseVoiceSmall 出最终文本（中英），比流式 ASR 更准，锁定一轮时用这个。"""

    def __init__(self, device: str) -> None:
        from funasr import AutoModel        # 懒 import：只有用非流式精转写才需要 funasr
        from voicemem.utils.common.paths import hf_model
        _name = hf_model("emotion", "FunAudioLLM/SenseVoiceSmall", "asr")
        self.model = AutoModel(model=_name, hub="hf",
                               device=device, disable_update=True,
                               trust_remote_code=False)

    def _generate(self, audio) -> str:
        res = self.model.generate(input=audio, cache={}, language="zh",
                                  use_itn=True, ban_emo_unk=True)
        if not res:
            return ""
        return res[0].get("text", "") or ""

    def run(self, audio) -> str:
        return re.sub(r"<\|[^|]*\|>", "", self._generate(audio)).strip()

    def run_with_emotion(self, audio) -> tuple[str, str]:
        """一次 SenseVoice 推理同时取得文本和声学情绪 token。"""
        raw = self._generate(audio)
        tags = re.findall(r"<\|([^|]+)\|>", raw.upper())
        emotion = next((SENSEVOICE_EMOTION_MAP[tag] for tag in tags
                        if tag in SENSEVOICE_EMOTION_MAP), "中性")
        return re.sub(r"<\|[^|]*\|>", "", raw).strip(), emotion
