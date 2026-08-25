"""一个双工音频设备 + 回声消除，04 / 05 共用。

外放时麦克风录到的是「你的声音 + 喇叭里助手的声音」。AEC 拿喇叭正在放的那份信号
（far-end）作参考，从麦克风信号里把它减掉，剩下的才是真正的人声。不做这一步：
转写会把助手的话当成你说的存进记忆，VAD 会一直以为有人在说话——助手一开口就把
自己掐了。

所以播放必须和录音走**同一个** `sd.Stream`：只有在同一个回调里，才拿得到跟这一帧
麦克风严格对齐的 far-end。

    audio = AudioIO(loop, on_barge_in=stop.set)
    audio.start()
    pcm = await audio.mic.get()      # 已经消过回声的 16k PCM16
    audio.play(pcm24k)               # 24k 的 TTS/realtime 音频，内部降到 16k

（examples/03 里内联了一份同样的东西，那个例子刻意保持单文件自足。）
"""
import asyncio
import threading

import numpy as np
import sounddevice as sd
from pywebrtc_audio import AudioProcessor

from voicemem.utils.audio.stream_io import resample

SR = 16000        # WebRTC APM 只吃 8/16/32/48k；voicemem 内部也是 16k，正好
BLOCK = 160       # 10ms，APM 的原生帧长
PLAY_SR = 24000   # TTS 和 realtime 吐出来的都是 24k


class AudioIO:
    """录音（消过回声）+ 播放 + 打断判定，一个设备全包。

    ``on_barge_in``：助手说话时听到人声持续 ``barge_s``，就在事件循环里回调一次
    （已经先把播放缓冲清了）。``grace_s`` 是刚开口那段宽限期——那时候麦克风里几乎
    只有助手自己的声音，AEC 还没收敛，很容易一出声就把自己掐了。
    """

    def __init__(self, loop, on_barge_in=None, *, threshold=0.3,
                 barge_s=0.5, grace_s=0.6, mic_backlog=100):
        self.loop = loop
        self.mic: asyncio.Queue = asyncio.Queue(maxsize=mic_backlog)
        self.on_barge_in = on_barge_in
        self.threshold, self.barge_s, self.grace_s = threshold, barge_s, grace_s

        self._buf = bytearray()                 # 播放缓冲，16k int16
        self._lock = threading.Lock()
        self._active = threading.Event()        # 助手正在说
        self._spoke_s = 0.0                     # 连续听到人声多久
        self._said_s = 0.0                      # 助手这一轮说了多久（宽限期用）

        self.aec = AudioProcessor(sample_rate=SR, echo_cancellation=True,
                                  noise_suppression=True, auto_gain_control=False,
                                  stream_delay_ms=0)
        self.stream = sd.Stream(samplerate=SR, blocksize=BLOCK, channels=1,
                                dtype="float32", callback=self._cb)

    # ── 音频回调（在音频线程上，不许有阻塞和网络）────────────────────────────
    def _pull(self, n) -> np.ndarray:
        with self._lock:
            take = bytes(self._buf[:n * 2])
            del self._buf[:n * 2]
        out = np.zeros(n, dtype=np.float32)
        f = np.frombuffer(take, np.int16).astype(np.float32) / 32768.0
        out[:len(f)] = f
        return out

    def _cb(self, indata, outdata, frames, _time, _status):
        far = self._pull(frames)                       # 这一帧喇叭要放的
        outdata[:, 0] = far
        clean = self.aec.process(indata[:, 0].copy(), far)   # 从麦克风里减掉它

        if self._active.is_set():
            self._said_s += frames / SR
            self._spoke_s = (self._spoke_s + frames / SR
                             if self.aec.speech_probability >= self.threshold else 0.0)
            if self._spoke_s >= self.barge_s and self._said_s >= self.grace_s:
                self._active.clear()
                self._spoke_s = 0.0
                self.stop_playing()                    # 先停播：本地操作，立刻生效
                if self.on_barge_in:
                    self.loop.call_soon_threadsafe(self.on_barge_in)
        else:
            self._spoke_s = 0.0

        pcm = (np.clip(clean, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        # 满了就丢：助手说话那几秒没人取，攒着也没用，反而让下一轮的 ASR 落后一大截
        self.loop.call_soon_threadsafe(
            lambda: None if self.mic.full() else self.mic.put_nowait(pcm))

    # ── 播放侧 ────────────────────────────────────────────────────────────
    def play(self, pcm24: bytes):
        """24k PCM16 进播放缓冲（内部降到 16k）。这份也正是 AEC 的参考信号。"""
        f = np.frombuffer(pcm24, np.int16).astype(np.float32) / 32768.0
        f16 = resample(f, src=PLAY_SR, dst=SR)
        with self._lock:
            self._buf += (np.clip(f16, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    def stop_playing(self):
        with self._lock:
            self._buf.clear()

    def busy(self) -> bool:
        with self._lock:
            return bool(self._buf)

    # ── 助手这一轮的起止（打断判定要知道它在不在说话）────────────────────────
    def assistant_started(self):
        self._spoke_s = self._said_s = 0.0
        self._active.set()

    def assistant_done(self):
        self._active.clear()

    def start(self):
        self.stream.start()

    def close(self):
        self.stream.stop()
        self.stream.close()
