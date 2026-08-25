"""voicemem 核心流式输入会话：边听边投机预取（EOU 0–300ms）。

和「文本」「wav」并列的第三种输入途径。两种喂法，每块都返回一个 ``StreamState``
（这块 ``<speak>``/``<silence>``、说完那块是 ``turn_over`` + 投机预取的记忆 + ``Turn``）：

    stream = vm.stream(on_partial=lambda t: print(t))

    # ① 喂音频块：voicemem 自带流式 ASR（FunASR paraformer）+ silero VAD
    st = await stream.feed(pcm_bytes)          # PCM16 @ src_rate（默认 24k）

    # ② 喂【外部 ASR】的 partial 文本（FunASR / Whisper / 任意）——换 ASR 只改喂进来的这行
    st = await stream.feed_partial(text, ended=is_final)

    st.state    # "<speak>" | "<silence>" | "turn_over"（这一轮说完了）
    st.memory   # 当前投机预取的记忆（SearchResult）；边说边有，没算好时 None
    st.turn     # 一轮说完才有 Turn（否则 None）

**只到记忆结果**——回复（tts/realtime）由调用方拿到 Turn/memory 后自理，核心不碰。
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voicemem.memory_api import build_memory_context
from voicemem.utils.audio.stream_io import resample


@dataclass
class Turn:
    """一轮说完（或打字）时、投机预取早已算好的记忆结果——调用方拿来直接回复，不再搜。"""
    text: str
    result: object

    @property
    def memory_context(self) -> str:
        return build_memory_context(self.result)


@dataclass
class StreamState:
    """每喂一块（音频或外部 ASR 文本）返回：这块静音/说话 + 当前投机记忆 + 说完了没。

    下面那组感知字段（``emotion`` / ``speaker_id`` / ``speaker_voiceprint`` /
    ``entity`` / ``schema`` / ``text_embedding``）全是**取用时才算**的 property：
    不读就一分钱一毫秒都不花，投机预取那条 0–300ms 的路径完全不受影响。
    """
    state: str                 # "<speak>" | "<silence>" | "turn_over"（一轮说完）
    text: str                  # 到目前为止的累积转写
    memory: object | None      # 当前投机预取到的记忆（SearchResult）；没算好时 None
    turn: Turn | None          # 一轮说完才有，否则 None
    _vm: object = None         # 惰性感知要用到的能力（utils）；调用方不用管
    _pcm: object = None        # 本轮 16k 单声道音频，供声纹/情绪按需分析

    @property
    def memory_context(self) -> str:
        m = self.turn.result if self.turn else self.memory
        return build_memory_context(m) if m is not None else ""

    # ── 记忆结果 ───────────────────────────────────────────────────────────────

    @property
    def _result(self):
        return self.turn.result if self.turn else self.memory

    @property
    def transcript(self) -> str:
        return self.text

    @property
    def result_leftbrain(self) -> list[str]:
        r = self._result
        return list(r.result_leftbrain) if r is not None else []

    @property
    def result_rightbrain(self) -> list[str]:
        r = self._result
        return list(r.result_rightbrain) if r is not None else []

    @property
    def entity(self) -> list[str]:
        """这句话里的命名实体（投机预取时已经分类过，直接取，不重算）。"""
        r = self._result
        return list(getattr(r.classification, "entities", []) or []) if r is not None else []

    @property
    def schema(self) -> list[str]:
        """这句话路由到的记忆槽位。"""
        r = self._result
        return list(getattr(r.classification, "slots", []) or []) if r is not None else []

    # ── 声学感知（取用时才算）───────────────────────────────────────────────────

    @property
    def _perception(self):
        if getattr(self, "_p_cache", None) is None:
            if self._vm is None or self._pcm is None or not len(self._pcm):
                return None
            self._p_cache = _perceive(self._vm, self._pcm, self.text)
        return self._p_cache

    @property
    def emotion(self) -> str:
        p = self._perception
        return getattr(p, "emotion", "") if p else ""

    @property
    def speaker_id(self) -> str:
        p = self._perception
        return (getattr(p, "person_id", None) or "") if p else ""

    @property
    def speaker_voiceprint(self):
        """本轮说话人的声纹向量（numpy 数组）；没开声纹或算不出时 None。"""
        if getattr(self, "_vp_cache", None) is None:
            if self._vm is None or self._pcm is None or not len(self._pcm):
                return None
            self._vp_cache = _voiceprint(self._vm, self._pcm)
        return self._vp_cache

    @property
    def text_embedding(self):
        """这句转写的文本向量；没有文本时 None。"""
        if getattr(self, "_emb_cache", None) is None:
            if self._vm is None or not self.text.strip():
                return None
            self._emb_cache = _embed(self._vm, self.text)
        return self._emb_cache


# ── StreamState 那几个感知字段的实现（都只在被读到时才跑）──────────────────────

def _tmp_wav(pcm) -> str:
    """把本轮音频落成临时 wav——声纹/情绪模块都按文件路径取音频。"""
    import tempfile, wave
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


def _perceive(vm, pcm, text):
    """跑一次音频感知（场景/声纹/情绪），复用编排层现成的 preprocess，不另起一套。"""
    path = _tmp_wav(pcm)
    try:
        return vm.preprocess(text or "", audio=path)
    except Exception as e:
        print(f"[stream] 感知失败: {e}", flush=True)
        return None
    finally:
        import os
        try: os.unlink(path)
        except OSError: pass


def _voiceprint(vm, pcm):
    path = _tmp_wav(pcm)
    try:
        return vm.utils.get("voiceprint").embed(Path(path))
    except Exception as e:
        print(f"[stream] 声纹提取失败: {e}", flush=True)
        return None
    finally:
        import os
        try: os.unlink(path)
        except OSError: pass


def _embed(vm, text):
    try:
        emb = vm.utils.get("embedding")
        fn = getattr(emb, "embed_texts", None)
        return fn([text])[0] if fn else emb.embed(text)
    except Exception as e:
        print(f"[stream] 文本向量失败: {e}", flush=True)
        return None


#: 每轮最多留这么长的音频给按需感知用（16k 单声道，30s ≈ 1.9MB）
_MAX_TURN_SAMPLES = 30 * 16000

#: 开口前保留这么长的音频，免得切掉第一个字（16k 单声道）
_PREROLL_SAMPLES = int(0.3 * 16000)

#: 没有转写文本、但持续这么久的声音也算一轮（对着麦克风放音乐的场景）。
#: 设长一点：短促的环境噪声（关门、咳嗽）不该变成一轮。
MIN_SOUND_ONLY_S = float(os.environ.get("VOICEMEM_SOUND_ONLY_S", "5"))

#: 纯声音的一轮，要静多久才算结束。
#:
#: 说话那一轮用 confirm_s（300ms）——对话就该这么快。但音乐不是对话：乐句之间的
#: 停顿、弱拍、前奏后的留白，随便就超过 300ms，于是一首歌被切成一地碎片（实测
#: 归档的录音全是 1.5~7.8 秒），回放时放出来只有开头几秒。
#: 放音乐的场景本来也不需要秒回，等久一点换一段完整的录音，划算。
SOUND_ONLY_SILENCE_S = float(os.environ.get("VOICEMEM_SOUND_ONLY_SILENCE_S", "3.0"))

#: 这一块音频有没有声音（RMS 阈值）。
#:
#: 判"一轮结束"平时看 VAD，而 silero VAD 判的是**有没有人声**——音乐不是人声，
#: 所以整段音乐在它眼里都是静音，静音计数一路涨，到 SOUND_ONLY_SILENCE_S 就把
#: 这轮切断了。实测放一首歌，归档的录音只有 3.0 秒，正好等于那个阈值。
#: 所以一个字都没转出来的时候改看能量：还有声音就不算静音，音乐放多久录多久。
SOUND_LEVEL = float(os.environ.get("VOICEMEM_SOUND_LEVEL", "0.01"))

#: 纯声音的一轮拿什么文本入库。
#:
#: 这一轮一个字都没转出来，直接 ingest("") 抽不出任何事实、也就没有记忆行，之后
#: 问「刚才那首歌帮我重播」什么都找不到。给它一句话当载体。
#:
#: 但**必须是这个常量**，别在调用方各写各的字面量：核心靠它认出"这一轮其实没有
#: 说话"（见 orchestrator 里的 _sound_only），认不出来就不会打 sound_only 标签，
#: 回放挑候选时分不清"用户说话那轮"和"音乐那轮"，放出来是用户自己的声音。
SOUND_ONLY_TEXT = "用户放了一段声音给我听。"


class VoiceStream:
    """核心流式输入会话：边喂边投机，说完时交出 Turn。

    ``vm.stream(on_partial=None, spec_min_chars=6, gamble_s=0.2, confirm_s=0.3,
    src_rate=24000)``。``feed`` 走内置流式 ASR + silero VAD；``feed_partial``
    接外部 ASR 的文本（换 ASR 只改喂进来的一行）。投机预取逻辑两者共用。

    两个时间参数是配套的，别单独调：

        静音 0ms ───────── 200ms ───────── 300ms
                            │               │
                     gamble_s：赌你说完了， confirm_s：VAD 确认说完，
                     后台开始检索记忆        记忆已经现成，直接开口

    中间那 100ms 就是留给检索的窗口——等 VAD 确认才开始查的话，这段时间会
    加在用户说完之后，变成他听得见的停顿。赌错了（用户只是停顿一下继续说）
    也不亏：下一帧有人声就把这次投机取消，白跑一次本地检索而已。
    """

    def __init__(self, vm, *, on_partial=None, spec_min_chars=6,
                 gamble_s=0.2, confirm_s=0.3, src_rate=24000, vad_threshold=None,
                 emotion=None):
        self.vm = vm
        #: 投机检索时带给 Search 的情绪提示（调用方可随时改写这个属性）。
        #:
        #: **右脑没有它就基本是空转的。** 右脑的情感记录几乎全部只挂在「情绪」
        #: 锚点上（实测一个库里 126 条 heartnote、124 条的锚点是 emotion），
        #: 不给情绪就一条都匹配不上，右脑只剩每轮都一样的静态 profile——回复里
        #: 读不出"它记得我这件事"，就是这么来的。实测同一个问题：
        #:     emotion=None   → 右脑 4 条，全是 profile
        #:     emotion="难过" → 情感记录 2 条 + 性格观察 1 条 + profile 2 条
        #:
        #: 但**本轮**的情绪读不得：``StreamState.emotion`` 是惰性属性，取一次要
        #: 同步跑整套声学感知（实测 2.1s），投机预取那 0–300ms 的预算根本不够。
        #: 所以调用方该把**上一轮** ingest 返回的 ``affect`` 写进来——情绪本来就有
        #: 连续性，代价是 0ms。
        self.emotion = emotion
        self.on_partial = on_partial
        self.spec_min_chars = spec_min_chars
        self.gamble_s = gamble_s
        self.confirm_s = confirm_s
        self.src_rate = src_rate
        self.vad_threshold = vad_threshold
        # ASR/VAD 懒加载：feed_text / feed_partial（外部 ASR）不碰音频模型。
        self._asr = None
        self._vad = None
        # 回合状态
        self._text = ""
        self._silence = 0.0
        self._spoke = False
        self._spec = None
        self._spec_text = ""
        self._last_memory = None   # 最新算好的投机记忆（SearchResult）
        self._pcm = []             # 本轮音频（16k 单声道），供 StreamState 按需做感知
        self._pcm_len = 0          # 已攒样本数，超上限就丢最早的（见 _MAX_TURN_S）
        self._preroll = []         # 开口前的一小段，接在本轮开头（见 _PREROLL_SAMPLES）

    @property
    def asr(self):
        if self._asr is None:
            self._asr = self.vm.utils.get("asr"); self._asr.reset()
        return self._asr

    @property
    def vad(self):
        if self._vad is None:
            if self.vad_threshold is None:
                self._vad = self.vm.utils.get("vad")   # 可注入：VoiceMem(vad=...) / config 的 vad 段
            else:
                from voicemem.utils.audio.stream_io import make_vad
                self._vad = make_vad(threshold=self.vad_threshold)
        return self._vad

    # ── 投机预取（本地分类器 + 本地向量 Search，0 LLM/网络，放线程里跟读麦克风并发）──
    async def _speculate(self, text) -> Turn:
        t0 = time.time()

        def work():
            c = self.vm.classify(text)
            return self.vm.search(text, slots=c.slots, entities=c.entities,
                                  emotion=self.emotion or None)

        result = await asyncio.to_thread(work)
        print(f"[speculate] {text[:24]!r} -> {len(result.hits)} hits  "
              f"{(time.time()-t0)*1000:.0f}ms", flush=True)
        return Turn(text, result)

    def _kick(self, text):
        """文本够长且变化了就（重）起后台投机。"""
        if text and text != self._spec_text and len(text) >= self.spec_min_chars:
            if self._spec:
                self._spec.cancel()
            self._spec_text = text
            self._spec = asyncio.create_task(self._speculate(text))

    def _ready_memory(self):
        """取最新算好的投机记忆（SearchResult）；没算好就保持上一份/None。"""
        if self._spec is not None and self._spec.done() and not self._spec.cancelled():
            try:
                self._last_memory = self._spec.result().result
            except Exception:
                pass
        return self._last_memory

    async def _confirm(self) -> Turn:
        try:
            turn = await (self._spec or self._speculate(self._text))
        except asyncio.CancelledError:
            turn = await self._speculate(self._text)
        # flush() 可能补出投机时还没解码出来的尾字：文本以最终版为准，记忆沿用已预取
        # 的结果（差的是最后几个字，为它重跑一次 Search 就把投机的收益还回去了）。
        if turn.text != self._text:
            turn = Turn(self._text, turn.result)
        return turn

    def _reset_turn(self):
        if self._asr is not None:
            self._asr.reset()
        self._text, self._silence, self._spoke = "", 0.0, False
        self._spec, self._spec_text, self._last_memory = None, "", None
        self._pcm, self._pcm_len = [], 0
        self._preroll = []

    async def feed_text(self, text) -> Turn:
        """打字轮：直接投机一次返回 Turn。"""
        return await self._speculate(text)

    async def feed_partial(self, text, ended: bool = False) -> StreamState:
        """接【外部 ASR】的 partial 文本（FunASR / Whisper / 任意流式 ASR）。

        换 ASR 只改「喂进来的这行文本」，本方法一字不用改。text 有新内容 = ``<speak>``
        并（重）起投机；``ended=True``（外部 VAD 判一句说完）→ 交出 Turn。
        """
        text = (text or "").strip()
        new = bool(text) and text != self._text
        if text:
            self._text = text
        if new and self.on_partial:
            self.on_partial(self._text)
        self._kick(self._text)
        if ended and self._text:
            turn = await self._confirm()
            self._reset_turn()
            return StreamState("turn_over", turn.text, None, turn, self.vm)
        return StreamState("<speak>" if new else "<silence>", self._text,
                           self._ready_memory(), None, self.vm)

    async def feed(self, pcm_bytes) -> StreamState:
        """喂一块 PCM16（``src_rate``，默认 24k）：内置流式 ASR + silero VAD + 投机。
        每块返回 ``StreamState``（``<speak>``/``<silence>`` + 当前投机记忆 + 说完时的 Turn）。
        """
        frame = resample(np.frombuffer(pcm_bytes, np.int16).astype(np.float32) / 32768.0,
                         src=self.src_rate)
        self._text = self.asr.feed(frame)
        speaking = self.vad.is_speech(frame)

        # 攒本轮音频：StreamState 的感知字段按需取用（声纹/情绪），也用来存档回放。
        #
        # 原来是**每一帧都攒**，包括你没说话的那些。于是两轮之间的静音一直往里堆，
        # 堆到 30 秒上限，一句「早上好呀」存出来是 29.9 秒、RMS 0.005 的音频。
        # 情绪模型拿到这个只会判「低能量 = 难过」——实测 emotion2vec / SenseVoice /
        # 韵律三个模型在这种音频上全判难过，看着像模型不准，其实是喂错了东西。
        #
        # 现在只从**开口那一刻**开始录，前面留 PREROLL 秒不切掉字头。
        if speaking or self._spoke:
            if not self._spoke and self._preroll:      # 刚开口：把前摇接上
                self._pcm.extend(self._preroll)
                self._pcm_len += sum(len(f) for f in self._preroll)
                self._preroll = []
            self._pcm.append(frame)
            self._pcm_len += len(frame)
            while self._pcm_len > _MAX_TURN_SAMPLES and len(self._pcm) > 1:
                self._pcm_len -= len(self._pcm.pop(0))
        else:
            self._preroll.append(frame)                # 还没开口：只留最近一小段
            while sum(len(f) for f in self._preroll) > _PREROLL_SAMPLES and len(self._preroll) > 1:
                self._preroll.pop(0)
        # 还一个字都没转出来时，"有声音"就不算静音——那多半是音乐，而 VAD 只认
        # 人声（见 SOUND_LEVEL）。已经有转写了就老老实实按 VAD 来，别让环境噪音
        # 把一句话的结束拖住。
        audible = speaking or (
            not self._text.strip() and self._spoke
            and float(np.sqrt(np.mean(frame * frame))) >= SOUND_LEVEL)
        if audible:
            if speaking and self._silence > 0 and self._spec:   # barge-in：又开口了 → 丢弃这次投机
                self._spec.cancel(); self._spec, self._spec_text = None, ""
            if speaking:
                self._spoke = True
            self._silence = 0.0
        else:
            self._silence += len(frame) / 16000.0
        if self._text.strip() and self.on_partial:
            self.on_partial(self._text)
        # 边说边预取 / 200ms 赌说完补发
        if self._spoke and self._text.strip() and \
                (self._silence == 0.0 and len(self._text) >= self.spec_min_chars
                 or self._silence >= self.gamble_s):
            self._kick(self._text)
        # 一轮成立要有转写文本——否则每段环境噪声都会变成一轮。
        # 但有个例外：**对着麦克风放音乐**。VAD 会把音乐判成人声（实测 357 块全
        # 中），ASR 却一个字都转不出，于是永远不成一轮：没有记忆、没有存档音频，
        # 之后问「刚才那首歌帮我重播」当然找不到。放够久（≥ MIN_SOUND_ONLY_S）
        # 就当作一轮交出去，text 为空，由上层决定怎么记。
        sound_only = (not self._text.strip()
                      and self._pcm_len >= MIN_SOUND_ONLY_S * 16000)
        # 一个字都没转出来的这一轮多半是音乐，别拿对话的 300ms 去切它，
        # 见 SOUND_ONLY_SILENCE_S。
        need_silence = self.confirm_s if self._text.strip() else SOUND_ONLY_SILENCE_S
        if self._spoke and self._silence >= need_silence and (self._text.strip() or sound_only):
            flush = getattr(self._asr, "flush", None)      # 块式 ASR（paraformer）把不足
            if flush is not None:                          # 一块的尾巴补零吐出来
                self._text = flush() or self._text
            turn = await self._confirm()                   # VAD 确认说完 → 交出预算记忆
            pcm = np.concatenate(self._pcm) if self._pcm else None
            self._reset_turn()
            return StreamState("turn_over", turn.text, None, turn, self.vm, pcm)
        return StreamState("<speak>" if speaking else "<silence>", self._text,
                           self._ready_memory(), None, self.vm)
