"""文本 → 语音。回复层只产出文本（见 ``voicemem/reply.py``），出声是这里、可选的一层。

四个内置后端，都吐 **24kHz 单声道 PCM16**：默认 OpenAI api；``local`` / ``piper``
走离线 piper；``voxcpm`` 走 VoxCPM2；``breeze`` 连 Breeze TTS 2 的流式服务
（自然语言指挥语气，见 ``BreezeTTS``；非商用许可，所以不是默认）。

**这是第九个可替换位**（前八个见 ``voicemem/utils/defaults.py``）。契约只有一条方法::

    class MyTTS:
        async def stream(self, text: str):     # 异步产出 24kHz 单声道 PCM16 bytes
            ...

    vm = VoiceMem(tts=lambda: MyTTS())                      # 写法 A：注入
    vm = VoiceMem.from_config({"tts": {"provider": "voxcpm"}})   # 写法 B：声明式

核心链路不碰它——记忆系统只到文本为止，出声是调用方的事。所以 ``tts`` 不在
``_NEED`` 里（不会被 warmup 拉起来），谁要出声谁 ``vm.utils.get("tts")``；
没装 piper / voxcpm 的用户不受影响。

``speak_stream()`` 是「边生成边合成」：吐满一句就送去合成，不等全文生成完——
等全文再合成的话，字早打完了音频还没起头（实测 TTS 首帧就要 ~1.2s）。
"""
from __future__ import annotations

import asyncio
import os

import numpy as np

from voicemem.utils.audio.stream_io import resample
from voicemem.llm_config import resolve_model

#: 兼容旧名字。真正的解析在 OpenAITTS.__init__ 里现算（见 llm_config）——
#: 原来这里是 import 时读 env，先 import 后设 env 就静默不生效。
TTS_MODEL = resolve_model(role="tts")
TTS_BACKEND = os.environ.get("TTS_BACKEND", "openai")   # openai(api) | local | voxcpm
#: 音色。原来这个值写死在合成函数里，走 llm_tts 的用户想换只能改源码——
#: 内置默认实现也该是可配的，不然「可替换」只剩换掉整个后端一条路。
TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "alloy")
#: 怎么念（语速、轻重、停顿）。gpt-4o-mini-tts 支持 instructions，是控制语气的地方。
#: 留空就不传这个参数——老模型（tts-1）不认它，传了会报错。
TTS_INSTRUCTIONS = os.environ.get("OPENAI_TTS_INSTRUCTIONS", "")
#: 只在「生成候选音色」时有意义（同 seed + 同文本 = 同一个人，方便挑）。
#: **它不保证跨句音色一致**，真正锁音色的是 ref_audio，见 BreezeTTS 文档。
BREEZE_SEED = int(os.environ.get("VOICEMEM_BREEZE_SEED", "42"))
SAMPLE_RATE = 24000

# 怎么切给 TTS 的段，决定了多久能出第一声。实测 gpt-4o-mini-tts 的首帧延迟随文本
# 长度涨：8字 615ms / 25字 902ms / 100字 1318ms——所以**第一段要尽量短**（早出声），
# 后面的段可以长（少调几次、语气连贯）。
_SENT_END  = "。！？!?…\n"          # 句末：正常的切段点
_SOFT_END  = "，,、；;：: "          # 句中停顿：只有第一段用，为了抢出第一声
#
# 下面几个数原来是写死的，照 gpt-4o-mini-tts 的首帧延迟标定的。换后端就得重标：
# 每段都是**独立合成**的，语调轮廓不跨段延续，所以段越多、接缝越明显——听感上
# 是"一句一句拼起来的"而不是连着说下来的。切得碎是拿流畅度换首帧延迟，
# 哪头更值取决于后端有多快。
_FIRST_MIN = int(os.environ.get("VOICEMEM_TTS_FIRST_MIN", "6"))
_FIRST_MAX = int(os.environ.get("VOICEMEM_TTS_FIRST_MAX", "20"))
_SENT_MIN  = int(os.environ.get("VOICEMEM_TTS_SENT_MIN", "12"))
_SENT_MAX  = int(os.environ.get("VOICEMEM_TTS_SENT_MAX", "60"))
#: 第一段允不允许在逗号处断开。默认允许——抢第一声用的。但这会**把一句话劈成两次
#: 独立合成**，接缝正好落在句子中间，是最难听的一种。后端首帧够快时设 0，
#: 让所有段都落在句末，句内不断。
_FIRST_SOFT = os.environ.get("VOICEMEM_TTS_FIRST_SOFT", "1") != "0"


def cut_point(buf: str, first: bool) -> bool:
    """这段够不够发去合成了。"""
    s = buf.strip()
    if not s:
        return False
    if first:                                  # 抢第一声：逗号也算，实在没有就按长度切
        ends = _SENT_END + _SOFT_END if _FIRST_SOFT else _SENT_END
        return (len(s) >= _FIRST_MIN and s[-1] in ends) or len(s) >= _FIRST_MAX
    return (len(s) >= _SENT_MIN and s[-1] in _SENT_END) or len(s) >= _SENT_MAX


# ── 内置后端 ──────────────────────────────────────────────────────────────────

class BaseTTS:
    """内置后端的公共壳：子类只写 ``_raw()``，样本对齐这里统一做。

    **按样本边界切块**：http 流是按网络包切的，实测 69 块里 62 块是奇数字节，而
    PCM16 一个样本占 2 字节——消费方 ``Int16Array``/``np.frombuffer`` 撞上奇数长度
    直接报错，那一整块音频就没了。这里把跨块的半个样本留到下一块，保证吐出去的
    每块都是完整样本。自己写后端不继承它也行，只要 ``stream()`` 吐的是整样本。
    """

    async def stream(self, text: str, instruction: str | None = None):
        """``instruction``：**这一轮**怎么念。感知层每轮判出的情绪要能进到声音里，
        而情绪是逐轮变的，写在实例上就成了全局常量。给 None 就用实例上的默认。
        后端不支持的（piper / voxcpm）忽略它即可。"""
        tail = b""
        async for chunk in self._raw(text, instruction):
            buf = tail + chunk
            cut = len(buf) & ~1                     # 向下取到偶数
            tail = buf[cut:]
            if cut:
                yield buf[:cut]
        if tail:
            yield tail + b"\x00"                    # 收尾那半个样本补齐

    def _raw(self, text: str, instruction: str | None = None):
        raise NotImplementedError


class OpenAITTS(BaseTTS):
    """在线 api：OpenAI TTS（默认 gpt-4o-mini-tts），response_format=pcm 就是 24k PCM16。

    ``base_url`` 默认**不跟着** ``VoiceMem(base_url=...)`` 走：那个通常指向自建的
    LLM / embedding 服务，多半没有 ``/audio/speech``，跟过去只会在出声时才报错。
    要换端点在这里显式给（或 ``OPENAI_TTS_BASE_URL``）。
    """

    def __init__(self, model=None, voice=None, instructions=None,
                 api_key=None, base_url=None):
        self.model = resolve_model(model, "tts")
        self.voice = voice or TTS_VOICE
        self.instructions = TTS_INSTRUCTIONS if instructions is None else instructions
        self._key = api_key
        self._base = base_url or os.environ.get("OPENAI_TTS_BASE_URL") or None
        self._client = None

    def _cli(self):
        """client 首次用到时才建，``import voicemem.tts`` 不会因此要求有 key。"""
        if self._client is None:
            from openai import AsyncOpenAI
            kw = {}
            if self._key:
                kw["api_key"] = self._key
            if self._base:
                kw["base_url"] = self._base
            self._client = AsyncOpenAI(**kw)
        return self._client

    async def _raw(self, text, instruction=None):
        kw = {"model": self.model, "voice": self.voice,
              "input": text, "response_format": "pcm"}
        ins = instruction or self.instructions
        if ins:
            kw["instructions"] = ins
        async with self._cli().audio.speech.with_streaming_response.create(**kw) as resp:
            async for chunk in resp.iter_bytes():
                yield chunk


class PiperTTS(BaseTTS):
    """离线小模型：piper（纯离线 onnx，中英皆可）。装：pip install piper-tts。

    ``model`` 指向 voice 的 .onnx（缺省读 ``VOICEMEM_TTS_MODEL``）。想换 kokoro /
    edge-tts 等，照着这个类写一个就行——外面只认 ``stream()``。
    """

    def __init__(self, model=None):
        self.model = resolve_model(model, "tts", default=None)
        self._voice = None

    def _load(self):
        if self._voice is None:
            if not self.model:
                raise ValueError(
                    "piper 后端要指定 voice 文件：设 VOICEMEM_TTS_MODEL 指向 .onnx，"
                    '或 config 里给 {"provider": "piper", "config": {"model": "…/x.onnx"}}')
            from piper import PiperVoice
            self._voice = PiperVoice.load(self.model)   # piper api 随版本，对照其文档
        return self._voice

    async def _raw(self, text, instruction=None):
        v = self._load()                      # piper 没有语气入口，instruction 忽略
        sr = getattr(getattr(v, "config", None), "sample_rate", 22050)
        for raw in v.synthesize_stream_raw(text):       # 同步生成器，int16 bytes @ sr
            f = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
            out = resample(f, src=sr, dst=SAMPLE_RATE)  # 统一到 24k
            yield (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


class VoxCPMTTS(BaseTTS):
    """离线大模型：VoxCPM2（2B，中英+多语）。装：pip install voxcpm。
    ``model`` 可指向本地目录，缺省用 HF 上的 openbmb/VoxCPM2（走本地缓存）。"""

    def __init__(self, model=None):
        self.model = resolve_model(model, "tts", default=None) or "openbmb/VoxCPM2"
        self._m = None

    def _load(self):
        if self._m is None:
            from voxcpm import VoxCPM
            self._m = VoxCPM.from_pretrained(self.model, load_denoiser=False)
        return self._m

    async def _raw(self, text, instruction=None):
        m = self._load()                      # voxcpm 同上
        sr = m.tts_model.sample_rate
        for f in m.generate_streaming(text=text):
            out = resample(np.asarray(f, np.float32).reshape(-1), src=sr, dst=SAMPLE_RATE)
            yield (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


class BreezeTTS(BaseTTS):
    """Breeze TTS 2（breezeblue-ai/breeze-tts）的流式服务客户端。

    它**是个服务、不是能 import 的库**：GPU 机器上起

        python -m breeze_infer.api <model_path> --host 0.0.0.0 --port 7860

    这边只是个 http 客户端，所以 VoiceMem 这侧不引入任何重依赖。要 Linux +
    NVIDIA（约 7.7 GiB 显存，建议 12GB），macOS 跑不了——``base_url`` 一般指向
    另一台机器（跟 examples/04_all_local_l40s.py 是同一个架构）。

    **不是默认后端，也不该设成默认**：代码是 Apache 2.0，但权重走 BreezeBlue 的
    研究/非商用许可，商用要 RESONIA, INC. 书面授权。设成默认等于把这个限制推给
    每一个 VoiceMem 用户。

    路径跟 OpenAI 一样是 ``/v1/audio/speech``，但收的是 form-data
    （``text`` / ``instruction`` / ``cfg_scale`` / ``ref_audio`` / ``ref_text`` /
    ``seed``），不是 JSON 的 ``input`` / ``voice``——所以不能拿 OpenAITTS 指过去。

    ``instruction`` 是用自然语言指挥语气的地方（"慢一点，说到难过的事压低声音"），
    这也是选它的理由：OpenAI realtime 那边只能在人设里夹一段文字，模型经常不理。
    正文里还能写发声事件：英文 ``(sigh)``、中文 ``[笑]``。

    **三种模式，别用错**（服务端按给没给 ref_audio 走两套不同模板）：

    ===============  ==================================  ==================
    模式             参数                                 音色
    ===============  ==================================  ==================
    Voice Design     instruction                          **每次都变**
    Voice Clone      ref_audio + ref_text                 固定
    Voice Direction  ref_audio + ref_text + instruction   固定 + 可指挥语气
    ===============  ==================================  ==================

    对话场景**必须给 ref_audio**（Voice Direction）。只给 instruction 的话说话人是
    跟文本一起生成出来的，逐句合成时每句话都会换一个人——这是实测踩到的。

    参考音频不用现录：先用 Voice Design 随机生成几段，挑一个顺耳的存下来当永久
    参考即可。要求 5~10 秒、单人、干净；``ref_text`` 必须是它的逐字转写，错了音色会飘。

    另外服务端 ``--fast-all`` 跟 ref_audio 这条路**不兼容**：text encoder 的 CUDA 图
    只按 warmup 时那几个形状捕获过，带参考音频的输入形状不在里面，会直接抛
    "text encoder CUDA graph (4, 32) was not declared in the warmup profile"。
    起服务时别加这个参数。
    """

    def __init__(self, base_url=None, instruction=None, cfg_scale=None,
                 ref_audio=None, ref_text=None, seed=None, timeout=60.0, model=None):
        self.base_url = (base_url or os.environ.get("VOICEMEM_BREEZE_URL")
                         or "http://127.0.0.1:7860").rstrip("/")
        self.instruction = instruction or os.environ.get("VOICEMEM_BREEZE_INSTRUCTION") or ""
        if cfg_scale is None:
            env_cfg = os.environ.get("VOICEMEM_BREEZE_CFG_SCALE")
            # 官方示例里 instruction 都配 cfg_scale=4（指令强度）；不给的话基本不跟指令。
            cfg_scale = float(env_cfg) if env_cfg else (4 if self.instruction else None)
        self.cfg_scale = cfg_scale
        # 参考音频这两项也给环境变量入口：web demo 的 --config 是**整体替换**内置
        # CONFIG 的，为了配一个 ref_audio 去写整份 json，很容易漏掉里面的人设
        # （reply.llm.config.system）——漏了就既没人设也不说中文。用环境变量配就
        # 不用碰那份 CONFIG。
        self.ref_audio = ref_audio or os.environ.get("VOICEMEM_BREEZE_REF_AUDIO") or None
        self.ref_text = ref_text or os.environ.get("VOICEMEM_BREEZE_REF_TEXT") or None
        # seed 只决定 voice design 从哪个随机点起步，**它锁不住音色**：说话人是跟
        # 文本一起自回归生成出来的，文本变了采样轨迹就变，同一个 seed 照样长出
        # 另一个嗓子。而这边是逐句合成的（speak_stream 攒够一句发一次请求），一段
        # 回复要发好几次——实测就是同一段话里每句话换一个人说。
        # 想固定音色只有 ref_audio 一条路，见类文档里的三种模式。
        # seed 留着是为了让「生成候选音色」可复现：同 seed + 同文本出同一个人，
        # 挑中了才好存下来当参考。
        self.seed = BREEZE_SEED if seed is None else seed
        self.timeout = timeout
        self._client = None                 # 连接复用，见 _cli()
        self._ref_bytes = None              # 参考音频只读一次盘
        # 服务端**单并发**：第二个请求在第一个还在流的时候打过去，直接 409 Conflict
        # （不是排队，是拒绝）。调用方是可以并发的——web 那边就提前给下一段起了任务，
        # 为的是省掉每段一个客户端→服务端的来回——所以这个约束由这里兜住：
        # 任务照样早创建，只是排在锁后面等，前一段一结束立刻发出去。
        # 锁只加在这个类里，OpenAI 那些能真并发的后端不受影响。
        self._lock = asyncio.Lock()
        # 服务端起服务时就把权重定死了，这里收下只为和别的后端对齐签名——
        # web/run.py 的 CONFIG 无论哪个 provider 都会传 model 进来。
        self.model = model

    def _cli(self):
        """复用同一个 client。一段回复是**逐句**发请求的（speak_stream 攒够一句就发），
        每句都新建连接的话，每次都要重走一遍 TCP 握手——服务在远端、又隔着 SSH 隧道
        时这一趟就是几十上百毫秒，直接听成卡顿。keep-alive 之后只有第一句付这个钱。
        """
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300.0))
        return self._client

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _raw(self, text, instruction=None):
        try:
            import httpx  # noqa: F401
        except ImportError as e:                # 跟着 openai 装的，正常都在
            raise ImportError("Breeze 后端要 httpx：pip install httpx") from e

        ins = instruction or self.instruction
        cfg = self.cfg_scale if self.cfg_scale is not None else (4 if ins else None)
        fields = {"text": text}
        if ins:
            fields["instruction"] = ins
        if cfg is not None:
            fields["cfg_scale"] = str(cfg)
        if self.ref_text:
            fields["ref_text"] = self.ref_text
        if self.seed is not None:
            fields["seed"] = str(self.seed)

        # 全部塞进 files 发 multipart：httpx 只在 files 非空时才发 multipart/form-data，
        # 光给 data= 会变成 urlencoded，而服务端那边示例是 curl -F。
        files = {k: (None, v) for k, v in fields.items()}
        if self.ref_audio:
            if self._ref_bytes is None:      # 每句话都去读一遍盘没必要
                from pathlib import Path
                ref = Path(self.ref_audio)
                self._ref_bytes = (ref.name, ref.read_bytes())
            files["ref_audio"] = self._ref_bytes

        url = f"{self.base_url}/v1/audio/speech"
        async with self._lock:                           # 单并发，见 __init__
            async with self._cli().stream("POST", url, files=files) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():   # 直接就是 24k 单声道 PCM16
                    yield chunk


#: provider 名 → 内置实现。``local`` 是 ``piper`` 的历史别名（TTS_BACKEND=local 一直
#: 是这个意思），两个都留着。
TTS_PROVIDERS = {
    "openai": OpenAITTS,
    "local":  PiperTTS,
    "piper":  PiperTTS,
    "voxcpm": VoxCPMTTS,
    "breeze": BreezeTTS,
}


#: 按 (provider, config) 复用实例。piper / voxcpm 要加载模型（VoxCPM2 是 2B），
#: 每次新建等于重新加载一遍——而 demo 里每个 Memory Space 一个 VoiceMem 实例，
#: 不共享的话开三个空间就是三份模型。后端本身除了那个模型没有别的状态，可以共享。
_INSTANCES: dict = {}


def make_tts(provider: str | None = None, **cfg):
    """按 provider 名建一个内置后端；``provider`` 省略就跟 ``TTS_BACKEND`` 环境变量。

    ``cfg`` 逐个 provider 不同（openai 认 model/voice/instructions/api_key/base_url，
    piper 和 voxcpm 只认 model），传错了直接 TypeError——比静默忽略好找。

    同样的 (provider, cfg) 返回同一个实例。要各自独立的就直接构造类。
    """
    name = (provider or TTS_BACKEND).lower()
    cls = TTS_PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"未知的 tts.provider={provider!r}；"
                         f"可选：{' / '.join(sorted(set(TTS_PROVIDERS)))}")
    key = (name, str(sorted(cfg.items())))
    if key not in _INSTANCES:
        _INSTANCES[key] = cls(**cfg)
    return _INSTANCES[key]


# ── 模块级入口（向后兼容）───────────────────────────────────────────────────────

def _tts_cfg(reply):
    seg = (reply or {}).get("tts") or {}
    return seg.get("provider"), (seg.get("config") or {})


async def tts_stream(text, reply=None, instruction=None):
    """从 ``reply.tts`` 那段配置解析后端并合成，吐 24kHz PCM16 流。

    注入进来的 TTS 走 ``vm.utils.get("tts").stream(text)``，不经过这里；这个函数
    是给只有一份 reply 配置、手上没有 VoiceMem 实例的调用方用的。
    """
    provider, cfg = _tts_cfg(reply)
    async for pcm in make_tts(provider, **cfg).stream(text, instruction):
        yield pcm


async def speak_stream(deltas, reply=None, on_delta=None, tts=None,
                       instruction=None):
    """文本增量流 → 语音流。合成跟生成**并行**：吐满一句就丢进队列，另一条协程
    取出来合成，边生成边出声。

    ``deltas``：异步迭代器（``vm.reply_stream(turn)`` 就是）。
    ``on_delta``：每收到一个文本增量回调一次（想边说边打字就传它）。
    ``tts``：合成用的对象（``vm.utils.get("tts")`` 或自己那个）；不给就按 ``reply``
    里的配置现解析一个。
    """
    queue: asyncio.Queue = asyncio.Queue()
    out: asyncio.Queue = asyncio.Queue()

    async def synth():
        while (seg := await queue.get()) is not None:
            gen = (tts.stream(seg, instruction) if tts is not None
                   else tts_stream(seg, reply, instruction))
            async for pcm in gen:
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
