"""voicemem 顶层门面：一个类 VoiceMem = 左脑 + 右脑 + 音频感知 + 一组可换的能力(utils)。

    左脑  事实记忆：实体 + 认知图（slot 分类/检索），底层 mem0 向量库
    右脑  情绪记忆：每轮 valence-arousal、情绪归因、人格画像
    utils 可插拔能力：embedding / schema(分类) / entity / emotion / voiceprint / asr / memory_engine
          每个都有内置默认，传一个函数就换成自己的（本地模型、别的向量库…）

    vm = VoiceMem(api_key="sk-...", mode="text_mode")
    vm.ingest("中午和 Alex 吃了拉面")
    vm.search("我中午吃了什么")                    # 左右脑一起检索
    vm.left_brain.search(...) / vm.right_brain.search(...)
    VoiceMem(embedding=lambda: MyE(), schema=lambda: MyClassifier())   # 换掉某个能力

mode 决定加载哪些能力：left_brain_single / text_mode / multi_modal(带音频)。

本文件只是「暴露给别人理解系统」的入口：VoiceMem 是一层薄门面，面向用户的小写
便捷方法（ingest/search/classify/preprocess/flush/test）各自一行委托给内部持有的
一个 Orchestrator 实例（self._o）。真正的整条 pipeline（Search/Ingest 编排、工具
方法、向左右脑/音频三组件的转发、SearchResult / Utils）都藏在 orchestrator.py。

想按老写法直接调大写编排方法（vm.Search / vm.Ingest / vm.Classify / vm.Flush …）
或访问内部转发方法也可以：门面的 __getattr__ 会透明地转到 self._o。
"""

from __future__ import annotations

from pathlib import Path

from voicemem.orchestrator import Orchestrator, SearchResult, Utils

# SearchResult / Utils 经此 re-export，保持 `from voicemem.core import ...` 与
# `from voicemem import SearchResult, Utils` 可用。
__all__ = ["VoiceMem", "SearchResult", "Utils"]


class VoiceMem:
    """顶层门面：左脑 + 右脑 + utils（一眼看懂系统）。实现见 orchestrator.py。

    输入侧 ``stream()`` → ``Turn``（voicemem/stream.py），输出侧 ``reply()`` /
    ``reply_stream()``（voicemem/reply.py）；``VoiceMem(reply=fn)`` 换成自己的模型。

    面向用户的小写便捷方法各自一行委托给内部 ``self._o``（一个 ``Orchestrator``
    实例）；``left_brain`` / ``right_brain`` / ``utils`` 直接指向真组件。构造参数原样
    透传给 ``Orchestrator``：``mode`` 走 mode，能力覆盖（``embedding`` / ``schema`` /
    ``memory_engine`` 等）与 ``enable_*`` / ``embedder`` / ``vector_store`` / ``classifier``
    都经 ``**kw`` 透传，语义与 ``Orchestrator.__init__`` 完全一致。
    """

    #: mode 的对外别名 → 内部名。README 用面向用户的说法（"normal" = 全都要，
    #: "leftbrain_only" = 只要事实记忆），内部名描述的是加载哪套 util。
    MODE_ALIASES = {
        "normal":         "multi_modal",
        "leftbrain_only": "left_brain_single",
        "text":           "text_mode",
    }

    def __init__(self, api_key=None, mode="text_mode", memory_root=None,
                 user_id="voice_user", base_url=None, reply=None,
                 openai_key=None, top_k=5, space=None, **kw):
        # space：一套记忆一个目录，落在 ./voicemem_memoryspace/<space>/。
        # 不给就是 "demo"。memory_root 显式给了就照用（评测要每段对话一个独立库）。
        self._o = Orchestrator(api_key=api_key or openai_key,
                               mode=self.MODE_ALIASES.get(mode, mode),
                               memory_root=memory_root, space=space,
                               user_id=user_id, base_url=base_url, **kw)
        # 回复层是门面级的事（编排层只到记忆结果为止），所以 reply 不往下透传。
        # None → 首次用到时回落到内置 openai provider，见 _reply_fn。
        self._reply_src = reply
        self._reply_norm = None
        self._top_k = top_k                  # search() 的默认取几条
        self.mode = self._o.mode
        self.utils = self._o.utils
        self.left_brain = self._o._left      # 真组件
        self.right_brain = self._o._right

    @classmethod
    def from_config(cls, config: dict) -> "VoiceMem":
        """声明式构造：一个统一 config dict 配齐所有本地/api 模型（仿 mem0）。

        每个组件写成 ``{"provider": ..., "config": {...}}``，打开一个 dict 就知道
        每个模型走本地还是 api。这是在现有 ``VoiceMem(embedding=fn, schema=fn, …)``
        注入机制之上的一层糖，现有构造方式照常工作。provider 映射表见
        ``voicemem.config``。::

            vm = VoiceMem.from_config({
                "mode": "multi_modal",
                "embedding": {"provider": "local"},   # 记忆向量走本地 E5
                "slots":     {"provider": "local"},   # slot 分类走本地 E5（0 LLM）
                "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
            })
        """
        from voicemem.config import build_kwargs
        return cls(**build_kwargs(config))

    # ── 面向用户的便捷方法（各自一行委托给 Orchestrator）─────────────────────────

    def ingest(self, text=None, audio=None, **kw):
        """记一句话。``ingest("文本")`` 存文本；``ingest(audio="x.wav")`` 只给音频时
        先本地转写再存（同一段音频照样跑声纹/场景/情绪感知）。两个都给就用给的文本。"""
        if text is None:
            if audio is None:
                raise ValueError("ingest() 要么给 text，要么给 audio")
            text = self.transcribe(audio)
        # 没有音频就没有声纹，说话人只能是账号主人本人——用 "user" 这个约定 id
        # （见 voice_input_to_messages），否则会走到给"未验证声纹"准备的防御标签上，
        # 存下来的每条事实都变成"Unidentified speaker Speaker 0 是素食主义者"。
        # 有音频时保持编排层默认，让声纹识别去定说话人（多人场景不能假设是主人）。
        if audio is None:
            kw.setdefault("speaker", "user")
        return self._o.Ingest(text, audio_path=audio, **kw)

    def transcribe(self, audio) -> str:
        """把一个音频文件整段转写成文本（复用 utils 里那个流式 ASR，不额外拉模型）。"""
        from voicemem.utils.audio.stream_io import transcribe_file
        return transcribe_file(self.utils.get("asr"), audio)

    def search(self, query, **kw):
        kw.setdefault("top_k", self._top_k)
        return self._o.Search(query, **kw)

    def classify(self, query):                return self._o.Classify(query)
    def preprocess(self, text, audio=None):   return self._o.preprocess(text, audio_path=audio)
    def flush(self):                          return self._o.Flush()

    def warmup(self, *, audio: bool = True, verbose: bool = True) -> None:
        """把本地模型全部先加载起来，别让第一次调用去等。

        模型都是懒加载的：不预热的话第一次 ``ingest(audio=...)`` 要多花二十几秒
        （本地 E5 ~1.7s、FunASR ~6.5s、感知那套 ~16s），而这几十秒正好落在用户
        第一次开口的时候——最不该卡的位置。web demo 一直是这么做的，现在挪到
        这儿，所有 demo 和自己写的脚本都能一行调到。

        ``audio=False`` 只热文本那条路（纯 leftbrain_only 用不着 ASR / 感知）。
        ``verbose=False`` 完全不输出。
        重复调用是安全的：模型已加载时每步都只是一次廉价的空跑。
        """
        import sys
        import time

        total = 4 if audio else 1
        # 只有真终端才画进度条：重定向到文件/管道时 \r 会糊成一行
        bar_ok = verbose and sys.stdout.isatty()
        done = [0]

        def draw(label, finished=False):
            if not verbose:
                return
            if not bar_ok:
                if finished:
                    print(f"[warmup] {label}", flush=True)
                return
            width = 24
            filled = int(width * done[0] / total)
            bar = "█" * filled + "░" * (width - filled)
            pct = int(100 * done[0] / total)
            end = "\n" if done[0] >= total else ""
            # 31 = 红色
            sys.stdout.write(f"\r\033[31m{bar}\033[0m {pct:3d}%  {label:<28}{end}")
            sys.stdout.flush()

        def step(name, fn):
            draw(f"正在加载 {name} …")
            t0 = time.time()
            try:
                fn()
                label = f"{name} {time.time() - t0:.1f}s"
            except Exception as e:                 # 少装了可选依赖 / 模型没下载
                label = f"{name} 跳过（{type(e).__name__}）"
            done[0] += 1
            draw(label if done[0] < total else "模型就绪", finished=True)

        step("embedding / slot 分类", lambda: self.classify("你好"))
        if not audio:
            return

        import numpy as np

        def warm_asr():
            asr = self.utils.get("asr")
            asr.feed(np.zeros(9600, dtype=np.float32))     # 拉起模型并真跑一块
            asr.reset()

        step("ASR", warm_asr)
        step("VAD", lambda: self.utils.get("vad").is_speech(np.zeros(512, dtype=np.float32)))

        # 感知那套（场景 AST / 声纹 3D-Speaker / 情绪 SenseVoice）要一个真文件。
        # 实测第一次 preprocess 2120ms 全在加载模型，之后稳定在 340–410ms。
        def warm_perceive():
            import tempfile
            import soundfile as sf
            from pathlib import Path
            p = Path(tempfile.gettempdir()) / "voicemem_warmup.wav"
            sf.write(p, np.zeros(16000, dtype=np.float32), 16000)
            try:
                self.preprocess("预热", audio=str(p))
            finally:
                p.unlink(missing_ok=True)

        step("感知（场景 / 声纹 / 情绪）", warm_perceive)

    def stream(self, **kw):
        """流式输入途径：喂音频块 / 文字 → 说完时得到 Turn（记忆结果）。见 voicemem/stream.py。"""
        from voicemem.stream import VoiceStream
        return VoiceStream(self, **kw)

    # ── 回复层（输出侧）：两条路一个口子，见 voicemem/reply.py ────────────────────

    def _reply_fn(self):
        """懒规格化：VoiceMem(reply=fn) 传进来的任意形状 → 统一的异步生成器函数。
        没传就用内置 openai provider（首次调用才建 client，不传 key 也能 import）。"""
        if self._reply_norm is None:
            from voicemem.reply import normalize, openai_reply
            self._reply_norm = normalize(self._reply_src or openai_reply())
        return self._reply_norm

    def reply_stream(self, turn_or_text, memory_context=""):
        """流式回复：``async for delta in vm.reply_stream(turn)``。

        第一个参数可直接给 ``Turn``/``StreamState``（自动拆出 text 与 memory_context），
        也可以给一段文本 + 自己渲染好的 memory_context。

        说完的这句自动登记给记忆层（``capture`` → ``remember_reply``），下次
        ``ingest()`` 就带上 agent 这半边，调用方一行不用改。
        """
        from voicemem.reply import capture, unpack
        text, ctx = unpack(turn_or_text, memory_context)
        return capture(self._reply_fn()(text, ctx),
                       lambda answer: self._o.remember_reply(text, answer))

    async def reply(self, turn_or_text, memory_context=""):
        """收全的回复：``answer = await vm.reply(turn)``。内部就是把 reply_stream 拼起来。"""
        return "".join([d async for d in self.reply_stream(turn_or_text, memory_context)])

    def test(self):
        """启动自检：只测本 mode 需要的 util，打印 4 档速度表。"""
        from voicemem.startup_check import run_util_report
        return run_util_report(self.utils)

    def __getattr__(self, name):
        # 老写法的大写编排方法（Search/Ingest/Classify/Flush…）与内部转发方法
        # 透明落到编排实现上；__getattr__ 只在常规属性查找失败时触发，故 self._o /
        # utils / left_brain / right_brain 这些已设的属性不会走到这里。
        return getattr(self.__dict__["_o"], name)
