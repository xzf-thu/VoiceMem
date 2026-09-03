"""启动自检：逐个独立测速每个组件，在 console 打印一份简单测评报告，用于门控启动。

用法（见 openai_voice_demo/backend/main.py 的接入）::

    from voicemem.startup_check import check_and_gate
    if not check_and_gate(vm):          # 打印报告；全达标返回 True 直接启动
        raise SystemExit("用户在自检后取消启动")

设计要点：
  · 每个组件独立探一次（构造 + 跑一次代表性操作），测单次延迟，互不影响。
  · 缺依赖 / 缺模型 / 缺 API key → 记为 SKIP（不参与速度门控），不是 FAIL。
  · 每类组件一个经验默认速度预算（ms），可用环境变量
    ``VOICEMEM_STARTUP_BUDGET_<KEY>`` 覆盖（如
    ``VOICEMEM_STARTUP_BUDGET_SPEAKER_ENCODER=2000``）。
  · 全部达标 → 直接启动；有组件偏慢（SLOW）或异常（FAIL）→ 询问是否仍启动。
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from math import sin, tau
from pathlib import Path
from typing import Any, Callable

# ── 状态 ─────────────────────────────────────────────────────────────────────
OK, SLOW, SKIP, FAIL = "ok", "slow", "skip", "fail"
_SYMBOL = {OK: "✓ 达标", SLOW: "⚠ 偏慢", SKIP: "– 跳过", FAIL: "✗ 异常"}


class SkipProbe(Exception):
    """探针主动跳过（依赖/模型/凭证缺失，不算失败）。"""


@dataclass
class ProbeResult:
    name: str
    status: str
    budget_ms: float
    elapsed_ms: float | None = None
    detail: str = ""


@dataclass
class StartupReport:
    results: list[ProbeResult] = field(default_factory=list)

    @property
    def slow(self) -> list[ProbeResult]:
        return [r for r in self.results if r.status == SLOW]

    @property
    def failed(self) -> list[ProbeResult]:
        return [r for r in self.results if r.status == FAIL]

    @property
    def needs_confirm(self) -> bool:
        """有偏慢或异常的组件时，需要向用户确认是否仍启动。"""
        return bool(self.slow or self.failed)

    def render(self) -> str:
        w = max((len(r.name) for r in self.results), default=4)
        lines = ["", "── VoiceMem 启动自检 ─────────────────────────────────────",
                 f"  {'组件'.ljust(w)}   {'状态':<6}  {'用时':>9}  {'预算':>9}"]
        for r in self.results:
            used = f"{r.elapsed_ms:.0f} ms" if r.elapsed_ms is not None else "—"
            budget = f"{r.budget_ms:.0f} ms"
            line = f"  {r.name.ljust(w)}   {_SYMBOL[r.status]:<6}  {used:>9}  {budget:>9}"
            if r.detail:
                line += f"   {r.detail}"
            lines.append(line)
        n_ok = sum(r.status == OK for r in self.results)
        n_slow, n_skip, n_fail = len(self.slow), sum(r.status == SKIP for r in self.results), len(self.failed)
        lines.append("  " + "─" * 54)
        lines.append(f"  合计 {len(self.results)} 项：达标 {n_ok} / 偏慢 {n_slow} / 跳过 {n_skip} / 异常 {n_fail}")
        lines.append("─────────────────────────────────────────────────────────")
        return "\n".join(lines)


# ── 默认速度预算（ms）─────────────────────────────────────────────────────────
# 每类组件一个经验值；音频模型首次推理含加载，预算给得宽。可用环境变量覆盖。
_DEFAULT_BUDGETS: dict[str, float] = {
    "preprocess_text":  400,    # 文本预处理（情绪兜底 + 声纹注册表）
    "scene_classifier": 60,     # 纯 python 场景归类
    "emotion_vad":      500,    # 韵律 V/A（RMS/ZCR，无 LLM）
    "environment_ast":  8000,   # AST 声学场景（首次含模型加载/下载）
    "speaker_encoder":  4000,   # 3D-Speaker 声纹（worker 子进程 + onnx）
    "dual_search":      300,    # 双脑检索稳态往返（本地 embedding 下 ~几十~300ms）
}


def _budget(key: str) -> float:
    env = os.environ.get(f"VOICEMEM_STARTUP_BUDGET_{key.upper()}")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return _DEFAULT_BUDGETS.get(key, 1000)


# ── 合成一段测试音频（stdlib，只用于给音频组件喂一次输入）────────────────────
def _synth_wav(path: Path, seconds: float = 0.5, rate: int = 16000, freq: float = 220.0) -> None:
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            frames += struct.pack("<h", int(0.3 * 32767 * sin(tau * freq * i / rate)))
        w.writeframes(bytes(frames))


# ── 探针上下文 ───────────────────────────────────────────────────────────────
@dataclass
class _Ctx:
    vm: Any
    wav_path: Path
    _vm_built: bool = False

    def get_vm(self):
        if self.vm is None and not self._vm_built:
            self._vm_built = True
            try:
                from voicemem.core import VoiceMem
                self.vm = VoiceMem()
            except Exception as e:
                raise SkipProbe(f"VoiceMem 构造失败: {e}")
        if self.vm is None:
            raise SkipProbe("无可用 VoiceMem 实例")
        return self.vm


# ── 各组件探针（返回 detail 或 (detail, 稳态ms)；raise SkipProbe 表示跳过）───────
# 约定：先预热一次（把模型加载/子进程启动/懒初始化等一次性开销吃掉），再测第二次
# 并把这个稳态耗时随 detail 一起返回——报告里显示的就是"正常状态"而非冷启动。
def _measure(op):
    """op 无参可调用：预热一次（不计时）再测一次，返回 (结果, 稳态ms)。"""
    op()
    t0 = time.perf_counter(); res = op(); return res, (time.perf_counter() - t0) * 1000


def _probe_preprocess_text(ctx: _Ctx):
    vm = ctx.get_vm()
    p, ms = _measure(lambda: vm.preprocess("我今天有点焦虑，工作压力好大"))
    return (f"emotion={p.emotion or '-'}", ms)


def _probe_scene_classifier(ctx: _Ctx):
    from voicemem.utils.audio.environment.scene_classifier import classify_scene
    pairs = [("Speech", 0.91), ("Music", 0.42), ("Vehicle", 0.30)]
    r, ms = _measure(lambda: classify_scene(pairs))
    return (f"scene={r.tag.value}", ms)


def _probe_emotion_vad(ctx: _Ctx):
    try:
        from voicemem.utils.audio.emotion.vad_audio import HeuristicWavVADEstimator
    except Exception as e:
        raise SkipProbe(f"依赖缺失: {e}")
    est = HeuristicWavVADEstimator()
    vad, ms = _measure(lambda: est.estimate(str(ctx.wav_path)))
    return (f"V={vad.valence:+.2f} A={vad.arousal:+.2f}", ms)


def _probe_environment_ast(ctx: _Ctx):
    try:
        from voicemem.utils.audio.environment.environment_detector_ast import ASTEnvironmentDetector
    except Exception as e:
        raise SkipProbe(f"依赖缺失(transformers/torch): {e}")
    try:
        det = ASTEnvironmentDetector()
        _, ms = _measure(lambda: det.detect_full(ctx.wav_path))   # 预热=首次加载模型
    except (ImportError, OSError, FileNotFoundError) as e:
        raise SkipProbe(f"模型不可用: {e}")
    return ("AST 推理成功", ms)


def _probe_speaker_encoder(ctx: _Ctx):
    try:
        from voicemem.utils.audio.voiceprint.speaker_encoder import SpeakerEncoder
    except Exception as e:
        raise SkipProbe(f"依赖缺失(sherpa-onnx): {e}")
    try:
        enc = SpeakerEncoder(device="cpu")
        vec, ms = _measure(lambda: enc.embed(ctx.wav_path))       # 预热=启动 worker 子进程
    except (ImportError, OSError, FileNotFoundError) as e:
        raise SkipProbe(f"模型不可用: {e}")
    if vec is None:
        raise SkipProbe("worker 未返回 embedding（模型/依赖缺失）")
    return (f"dim={len(vec)}", ms)


def _probe_dual_search(ctx: _Ctx) -> str:
    # Search() 一次同时召回左脑(事实 hits)+ 右脑(画像 rb_hits)——双脑检索。
    # 先预热一次（懒加载 repo/图谱/索引都在首调发生），再测第二次，测的才是
    # 稳态延迟（本地 embedding 下应在 ~200ms 内）；否则测到的是冷启动。
    vm = ctx.get_vm()
    try:
        vm.Search("测试检索")          # 预热，不计时
    except Exception as e:
        raise SkipProbe(f"检索后端不可用: {e}")
    t0 = time.perf_counter()
    r = vm.Search("测试检索")           # 计时的这次才是稳态
    ms = (time.perf_counter() - t0) * 1000
    detail = f"左{len(r.hits)}/右{len(getattr(r,'rb_hits',[]) or [])}"
    return (detail, ms)                 # 自报稳态耗时（不含上面的预热）


_PROBES: list[tuple[str, str, Callable[[_Ctx], str]]] = [
    # (显示名, budget key, 探针)
    ("文本预处理 preprocess",   "preprocess_text",  _probe_preprocess_text),
    ("场景分类 scene",          "scene_classifier", _probe_scene_classifier),
    ("韵律情绪 VAD",            "emotion_vad",      _probe_emotion_vad),
    ("声学场景 AST",            "environment_ast",  _probe_environment_ast),
    ("声纹编码 speaker",        "speaker_encoder",  _probe_speaker_encoder),
    ("双脑检索 search",         "dual_search",      _probe_dual_search),
]


def run_startup_check(vm: Any = None) -> StartupReport:
    """逐个独立测速每个组件，返回测评报告（不打印、不门控）。

    vm: 可选，复用已构造好的 VoiceMem（如 demo 的 memory_bridge.vm）；不传则
        在需要时惰性构造一个默认实例，构造失败的相关探针记为 SKIP。
    """
    report = StartupReport()
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "probe.wav"
        try:
            _synth_wav(wav)
        except Exception:
            wav = Path(td) / "missing.wav"   # 音频探针会因此 SKIP
        ctx = _Ctx(vm=vm, wav_path=wav)
        for name, key, fn in _PROBES:
            budget = _budget(key)
            t0 = time.perf_counter()
            try:
                detail = fn(ctx)
                elapsed = (time.perf_counter() - t0) * 1000
                # 探针可返回 (detail, measured_ms) 自报稳态耗时（如预热后再测），
                # 此时以自报值判定，而不是含预热的墙钟时间。
                if isinstance(detail, tuple):
                    detail, elapsed = detail[0], float(detail[1])
                status = OK if elapsed <= budget else SLOW
                report.results.append(ProbeResult(name, status, budget, elapsed, detail))
            except SkipProbe as e:
                report.results.append(ProbeResult(name, SKIP, budget, None, str(e)))
            except Exception as e:  # noqa: BLE001 — 未预期异常记为 FAIL
                elapsed = (time.perf_counter() - t0) * 1000
                report.results.append(ProbeResult(name, FAIL, budget, elapsed, f"{type(e).__name__}: {e}"))
    return report


def check_and_gate(vm: Any = None, *, interactive: bool | None = None) -> bool:
    """跑自检、打印报告，并据结果决定是否启动。

    返回 True 表示可以启动。全部达标 → 直接 True；有偏慢/异常组件 → 询问用户
    （非交互终端下默认放行并打印告警，避免无人值守部署被卡住）。
    """
    report = run_startup_check(vm)
    print(report.render(), flush=True)

    if not report.needs_confirm:
        print("  ✓ 所有组件均满足速度要求，直接启动。\n", flush=True)
        return True

    slow_names = "、".join(r.name for r in report.slow + report.failed)
    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        print(f"  ⚠ 组件偏慢/异常（{slow_names}），非交互终端默认继续启动。\n", flush=True)
        return True

    try:
        answer = input(f"  ⚠ 以下组件偏慢/异常：{slow_names}。仍要启动吗？[y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  已取消启动。", flush=True)
        return False
    go = answer in ("y", "yes")
    print(("  继续启动。\n" if go else "  已取消启动。\n"), flush=True)
    return go


# ── util 自检（VoiceMem.test()）：4 档 = 不可用 / 较慢 / 正常 / 极快 ─────────────
_UTIL_BUDGET = {   # ms，每个 util 的"正常"上限
    "embedding": 200, "slots": 400, "entity": 400, "emotion": 500,
    "voiceprint": 3000, "asr": 5000, "memory_engine": 300,
}
_TIER = {"极快": "⚡极快", "正常": "✓正常", "较慢": "⚠较慢", "不可用": "✗不可用"}


def _exercise(name, obj, wav):
    """跑一次代表性操作（没有廉价代表操作的 util 只算构建耗时）。"""
    if name == "embedding":       obj.embed_texts(["测试"])
    elif name == "slots":         obj.classify("我今天有点累")
    elif name == "emotion":       obj.detect(wav)
    elif name == "voiceprint":    obj.embed(wav)
    elif name == "memory_engine": obj.search("测试", user_id="u")


def run_util_report(utils):
    """对本 mode 需要的每个 util 加载+跑一次、测延迟，打印 4 档表格。返回结果列表。"""
    rows = []
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "probe.wav"
        try:
            _synth_wav(wav)
        except Exception:
            pass
        for name in utils.need:
            budget = _UTIL_BUDGET.get(name, 1000)
            t0 = time.perf_counter()
            try:
                _exercise(name, utils.get(name), wav)
                ms = (time.perf_counter() - t0) * 1000
                tier = "极快" if ms < budget * 0.25 else "正常" if ms <= budget else "较慢"
                detail = ""
            except Exception as e:
                ms, tier, detail = None, "不可用", f"{type(e).__name__}: {e}"
            rows.append({"util": name, "tier": tier, "ms": ms, "budget": budget, "detail": detail})

    print("\n── VoiceMem utils 自检 ──────────────────────────────────")
    print(f"  {'util':<15}{'状态':<9}{'延迟':>9}{'预算':>9}")
    for r in rows:
        used = f"{r['ms']:.0f} ms" if r["ms"] is not None else "—"
        line = f"  {r['util']:<15}{_TIER[r['tier']]:<9}{used:>9}{str(r['budget'])+'ms':>9}"
        if r["detail"]:
            line += "   " + r["detail"][:40]
        print(line)
    print("─────────────────────────────────────────────────────────")
    return rows


if __name__ == "__main__":  # python -m voicemem.startup_check
    raise SystemExit(0 if check_and_gate() else 1)
