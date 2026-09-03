"""SpeakerEncoder — 本地 3D-Speaker 声纹向量提取（可选跨环境运行）。

3D-Speaker 的 ERes2Net ONNX 模型通过一个常驻子进程
（``voicemem/voiceprint/campplus_worker.py``）调用，模型只加载一次、常驻
复用，避免每次调用都重新加载的开销。

默认子进程用当前解释器（``sys.executable``）启动，即 funasr/modelscope 装
在同一环境即可直接用。如果你想把它隔离到独立环境（例如单独一个装了
funasr 的 conda env），设置环境变量 ``VOICEMEM_AUDIOMEM_PYTHON`` 指向那个
环境的 python 可执行文件。

用法::

    enc = SpeakerEncoder()
    vec = enc.embed(Path("recording.wav"))  # np.ndarray [192]，None 表示失败
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from voicemem.utils.audio.voiceprint import l2norm

# worker 就在本模块旁边。之前多拼了一层 "voiceprint/"（重构搬目录时留下的），
# 指向一个不存在的文件：子进程起来就退，readline 拿到空串，报成
# "campplus_worker 启动失败: ''" ——错误信息里连原因都没有，声纹整个不可用。
_WORKER_SCRIPT = Path(__file__).resolve().parent / "campplus_worker.py"
_WORKER_PYTHON = os.environ.get("VOICEMEM_AUDIOMEM_PYTHON", sys.executable)


class SpeakerEncoder:
    """常驻 3D-Speaker ERes2Net worker 子进程的懒加载封装，线程安全。"""

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        # models 目录在父进程这边解析好再传下去。worker 是独立脚本，自己算过一份，
        # 但那份漏了 pip 装之后的回退——两份逻辑迟早分叉，这里定死一份。
        from voicemem.utils.common.paths import local_model_override, models_dir

        env = dict(os.environ)
        env.setdefault("VOICEMEM_MODELS_DIR", str(models_dir()))
        # worker 不 import voicemem，只认旧的 VOICEMEM_SPEAKER_MODEL；新名字
        # （VOICEMEM_VOICEPRINT_MODEL）在这儿解析好翻译过去。
        picked = local_model_override("voiceprint")
        if picked:
            env["VOICEMEM_SPEAKER_MODEL"] = picked

        # stderr 收到管道里，别丢。之前是 DEVNULL，worker 因为找不到模型文件退出时
        # 外面只看到 "campplus_worker 启动失败: ''"——连原因都没有，而 worker 其实
        # 已经打印了"模型找不到：<路径>，下载：bash scripts/download_models.sh"。
        self._proc = subprocess.Popen(
            [_WORKER_PYTHON, str(_WORKER_SCRIPT), self._device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env,
        )
        ready = self._proc.stdout.readline().strip()
        if ready != "READY":
            why = ""
            if self._proc.poll() is not None:      # 已经退了，stderr 读得完，不会卡住
                err = (self._proc.stderr.read() or "").strip().splitlines()
                # 只要最后那几行——前面是 traceback 的调用栈，对用户没用
                why = "\n".join(err[-4:])
            raise RuntimeError("campplus_worker 启动失败" + (f"：\n{why}" if why else f": {ready!r}"))

    def embed(self, audio_path: Path) -> np.ndarray | None:
        """返回 192 维 L2 归一化声纹向量，失败时返回 None。"""
        try:
            with self._lock:
                self._ensure_started()
                self._proc.stdin.write(f"{audio_path}\n")
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("campplus_worker 无响应（进程可能已退出）")
            resp = json.loads(line)
            if "error" in resp:
                raise RuntimeError(resp["error"])

            vec = np.asarray(resp["embedding"], dtype=np.float64)
            return l2norm(vec)
        except Exception as e:
            print(f"  [speaker_encoder] embed failed: {e}", flush=True)
            return None
