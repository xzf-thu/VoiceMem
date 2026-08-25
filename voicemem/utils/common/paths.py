"""本地模型目录的唯一解析入口。

在这之前三处各写各的：``defaults.py`` 用 ``"models"``（cwd 相对）、``stream_io.py`` 用
``"../models"``（假设从 web/ 起）、``campplus_worker.py`` 从 ``__file__`` 往上数——而且
数错了层数，指到了 ``voicemem/utils/models/``（不存在）。统一到这里。
"""
from __future__ import annotations

import os
from pathlib import Path


def models_dir() -> Path:
    """``VOICEMEM_MODELS_DIR`` > 仓库根的 ``models/`` > 从 cwd 往上找 ``models/``。

    往上找而不是只看 cwd：pip 装之后 ``parents[3]`` 是 site-packages，那底下没有
    ``models/``，于是只能靠 cwd 相对路径——而用户在项目里换个子目录跑（比如
    ``cd tests && python test_streaming.py``）就找不到了，报的还是"模型没下载"，
    实际上模型就在上一层。
    """
    env = os.environ.get("VOICEMEM_MODELS_DIR")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[3] / "models"   # utils/common/paths.py → 仓库根
    if repo.is_dir():
        return repo
    here = Path.cwd().resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "models"
        if candidate.is_dir():
            return candidate
    return Path("models")          # 都没有：交给 require() 报一句人能看懂的话


def model_path(name: str, env_override: str | None = None, kind: str = "") -> Path:
    """取一个具体模型文件的路径；``env_override`` 指定的环境变量优先级最高。

    ``kind`` 是按用途分的子目录（``vad`` / ``asr`` / ``speaker``），跟发布仓库
    zhifeixie/VoiceMem_Default_Models_Env 的布局一致。找不到就退回上一版的平铺布局——早先
    下载过的人不该因为换了组织方式就突然找不到模型。
    """
    if env_override:
        explicit = os.environ.get(env_override)
        if explicit:
            return Path(explicit)
    root = models_dir()
    if kind:
        grouped = root / kind / name
        if grouped.exists():
            return grouped
    return root / name          # 旧的平铺布局；真不存在时由 require() 报错


def hf_model(kind: str, default_id: str, env_override: str | None = None) -> str:
    """一个 HF 模型该从哪儿加载：``env`` > 本地离线包 > HF 仓库 id（自动下载）。

    离线包就是 ``<models>/<kind>/``，布局跟发布仓库 zhifeixie/VoiceMem_Default_Models_Env 一致
    （``embedding`` / ``scene`` / ``emotion`` …，一个用途一个文件夹）。没下载过就返回
    HF id，transformers 照旧首次运行时自动拉——**零配置的默认行为不变**，下载了
    离线包的人则整条链路不再需要网络。

    判定"这个目录里确实有个模型"看的是 config.json 或 .onnx，而不是目录是否存在：
    空目录（比如下载中断留下的）应该继续回退到 HF，而不是让加载器对着空目录报错。
    """
    if env_override:
        explicit = os.environ.get(env_override)
        if explicit:
            return explicit
    local = models_dir() / kind
    if (local / "config.json").exists() or any(local.glob("*.onnx")):
        return str(local)
    return default_id


def require(path: Path, what: str, how: str = "bash scripts/download_models.sh models") -> Path:
    """模型文件不在就报一句人能看懂的话，而不是让底层库抛个看不懂的错。"""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{what} 找不到：{path}\n"
            f"下载：{how}\n"
            f"或用 VOICEMEM_MODELS_DIR 指向你已有的模型目录。"
        )
    return Path(path)
