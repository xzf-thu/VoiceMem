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


#: 本地模型的旧环境变量名。正式名由能力名派生（见 local_model_env），旧名字仍认。
#: 这些名字当初一个模型起一个，规律全靠记——``VOICEMEM_SILERO_VAD`` 换个 VAD
#: 实现就名不副实，``VOICEMEM_SENSEVOICE_MODEL`` 同理。
_LOCAL_LEGACY = {
    "asr":        "VOICEMEM_SENSEVOICE_MODEL",
    "vad":        "VOICEMEM_SILERO_VAD",
    "voiceprint": "VOICEMEM_SPEAKER_MODEL",
    "scene":      "VOICEMEM_ENVIRONMENT_MODEL",
    "e5":         "VOICEMEM_E5_MODEL",
}


def local_model_env(cap: str) -> str:
    """本地模型的环境变量名，由能力名派生：``vad`` → ``VOICEMEM_VAD_MODEL``。

    和 API 模型那边（``llm_config.env_name``）同一个规律。本地模型没有并进那张
    角色表，是因为两者填的东西不是一类：那边填模型名（``gpt-4o-mini``），这边
    填的是文件路径或 HF 仓库 id，而且和具体实现绑定——同样是 embedding 能力，
    走 API 填 ``text-embedding-3-small``，走本地填 ``intfloat/multilingual-e5-small``，
    合成一个变量只会让人填错。规律统一，语义分开。
    """
    return f"VOICEMEM_{cap.upper()}_MODEL"


def local_model_override(cap: str) -> str:
    """用户给这个能力指定的本地模型（新名字优先，旧名字仍认）；没给返回 ""。"""
    legacy = _LOCAL_LEGACY.get(cap, "")
    return (os.environ.get(local_model_env(cap), "").strip()
            or (os.environ.get(legacy, "").strip() if legacy else ""))


def model_path(name: str, cap: str | None = None, kind: str = "") -> Path:
    """取一个具体模型文件的路径；``cap`` 对应的环境变量优先级最高。

    ``cap`` 是能力名（``vad`` / ``asr`` / ``voiceprint`` …），对应的环境变量由它派生。
    ``kind`` 是按用途分的子目录（``vad`` / ``asr`` / ``speaker``），跟发布仓库
    zhifeixie/VoiceMem_Default_Models_Env 的布局一致。找不到就退回上一版的平铺布局——早先
    下载过的人不该因为换了组织方式就突然找不到模型。
    """
    if cap:
        explicit = local_model_override(cap)
        if explicit:
            return Path(explicit)
    root = models_dir()
    if kind:
        grouped = root / kind / name
        if grouped.exists():
            return grouped
    return root / name          # 旧的平铺布局；真不存在时由 require() 报错


def hf_model(kind: str, default_id: str, cap: str | None = None) -> str:
    """一个 HF 模型该从哪儿加载：``env`` > 本地离线包 > HF 仓库 id（自动下载）。

    离线包就是 ``<models>/<kind>/``，布局跟发布仓库 zhifeixie/VoiceMem_Default_Models_Env 一致
    （``embedding`` / ``scene`` / ``emotion`` …，一个用途一个文件夹）。没下载过就返回
    HF id，transformers 照旧首次运行时自动拉——**零配置的默认行为不变**，下载了
    离线包的人则整条链路不再需要网络。

    判定"这个目录里确实有个模型"看的是 config.json 或 .onnx，而不是目录是否存在：
    空目录（比如下载中断留下的）应该继续回退到 HF，而不是让加载器对着空目录报错。
    """
    if cap:
        explicit = local_model_override(cap)
        if explicit:
            return explicit
    local = models_dir() / kind
    if (local / "config.json").exists() or any(local.glob("*.onnx")):
        return str(local)
    return default_id


def require(path: Path, what: str, how: str = "") -> Path:
    """模型文件不在就报一句人能看懂的话，而不是让底层库抛个看不懂的错。

    提示要分两种人说：clone 了仓库的人跑 scripts/download_models.sh 就行；
    ``pip install voicemem`` 的人根本没有 scripts/ 目录，让他跑那条命令等于没说。
    看当前目录有没有那个脚本来决定说哪句。
    """
    if not Path(path).exists():
        if not how:
            if Path("scripts/download_models.sh").is_file():
                how = "bash scripts/download_models.sh models"
            else:
                how = ("git clone https://github.com/xzf-thu/VoiceMem && "
                       "bash VoiceMem/scripts/download_models.sh models")
        raise FileNotFoundError(
            f"{what} 找不到：{path}\n"
            f"下载：{how}\n"
            f"或用 VOICEMEM_MODELS_DIR 指向你已有的模型目录。"
        )
    return Path(path)
