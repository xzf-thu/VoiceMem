"""3D-Speaker ERes2Net 声纹向量常驻 worker。

voicemem 主程序（另一个 conda 环境，没装 funasr/modelscope）通过 subprocess
启动这个脚本，用「一行路径 -> 一行 JSON」的协议跨环境拿声纹向量：

    请求（stdin，一行一个）：  /path/to/audio.wav
    响应（stdout，一行一个）：  {"embedding": [0.01, -0.02, ...]}   # 192 维
                              或 {"error": "..."}

启动完成后先打印一行 "READY"，调用方据此确认模型已加载完毕。
所有非协议输出（下载进度条、版本检查提示等）必须走 stderr，不能污染 stdout。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def main() -> None:
    import soundfile as sf
    import sherpa_onnx
    from scipy.signal import resample_poly

    # 保留旧的 device 参数兼容性；sherpa-onnx 的 CPU 推理已足够快。
    _device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    # 这里是独立子进程（可能跑在另一个 conda 环境里），所以不 import voicemem，
    # 路径解析逻辑与 voicemem/utils/common/paths.py 保持一致：
    #   VOICEMEM_SPEAKER_MODEL > <models 目录>/speaker/<name> > <models 目录>/<name>
    # speaker/ 子目录是发布仓库 zhifeixie/VoiceMem_default 的布局；平铺那条是早先
    # 下载过的人的位置，一并认，免得换了组织方式就突然找不到。
    # 注：旧代码用 parents[2]，那是 voicemem/utils/，指到了一个不存在的目录——
    # 不设 VOICEMEM_SPEAKER_MODEL 时声纹开箱就找不到模型。仓库根是 parents[4]。
    #
    # parents[4] 只在**从仓库跑**时是仓库根；pip 装之后它是 site-packages，那底下
    # 没有 models/。所以跟 paths.py:19 一样要回退到 cwd 下的 models/——否则装了包
    # 的用户即使 models/ 就在手边，声纹也只会报 "campplus_worker 启动失败: ''"
    # （stderr 被 DEVNULL 吞了，连原因都看不到）。
    from pathlib import Path

    _NAME = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    model_path = os.environ.get("VOICEMEM_SPEAKER_MODEL")
    if not model_path:
        _root = os.environ.get("VOICEMEM_MODELS_DIR")
        _repo = Path(__file__).resolve().parents[4] / "models"
        _dir = Path(_root) if _root else (_repo if _repo.is_dir() else Path("models"))
        _grouped = _dir / "speaker" / _NAME
        model_path = str(_grouped if _grouped.exists() else _dir / _NAME)
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"声纹模型 {_NAME} 找不到：{model_path}\n"
            f"下载：bash scripts/download_models.sh models\n"
            f"或用 VOICEMEM_SPEAKER_MODEL / VOICEMEM_MODELS_DIR 指到已有的位置。"
        )
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model_path, num_threads=2)
    )
    window = int(float(os.environ.get("VOICEMEM_SPEAKER_WINDOW", "3.0")) * 16000)
    hop = int(float(os.environ.get("VOICEMEM_SPEAKER_HOP", "1.5")) * 16000)

    print("READY", flush=True)

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        if path == "__exit__":
            break
        try:
            audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if sample_rate != 16000:
                audio = resample_poly(audio, 16000, int(sample_rate))
            audio = np.asarray(audio, dtype=np.float32)
            # 一整段只取一个向量很容易被开头噪声/尾部静音带偏；切成短窗，
            # 用 RMS 门控过滤静音，再平均多个有效窗口。
            if len(audio) <= window:
                chunks = [audio]
            else:
                chunks = [audio[i:i + window] for i in range(0, len(audio) - window + 1, hop)]
            embeddings = []
            for chunk in chunks:
                if len(chunk) < 16000 or float(np.sqrt(np.mean(chunk * chunk))) < 0.005:
                    continue
                stream = extractor.create_stream()
                stream.accept_waveform(16000, chunk)
                stream.input_finished()
                if extractor.is_ready(stream):
                    embeddings.append(np.asarray(extractor.compute(stream), dtype=np.float32))
            if not embeddings:
                raise RuntimeError("speaker embedding 未达到最小语音长度")
            vec = np.mean(embeddings, axis=0)
            vec /= np.linalg.norm(vec) + 1e-8
            vec = vec.tolist()
            print(json.dumps({"embedding": vec}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
