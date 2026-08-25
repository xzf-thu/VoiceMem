#!/usr/bin/env bash
# 下载 voicemem 用到的**本地模型**，按用途分目录：
#
#   models/
#     vad/        silero_vad.onnx                                  判「说完了」
#     asr/        funasr-paraformer-zh-streaming                   流式 ASR（默认，中文更准）
#                 sherpa-onnx-streaming-zipformer-bilingual-zh-en… 流式 ASR（回退，纯 onnx 不依赖 torch）
#     speaker/    3dspeaker_speech_eres2net_base_sv_zh-cn…onnx     声纹
#     embedding/  intfloat/multilingual-e5-small                   记忆向量 + slot 分类（共用一份）
#     scene/      MIT/ast-finetuned-audioset-10-10-0.4593          声学场景
#     emotion/    FunAudioLLM/SenseVoiceSmall                      情绪 + 精转写
#     tts/        （可选）piper voice .onnx，只有 TTS_BACKEND=local 才要
#
# 下完这些整条链路不再需要网络（除了回复模型那次 API 调用）。不下也能跑——
# 代码会回退到 HF id，首次用到时 transformers 自动拉。
#
# 不在这里的两样：
#   · 回复模型 —— PEFT adapter（180MB），挂在 Qwen/Qwen3.6-35B-A3B 上，基座另取。
#     加 --reply-adapter 拉，见下面。
#   · 情绪归因用的 Qwen2.5-Omni —— 目前没有发布微调版，代码默认用官方
#     Qwen/Qwen2.5-Omni-3B，首次用到时自动拉。有自己的微调版就
#     export VOICEMEM_OMNI_MODEL=/你的/路径 指过去。
#
# 用法（从仓库根目录）:
#   bash scripts/download_models.sh                  # 拉 models/ 那一套
#   bash scripts/download_models.sh /path/to/models  # 换目标目录
#   bash scripts/download_models.sh --reply-adapter  # 额外拉回复模型 adapter
#   VOICEMEM_FROM_UPSTREAM=1 bash scripts/download_models.sh   # 改从各家官方源逐个拉
set -euo pipefail

DEST="models"
WANT_SLM=0
for arg in "$@"; do
  case "$arg" in
    --reply-adapter|--slm) WANT_SLM=1 ;;   # --slm 是旧名字，留着不破坏老命令
    *)     DEST="$arg" ;;
  esac
done

REPO="${VOICEMEM_MODELS_REPO:-zhifeixie/VoiceMem_Default_Models_Env}"
# 注意：仓库名里写着 Qwen25_omni，但内容是 Qwen3.6-35B 的回复 adapter。
ADAPTER_REPO="${VOICEMEM_REPLY_ADAPTER_REPO:-${VOICEMEM_SLM_REPO:-zhifeixie/VoiceMem_SLM_Qwen25_omni}}"
mkdir -p "${DEST}"

if [ "${VOICEMEM_FROM_UPSTREAM:-0}" != "1" ]; then
  echo "[1/2] 从 ${REPO} 拉全套本地模型 …"
  python3 - "${REPO}" "${DEST}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])   # 断点续传，重复跑不重下
PY
else
  # 逐个从各家官方公开来源拉（HF 不可达、或想核对来源与许可时用）
  REL="https://github.com/k2-fsa/sherpa-onnx/releases/download"
  ASR_DIR="sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
  mkdir -p "${DEST}"/{vad,asr,speaker,embedding,scene,emotion}

  echo "[1/2] 官方源：VAD silero（MIT）…"
  curl -L -o "${DEST}/vad/silero_vad.onnx" "${REL}/asr-models/silero_vad.onnx"

  echo "      官方源：回退流式 ASR ${ASR_DIR}（Apache-2.0, k2-fsa）…"
  [ -d "${DEST}/asr/${ASR_DIR}" ] || curl -L "${REL}/asr-models/${ASR_DIR}.tar.bz2" | tar xj -C "${DEST}/asr"

  # 注意官方 release tag 拼写就是 recongition
  echo "      官方源：声纹 3D-Speaker ERes2Net（Apache-2.0）…"
  curl -L -o "${DEST}/speaker/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" \
    "${REL}/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

  echo "      官方源：embedding / scene / emotion（HF 各自原仓库）…"
  python3 - "${DEST}" <<'PY'
import sys
from huggingface_hub import snapshot_download
dest = sys.argv[1]
# 只拉真正会被加载的那一份权重。这些仓库同时放了 safetensors / pytorch_model.bin /
# onnx 各种量化变体 / openvino，整仓拉是 4.2G，只取需要的约 1.4G——下载、以及
# 之后传到发布仓库都省一大截。
SKIP = ["*.bin", "onnx/*", "openvino/*", "*.tflite", "*.h5", "*.msgpack", "coreml/*"]
for kind, repo, skip in [# 默认流式 ASR。不带它的话 funasr 会在用户说第一句话时才现下 848M
                         ("asr/funasr-paraformer-zh-streaming",
                          "funasr/paraformer-zh-streaming", ["example/*", "fig/*"]),
                         ("embedding", "intfloat/multilingual-e5-small", SKIP),
                         ("scene",     "MIT/ast-finetuned-audioset-10-10-0.4593", SKIP),
                         # SenseVoice 的权重就是 model.pt，不能按 *.bin 那套排除
                         ("emotion",   "FunAudioLLM/SenseVoiceSmall", ["*.onnx", "*.tflite"])]:
    print(f"        {kind} ← {repo}")
    snapshot_download(repo_id=repo, local_dir=f"{dest}/{kind}", ignore_patterns=skip)
PY
fi

if [ "${WANT_SLM}" = "1" ]; then
  echo "[2/2] 回复模型 adapter ${ADAPTER_REPO} …"
  python3 - "${ADAPTER_REPO}" "${DEST}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=f"{sys.argv[2]}/reply_adapter")
PY
  echo "      用它：export VOICEMEM_REPLY_ADAPTER=${DEST}/reply_adapter"
  echo "      这是 adapter 不是完整模型——基座 Qwen/Qwen3.6-35B-A3B 要另取，按它自己的 license。"
  echo "      跑：python examples/03_simple_agent_with_voicemem_memory.py"
else
  echo "[2/2] 跳过回复模型 adapter（要的话加 --reply-adapter）"
fi

echo
echo "完成：${DEST}/ 下按用途分好（vad / asr / speaker / embedding / scene / emotion）。"
echo "代码会优先用这里的本地模型，没有的项自动回退到 HF 下载。"
