"""EmotionDetector 的论文对齐替代实现。

论文的 φ(x_t) 是"从原始音频算连续 V/A（valence/arousal），负面显著的轮次
再做多模态归因"——这跟 ``emotion_detector.py`` 用的 emotion2vec+
（一个独立的、九分类的音频情感分类器，模型本身跟 φ(x_t) 的描述对不上）
是两回事：emotion2vec 是"这段音频属于哪个固定情绪类别"，φ(x_t) 是"这段
音频的声学状态是什么，且只有值得关注的轮次才值得花大代价去归因"。

真实实现分两层，对应 ``voicemem/emotion/`` 已有的、之前完全没被
``core.py`` 引用过的组件：
  1. 韵律 VAD（``vad_audio.HeuristicWavVADEstimator``）——纯声学特征，
     每轮都跑，便宜。
  2. Qwen2.5-Omni 多模态归因（``attribution_qwen_omni.QwenOmniEmotionAttributor``）
     ——只有 VAD 判定"负面显著"（``vad_trigger.is_negative_vad_significant``）
     的轮次才加载/调用，避免每轮都背上多模态 LLM 的推理成本。模型懒加载，
     加载失败（没有 GPU/权重未下载/显存不够）时优雅退回纯 VAD 的粗分类，
     不会让 Ingest() 崩掉。

返回值签名保持跟 ``EmotionDetector.detect() -> str`` 完全一样（返回一个
情绪标签字符串），``core.py::Ingest()`` 里十几处消费 ``emotion`` 字符串的
下游逻辑（右脑 heartnote 情绪锚点、inner OS 生成、情绪特质抽取……）不需要
跟着改，只是这个字符串现在是真的从原始音频声学信号 + （必要时）多模态
LLM 算出来的，不再是一个跟音频波形无关的九分类标签。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from voicemem.utils.audio.emotion.layer import EmotionLayerConfig
from voicemem.utils.audio.emotion.vad_audio import HeuristicWavVADEstimator, VADEstimator
from voicemem.utils.audio.emotion.vad_trigger import is_negative_vad_significant


def _vad_to_label(valence: float, arousal: float) -> str:
    """VAD 象限 → 粗粒度情绪标签，供「不显著」或 Qwen-Omni 不可用时兜底。

    标签词表对齐 ``anchor_router._CANONICAL_EMOTIONS``（右脑情绪锚点用的
    固定 8 类），保证兜底路径产出的标签也能被下游正确识别/归一化，而不是
    像 emotion2vec 的"中性"兜底那样，模型没触发就永远给同一个值。

    输出按**库语言**写（见 ``voicemem.lang``）：内部值是中文，但这个标签会存进
    记忆、也会显示给用户，英文库里冒出「开心」就是 bug。英文写法同样在
    anchor_router 的英文词表里，读回来能归一回同一个内部值。
    """
    from voicemem.lang import display_emotion
    if valence >= 0.15:
        canon = "开心" if arousal >= 0.4 else "平静"
    elif valence <= -0.15:
        if arousal >= 0.55:
            canon = "焦虑"
        elif arousal >= 0.35:
            canon = "委屈"
        else:
            canon = "悲伤"
    else:
        canon = "平静" if arousal < 0.4 else "纠结"
    return display_emotion(canon)


def _load_omni(model_path: str, *, device_map: str) -> tuple[Any, Any, Any]:
    """加载 Qwen2.5-Omni processor/tokenizer/model 三件套（bf16，只要文本
    输出，用 Thinker 子模型比完整 Omni 省显存）。跟
    ``examples/load_qwen_omni_attributor.py::load_omni`` 逻辑一致，内联在
    这里而不是 import 那个脚本——``examples/`` 不是包的一部分，不应该被
    运行时代码依赖。"""
    import torch
    from transformers import AutoTokenizer, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    try:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True,
        )
    except TypeError:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True,
        )
    model.eval()
    return processor, tokenizer, model


class PaperAlignedEmotionDetector:
    """常驻 VAD 估计器 + 懒加载 Qwen2.5-Omni 归因器，接口跟
    ``EmotionDetector`` 兼容（``detect(audio_path) -> str``），
    ``core.py::_get_emotion_detector()`` 直接换这一个类即可，不用改调用方。
    """

    def __init__(
        self,
        *,
        vad_estimator: VADEstimator | None = None,
        layer_config: EmotionLayerConfig | None = None,
        omni_model_path: str | None = None,
        omni_device_map: str | None = None,
    ) -> None:
        self._vad = vad_estimator or HeuristicWavVADEstimator()
        self._config = layer_config or EmotionLayerConfig()
        # 环境变量优先，未设置时退回 3B（比 7B 省显存，这台机器上两个都已
        # 缓存过，见 Phase 5 隔离测试）——真实部署按显卡情况自己配置。
        self._omni_model_path = omni_model_path or os.environ.get("VOICEMEM_OMNI_MODEL", "Qwen/Qwen2.5-Omni-3B")
        # "auto" 在显存被其它进程占满的共享多卡机器上可能把模型切分到繁忙
        # 的卡上导致 OOM（Phase 5 隔离测试实测复现）——真实部署应通过
        # VOICEMEM_OMNI_DEVICE 指定一张有空闲显存的卡；这里默认 "auto"
        # 只是不强加假设，不代表对每台机器都是安全默认值。
        self._omni_device_map = omni_device_map or os.environ.get("VOICEMEM_OMNI_DEVICE", "auto")
        self._attributor: Any = None       # 懒加载；None=还没试过，False=加载失败过（不重试）
        self._attributor_failed = False

    def _ensure_attributor(self):
        if self._attributor is not None:
            return self._attributor
        if self._attributor_failed:
            return None
        try:
            from voicemem.utils.audio.emotion.attribution_qwen_omni import QwenOmniEmotionAttributor

            processor, tokenizer, model = _load_omni(self._omni_model_path, device_map=self._omni_device_map)
            self._attributor = QwenOmniEmotionAttributor(processor=processor, model=model, tokenizer=tokenizer)
        except Exception as e:
            # 退回纯声学象限分类之后，情绪只看语音的 valence/arousal，完全不看内容——
            # 平铺直叙说"我喜欢草莓"会落进低 valence 象限判成悲伤。右脑的情感记录
            # 全建立在这个标签上，所以它一瘸，右脑那半就整个不可信了。
            # 缺 torchvision 是最常见的原因（transformers 加载 Qwen2.5-Omni 时要
            # Qwen2VLVideoProcessor，它 import torchvision），而且没人会想到。
            # 所以这条必须显眼、且要直接给出解法，不能只打一行异常了事。
            hint = ""
            if "torchvision" in str(e).lower() or "Torchvision" in str(e):
                hint = ("\n  [emotion]   → 缺 torchvision。装上就好：pip install torchvision"
                        "\n  [emotion]     （不装的话情绪只靠声学象限判，右脑的情感记忆不可信）")
            print(f"  [emotion] ⚠ Qwen-Omni 归因器加载失败，情绪退回纯声学粗分类: {e}{hint}",
                  flush=True)
            self._attributor_failed = True
            return None
        return self._attributor

    def detect(self, audio_path: Path) -> str:
        """返回情绪标签字符串（跟 ``EmotionDetector.detect()`` 接口兼容）。"""
        try:
            vad = self._vad.estimate(str(audio_path))
        except Exception as e:
            print(f"  [emotion] VAD 估计失败: {e}", flush=True)
            return "未知"

        if not is_negative_vad_significant(vad, self._config):
            return _vad_to_label(vad.valence, vad.arousal)

        attributor = self._ensure_attributor()
        if attributor is None:
            return _vad_to_label(vad.valence, vad.arousal)

        try:
            import uuid as _uuid

            from voicemem.utils.audio.emotion.types import TurnEmotionRecord

            turn = TurnEmotionRecord(turn_id=_uuid.uuid4().hex, session_id="ingest", vad=vad)
            result = attributor.analyze_turn_with_audio(
                audio_path=str(audio_path), asr_text=None,
                left_memory_block="", emotion_graph_context=None, turn=turn,
            )
            label = (result.emotion.label or "").strip()
            print(f"  [emotion] Qwen-Omni 归因 → {label!r} (VAD={vad})", flush=True)
            return label or _vad_to_label(vad.valence, vad.arousal)
        except Exception as e:
            print(f"  [emotion] Qwen-Omni 归因失败，退回 VAD 粗分类: {e}", flush=True)
            return _vad_to_label(vad.valence, vad.arousal)
