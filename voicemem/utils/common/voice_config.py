"""声纹与原声存储配置。

通过环境变量选场景：
    VOICE_SCENE=medical   → 声纹 + 原声都保留
    VOICE_SCENE=companion → 只保留声纹
    VOICE_SCENE=diary     → 两者都不保留
    VOICE_SCENE=default   → 只保留声纹（默认）

也可单独覆盖：
    VOICE_ENABLE_VOICEPRINT=true/false
    VOICE_RETAIN_RAW_AUDIO=true/false
    VOICE_RAW_AUDIO_DIR=/path/to/dir
    VOICE_MATCH_THR=0.50
    VOICE_CAND_THR=0.40
    VOICE_MERGE_THR=0.65
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class VoiceStoreConfig:
    retain_voiceprint: bool = True
    retain_raw_audio: bool = False
    raw_audio_dir: Optional[Path] = None
    match_threshold: float = 0.50
    candidate_threshold: float = 0.40
    # 两个 person_id 的候选声纹永久合并（并改名）用的门槛，故意比 match_threshold
    # 更高：match_threshold 只是"这一句话像不像"，判错了顶多软性地污染一点画像
    # （下次真实观测还能把中心拉回来）；而合并是硬性的、几乎不可逆的操作——
    # 两个画像焊成一个、名字互相覆盖。实测过 0.542（刚过 match_threshold 一点点）
    # 就足够把两个真实的不同人（Nancy/Jennifer，本身声音就分不太开）永久焊死，
    # 合并完再也拆不回去，所以要求明显更高的确信度才敢做。
    merge_threshold: float = 0.65

    _PRESETS: dict = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def for_scene(cls, scene: str) -> "VoiceStoreConfig":
        presets = {
            "medical":   cls(retain_voiceprint=True,  retain_raw_audio=True),
            "legal":     cls(retain_voiceprint=True,  retain_raw_audio=True),
            "companion": cls(retain_voiceprint=True,  retain_raw_audio=False),
            "diary":     cls(retain_voiceprint=False, retain_raw_audio=False),
            "default":   cls(retain_voiceprint=True,  retain_raw_audio=False),
        }
        return presets.get(scene, cls())

    @classmethod
    def from_env(cls) -> "VoiceStoreConfig":
        scene = os.environ.get("VOICE_SCENE", "default")
        cfg = cls.for_scene(scene)

        # 环境变量可逐字段覆盖场景预设
        if "VOICE_ENABLE_VOICEPRINT" in os.environ:
            cfg.retain_voiceprint = os.environ["VOICE_ENABLE_VOICEPRINT"].lower() == "true"
        if "VOICE_RETAIN_RAW_AUDIO" in os.environ:
            cfg.retain_raw_audio = os.environ["VOICE_RETAIN_RAW_AUDIO"].lower() == "true"
        if "VOICE_RAW_AUDIO_DIR" in os.environ:
            cfg.raw_audio_dir = Path(os.environ["VOICE_RAW_AUDIO_DIR"])
        if "VOICE_MATCH_THR" in os.environ:
            cfg.match_threshold = float(os.environ["VOICE_MATCH_THR"])
        if "VOICE_CAND_THR" in os.environ:
            cfg.candidate_threshold = float(os.environ["VOICE_CAND_THR"])
        if "VOICE_MERGE_THR" in os.environ:
            cfg.merge_threshold = float(os.environ["VOICE_MERGE_THR"])

        return cfg
