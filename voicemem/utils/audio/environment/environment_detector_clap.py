"""Optional CLAP backend used only for final environment-memory writes.

AST remains the low-latency provisional detector.  This module is deliberately
optional: it is loaded only when ``VOICEMEM_CLAP_CHECKPOINT`` is configured.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


PROMPTS = {
    "air_conditioner": ["air conditioner hum", "HVAC ventilation fan noise"],
    "airplane": ["airplane flying overhead", "aircraft engine sound"],
    "car_horn": ["car horn honking", "sound of a car horn"],
    "cat": ["cat meowing", "sound of a cat"],
    "chirping_birds": ["birds chirping", "sound of birds singing"],
    "children_playing": ["children playing and shouting", "sound of children at play"],
    "coughing": ["a person coughing", "sound of coughing"],
    "footsteps": ["footsteps walking", "sound of footsteps"],
    "helicopter": ["helicopter flying overhead", "helicopter rotor blades sound"],
    "keyboard_typing": ["computer keyboard typing", "typing on a keyboard"],
    "mouse_click": ["computer mouse clicking", "clicking a computer mouse button"],
    "rain": ["rain falling", "sound of rainfall"],
    "siren": ["emergency vehicle siren", "police or ambulance siren"],
    "street_music": ["street music performance", "music playing in the street"],
    "thunderstorm": ["thunderstorm with thunder", "sound of thunder"],
    "train": ["train passing on railway tracks", "railroad train sound"],
    "vacuum_cleaner": ["vacuum cleaner running", "sound of a vacuum cleaner"],
    "washing_machine": ["washing machine running", "laundry washing machine noise"],
    "wind": ["wind blowing", "sound of wind"],
}


class CLAPEnvironmentDetector:
    """Fixed-label, long-window CLAP detector.

    The text prototypes are computed once.  Each input is split into 4-second
    windows with a 2-second hop and the audio/text cosine scores are averaged.
    """

    def __init__(self, checkpoint: str | Path | None = None, min_score: float | None = None,
                 min_margin: float | None = None) -> None:
        self._checkpoint = str(checkpoint or os.environ.get("VOICEMEM_CLAP_CHECKPOINT", ""))
        self._min_score = float(min_score if min_score is not None else os.environ.get("VOICEMEM_CLAP_MIN_SCORE", "0.15"))
        self._min_margin = float(min_margin if min_margin is not None else os.environ.get("VOICEMEM_CLAP_MIN_MARGIN", "0.03"))
        self._model = None
        self._labels = list(PROMPTS)
        self._text = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self._checkpoint:
            raise RuntimeError("VOICEMEM_CLAP_CHECKPOINT is not configured")
        import laion_clap

        self._model = laion_clap.CLAP_Module(
            enable_fusion=False,
            amodel="HTSAT-tiny",
            device=os.environ.get("VOICEMEM_CLAP_DEVICE") or None,
        )
        self._model.load_ckpt(self._checkpoint, verbose=False)
        prompts = [prompt for label in self._labels for prompt in PROMPTS[label]]
        text = np.asarray(self._model.get_text_embedding(prompts)).reshape(len(self._labels), 2, -1).mean(axis=1)
        self._text = text / (np.linalg.norm(text, axis=1, keepdims=True) + 1e-8)

    @staticmethod
    def _load_audio(path: Path) -> np.ndarray:
        import soundfile as sf
        from scipy.signal import resample_poly

        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sample_rate != 48000:
            from math import gcd
            divisor = gcd(int(sample_rate), 48000)
            audio = resample_poly(audio, 48000 // divisor, int(sample_rate) // divisor)
        return np.asarray(audio, dtype=np.float32)

    def detect_full(self, audio_path: Path) -> dict:
        self._load()
        audio = self._load_audio(audio_path)
        window, hop = 48000 * 4, 48000 * 2
        starts = range(0, max(1, len(audio) - window + 1), hop)
        chunks = [audio[start:start + window] for start in starts]
        if not chunks:
            chunks = [audio]
        embeddings = np.asarray(self._model.get_audio_embedding_from_data(chunks))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        scores = (embeddings @ self._text.T).mean(axis=0)
        order = np.argsort(scores)[::-1]
        top = float(scores[order[0]])
        second = float(scores[order[1]]) if len(order) > 1 else -1.0
        pairs = []
        if top >= self._min_score and top - second >= self._min_margin:
            pairs = [(self._labels[int(index)], float(scores[index])) for index in order[:5]]
        embedding = embeddings.mean(axis=0)
        embedding /= np.linalg.norm(embedding) + 1e-8
        return {"pairs": pairs, "music": None, "abnormal": [], "embedding": embedding}
