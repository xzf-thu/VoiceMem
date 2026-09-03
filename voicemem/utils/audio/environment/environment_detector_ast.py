"""AST-based environmental sound detector.

This is the default, immediate-hint environmental sound backend: the Audio
Spectrogram Transformer fine-tuned on AudioSet, covering background-scene
description, music/humming detection, abnormal-sound detection, and the raw
embedding used for familiar-location clustering. It keeps the same compact
``detect_full`` contract used by :class:`VoiceMem`, so callers do not need to
know which audio model is configured.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from voicemem.utils.audio.environment.audioset_labels import (
    _ABNORMAL_KEYWORDS,
    _MUSIC_KEYWORDS,
    _SPEECH_LABEL_INDICES,
)


def _default_model() -> str:
    from voicemem.utils.common.paths import hf_model
    return hf_model("scene", "MIT/ast-finetuned-audioset-10-10-0.4593", "scene")


DEFAULT_MODEL = _default_model()


class ASTEnvironmentDetector:
    """AudioSet AST detector, lazily loaded once per VoiceMem instance."""

    def __init__(self, device: str | None = None, threshold: float = 0.20, top_k: int = 8) -> None:
        self._device = device
        self._threshold = threshold
        self._top_k = top_k
        self._processor = None
        self._model = None
        self._labels: list[str] = []

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import ASTForAudioClassification, AutoProcessor

        source = os.environ.get("VOICEMEM_ENVIRONMENT_MODEL_DIR") or os.environ.get(
            "VOICEMEM_ENVIRONMENT_MODEL", DEFAULT_MODEL
        )
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoProcessor.from_pretrained(source)
        self._model = ASTForAudioClassification.from_pretrained(source).to(device).eval()
        self._device = device
        self._labels = [self._model.config.id2label[i] for i in range(self._model.config.num_labels)]

    @staticmethod
    def _load_audio(audio_path: Path) -> np.ndarray:
        from math import gcd
        from scipy.signal import resample_poly
        import soundfile as sf

        audio, sample_rate = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            divisor = gcd(int(sample_rate), 16000)
            audio = resample_poly(audio, 16000 // divisor, int(sample_rate) // divisor)
        return np.asarray(audio, dtype=np.float32)

    def _infer(self, audio_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
        self._load()
        try:
            audio = self._load_audio(audio_path)
            if not len(audio):
                return None
            import torch

            inputs = self._processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {name: value.to(self._device) for name, value in inputs.items()}
            with torch.inference_mode():
                output = self._model(**inputs, output_hidden_states=True)
                scores = torch.sigmoid(output.logits[0]).cpu().numpy()
                embedding = output.hidden_states[-1][0].mean(dim=0).cpu().numpy()
            return scores, embedding
        except Exception as exc:
            print(f"  [env] AST inference skipped: {exc}", flush=True)
            return None

    def _pairs_from_scores(self, scores: np.ndarray) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        for index in np.argsort(scores)[::-1]:
            if len(results) >= self._top_k:
                break
            # AudioSet keeps its speech family at these canonical indices.  AST
            # is trained on the same ontology, so preserve the previous policy.
            if int(index) in _SPEECH_LABEL_INDICES or scores[index] < self._threshold:
                continue
            results.append((self._labels[int(index)], float(scores[index])))
        return results

    def _keyword_hits(self, scores: np.ndarray, keywords: list[str]) -> list[tuple[str, float]]:
        hits = [
            (label, float(scores[index]))
            for index, label in enumerate(self._labels)
            if scores[index] >= self._threshold and any(word in label.lower() for word in keywords)
        ]
        return sorted(hits, key=lambda item: -item[1])[: self._top_k]

    def detect_full(self, audio_path: Path) -> dict:
        inferred = self._infer(audio_path)
        if inferred is None:
            return {"pairs": [], "music": None, "abnormal": [], "embedding": None}
        scores, embedding = inferred
        music = self._keyword_hits(scores, _MUSIC_KEYWORDS)
        return {
            "pairs": self._pairs_from_scores(scores),
            "music": {"labels": music, "embedding": embedding} if music else None,
            "abnormal": self._keyword_hits(scores, _ABNORMAL_KEYWORDS),
            "embedding": embedding,
        }
