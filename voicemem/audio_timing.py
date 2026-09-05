"""Provider-neutral timing metadata for streamed speech output."""
from __future__ import annotations

from dataclasses import dataclass


OUTPUT_SAMPLE_RATE = 24000


@dataclass(frozen=True)
class TextTimestamp:
    """Map a half-open code-point range to samples relative to one segment."""

    text_start: int
    text_end: int
    audio_start_samples: int
    audio_end_samples: int
    sample_rate: int = OUTPUT_SAMPLE_RATE
    confidence: float = 1.0


@dataclass(frozen=True)
class TimedAudioChunk:
    """Optional enriched TTS chunk after conversion to the output sample rate."""

    pcm: bytes
    timestamps: tuple[TextTimestamp, ...] = ()
    sample_rate: int = OUTPUT_SAMPLE_RATE
