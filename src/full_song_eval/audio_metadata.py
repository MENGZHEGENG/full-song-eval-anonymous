from __future__ import annotations

import contextlib
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioMetadata:
    path: Path
    duration_seconds: float | None
    sample_rate: int | None
    channels: int | None
    frames: int | None


def read_audio_metadata(path: Path) -> AudioMetadata:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        with contextlib.closing(wave.open(str(path), "rb")) as handle:
            frames = handle.getnframes()
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            duration = frames / sample_rate if sample_rate else None
            return AudioMetadata(path=path, duration_seconds=duration, sample_rate=sample_rate, channels=channels, frames=frames)
    return AudioMetadata(path=path, duration_seconds=None, sample_rate=None, channels=None, frames=None)
