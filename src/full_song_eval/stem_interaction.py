from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any


def _load_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required for stem interaction diagnostics") from exc
    return np


def read_wav_mono_float(path: Path, max_seconds: float | None = None) -> tuple[Any, int]:
    np = _load_numpy()
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        if max_seconds is not None:
            frames = min(frames, int(max_seconds * sample_rate))
        raw = handle.readframes(frames)
    if sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sample_width} for {path}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def _safe_mean_square(audio: Any) -> float | None:
    if len(audio) == 0:
        return None
    np = _load_numpy()
    return float(np.mean(audio * audio))


def _safe_rms(audio: Any) -> float | None:
    mean_square = _safe_mean_square(audio)
    return math.sqrt(mean_square) if mean_square is not None else None


def _safe_cosine(first: Any, second: Any) -> float | None:
    np = _load_numpy()
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom == 0.0:
        return None
    return float(np.dot(first, second) / denom)


def _safe_centered_correlation(first: Any, second: Any) -> float | None:
    np = _load_numpy()
    first_centered = first - float(np.mean(first))
    second_centered = second - float(np.mean(second))
    return _safe_cosine(first_centered, second_centered)


def paired_stem_interaction_features(
    vocal_path: Path, accompaniment_path: Path, max_seconds: float | None = None
) -> dict[str, Any]:
    vocal, vocal_sample_rate = read_wav_mono_float(vocal_path, max_seconds=max_seconds)
    accompaniment, accompaniment_sample_rate = read_wav_mono_float(accompaniment_path, max_seconds=max_seconds)
    sample_count = min(len(vocal), len(accompaniment))
    vocal = vocal[:sample_count]
    accompaniment = accompaniment[:sample_count]
    vocal_mean_square = _safe_mean_square(vocal)
    accompaniment_mean_square = _safe_mean_square(accompaniment)
    energy_total = None
    vocal_energy_share = None
    if vocal_mean_square is not None and accompaniment_mean_square is not None:
        energy_total = vocal_mean_square + accompaniment_mean_square
        if energy_total > 0.0:
            vocal_energy_share = vocal_mean_square / energy_total
    centered_correlation = _safe_centered_correlation(vocal, accompaniment) if sample_count else None
    return {
        "vocal_stem_uri": str(vocal_path),
        "accompaniment_stem_uri": str(accompaniment_path),
        "vocal_sample_rate": vocal_sample_rate,
        "accompaniment_sample_rate": accompaniment_sample_rate,
        "aligned_sample_count": int(sample_count),
        "aligned_duration_seconds": sample_count / vocal_sample_rate if vocal_sample_rate else None,
        "vocal_rms_amplitude": _safe_rms(vocal),
        "accompaniment_rms_amplitude": _safe_rms(accompaniment),
        "vocal_energy_share": vocal_energy_share,
        "raw_cosine_similarity": _safe_cosine(vocal, accompaniment) if sample_count else None,
        "centered_correlation": centered_correlation,
        "absolute_centered_correlation": abs(centered_correlation) if centered_correlation is not None else None,
    }
