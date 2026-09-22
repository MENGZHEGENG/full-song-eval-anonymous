from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any, Iterable


def is_valid_pitch(value: float | int | None) -> bool:
    if value is None:
        return False
    try:
        pitch = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(pitch) and pitch > 0.0


def hz_to_midi(hz: float) -> float:
    if hz <= 0.0 or not math.isfinite(hz):
        raise ValueError("hz must be a positive finite value")
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def summarize_pitch_track(
    f0_hz: Iterable[float | int | None], *, large_jump_threshold_semitones: float = 7.0
) -> dict[str, Any]:
    values = list(f0_hz)
    voiced_hz = [float(value) for value in values if is_valid_pitch(value)]
    voiced_midi = [hz_to_midi(value) for value in voiced_hz]
    frame_count = len(values)
    voiced_frame_count = len(voiced_hz)
    consecutive_steps = [
        abs(next_midi - prev_midi)
        for prev_midi, next_midi in zip(voiced_midi, voiced_midi[1:])
        if math.isfinite(prev_midi) and math.isfinite(next_midi)
    ]
    hz_stats = _stats(voiced_hz)
    midi_stats = _stats(voiced_midi)
    pitch_range = None
    if midi_stats["min"] is not None and midi_stats["max"] is not None:
        pitch_range = midi_stats["max"] - midi_stats["min"]
    mean_step = statistics.mean(consecutive_steps) if consecutive_steps else None
    max_step = max(consecutive_steps) if consecutive_steps else None
    large_jump_rate = None
    if consecutive_steps:
        large_jump_count = sum(step >= large_jump_threshold_semitones for step in consecutive_steps)
        large_jump_rate = large_jump_count / len(consecutive_steps)
    return {
        "pitch_frame_count": frame_count,
        "voiced_frame_count": voiced_frame_count,
        "voiced_fraction": voiced_frame_count / frame_count if frame_count else None,
        "median_pitch_hz": hz_stats["median"],
        "mean_pitch_hz": hz_stats["mean"],
        "min_pitch_hz": hz_stats["min"],
        "max_pitch_hz": hz_stats["max"],
        "median_pitch_midi": midi_stats["median"],
        "mean_pitch_midi": midi_stats["mean"],
        "pitch_range_semitones": pitch_range,
        "mean_abs_pitch_step_semitones": mean_step,
        "max_abs_pitch_step_semitones": max_step,
        "large_jump_rate": large_jump_rate,
    }


def extract_librosa_pyin_features(
    path: Path,
    *,
    sample_rate: int = 16000,
    fmin_hz: float = 65.0,
    fmax_hz: float = 1047.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    try:
        import librosa
    except ModuleNotFoundError as exc:
        raise RuntimeError("librosa is required for vocal melody extraction; install a project environment with the melody dependencies") from exc

    audio, loaded_sample_rate = librosa.load(path, sr=sample_rate, mono=True, duration=max_seconds)
    f0, voiced_flag, voiced_probability = librosa.pyin(
        audio,
        fmin=fmin_hz,
        fmax=fmax_hz,
        sr=loaded_sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    summary = summarize_pitch_track(f0.tolist())
    valid_probs = [
        float(probability) for probability in voiced_probability.tolist() if math.isfinite(float(probability))
    ]
    summary.update(
        {
            "audio_uri": str(path),
            "analysis_sample_rate": loaded_sample_rate,
            "analysis_duration_seconds": len(audio) / loaded_sample_rate if loaded_sample_rate else None,
            "pitch_algorithm": "librosa.pyin",
            "fmin_hz": fmin_hz,
            "fmax_hz": fmax_hz,
            "frame_length": frame_length,
            "hop_length": hop_length,
            "voiced_probability_mean": statistics.mean(valid_probs) if valid_probs else None,
            "librosa_voiced_flag_fraction": float(voiced_flag.mean()) if len(voiced_flag) else None,
        }
    )
    return summary
