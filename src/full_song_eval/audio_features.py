from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import Any

from full_song_eval.audio_metadata import read_audio_metadata


def _pcm_values(path: Path, max_frames: int | None = None) -> tuple[list[int], int] | tuple[None, int]:
    with wave.open(str(path), "rb") as handle:
        sample_width = handle.getsampwidth()
        frames_to_read = handle.getnframes() if max_frames is None else min(max_frames, handle.getnframes())
        raw = handle.readframes(frames_to_read)
    if sample_width == 1:
        values = [sample - 128 for sample in raw]
        return values, 128
    if sample_width == 2:
        count = len(raw) // 2
        return list(struct.unpack(f"<{count}h", raw)), 32768
    if sample_width == 4:
        count = len(raw) // 4
        return list(struct.unpack(f"<{count}i", raw)), 2147483648
    return None, 1


def extract_wav_features(path: Path, max_seconds: float | None = None) -> dict[str, Any]:
    metadata = read_audio_metadata(path)
    max_frames = None
    if max_seconds is not None and metadata.sample_rate is not None:
        max_frames = int(max_seconds * metadata.sample_rate)
    values, scale = _pcm_values(path, max_frames=max_frames)
    features: dict[str, Any] = {
        "audio_uri": str(path),
        "duration_seconds": metadata.duration_seconds,
        "sample_rate": metadata.sample_rate,
        "channels": metadata.channels,
        "frames": metadata.frames,
    }
    if values:
        peak = max(abs(value) for value in values) / scale
        rms = math.sqrt(sum(value * value for value in values) / len(values)) / scale
        features.update({"peak_amplitude": peak, "rms_amplitude": rms})
    else:
        features.update({"peak_amplitude": None, "rms_amplitude": None})
    return features


def extract_wav_section_features(path: Path, sections: int = 3) -> dict[str, Any]:
    if sections <= 0:
        raise ValueError("sections must be positive")
    metadata = read_audio_metadata(path)
    values, scale = _pcm_values(path)
    features: dict[str, Any] = {
        "audio_uri": str(path),
        "duration_seconds": metadata.duration_seconds,
        "sample_rate": metadata.sample_rate,
        "channels": metadata.channels,
        "frames": metadata.frames,
        "section_count": sections,
    }
    if not values or metadata.frames is None or metadata.channels is None or metadata.frames <= 0:
        for section_index in range(sections):
            features.update(_empty_section_features(section_index))
        features.update({"section_rms_range": None, "section_peak_range": None, "section_rms_max_abs_delta": None})
        return features

    rms_values = []
    peak_values = []
    for section_index in range(sections):
        start_frame = round(section_index * metadata.frames / sections)
        end_frame = round((section_index + 1) * metadata.frames / sections)
        start_sample = start_frame * metadata.channels
        end_sample = end_frame * metadata.channels
        section_values = values[start_sample:end_sample]
        if section_values:
            peak = max(abs(value) for value in section_values) / scale
            rms = math.sqrt(sum(value * value for value in section_values) / len(section_values)) / scale
            rms_values.append(rms)
            peak_values.append(peak)
        else:
            peak = None
            rms = None
        start_seconds = start_frame / metadata.sample_rate if metadata.sample_rate else None
        end_seconds = end_frame / metadata.sample_rate if metadata.sample_rate else None
        features.update(
            {
                f"section_{section_index}_start_seconds": start_seconds,
                f"section_{section_index}_end_seconds": end_seconds,
                f"section_{section_index}_peak_amplitude": peak,
                f"section_{section_index}_rms_amplitude": rms,
            }
        )
    features["section_rms_range"] = max(rms_values) - min(rms_values) if rms_values else None
    features["section_peak_range"] = max(peak_values) - min(peak_values) if peak_values else None
    if len(rms_values) >= 2:
        features["section_rms_max_abs_delta"] = max(
            abs(current - previous) for previous, current in zip(rms_values, rms_values[1:])
        )
    else:
        features["section_rms_max_abs_delta"] = None
    return features


def _empty_section_features(section_index: int) -> dict[str, Any]:
    return {
        f"section_{section_index}_start_seconds": None,
        f"section_{section_index}_end_seconds": None,
        f"section_{section_index}_peak_amplitude": None,
        f"section_{section_index}_rms_amplitude": None,
    }
