from __future__ import annotations

from pathlib import Path
from typing import Any


def select_records(records: list[dict[str, Any]], offset: int, limit: int | None) -> list[dict[str, Any]]:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    end = None if limit is None else offset + limit
    return records[offset:end]


def demucs_stem_paths(output_dir: Path, separator_model: str, audio_path: Path) -> tuple[Path, Path]:
    track_dir = output_dir / separator_model / audio_path.stem
    return track_dir / "vocals.wav", track_dir / "no_vocals.wav"


def build_stem_record(
    generation_record: dict[str, Any],
    *,
    source_audio_path: Path,
    vocal_stem_path: Path,
    accompaniment_stem_path: Path,
    separator_model: str,
) -> dict[str, Any]:
    return {
        "generation_id": generation_record["generation_id"],
        "prompt_id": generation_record["prompt_id"],
        "model_id": generation_record["model_id"],
        "separator_id": f"demucs:{separator_model}",
        "source_audio_uri": str(source_audio_path),
        "vocal_stem_uri": str(vocal_stem_path),
        "accompaniment_stem_uri": str(accompaniment_stem_path),
        "metadata": {
            "genre": generation_record.get("metadata", {}).get("genre"),
            "language": generation_record.get("metadata", {}).get("language"),
            "candidate_index": generation_record.get("metadata", {}).get("candidate_index"),
        },
    }
