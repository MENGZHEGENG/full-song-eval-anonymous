from __future__ import annotations

from pathlib import Path
from typing import Any

from full_song_eval.jsonl import read_jsonl

BASE_FIELDS = ["generation_id", "prompt_id", "model_id", "genre", "language", "candidate_index"]
SKIP_SOURCE_FIELDS = {
    "generation_id",
    "prompt_id",
    "model_id",
    "genre",
    "language",
    "candidate_index",
    "separator_id",
    "audio_uri",
    "vocal_stem_uri",
    "accompaniment_stem_uri",
    "source_audio_uri",
    "reference_lyrics",
    "asr_transcript",
}


def load_rows_by_generation_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        generation_id = str(row.get("generation_id", ""))
        if not generation_id:
            raise ValueError(f"Missing generation_id in {path}")
        if generation_id in rows:
            raise ValueError(f"Duplicate generation_id {generation_id!r} in {path}")
        rows[generation_id] = row
    return rows


def base_feature_row(generation_record: dict[str, Any]) -> dict[str, Any]:
    metadata = generation_record.get("metadata", {})
    return {
        "generation_id": generation_record.get("generation_id"),
        "prompt_id": generation_record.get("prompt_id"),
        "model_id": generation_record.get("model_id"),
        "genre": metadata.get("genre"),
        "language": metadata.get("language"),
        "candidate_index": metadata.get("candidate_index"),
        "duration_seconds": generation_record.get("duration_seconds"),
        "sample_rate": generation_record.get("sample_rate"),
        "channels": generation_record.get("channels"),
    }


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def merge_feature_sources(
    generation_records: list[dict[str, Any]], sources: dict[str, dict[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    merged = []
    for generation_record in generation_records:
        generation_id = str(generation_record.get("generation_id", ""))
        row = base_feature_row(generation_record)
        for prefix, source_rows in sources.items():
            source_row = source_rows.get(generation_id)
            if source_row is None:
                continue
            for key, value in source_row.items():
                if key in SKIP_SOURCE_FIELDS or not is_scalar(value):
                    continue
                row[source_feature_key(prefix, key)] = value
        merged.append(row)
    return merged


def source_feature_key(prefix: str, key: str) -> str:
    if key.startswith(f"{prefix}_"):
        return key
    return f"{prefix}_{key}"


def collect_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    base = ["generation_id", "prompt_id", "model_id", "genre", "language", "candidate_index"]
    extra = sorted({key for row in rows for key in row if key not in base})
    return base + extra
