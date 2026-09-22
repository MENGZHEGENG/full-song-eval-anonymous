from __future__ import annotations

from pathlib import Path
from typing import Any

from full_song_eval.jsonl import read_jsonl, write_jsonl


def merge_feature_table_rows(inputs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_generation_ids: set[str] = set()
    for path in inputs:
        for row in read_jsonl(path):
            generation_id = str(row.get("generation_id", ""))
            if not generation_id:
                raise ValueError(f"Feature row in {path} is missing generation_id")
            if generation_id in seen_generation_ids:
                raise ValueError(f"Duplicate generation_id {generation_id!r} in {path}")
            seen_generation_ids.add(generation_id)
            rows.append(row)
    return sorted(rows, key=_sort_key)


def merge_feature_tables(inputs: list[Path], output: Path) -> int:
    rows = merge_feature_table_rows(inputs)
    return write_jsonl(output, rows)


def _sort_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row.get("prompt_id", "")),
        _candidate_index(row),
        str(row.get("model_id", "")),
        str(row.get("generation_id", "")),
    )


def _candidate_index(row: dict[str, Any]) -> int:
    value = row.get("candidate_index")
    if value is None:
        return -1
    return int(value)
