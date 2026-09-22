from __future__ import annotations

import math
import statistics
from typing import Any

ID_FIELDS = {
    "generation_id",
    "prompt_id",
    "model_id",
    "genre",
    "language",
    "candidate_index",
}


def rows_by_generation_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        generation_id = str(row.get("generation_id", ""))
        if not generation_id:
            raise ValueError("Feature row is missing generation_id")
        if generation_id in by_id:
            raise ValueError(f"Duplicate generation_id {generation_id!r}")
        by_id[generation_id] = row
    return by_id


def is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def numeric_feature_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields = set()
    for row in rows:
        for key, value in row.items():
            if key not in ID_FIELDS and is_number(value):
                fields.add(key)
    return sorted(fields)


def build_pairwise_delta_rows(
    tasks: list[dict[str, Any]], feature_rows: list[dict[str, Any]], feature_fields: list[str] | None = None
) -> list[dict[str, Any]]:
    by_id = rows_by_generation_id(feature_rows)
    fields = feature_fields or numeric_feature_fields(feature_rows)
    rows = []
    for task in tasks:
        a_id = str(task["candidate_a"]["generation_id"])
        b_id = str(task["candidate_b"]["generation_id"])
        if a_id not in by_id:
            raise KeyError(f"Missing feature row for candidate A {a_id}")
        if b_id not in by_id:
            raise KeyError(f"Missing feature row for candidate B {b_id}")
        a_row = by_id[a_id]
        b_row = by_id[b_id]
        output: dict[str, Any] = {
            "task_id": task.get("task_id"),
            "prompt_id": task.get("prompt_id"),
            "genre": task.get("metadata", {}).get("genre") or a_row.get("genre"),
            "language": task.get("metadata", {}).get("language") or a_row.get("language"),
            "candidate_a_generation_id": a_id,
            "candidate_b_generation_id": b_id,
        }
        complete_fields = 0
        for field in fields:
            a_value = a_row.get(field)
            b_value = b_row.get(field)
            if is_number(a_value) and is_number(b_value):
                delta = float(b_value) - float(a_value)
                output[f"delta_{field}"] = delta
                output[f"abs_delta_{field}"] = abs(delta)
                complete_fields += 1
            else:
                output[f"delta_{field}"] = None
                output[f"abs_delta_{field}"] = None
        output["proxy_complete_feature_count"] = complete_fields
        rows.append(output)
    add_proxy_contrast_scores(rows, fields)
    return rows


def add_proxy_contrast_scores(rows: list[dict[str, Any]], fields: list[str]) -> None:
    scales: dict[str, float] = {}
    for field in fields:
        values = [float(row[f"abs_delta_{field}"]) for row in rows if row.get(f"abs_delta_{field}") is not None]
        if len(values) >= 2:
            std = statistics.pstdev(values)
            if std > 0.0:
                scales[field] = std
    for row in rows:
        normalized = []
        for field, scale in scales.items():
            value = row.get(f"abs_delta_{field}")
            if value is not None:
                normalized.append(float(value) / scale)
        row["proxy_contrast_score"] = statistics.mean(normalized) if normalized else None
        row["proxy_contrast_feature_count"] = len(normalized)


def summarize_pairwise_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row["proxy_contrast_score"]) for row in rows if row.get("proxy_contrast_score") is not None]
    by_language: dict[str, list[float]] = {}
    for row in rows:
        if row.get("proxy_contrast_score") is None:
            continue
        language = str(row.get("language") or "unknown")
        by_language.setdefault(language, []).append(float(row["proxy_contrast_score"]))
    return {
        "pairs": len(rows),
        "pairs_with_proxy_contrast": len(scores),
        "proxy_contrast_min": min(scores) if scores else None,
        "proxy_contrast_mean": statistics.mean(scores) if scores else None,
        "proxy_contrast_max": max(scores) if scores else None,
        "by_language": {
            language: {"pairs": len(values), "proxy_contrast_mean": statistics.mean(values)}
            for language, values in sorted(by_language.items())
        },
    }
