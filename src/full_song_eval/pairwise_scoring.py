from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from full_song_eval.jsonl import read_jsonl, write_jsonl
from full_song_eval.pairwise_baseline import LogisticModel, predict_probability


def score_pairwise_deltas(
    *,
    model_summary_path: Path,
    pairwise_deltas_path: Path,
    output_jsonl: Path,
    output_summary: Path,
) -> dict[str, Any]:
    model_summary = _load_json(model_summary_path)
    if not model_summary:
        report = _report(
            status="fail",
            model_summary_path=model_summary_path,
            pairwise_deltas_path=pairwise_deltas_path,
            output_jsonl=output_jsonl,
            output_summary=output_summary,
            reason="missing or invalid model summary",
            rows_scored=0,
            rows_skipped=0,
        )
        output_summary.parent.mkdir(parents=True, exist_ok=True)
        output_summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    if model_summary.get("status") != "ok":
        report = _report(
            status="insufficient_model",
            model_summary_path=model_summary_path,
            pairwise_deltas_path=pairwise_deltas_path,
            output_jsonl=output_jsonl,
            output_summary=output_summary,
            reason=str(model_summary.get("reason", model_summary.get("status", "model is not trainable"))),
            rows_scored=0,
            rows_skipped=0,
        )
        output_summary.parent.mkdir(parents=True, exist_ok=True)
        output_summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    model = _model_from_summary(model_summary)
    if model is None:
        report = _report(
            status="fail",
            model_summary_path=model_summary_path,
            pairwise_deltas_path=pairwise_deltas_path,
            output_jsonl=output_jsonl,
            output_summary=output_summary,
            reason="model summary is missing feature names, weights, means, scales, or bias",
            rows_scored=0,
            rows_skipped=0,
        )
        output_summary.parent.mkdir(parents=True, exist_ok=True)
        output_summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    delta_rows = read_jsonl(pairwise_deltas_path)
    scored_rows = []
    skipped = 0
    for row in delta_rows:
        values = _feature_values(row, model.feature_names)
        if values is None:
            skipped += 1
            continue
        probability_b = predict_probability(model, values)
        preferred_role = "candidate_b" if probability_b >= 0.5 else "candidate_a"
        preferred_generation_id = row.get("candidate_b_generation_id") if preferred_role == "candidate_b" else row.get("candidate_a_generation_id")
        scored_rows.append(
            {
                "task_id": row.get("task_id"),
                "prompt_id": row.get("prompt_id"),
                "candidate_a_generation_id": row.get("candidate_a_generation_id"),
                "candidate_b_generation_id": row.get("candidate_b_generation_id"),
                "probability_b_preferred": probability_b,
                "preferred_candidate_role": preferred_role,
                "preferred_generation_id": preferred_generation_id,
                "score_margin": abs(probability_b - 0.5),
            }
        )
    write_jsonl(output_jsonl, scored_rows)
    report = _report(
        status="ok" if scored_rows else "no_scoreable_rows",
        model_summary_path=model_summary_path,
        pairwise_deltas_path=pairwise_deltas_path,
        output_jsonl=output_jsonl,
        output_summary=output_summary,
        reason=None if scored_rows else "no rows contained all model features",
        rows_scored=len(scored_rows),
        rows_skipped=skipped,
    )
    report["features"] = model.feature_names
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _model_from_summary(summary: dict[str, Any]) -> LogisticModel | None:
    model = summary.get("model")
    if not isinstance(model, dict):
        return None
    feature_names = model.get("feature_names")
    weights = model.get("weights")
    means = model.get("means")
    scales = model.get("scales")
    bias = model.get("bias")
    if not all(isinstance(value, list) for value in [feature_names, weights, means, scales]):
        return None
    if not isinstance(bias, int | float) or not math.isfinite(float(bias)):
        return None
    if not (len(feature_names) == len(weights) == len(means) == len(scales)):
        return None
    if not all(isinstance(name, str) and name for name in feature_names):
        return None
    numeric_weights = _numeric_list(weights)
    numeric_means = _numeric_list(means)
    numeric_scales = _numeric_list(scales)
    if numeric_weights is None or numeric_means is None or numeric_scales is None:
        return None
    return LogisticModel(
        feature_names=list(feature_names),
        means=numeric_means,
        scales=[scale if scale > 0.0 else 1.0 for scale in numeric_scales],
        weights=numeric_weights,
        bias=float(bias),
    )


def _numeric_list(values: list[Any]) -> list[float] | None:
    numeric_values = []
    for value in values:
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            return None
        numeric_values.append(float(value))
    return numeric_values


def _feature_values(row: dict[str, Any], feature_names: list[str]) -> list[float] | None:
    values = []
    for feature_name in feature_names:
        value = row.get(feature_name)
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            return None
        values.append(float(value))
    return values


def _report(
    *,
    status: str,
    model_summary_path: Path,
    pairwise_deltas_path: Path,
    output_jsonl: Path,
    output_summary: Path,
    reason: str | None,
    rows_scored: int,
    rows_skipped: int,
) -> dict[str, Any]:
    return {
        "status": status,
        "model_summary": str(model_summary_path),
        "pairwise_deltas": str(pairwise_deltas_path),
        "output_jsonl": str(output_jsonl),
        "output_summary": str(output_summary),
        "reason": reason,
        "rows_scored": rows_scored,
        "rows_skipped": rows_skipped,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}
