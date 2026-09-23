"""CPU-only Stage-1 diagnostic for MusicCaps metadata proxy validity.

The diagnostic is intentionally self-contained and has no third-party runtime
dependencies.  A representative local CPU command is::

    PYTHONPATH=src python -m full_song_eval.musiccaps_stage1_diagnostic \
      --manifest data/musiccaps_manifest.jsonl \
      --output reports/musiccaps_stage1_diagnostic.json \
      --top-label-count 16 --min-label-count 25 \
      --split-unit author_id --fold-count 5 --seed stage1-v1

The held-out group, not a model or random seed, is the uncertainty unit.
Label vocabularies are selected separately from each training fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_MANIFEST = Path("data/musiccaps_manifest.jsonl")
DEFAULT_OUTPUT = Path("reports/musiccaps_stage1_diagnostic.json")
DEFAULT_SEED = "full-song-eval-stage1-v1"
CONDITION_NAMES = (
    "raw",
    "normalized_unmasked",
    "target_label_masked",
    "global_union_masked",
    "matched_random_deletion",
)
PREDICTOR_NAMES = ("prior", "text", "prior_plus_text")
T_CRITICAL_95_BY_DF = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

RAW_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
}


def _raw_tokens(text: str) -> list[str]:
    return RAW_TOKEN_RE.findall(text.lower())


def _normalized_tokens(text: str) -> list[str]:
    return [token for token in _raw_tokens(text) if len(token) > 1 and token not in STOPWORDS]


def _stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def condition_tokens(
    text: str,
    *,
    condition: str,
    labels: Sequence[str],
    target_label: str,
    seed: str,
    row_key: str,
) -> list[str]:
    """Return the tokens seen by one candidate-label classifier.

    ``matched_random_deletion`` removes exactly as many normalized token
    positions as the union mask, while choosing positions independently of the
    label strings.  The stable hash makes the intervention reproducible.
    """

    if condition not in CONDITION_NAMES:
        raise ValueError(f"unsupported text condition: {condition}")
    if condition == "raw":
        return _raw_tokens(text)

    tokens = _normalized_tokens(text)
    if condition == "normalized_unmasked":
        return tokens
    if condition == "target_label_masked":
        target_tokens = set(_normalized_tokens(target_label))
        return [token for token in tokens if token not in target_tokens]

    union_tokens = {token for label in labels for token in _normalized_tokens(label)}
    union_masked = [token for token in tokens if token not in union_tokens]
    if condition == "global_union_masked":
        return union_masked

    removal_count = len(tokens) - len(union_masked)
    ranked_positions = sorted(
        range(len(tokens)),
        key=lambda index: (_stable_hash(f"{seed}|{row_key}|{index}|{tokens[index]}"), index),
    )
    removed_positions = set(ranked_positions[:removal_count])
    return [token for index, token in enumerate(tokens) if index not in removed_positions]


def build_musiccaps_stage1_diagnostic(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    top_label_count: int = 16,
    min_label_count: int = 25,
    fold_count: int = 5,
    split_unit: str = "author_id",
    target_field: str = "aspect_list",
    seed: str = DEFAULT_SEED,
    reproduction_output_path: Path = DEFAULT_OUTPUT,
    code_commit: str | None = None,
) -> dict[str, Any]:
    """Build the complete Stage-1 report without writing files."""

    _validate_positive("top_label_count", top_label_count)
    _validate_positive("min_label_count", min_label_count)
    _validate_positive("fold_count", fold_count)
    rows = _load_manifest(manifest_path)
    resolved_code_commit = _resolve_code_commit(code_commit)
    reproduction = _reproduction(
        manifest_path=manifest_path,
        top_label_count=top_label_count,
        min_label_count=min_label_count,
        fold_count=fold_count,
        split_unit=split_unit,
        target_field=target_field,
        seed=seed,
        output_path=reproduction_output_path,
        code_commit=resolved_code_commit,
    )
    if not rows:
        return _blocked_report(
            manifest_path=manifest_path,
            row_count=0,
            split_unit=split_unit,
            target_field=target_field,
            blocker="missing_or_empty_manifest",
            reproduction=reproduction,
        )

    missing_group_rows = sum(1 for row in rows if not _unit_value(row, split_unit))
    groups = sorted({_unit_value(row, split_unit) for row in rows if _unit_value(row, split_unit)})
    if missing_group_rows:
        return _blocked_report(
            manifest_path=manifest_path,
            row_count=len(rows),
            split_unit=split_unit,
            target_field=target_field,
            blocker="missing_independent_unit_values",
            reproduction=reproduction,
            extra={"missing_independent_unit_row_count": missing_group_rows},
        )
    if len(groups) < 2:
        return _blocked_report(
            manifest_path=manifest_path,
            row_count=len(rows),
            split_unit=split_unit,
            target_field=target_field,
            blocker="insufficient_independent_groups",
            reproduction=reproduction,
        )

    effective_fold_count = min(fold_count, len(groups))
    assignments = _assign_group_folds(rows, split_unit=split_unit, fold_count=effective_fold_count, seed=seed)
    folds: list[dict[str, Any]] = []
    all_unit_metrics: list[dict[str, Any]] = []
    blockers: set[str] = set()
    for heldout_fold in range(effective_fold_count):
        fold = _evaluate_fold(
            rows,
            assignments=assignments,
            heldout_fold=heldout_fold,
            split_unit=split_unit,
            target_field=target_field,
            top_label_count=top_label_count,
            min_label_count=min_label_count,
            seed=seed,
        )
        folds.append(fold)
        if fold.get("blocker"):
            blockers.add(str(fold["blocker"]))
        all_unit_metrics.extend(fold.get("unit_metrics", []))

    leakage_count = sum(not fold.get("groups_disjoint", False) for fold in folds)
    if leakage_count:
        blockers.add("independent_unit_split_leakage")
    complete_folds = [fold for fold in folds if not fold.get("blocker")]
    aggregate = _aggregate_independent_units(all_unit_metrics, split_unit=split_unit) if not blockers else {}
    return {
        "schema_version": 1,
        "status": "complete" if not blockers and len(complete_folds) == effective_fold_count else "blocked",
        "claim_scope": "MusicCaps caption-to-label Stage-1 diagnostic; not audio quality, preference, or full-song evidence",
        "listener_preference_claim_allowed": False,
        "audio_used": False,
        "model_checkpoint_used": False,
        "gpu_used": False,
        "manifest_path": str(manifest_path),
        "code_commit": reproduction["code_commit"],
        "implementation_sha256": reproduction["implementation_sha256"],
        "row_count": len(rows),
        "target_field": target_field,
        "label_vocabulary_scope": "training_fold_only",
        "independent_unit": split_unit,
        "independent_unit_count": len(groups),
        "requested_fold_count": fold_count,
        "effective_fold_count": effective_fold_count,
        "seed": seed,
        "top_label_count": top_label_count,
        "min_label_count": min_label_count,
        "conditions": list(CONDITION_NAMES),
        "predictors": list(PREDICTOR_NAMES),
        "primary_metric": "macro_auprc_across_heldout_independent_units",
        "split_leakage_violation_count": leakage_count,
        "folds": folds,
        "aggregate_metrics": aggregate,
        "known_blockers": sorted(blockers),
        "reproduction": reproduction,
    }


def write_musiccaps_stage1_diagnostic(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    **kwargs: Any,
) -> dict[str, Any]:
    report = build_musiccaps_stage1_diagnostic(reproduction_output_path=output_path, **kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate_stage1_report(report: dict[str, Any]) -> list[str]:
    """Return fail-closed validation errors for a completed report."""

    errors: list[str] = []
    if report.get("status") != "complete":
        errors.append(f"status_not_complete:{report.get('status')}")
    for condition in CONDITION_NAMES:
        if condition not in report.get("conditions", []):
            errors.append(f"missing_condition:{condition}")
    for predictor in PREDICTOR_NAMES:
        if predictor not in report.get("predictors", []):
            errors.append(f"missing_predictor:{predictor}")
    blockers = [str(blocker) for blocker in report.get("known_blockers", [])]
    if blockers:
        errors.append("known_blockers_present:" + ",".join(sorted(blockers)))
    if int(report.get("split_leakage_violation_count", 0)) != 0:
        errors.append("split_leakage_detected")
    if report.get("label_vocabulary_scope") != "training_fold_only":
        errors.append("label_vocabulary_not_training_only")
    if not report.get("code_commit"):
        errors.append("missing_code_commit")
    if not report.get("implementation_sha256"):
        errors.append("missing_implementation_sha256")
    folds = report.get("folds")
    if not isinstance(folds, list) or not folds:
        errors.append("missing_fold_outputs")
    else:
        for fold in folds:
            if not fold.get("groups_disjoint"):
                errors.append(f"fold_group_overlap:{fold.get('fold')}")
    aggregate = report.get("aggregate_metrics", {})
    for condition in CONDITION_NAMES:
        condition_metrics = aggregate.get(condition, {}) if isinstance(aggregate, dict) else {}
        for predictor in PREDICTOR_NAMES:
            if predictor not in condition_metrics:
                errors.append(f"missing_aggregate_cell:{condition}:{predictor}")
    return sorted(set(errors))


def validate_stage1_report_path(path: Path) -> list[str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return [f"unreadable_report:{type(exc).__name__}"]
    if not isinstance(value, dict):
        return ["report_is_not_json_object"]
    return validate_stage1_report(value)


def _evaluate_fold(
    rows: list[dict[str, Any]],
    *,
    assignments: dict[str, int],
    heldout_fold: int,
    split_unit: str,
    target_field: str,
    top_label_count: int,
    min_label_count: int,
    seed: str,
) -> dict[str, Any]:
    train_rows = [row for row in rows if assignments[_unit_value(row, split_unit)] != heldout_fold]
    test_rows = [row for row in rows if assignments[_unit_value(row, split_unit)] == heldout_fold]
    train_units = sorted({_unit_value(row, split_unit) for row in train_rows})
    test_units = sorted({_unit_value(row, split_unit) for row in test_rows})
    label_counts = Counter(label for row in train_rows for label in _row_labels(row, target_field))
    labels = [
        label
        for label, _count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
        if label_counts[label] >= min_label_count
    ][:top_label_count]
    base = {
        "fold": heldout_fold,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_units": train_units,
        "test_units": test_units,
        "groups_disjoint": set(train_units).isdisjoint(test_units),
        "labels": labels,
        "train_label_counts": {label: label_counts[label] for label in labels},
        "label_vocabulary_scope": "training_fold_only",
    }
    if not labels:
        return {**base, "blocker": "insufficient_train_label_frequency", "metrics": {}, "unit_metrics": [], "mask_diagnostics": {}}

    train_target_sets = [_target_set(row, labels, target_field) for row in train_rows]
    prediction_count = max(
        1,
        min(len(labels), round(sum(len(values) for values in train_target_sets) / max(len(train_target_sets), 1))),
    )
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    unit_metrics_by_id: dict[str, dict[str, Any]] = {
        unit_id: {
            "unit_id": unit_id,
            "row_count": sum(1 for row in test_rows if _unit_value(row, split_unit) == unit_id),
            "metrics": {},
        }
        for unit_id in test_units
    }
    unit_indices = {unit_id: unit_index for unit_index, unit_id in enumerate(test_units)}
    for condition in CONDITION_NAMES:
        model = _fit_sparse_text_model(
            train_rows,
            labels=labels,
            target_field=target_field,
            condition=condition,
            seed=f"{seed}|fold:{heldout_fold}",
        )
        buffer = _new_score_buffer()
        for row_index, row in enumerate(test_rows):
            row_key = _row_key(row, row_index)
            unit_id = _unit_value(row, split_unit)
            unit_index = unit_indices[unit_id]
            actual = _target_set(row, labels, target_field)
            base_tokens = _base_condition_tokens(
                str(row.get("caption", "")),
                condition=condition,
                labels=labels,
                seed=f"{seed}|fold:{heldout_fold}",
                row_key=row_key,
            )
            score_vectors = {predictor: array("d") for predictor in PREDICTOR_NAMES}
            for label in labels:
                tokens = base_tokens
                prior_score, text_score, combined_score = _score_label(model, label, tokens)
                score_vectors["prior"].append(prior_score)
                score_vectors["text"].append(text_score)
                score_vectors["prior_plus_text"].append(combined_score)
            predicted_indices = {
                predictor: _top_k_indices(score_vectors[predictor], labels, prediction_count)
                for predictor in PREDICTOR_NAMES
            }
            for label_index, label in enumerate(labels):
                buffer["actual"].append(label in actual)
                buffer["label_index"].append(label_index)
                buffer["unit_index"].append(unit_index)
                for predictor in PREDICTOR_NAMES:
                    buffer["scores"][predictor].append(score_vectors[predictor][label_index])
                    buffer["predicted"][predictor].append(label_index in predicted_indices[predictor])
        metrics[condition] = {
            predictor: _metrics_from_buffer(buffer, labels=labels, predictor=predictor)
            for predictor in PREDICTOR_NAMES
        }
        for unit_id, unit_index in unit_indices.items():
            unit_metrics_by_id[unit_id]["metrics"][condition] = {
                predictor: _metrics_from_buffer(buffer, labels=labels, predictor=predictor, unit_index=unit_index)
                for predictor in PREDICTOR_NAMES
            }
    return {
        **base,
        "prediction_count": prediction_count,
        "train_label_cardinality": round(sum(len(values) for values in train_target_sets) / max(len(train_target_sets), 1), 6),
        "mask_diagnostics": _mask_diagnostics(rows, labels=labels, seed=f"{seed}|fold:{heldout_fold}"),
        "metrics": metrics,
        "unit_metrics": [unit_metrics_by_id[unit_id] for unit_id in test_units],
        "blocker": None,
    }


def _fit_sparse_text_model(
    rows: list[dict[str, Any]],
    *,
    labels: list[str],
    target_field: str,
    condition: str,
    seed: str,
) -> dict[str, Any]:
    allowed_labels = set(labels)
    all_token_rows: Counter[str] = Counter()
    positive_rows: Counter[str] = Counter()
    positive_tokens: dict[str, Counter[str]] = {label: Counter() for label in labels}
    for row_index, row in enumerate(rows):
        tokens = set(
            _base_condition_tokens(
                str(row.get("caption", "")),
                condition=condition,
                labels=labels,
                seed=seed,
                row_key=_row_key(row, row_index),
            )
        )
        all_token_rows.update(tokens)
        for label in set(_row_labels(row, target_field)) & allowed_labels:
            positive_rows[label] += 1
            positive_tokens[label].update(tokens)
    return {
        "labels": labels,
        "condition": condition,
        "row_count": len(rows),
        "all_token_rows": all_token_rows,
        "positive_rows": positive_rows,
        "positive_tokens": positive_tokens,
        "target_label_tokens": {label: set(_normalized_tokens(label)) for label in labels},
        "alpha": 0.5,
    }


def _base_condition_tokens(text: str, *, condition: str, labels: Sequence[str], seed: str, row_key: str) -> list[str]:
    if condition == "target_label_masked":
        return _normalized_tokens(text)
    return condition_tokens(
        text,
        condition=condition,
        labels=labels,
        target_label=labels[0],
        seed=seed,
        row_key=row_key,
    )


def _score_label(model: dict[str, Any], label: str, tokens: Sequence[str]) -> tuple[float, float, float]:
    alpha = float(model["alpha"])
    positive_rows = int(model["positive_rows"][label])
    negative_rows = int(model["row_count"]) - positive_rows
    prior_probability = (positive_rows + alpha) / (positive_rows + negative_rows + 2.0 * alpha)
    prior_logit = math.log((positive_rows + alpha) / (negative_rows + alpha))
    text_score = 0.0
    effective_tokens = set(tokens)
    if model["condition"] == "target_label_masked":
        effective_tokens -= model["target_label_tokens"][label]
    for token in effective_tokens:
        positive_token_rows = model["positive_tokens"][label][token]
        negative_token_rows = model["all_token_rows"][token] - positive_token_rows
        positive_rate = (positive_token_rows + alpha) / (positive_rows + 2.0 * alpha)
        negative_rate = (negative_token_rows + alpha) / (negative_rows + 2.0 * alpha)
        text_score += math.log(positive_rate / negative_rate)
    return prior_probability, text_score, prior_logit + text_score


def _new_score_buffer() -> dict[str, Any]:
    return {
        "actual": bytearray(),
        "label_index": array("I"),
        "unit_index": array("I"),
        "scores": {predictor: array("d") for predictor in PREDICTOR_NAMES},
        "predicted": {predictor: bytearray() for predictor in PREDICTOR_NAMES},
    }


def _top_k_indices(scores: Sequence[float], labels: Sequence[str], count: int) -> set[int]:
    return set(sorted(range(len(labels)), key=lambda index: (-scores[index], labels[index]))[:count])


def _metrics_from_buffer(
    buffer: dict[str, Any],
    *,
    labels: Sequence[str],
    predictor: str,
    unit_index: int | None = None,
) -> dict[str, Any]:
    actual = buffer["actual"]
    predicted = buffer["predicted"][predictor]
    scores = buffer["scores"][predictor]
    units = buffer["unit_index"]
    label_indices = buffer["label_index"]

    def selected(index: int) -> bool:
        return unit_index is None or units[index] == unit_index

    true_positive = false_positive = false_negative = 0
    evaluated_pair_count = 0
    positive_count = 0
    for index in range(len(actual)):
        if not selected(index):
            continue
        evaluated_pair_count += 1
        positive_count += bool(actual[index])
        true_positive += bool(actual[index]) and bool(predicted[index])
        false_positive += (not bool(actual[index])) and bool(predicted[index])
        false_negative += bool(actual[index]) and (not bool(predicted[index]))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    label_ap: list[float] = []
    label_f1: list[float] = []
    label_support: dict[str, int] = {}
    for label_index, label in enumerate(labels):
        indices = [
            index
            for index in range(label_index, len(actual), len(labels))
            if selected(index) and label_indices[index] == label_index
        ]
        support = sum(bool(actual[index]) for index in indices)
        label_support[label] = support
        if support:
            label_ap.append(_average_precision_pairs((scores[index], bool(actual[index])) for index in indices))
        label_tp = sum(bool(actual[index]) and bool(predicted[index]) for index in indices)
        label_fp = sum((not bool(actual[index])) and bool(predicted[index]) for index in indices)
        label_fn = sum(bool(actual[index]) and (not bool(predicted[index])) for index in indices)
        label_f1.append(_f1(_safe_divide(label_tp, label_tp + label_fp), _safe_divide(label_tp, label_tp + label_fn)))
    return {
        "macro_auprc": round(sum(label_ap) / len(label_ap), 6) if label_ap else None,
        "micro_auprc": round(
            _average_precision_pairs((scores[index], bool(actual[index])) for index in range(len(actual)) if selected(index)),
            6,
        )
        if positive_count
        else None,
        "micro_precision": round(precision, 6),
        "micro_recall": round(recall, 6),
        "micro_f1": round(_f1(precision, recall), 6),
        "macro_f1": round(sum(label_f1) / len(label_f1), 6) if label_f1 else None,
        "positive_pair_count": positive_count,
        "evaluated_pair_count": evaluated_pair_count,
        "labels_with_positive_support": len(label_ap),
        "label_support": label_support,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }


def _average_precision_pairs(pairs: Iterable[tuple[float, bool]]) -> float:
    """Threshold-grouped average precision, so tied scores are order invariant."""

    values = list(pairs)
    positive_count = sum(actual for _score, actual in values)
    if positive_count == 0:
        return 0.0
    grouped: dict[float, list[bool]] = defaultdict(list)
    for score, actual in values:
        grouped[float(score)].append(actual)
    true_positive = false_positive = 0
    previous_recall = 0.0
    area = 0.0
    for score in sorted(grouped, reverse=True):
        group = grouped[score]
        true_positive += sum(group)
        false_positive += sum(not actual for actual in group)
        recall = true_positive / positive_count
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def _aggregate_independent_units(unit_rows: list[dict[str, Any]], *, split_unit: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in CONDITION_NAMES:
        result[condition] = {}
        for predictor in PREDICTOR_NAMES:
            unit_metrics = [row["metrics"][condition][predictor] for row in unit_rows]
            summary: dict[str, Any] = {
                "uncertainty_unit": f"heldout_{split_unit}_group",
                "independent_unit_count": len(unit_metrics),
            }
            for metric_name in ("macro_auprc", "micro_auprc", "macro_f1", "micro_f1"):
                values = [float(metric[metric_name]) for metric in unit_metrics if metric[metric_name] is not None]
                summary.update(_mean_ci(values, prefix=metric_name))
            result[condition][predictor] = summary
    return result


def _mean_ci(values: Sequence[float], *, prefix: str) -> dict[str, Any]:
    if not values:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_ci95_low": None,
            f"{prefix}_ci95_high": None,
            f"{prefix}_ci95_method": "student_t_95",
        }
    mean = sum(values) / len(values)
    if len(values) == 1:
        half_width = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        degrees_of_freedom = len(values) - 1
        critical_value = T_CRITICAL_95_BY_DF.get(degrees_of_freedom, 1.96)
        half_width = critical_value * math.sqrt(variance / len(values))
    return {
        f"{prefix}_mean": round(mean, 6),
        f"{prefix}_ci95_low": round(max(0.0, mean - half_width), 6),
        f"{prefix}_ci95_high": round(min(1.0, mean + half_width), 6),
        f"{prefix}_ci95_method": "student_t_95",
    }


def _mask_diagnostics(rows: list[dict[str, Any]], *, labels: list[str], seed: str) -> dict[str, dict[str, Any]]:
    del seed  # Counts are deterministic and do not depend on which matched positions were removed.
    raw_rows = [_raw_tokens(str(row.get("caption", ""))) for row in rows]
    normalized_rows = [_normalized_tokens(str(row.get("caption", ""))) for row in rows]
    raw_total = sum(map(len, raw_rows))
    normalized_total = sum(map(len, normalized_rows))
    union_tokens = {token for label in labels for token in _normalized_tokens(label)}
    globally_removed = sum(token in union_tokens for tokens in normalized_rows for token in tokens)
    labels_per_token = Counter(token for label in labels for token in set(_normalized_tokens(label)))
    target_removed_pairs = sum(labels_per_token[token] for tokens in normalized_rows for token in tokens)
    target_removed_mean = target_removed_pairs / max(len(labels), 1)
    totals = {
        "raw": (raw_total, raw_total),
        "normalized_unmasked": (normalized_total, normalized_total),
        "target_label_masked": (normalized_total, normalized_total - target_removed_mean),
        "global_union_masked": (normalized_total, normalized_total - globally_removed),
        "matched_random_deletion": (normalized_total, normalized_total - globally_removed),
    }
    diagnostics: dict[str, dict[str, Any]] = {}
    for condition, (reference_total, retained) in totals.items():
        removed = reference_total - retained
        diagnostics[condition] = {
            "row_count": len(rows),
            "reference_token_count": reference_total,
            "retained_token_count": round(retained, 6),
            "removed_token_count": round(removed, 6),
            "removed_fraction": round(_safe_divide(removed, reference_total), 6),
        }
    return diagnostics


def _assign_group_folds(rows: list[dict[str, Any]], *, split_unit: str, fold_count: int, seed: str) -> dict[str, int]:
    group_sizes = Counter(_unit_value(row, split_unit) for row in rows)
    ordered = sorted(group_sizes, key=lambda group: (-group_sizes[group], _stable_hash(f"{seed}|{group}"), group))
    loads = [0] * fold_count
    assignments: dict[str, int] = {}
    for group in ordered:
        fold = min(range(fold_count), key=lambda index: (loads[index], index))
        assignments[group] = fold
        loads[fold] += group_sizes[group]
    return assignments


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    rows = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"manifest line {line_number} is not a JSON object")
        rows.append(value)
    return rows


def _row_labels(row: dict[str, Any], target_field: str) -> list[str]:
    values = row.get(target_field, [])
    if not isinstance(values, list):
        return []
    return sorted({str(value).strip().lower() for value in values if str(value).strip()})


def _target_set(row: dict[str, Any], labels: Sequence[str], target_field: str) -> set[str]:
    return set(_row_labels(row, target_field)) & set(labels)


def _unit_value(row: dict[str, Any], split_unit: str) -> str:
    value = row.get(split_unit)
    return str(value).strip() if value is not None else ""


def _row_key(row: dict[str, Any], row_index: int) -> str:
    return str(row.get("row_id") or row.get("youtube_id") or f"row:{row_index}")


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _manifest_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _reproduction(
    *,
    manifest_path: Path,
    top_label_count: int,
    min_label_count: int,
    fold_count: int,
    split_unit: str,
    target_field: str,
    seed: str,
    output_path: Path,
    code_commit: str | None,
) -> dict[str, Any]:
    arguments = [
        "python",
        "-m",
        "full_song_eval.musiccaps_stage1_diagnostic",
        "--manifest",
        str(manifest_path),
        "--output",
        str(output_path),
        "--top-label-count",
        str(top_label_count),
        "--min-label-count",
        str(min_label_count),
        "--fold-count",
        str(fold_count),
        "--split-unit",
        split_unit,
        "--target-field",
        target_field,
        "--seed",
        seed,
    ]
    if code_commit:
        arguments.extend(["--code-commit", code_commit])
    return {
        "command": "PYTHONPATH=src " + " ".join(shlex.quote(argument) for argument in arguments),
        "validation_command": "PYTHONPATH=src python -m full_song_eval.musiccaps_stage1_diagnostic --validate-report "
        + shlex.quote(str(output_path)),
        "manifest_sha256": _manifest_sha256(manifest_path),
        "code_commit": code_commit,
        "implementation_sha256": _manifest_sha256(Path(__file__)),
    }


def _resolve_code_commit(explicit: str | None) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip()
    environment_value = os.environ.get("FULL_SONG_EVAL_CODE_COMMIT", "").strip()
    if environment_value:
        return environment_value
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value or None


def _blocked_report(
    *,
    manifest_path: Path,
    row_count: int,
    split_unit: str,
    target_field: str,
    blocker: str,
    reproduction: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "status": "blocked",
        "claim_scope": "MusicCaps caption-to-label Stage-1 diagnostic; not audio quality, preference, or full-song evidence",
        "listener_preference_claim_allowed": False,
        "audio_used": False,
        "model_checkpoint_used": False,
        "gpu_used": False,
        "manifest_path": str(manifest_path),
        "code_commit": reproduction["code_commit"],
        "implementation_sha256": reproduction["implementation_sha256"],
        "row_count": row_count,
        "target_field": target_field,
        "label_vocabulary_scope": "training_fold_only",
        "independent_unit": split_unit,
        "conditions": list(CONDITION_NAMES),
        "predictors": list(PREDICTOR_NAMES),
        "folds": [],
        "aggregate_metrics": {},
        "known_blockers": [blocker],
        "reproduction": reproduction,
    }
    if extra:
        report.update(extra)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-label-count", type=int, default=16)
    parser.add_argument("--min-label-count", type=int, default=25)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--split-unit", default="author_id")
    parser.add_argument("--target-field", default="aspect_list")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--code-commit")
    parser.add_argument("--validate-report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_report is not None:
        errors = validate_stage1_report_path(args.validate_report)
        print(json.dumps({"errors": errors, "report": str(args.validate_report), "valid": not errors}, sort_keys=True))
        return 0 if not errors else 3
    report = write_musiccaps_stage1_diagnostic(
        manifest_path=args.manifest,
        output_path=args.output,
        top_label_count=args.top_label_count,
        min_label_count=args.min_label_count,
        fold_count=args.fold_count,
        split_unit=args.split_unit,
        target_field=args.target_field,
        seed=args.seed,
        code_commit=args.code_commit,
    )
    print(json.dumps({"output": str(args.output), "status": report["status"]}, sort_keys=True))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
