"""Random-ranking baselines for the fixed metadata calibration folds."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from full_song_eval.musiccaps_gate_calibration import (
    _average_precision_pairs,
    _fit_label_model,
    _label_set_permutation,
    _normalized_tokens,
    _partial_label_set_permutation,
    _row_key,
    _score_label_model,
    _stable_fraction,
    _stable_hash,
    _target_view_from_tokens,
)


MUSICCAPS_SEED = "full-song-eval-gate-calibration-v1"
MTG_JAMENDO_SEED = "full-song-eval-mtg-jamendo-metadata-v1"
T_CRITICAL_95_DF9 = 2.2621571627409915
SCORE_REPLAY_TOLERANCE = 1e-4


def expected_average_precision(*, n: int, k: int) -> float:
    """Return exact E[AP] for a strict random order with ``k`` positives."""

    if n < 1 or k < 1 or k > n:
        raise ValueError("expected 1 <= k <= n")
    harmonic_n = math.fsum(1.0 / rank for rank in range(1, n + 1))
    if n == 1:
        return 1.0
    return k / n + (n - k) * (harmonic_n - 1.0) / (n * (n - 1))


def expected_tie_aware_average_precision(*, n: int, k: int, tie_group_sizes: list[int]) -> float:
    """Return random-label E[AP] for ordered score-tie blocks."""

    if n < 1 or k < 1 or k > n:
        raise ValueError("expected 1 <= k <= n")
    if not tie_group_sizes or any(size < 1 for size in tie_group_sizes) or sum(tie_group_sizes) != n:
        raise ValueError("tie-group sizes must be positive and sum to n")
    if n == 1:
        return 1.0
    contributions: list[float] = []
    before = 0
    for size in tie_group_sizes:
        numerator = size * (before + size - 1) * (k - 1) / (n * (n - 1)) + size / n
        contributions.append(numerator / (before + size))
        before += size
    return math.fsum(contributions)


def rotation_donor_indices(
    rows: Sequence[dict[str, Any]], *, seed: str, rotation_kind: str
) -> dict[int, int]:
    """Reproduce the recipient-to-donor mapping used by one invalid control."""

    if rotation_kind not in {"label_set", "full_corruption"}:
        raise ValueError(f"unsupported rotation kind: {rotation_kind}")
    if len(rows) < 2:
        return {}
    if rotation_kind == "label_set":
        selected = list(range(len(rows)))
        namespace = "label-set"
    else:
        selected = [
            index
            for index, row in enumerate(rows)
            if _stable_fraction(f"{seed}|corruption-select|{_row_key(row, index)}") < 1.0
        ]
        namespace = "corruption-order"
    if len(selected) < 2:
        return {}
    order = sorted(
        selected,
        key=lambda index: (
            _stable_hash(f"{seed}|{namespace}|{_row_key(rows[index], index)}"),
            index,
        ),
    )
    donors = order[1:] + order[:1]
    return dict(zip(order, donors))


def _normalized_label_set(row: dict[str, Any], target_field: str) -> set[str]:
    values = row.get(target_field)
    if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{target_field} must be a list of nonempty strings")
    return {value.strip().lower() for value in values}


def analyze_rotation_integrity(
    rows: Sequence[dict[str, Any]],
    *,
    target_field: str,
    selected_labels: Sequence[str],
    seed: str,
    rotation_kind: str,
) -> dict[str, Any]:
    """Audit the selected-label vectors transferred by a test-set rotation."""

    labels = [str(label).strip().lower() for label in selected_labels]
    if not rows or not labels or any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("rows and unique selected labels are required")
    selected = set(labels)
    before = [_normalized_label_set(row, target_field).intersection(selected) for row in rows]
    if rotation_kind == "label_set":
        transformed_rows = _label_set_permutation(rows, target_field=target_field, seed=seed)
    else:
        transformed_rows = _partial_label_set_permutation(
            rows, target_field=target_field, level=1.0, seed=seed
        )
    after = [
        _normalized_label_set(row, target_field).intersection(selected) for row in transformed_rows
    ]
    donors = rotation_donor_indices(rows, seed=seed, rotation_kind=rotation_kind)

    jaccards: list[float] = []
    for original, transferred in zip(before, after):
        union = original.union(transferred)
        jaccards.append(len(original.intersection(transferred)) / len(union) if union else 1.0)
    before_prevalence = Counter(label for values in before for label in values)
    after_prevalence = Counter(label for values in after for label in values)
    changed = sum(original != transferred for original, transferred in zip(before, after))
    return {
        "row_count": len(rows),
        "assigned_recipient_count": len(donors),
        "donor_self_matches": sum(recipient == donor for recipient, donor in donors.items()),
        "exact_selected_label_vector_transfers": sum(
            after[recipient] == before[donor] for recipient, donor in donors.items()
        ),
        "changed_selected_label_vectors": changed,
        "percent_changed": round(100.0 * changed / len(rows), 12),
        "mean_selected_label_jaccard": round(math.fsum(jaccards) / len(jaccards), 12),
        "prevalence_preserved": before_prevalence == after_prevalence,
    }


def replay_target_masked_condition(
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    *,
    labels: Sequence[str],
    target_field: str,
    scenario_seed: str,
) -> dict[str, Any]:
    """Replay target-masked scores and derive strict and tie-aware baselines."""

    if not train_rows or not test_rows or not labels:
        raise ValueError("nonempty training rows, test rows, and labels are required")
    normalized_labels = [str(label).strip().lower() for label in labels]
    if any(not label for label in normalized_labels) or len(set(normalized_labels)) != len(normalized_labels):
        raise ValueError("labels must be unique nonempty strings")
    train_tokens = [
        (_row_key(row, index), _normalized_tokens(str(row.get("caption", ""))))
        for index, row in enumerate(train_rows)
    ]
    test_tokens = [
        (_row_key(row, index), _normalized_tokens(str(row.get("caption", ""))))
        for index, row in enumerate(test_rows)
    ]
    label_diagnostics: list[dict[str, Any]] = []
    for label in normalized_labels:
        model = _fit_label_model(
            train_rows,
            train_tokens,
            label=label,
            target_field=target_field,
            view="target_label_masked",
            seed=f"{scenario_seed}|train",
        )
        scores: list[float] = []
        actual: list[bool] = []
        for row, (row_key, tokens) in zip(test_rows, test_tokens):
            viewed, _diagnostic = _target_view_from_tokens(
                tokens,
                target_label=label,
                view="target_label_masked",
                seed=f"{scenario_seed}|test",
                row_key=row_key,
            )
            scores.append(_score_label_model(model, viewed))
            actual.append(label in _normalized_label_set(row, target_field))
        positive_count = sum(actual)
        if positive_count == 0:
            continue
        tie_sizes = [
            count
            for _score, count in sorted(Counter(scores).items(), key=lambda item: item[0], reverse=True)
        ]
        n = len(test_rows)
        label_diagnostics.append(
            {
                "label": label,
                "test_record_count": n,
                "positive_count": positive_count,
                "prevalence": positive_count / n,
                "actual_average_precision": _average_precision_pairs(zip(scores, actual)),
                "strict_random_ranking_expected_average_precision": expected_average_precision(
                    n=n, k=positive_count
                ),
                "tie_aware_random_label_expected_average_precision": expected_tie_aware_average_precision(
                    n=n, k=positive_count, tie_group_sizes=tie_sizes
                ),
                "ordered_score_tie_group_sizes": tie_sizes,
            }
        )
    if not label_diagnostics:
        raise ValueError("no selected label has positive support in the test partition")
    count = len(label_diagnostics)
    return {
        "macro_auprc": math.fsum(item["actual_average_precision"] for item in label_diagnostics) / count,
        "strict_random_ranking_expected_macro_auprc": math.fsum(
            item["strict_random_ranking_expected_average_precision"] for item in label_diagnostics
        )
        / count,
        "tie_aware_random_label_expected_macro_auprc": math.fsum(
            item["tie_aware_random_label_expected_average_precision"] for item in label_diagnostics
        )
        / count,
        "labels": label_diagnostics,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable {name}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _read_manifest(
    path: Path, *, name: str, group_field: str, target_field: str
) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"unreadable {name}: {type(exc).__name__}") from exc
    rows: list[dict[str, Any]] = []
    keys: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} line {line_number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{name} line {line_number} is not a JSON object")
        if not str(row.get(group_field, "")).strip():
            raise ValueError(f"{name} line {line_number} lacks {group_field}")
        _normalized_label_set(row, target_field)
        if not isinstance(row.get("caption"), str):
            raise ValueError(f"{name} line {line_number} lacks a string caption")
        key = _row_key(row, len(rows))
        if key in keys:
            raise ValueError(f"{name} has duplicate row key: {key}")
        keys.add(key)
        rows.append(row)
    if not rows:
        raise ValueError(f"{name} is empty")
    return rows


def _selected_labels(
    rows: Sequence[dict[str, Any]], *, target_field: str, top_label_count: int, min_label_count: int
) -> tuple[list[str], Counter[str]]:
    counts = Counter(label for row in rows for label in _normalized_label_set(row, target_field))
    labels = [
        label
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_label_count
    ][:top_label_count]
    return labels, counts


def _reported_score(fold: dict[str, Any], *, source: str, condition: str) -> float:
    metrics = fold.get("condition_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{source} fold lacks condition metrics")
    if condition == "observed":
        node = metrics.get("observed")
    elif condition == "permutation_a":
        node = metrics.get("label_set_permutation")
    elif condition == "permutation_b" and source == "musiccaps":
        levels = metrics.get("label_corruption")
        node = levels.get("1.0") if isinstance(levels, dict) else None
    elif condition == "permutation_b":
        node = metrics.get("full_label_corruption")
    else:  # pragma: no cover - internal misuse guard
        raise ValueError(f"unsupported condition: {source}/{condition}")
    view = node.get("target_label_masked") if isinstance(node, dict) else None
    value = view.get("macro_auprc") if isinstance(view, dict) else None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{source} fold lacks a finite {condition} target-masked score")
    return float(value)


def _condition_inputs(
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    *,
    source: str,
    target_field: str,
    fold_seed: str,
    condition: str,
) -> tuple[Sequence[dict[str, Any]], Sequence[dict[str, Any]], str, str | None, str | None]:
    if condition == "observed":
        return train_rows, test_rows, f"{fold_seed}|observed", None, None
    if source == "musiccaps" and condition == "permutation_a":
        train_seed = f"{fold_seed}|label-permutation|train"
        test_seed = f"{fold_seed}|label-permutation|test"
        return (
            _label_set_permutation(train_rows, target_field=target_field, seed=train_seed),
            _label_set_permutation(test_rows, target_field=target_field, seed=test_seed),
            f"{fold_seed}|label-permutation",
            test_seed,
            "label_set",
        )
    if source == "musiccaps" and condition == "permutation_b":
        train_seed = f"{fold_seed}|corruption|train"
        test_seed = f"{fold_seed}|corruption|test"
        return (
            _partial_label_set_permutation(
                train_rows, target_field=target_field, level=1.0, seed=train_seed
            ),
            _partial_label_set_permutation(
                test_rows, target_field=target_field, level=1.0, seed=test_seed
            ),
            f"{fold_seed}|corruption|1.0",
            test_seed,
            "full_corruption",
        )
    if source == "mtg_jamendo" and condition == "permutation_a":
        train_seed = f"{fold_seed}|permuted-train"
        test_seed = f"{fold_seed}|permuted-test"
        return (
            _label_set_permutation(train_rows, target_field=target_field, seed=train_seed),
            _label_set_permutation(test_rows, target_field=target_field, seed=test_seed),
            f"{fold_seed}|permuted",
            test_seed,
            "label_set",
        )
    if source == "mtg_jamendo" and condition == "permutation_b":
        train_seed = f"{fold_seed}|corrupt-train"
        test_seed = f"{fold_seed}|corrupt-test"
        return (
            _partial_label_set_permutation(
                train_rows, target_field=target_field, level=1.0, seed=train_seed
            ),
            _partial_label_set_permutation(
                test_rows, target_field=target_field, level=1.0, seed=test_seed
            ),
            f"{fold_seed}|corrupt",
            test_seed,
            "full_corruption",
        )
    raise ValueError(f"unsupported condition: {source}/{condition}")


def _compact_replay(replay: dict[str, Any]) -> dict[str, Any]:
    labels: list[dict[str, Any]] = []
    for item in replay["labels"]:
        sizes = item["ordered_score_tie_group_sizes"]
        sizes_bytes = json.dumps(sizes, separators=(",", ":")).encode("utf-8")
        labels.append(
            {
                key: item[key]
                for key in (
                    "label",
                    "test_record_count",
                    "positive_count",
                    "prevalence",
                    "actual_average_precision",
                    "strict_random_ranking_expected_average_precision",
                    "tie_aware_random_label_expected_average_precision",
                )
            }
            | {
                "score_tie_group_count": len(sizes),
                "singleton_score_tie_group_count": sum(size == 1 for size in sizes),
                "largest_score_tie_group": max(sizes),
                "ordered_score_tie_group_sizes_sha256": hashlib.sha256(sizes_bytes).hexdigest(),
            }
        )
    return {
        "macro_auprc": replay["macro_auprc"],
        "strict_random_ranking_expected_macro_auprc": replay[
            "strict_random_ranking_expected_macro_auprc"
        ],
        "tie_aware_random_label_expected_macro_auprc": replay[
            "tie_aware_random_label_expected_macro_auprc"
        ],
        "labels": labels,
    }


def _paired_interval(differences: dict[str, float]) -> dict[str, Any]:
    if len(differences) != 10:
        raise ValueError(f"expected ten held-out-group differences, found {len(differences)}")
    values = list(differences.values())
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    standard_error = standard_deviation / math.sqrt(len(values))
    half_width = T_CRITICAL_95_DF9 * standard_error
    ci95_low = mean - half_width
    ci95_high = mean + half_width
    return {
        "mean_difference": round(mean, 9),
        "ci95_low": round(ci95_low, 9),
        "ci95_high": round(ci95_high, 9),
        "sample_standard_deviation": round(standard_deviation, 9),
        "standard_error": round(standard_error, 9),
        "positive_group_count": sum(value > 0.0 for value in values),
        "group_count": len(values),
        "accepted_above_random": ci95_low > 0.0,
        "group_differences": {
            group: round(value, 9) for group, value in sorted(differences.items())
        },
        "interval_method": "paired_descriptive_student_t_95",
    }


def _aggregate_rotation(folds: Sequence[dict[str, Any]], condition: str) -> dict[str, Any]:
    values = [fold["rotation_integrity"][condition] for fold in folds]
    row_count = sum(int(value["row_count"]) for value in values)
    changed = sum(int(value["changed_selected_label_vectors"]) for value in values)
    return {
        "fold_count": len(values),
        "row_count": row_count,
        "assigned_recipient_count": sum(int(value["assigned_recipient_count"]) for value in values),
        "donor_self_matches": sum(int(value["donor_self_matches"]) for value in values),
        "exact_selected_label_vector_transfers": sum(
            int(value["exact_selected_label_vector_transfers"]) for value in values
        ),
        "changed_selected_label_vectors": changed,
        "percent_changed": round(100.0 * changed / row_count, 12),
        "mean_selected_label_jaccard": round(
            math.fsum(float(value["mean_selected_label_jaccard"]) * int(value["row_count"]) for value in values)
            / row_count,
            12,
        ),
        "prevalence_preserved": all(bool(value["prevalence_preserved"]) for value in values),
    }


def _analyze_source(
    *,
    source: str,
    rows: Sequence[dict[str, Any]],
    report: dict[str, Any],
    group_field: str,
    report_group_field: str,
    target_field: str,
    seed: str,
) -> dict[str, Any]:
    if report.get("status") != "complete":
        raise ValueError(f"{source} report is not complete")
    report_folds = report.get("folds")
    if not isinstance(report_folds, list) or len(report_folds) != 10:
        raise ValueError(f"{source} report must contain ten folds")
    groups = sorted({str(row[group_field]).strip() for row in rows})
    if len(groups) != 10:
        raise ValueError(f"{source} manifest must contain ten nonempty groups")
    design = report.get("design")
    if not isinstance(design, dict):
        raise ValueError(f"{source} report lacks its design")
    try:
        top_label_count = int(design["top_label_count"])
        min_label_count = int(design["min_label_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source} report has invalid label-selection settings") from exc
    if top_label_count < 1 or min_label_count < 1:
        raise ValueError(f"{source} report has invalid label-selection settings")

    fold_outputs: list[dict[str, Any]] = []
    strict_differences = {condition: {} for condition in ("observed", "permutation_a", "permutation_b")}
    tie_differences = {condition: {} for condition in ("observed", "permutation_a", "permutation_b")}
    replay_errors: list[float] = []
    for fold_index, group in enumerate(groups):
        report_fold = report_folds[fold_index]
        if not isinstance(report_fold, dict):
            raise ValueError(f"{source} fold {fold_index} is not an object")
        if int(report_fold.get("fold", -1)) != fold_index or str(report_fold.get(report_group_field, "")) != group:
            raise ValueError(f"{source} fold {fold_index} does not match manifest partition {group}")
        train_rows = [row for row in rows if str(row[group_field]).strip() != group]
        test_rows = [row for row in rows if str(row[group_field]).strip() == group]
        if int(report_fold.get("train_record_count", report_fold.get("train_row_count", -1))) != len(train_rows):
            raise ValueError(f"{source} fold {fold_index} training size mismatch")
        if int(report_fold.get("test_record_count", report_fold.get("test_row_count", -1))) != len(test_rows):
            raise ValueError(f"{source} fold {fold_index} test size mismatch")
        labels, train_counts = _selected_labels(
            train_rows,
            target_field=target_field,
            top_label_count=top_label_count,
            min_label_count=min_label_count,
        )
        if len(labels) != top_label_count:
            raise ValueError(f"{source} fold {fold_index} lacks the requested selected labels")
        if labels != report_fold.get("labels"):
            raise ValueError(f"{source} fold {fold_index} selected-label mismatch")
        fold_seed = f"{seed}|fold:{fold_index}"
        condition_outputs: dict[str, Any] = {}
        rotation_outputs: dict[str, Any] = {}
        strict_expected: float | None = None
        for condition in ("observed", "permutation_a", "permutation_b"):
            transformed_train, transformed_test, scenario_seed, rotation_seed, rotation_kind = _condition_inputs(
                train_rows,
                test_rows,
                source=source,
                target_field=target_field,
                fold_seed=fold_seed,
                condition=condition,
            )
            replay = replay_target_masked_condition(
                transformed_train,
                transformed_test,
                labels=labels,
                target_field=target_field,
                scenario_seed=scenario_seed,
            )
            score = _reported_score(report_fold, source=source, condition=condition)
            replay_error = abs(float(replay["macro_auprc"]) - score)
            replay_errors.append(replay_error)
            if replay_error > SCORE_REPLAY_TOLERANCE:
                raise ValueError(
                    f"{source} fold {fold_index} {condition} replay mismatch: {replay_error}"
                )
            condition_strict = float(replay["strict_random_ranking_expected_macro_auprc"])
            if strict_expected is None:
                strict_expected = condition_strict
            elif not math.isclose(strict_expected, condition_strict, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{source} fold {fold_index} control prevalence was not preserved")
            tie_expected = float(replay["tie_aware_random_label_expected_macro_auprc"])
            strict_differences[condition][group] = score - condition_strict
            tie_differences[condition][group] = score - tie_expected
            condition_outputs[condition] = {
                "reported_macro_auprc": score,
                "replayed_macro_auprc": replay["macro_auprc"],
                "replay_absolute_error": replay_error,
                "scenario_seed": scenario_seed,
                "strict_random_ranking_expected_macro_auprc": condition_strict,
                "tie_aware_random_label_expected_macro_auprc": tie_expected,
                "score_minus_strict_random_ranking_expected": score - condition_strict,
                "score_minus_tie_aware_random_label_expected": score - tie_expected,
                "label_diagnostics": _compact_replay(replay)["labels"],
            }
            if rotation_seed is not None and rotation_kind is not None:
                rotation_outputs[condition] = {
                    "seed": rotation_seed,
                    "rotation_kind": rotation_kind,
                    **analyze_rotation_integrity(
                        test_rows,
                        target_field=target_field,
                        selected_labels=labels,
                        seed=rotation_seed,
                        rotation_kind=rotation_kind,
                    ),
                }
        if strict_expected is None:  # pragma: no cover - fixed loop guard
            raise ValueError(f"{source} fold {fold_index} lacks a strict baseline")
        supported = condition_outputs["observed"]["label_diagnostics"]
        prior = math.fsum(float(item["prevalence"]) for item in supported) / len(supported)
        reported_prior = report_fold.get("prior_macro_auprc")
        if not isinstance(reported_prior, (int, float)) or abs(float(reported_prior) - prior) > 1e-6:
            raise ValueError(f"{source} fold {fold_index} prior mismatch")
        fold_outputs.append(
            {
                "fold": fold_index,
                "heldout_group": group,
                "train_record_count": len(train_rows),
                "test_record_count": len(test_rows),
                "selected_labels": labels,
                "train_label_counts": {label: train_counts[label] for label in labels},
                "labels_with_positive_test_support": len(supported),
                "prior_macro_auprc": float(reported_prior),
                "recomputed_prior_macro_auprc": prior,
                "strict_random_ranking_expected_macro_auprc": strict_expected,
                "per_label_strict_random_ranking": [
                    {
                        key: item[key]
                        for key in (
                            "label",
                            "test_record_count",
                            "positive_count",
                            "prevalence",
                            "strict_random_ranking_expected_average_precision",
                        )
                    }
                    for item in supported
                ],
                "conditions": condition_outputs,
                "rotation_integrity": rotation_outputs,
            }
        )

    return {
        "seed": seed,
        "design": {
            "group_field": group_field,
            "target_field": target_field,
            "fold_count": 10,
            "top_label_count": top_label_count,
            "min_label_count": min_label_count,
            "text_view": "target_label_masked",
        },
        "folds": fold_outputs,
        "strict_random_ranking": {
            "score_minus_expected": {
                condition: _paired_interval(strict_differences[condition])
                for condition in ("observed", "permutation_a", "permutation_b")
            }
        },
        "tie_aware_random_label": {
            "score_minus_expected": {
                condition: _paired_interval(tie_differences[condition])
                for condition in ("observed", "permutation_a", "permutation_b")
            }
        },
        "rotation_integrity": {
            condition: _aggregate_rotation(fold_outputs, condition)
            for condition in ("permutation_a", "permutation_b")
        },
        "score_replay_validation": {
            "passed": True,
            "absolute_tolerance": SCORE_REPLAY_TOLERANCE,
            "maximum_absolute_error": max(replay_errors),
            "comparison_count": len(replay_errors),
            "note": (
                "The source scorer sums token sets; PYTHONHASHSEED=0 is used by the "
                "reproducibility command to stabilize floating-point accumulation order."
            ),
        },
    }


def build_random_ranking_baseline(
    *,
    musiccaps_manifest_path: Path,
    mtg_manifest_path: Path,
    musiccaps_report_path: Path,
    mtg_report_path: Path,
    reproduction_command: str,
) -> dict[str, Any]:
    """Build the two-source strict-ranking and tie-aware calibration artifact."""

    musiccaps_rows = _read_manifest(
        musiccaps_manifest_path,
        name="MusicCaps manifest",
        group_field="author_id",
        target_field="aspect_list",
    )
    mtg_rows = _read_manifest(
        mtg_manifest_path,
        name="MTG-Jamendo manifest",
        group_field="artist_fold",
        target_field="genre_list",
    )
    musiccaps_report = _read_json(musiccaps_report_path, name="MusicCaps report")
    mtg_report = _read_json(mtg_report_path, name="MTG-Jamendo report")
    input_sha256 = {
        "musiccaps_manifest": _sha256(musiccaps_manifest_path),
        "mtg_jamendo_manifest": _sha256(mtg_manifest_path),
        "musiccaps_report": _sha256(musiccaps_report_path),
        "mtg_jamendo_report": _sha256(mtg_report_path),
    }
    if musiccaps_report.get("provenance", {}).get("manifest_sha256") != input_sha256[
        "musiccaps_manifest"
    ]:
        raise ValueError("MusicCaps report-manifest digest mismatch")
    if mtg_report.get("dataset", {}).get("manifest_sha256") != input_sha256["mtg_jamendo_manifest"]:
        raise ValueError("MTG-Jamendo report-manifest digest mismatch")
    if musiccaps_report.get("seed") != MUSICCAPS_SEED:
        raise ValueError("MusicCaps report seed mismatch")

    sources = {
        "musiccaps": _analyze_source(
            source="musiccaps",
            rows=musiccaps_rows,
            report=musiccaps_report,
            group_field="author_id",
            report_group_field="heldout_author",
            target_field="aspect_list",
            seed=MUSICCAPS_SEED,
        ),
        "mtg_jamendo": _analyze_source(
            source="mtg_jamendo",
            rows=mtg_rows,
            report=mtg_report,
            group_field="artist_fold",
            report_group_field="heldout_artist_fold",
            target_field="genre_list",
            seed=MTG_JAMENDO_SEED,
        ),
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "scope": "Metadata-only calibration of observed and invalid-control target-masked macro average precision.",
        "metric": "macro_auprc",
        "formulae": {
            "strict_random_ranking_expected_average_precision": "k/n + (n-k)(H_n-1)/(n(n-1))",
            "tie_aware_random_label_expected_average_precision": (
                "sum_g ([m_g(M_g+m_g-1)(k-1)/(n(n-1)) + m_g/n] / (M_g+m_g))"
            ),
            "symbols": {
                "n": "held-out records",
                "k": "held-out positives for one selected label",
                "H_n": "nth harmonic number",
                "m_g": "size of ordered score-tie group g",
                "M_g": "records in earlier score-tie groups",
            },
        },
        "inference_note": (
            "Paired Student-t intervals describe the ten fixed held-out-group score "
            "differences; training sets overlap."
        ),
        "inputs": {
            "musiccaps_manifest": {
                "identifier": "musiccaps_manifest.jsonl",
                "sha256": input_sha256["musiccaps_manifest"],
            },
            "mtg_jamendo_manifest": {
                "identifier": "verified_mtg_manifest/mtg_manifest.jsonl",
                "sha256": input_sha256["mtg_jamendo_manifest"],
            },
            "musiccaps_report": {
                "identifier": "musiccaps_gate_calibration.json",
                "sha256": input_sha256["musiccaps_report"],
            },
            "mtg_jamendo_report": {
                "identifier": "mtg_jamendo_metadata_calibration.json",
                "sha256": input_sha256["mtg_jamendo_report"],
            },
        },
        "sources": sources,
        "reproducibility_command": reproduction_command,
    }
