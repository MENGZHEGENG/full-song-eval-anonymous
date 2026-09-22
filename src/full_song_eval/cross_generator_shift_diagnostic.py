from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable


CLAIM_SCOPE = "automatic_proxy_only"

# These fields describe recording, extraction, or array shape rather than an
# automatic song diagnostic.  Excluding them prevents the classifier from
# succeeding only because one pilot is mono/15 s and the other is stereo/30 s.
_TECHNICAL_EXACT_FIELDS = {
    "candidate_index",
    "channels",
    "duration_seconds",
    "sample_rate",
    "section_count",
}
_TECHNICAL_SUFFIXES = (
    "_aligned_duration_seconds",
    "_analysis_duration_seconds",
    "_channels",
    "_duration_seconds",
    "_end_seconds",
    "_fmax_hz",
    "_fmin_hz",
    "_frame_count",
    "_frame_length",
    "_frames",
    "_hop_length",
    "_sample_count",
    "_sample_rate",
    "_start_seconds",
)
_IDENTITY_FIELDS = {"generation_id", "model_id", "prompt_id", "genre", "language"}


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _candidate_index(record: dict[str, Any]) -> int | None:
    value = record.get("candidate_index")
    if value is None and isinstance(record.get("metadata"), dict):
        value = record["metadata"].get("candidate_index")
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid candidate_index: {value!r}") from exc


def _select_candidate_rows(
    label: str,
    records: Iterable[dict[str, Any]],
    *,
    selected_candidate_index: int,
    missing_candidate_index: int | None,
) -> tuple[dict[str, dict[str, Any]], int]:
    selected: dict[str, dict[str, Any]] = {}
    missing_count = 0
    for record in records:
        current_index = _candidate_index(record)
        if current_index is None:
            missing_count += 1
            current_index = missing_candidate_index
        if current_index != selected_candidate_index:
            continue
        prompt_id = str(record.get("prompt_id", "")).strip()
        if not prompt_id:
            raise ValueError(f"Missing prompt_id in selected {label} record")
        if prompt_id in selected:
            raise ValueError(f"Duplicate selected candidates for prompt IDs: {prompt_id}")
        selected[prompt_id] = record
    return selected, missing_count


def _is_usable_feature(field: str) -> bool:
    if field in _IDENTITY_FIELDS or field in _TECHNICAL_EXACT_FIELDS:
        return False
    return not field.endswith(_TECHNICAL_SUFFIXES)


def _common_complete_numeric_features(
    selected: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    labels = list(selected)
    if not labels or not selected[labels[0]]:
        return []
    features: list[str] = []
    all_rows = [row for rows in selected.values() for row in rows.values()]
    common = set(all_rows[0])
    for row in all_rows[1:]:
        common &= set(row)
    for field in sorted(common):
        if not _is_usable_feature(field):
            continue
        if all(_finite_float(row.get(field)) is not None for row in all_rows):
            features.append(field)
    return features


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _population_sd(values: list[float], mean: float | None = None) -> float:
    center = _mean(values) if mean is None else mean
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def _standardized_vectors(
    samples: list[tuple[str, str, list[float]]],
) -> list[tuple[str, str, list[float]]]:
    if not samples:
        return []
    means = [_mean([sample[2][index] for sample in samples]) for index in range(len(samples[0][2]))]
    scales = [_population_sd([sample[2][index] for sample in samples], means[index]) for index in range(len(means))]
    scales = [scale if scale > 0 else 1.0 for scale in scales]
    return [
        (label, prompt_id, [(value - means[index]) / scales[index] for index, value in enumerate(values)])
        for label, prompt_id, values in samples
    ]


def _pair_centered_vectors(
    samples: list[tuple[str, str, list[float]]],
) -> list[tuple[str, str, list[float]]]:
    by_prompt: dict[str, list[tuple[str, str, list[float]]]] = defaultdict(list)
    for sample in samples:
        by_prompt[sample[1]].append(sample)
    centered: list[tuple[str, str, list[float]]] = []
    for prompt_id in sorted(by_prompt):
        prompt_samples = by_prompt[prompt_id]
        if len(prompt_samples) != 2:
            raise ValueError(f"Expected exactly two selected records for prompt {prompt_id}")
        center = [
            _mean([sample[2][index] for sample in prompt_samples])
            for index in range(len(prompt_samples[0][2]))
        ]
        centered.extend(
            (label, prompt_id, [value - center[index] for index, value in enumerate(values)])
            for label, _, values in prompt_samples
        )
    return centered


def _nearest_centroid_accuracy(
    samples: list[tuple[str, str, list[float]]],
    labels: list[str],
) -> dict[str, float | int]:
    prompts = sorted({sample[1] for sample in samples})
    if len(prompts) < 2:
        raise ValueError("At least two prompt groups are required for leave-one-prompt-out prediction")
    correct = 0
    total = 0
    for held_out_prompt in prompts:
        training = [sample for sample in samples if sample[1] != held_out_prompt]
        test = [sample for sample in samples if sample[1] == held_out_prompt]
        standardized_training = _standardized_vectors(training)
        train_means = {
            label: [
                _mean([sample[2][index] for sample in standardized_training if sample[0] == label])
                for index in range(len(training[0][2]))
            ]
            for label in labels
        }
        train_scales = [
            _population_sd([sample[2][index] for sample in training])
            for index in range(len(training[0][2]))
        ]
        train_scales = [scale if scale > 0 else 1.0 for scale in train_scales]
        for label, _, values in test:
            standardized_test = [
                (value - _mean([sample[2][index] for sample in training])) / train_scales[index]
                for index, value in enumerate(values)
            ]
            distances = {
                centroid_label: math.sqrt(
                    sum(
                        (standardized_test[index] - centroid[index]) ** 2
                        for index in range(len(standardized_test))
                    )
                )
                for centroid_label, centroid in train_means.items()
            }
            predicted = min(labels, key=lambda candidate: (distances[candidate], candidate))
            correct += int(predicted == label)
            total += 1
    return {"correct": correct, "total": total, "accuracy": correct / total}


def _feature_shift(
    selected: dict[str, dict[str, dict[str, Any]]],
    labels: list[str],
    fields: list[str],
) -> list[dict[str, float | int | str]]:
    left, right = labels
    shifts: list[dict[str, float | int | str]] = []
    for field in fields:
        left_values = [_finite_float(row[field]) for row in selected[left].values()]
        right_values = [_finite_float(row[field]) for row in selected[right].values()]
        if not all(value is not None for value in left_values + right_values):
            raise ValueError(f"Feature {field!r} unexpectedly contains a nonfinite value")
        left_numeric = [value for value in left_values if value is not None]
        right_numeric = [value for value in right_values if value is not None]
        left_mean = _mean(left_numeric)
        right_mean = _mean(right_numeric)
        left_sd = _population_sd(left_numeric, left_mean)
        right_sd = _population_sd(right_numeric, right_mean)
        pooled_sd = math.sqrt((left_sd**2 + right_sd**2) / 2)
        if pooled_sd == 0:
            smd: float | None = 0.0 if left_mean == right_mean else None
        else:
            smd = (right_mean - left_mean) / pooled_sd
        absolute_smd = None if smd is None else abs(smd)
        shifts.append(
            {
                "field": field,
                "left_mean": left_mean,
                "right_mean": right_mean,
                "left_sd": left_sd,
                "right_sd": right_sd,
                "smd_right_minus_left": smd,
                "absolute_smd": absolute_smd,
                "left_records": len(left_numeric),
                "right_records": len(right_numeric),
            }
        )
    return sorted(
        shifts,
        key=lambda row: (
            row["absolute_smd"] is None,
            -float(row["absolute_smd"]) if row["absolute_smd"] is not None else 0.0,
            str(row["field"]),
        ),
    )


def _permutation_null(
    samples: list[tuple[str, str, list[float]]],
    labels: list[str],
    observed_accuracy: float,
    *,
    permutations: int,
    seed: int,
) -> dict[str, float | int | None]:
    if permutations <= 0:
        return {
            "permutations": 0,
            "seed": seed,
            "mean_accuracy": None,
            "max_accuracy": None,
            "p_value_plus_one": None,
        }
    rng = random.Random(seed)
    accuracies: list[float] = []
    samples_by_prompt: dict[str, list[tuple[str, str, list[float]]]] = defaultdict(list)
    for sample in samples:
        samples_by_prompt[sample[1]].append(sample)
    for _ in range(permutations):
        permuted: list[tuple[str, str, list[float]]] = []
        for prompt_id in sorted(samples_by_prompt):
            swap = rng.random() < 0.5
            for original_label, sample_prompt, values in samples_by_prompt[prompt_id]:
                permuted_label = labels[1] if swap and original_label == labels[0] else labels[0] if swap else original_label
                permuted.append((permuted_label, sample_prompt, values))
        accuracies.append(float(_nearest_centroid_accuracy(permuted, labels)["accuracy"]))
    exceedances = sum(accuracy >= observed_accuracy for accuracy in accuracies)
    return {
        "permutations": permutations,
        "seed": seed,
        "mean_accuracy": _mean(accuracies),
        "max_accuracy": max(accuracies),
        "p_value_plus_one": (exceedances + 1) / (permutations + 1),
    }


def analyze_generator_shift(
    sources: dict[str, list[dict[str, Any]]],
    *,
    selected_candidate_indices: dict[str, int] | None = None,
    missing_candidate_indices: dict[str, int | None] | None = None,
    permutations: int = 99,
    seed: int = 20260912,
) -> dict[str, Any]:
    """Measure prompt-held-out automatic-feature shift between two generators.

    The classifier predicts generator family from automatic features. It is a
    protocol and distribution-shift diagnostic; it is not trained to predict
    listener preference and cannot establish quality or generator superiority.
    """
    labels = list(sources)
    if len(labels) != 2:
        raise ValueError("Exactly two generator families are required")
    selected_candidate_indices = selected_candidate_indices or {}
    missing_candidate_indices = missing_candidate_indices or {}
    selected: dict[str, dict[str, dict[str, Any]]] = {}
    source_summaries: dict[str, dict[str, int]] = {}
    for label in labels:
        selected_rows, missing_count = _select_candidate_rows(
            label,
            sources[label],
            selected_candidate_index=int(selected_candidate_indices.get(label, 0)),
            missing_candidate_index=missing_candidate_indices.get(label),
        )
        selected[label] = selected_rows
        source_summaries[label] = {
            "input_records": len(sources[label]),
            "selected_records": len(selected_rows),
            "missing_candidate_index_records": missing_count,
        }
    prompt_sets = [set(rows) for rows in selected.values()]
    if prompt_sets[0] != prompt_sets[1]:
        raise ValueError(
            f"Prompt mismatch: missing_left={sorted(prompt_sets[1] - prompt_sets[0])[:10]}, "
            f"missing_right={sorted(prompt_sets[0] - prompt_sets[1])[:10]}"
        )
    prompts = sorted(prompt_sets[0])
    if len(prompts) < 2:
        raise ValueError("At least two prompt groups are required")
    fields = _common_complete_numeric_features(selected)
    if not fields:
        raise ValueError("No usable common numeric features after technical-field exclusions")
    samples = [
        (label, prompt_id, [float(selected[label][prompt_id][field]) for field in fields])
        for prompt_id in prompts
        for label in labels
    ]
    raw_prediction = _nearest_centroid_accuracy(samples, labels)
    centered_samples = _pair_centered_vectors(samples)
    centered_prediction = _nearest_centroid_accuracy(centered_samples, labels)
    return {
        "schema_version": 1,
        "status": "pass",
        "claim_scope": CLAIM_SCOPE,
        "generator_labels": labels,
        "inputs": source_summaries,
        "pairing": {
            "prompts": len(prompts),
            "selected_records": {label: len(selected[label]) for label in labels},
            "candidate_indices": {
                label: int(selected_candidate_indices.get(label, 0)) for label in labels
            },
            "prompt_disjoint_leave_one_out": True,
        },
        "feature_policy": {
            "fields": fields,
            "excluded_capture_metadata": sorted(_TECHNICAL_EXACT_FIELDS),
            "excluded_suffixes": list(_TECHNICAL_SUFFIXES),
        },
        "feature_shift": _feature_shift(selected, labels, fields),
        "family_prediction": {
            "raw": raw_prediction,
            "pair_centered": centered_prediction,
            "permutation_null_raw": _permutation_null(
                samples,
                labels,
                float(raw_prediction["accuracy"]),
                permutations=permutations,
                seed=seed,
            ),
        },
        "interpretation": (
            "Family predictability indicates automatic-feature distribution shift across the evaluated "
            "generator snapshots. It does not identify which family is better and does not provide listener "
            "preference, evaluator-human agreement, or deployment evidence."
        ),
    }
