from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable

from full_song_eval.cross_generator_shift_diagnostic import (
    CLAIM_SCOPE,
    _IDENTITY_FIELDS,
    _TECHNICAL_EXACT_FIELDS,
    _TECHNICAL_SUFFIXES,
    _finite_float,
    _is_usable_feature,
    _mean,
    _pair_centered_vectors,
    _population_sd,
    _select_candidate_rows,
)


SCHEMA_VERSION = 1
DEFAULT_BOOTSTRAP_REPLICATES = 2000

# These names denote signal level or amplitude.  The policy is intentionally
# explicit so that a future feature-table schema cannot silently change the
# meaning of the audit.
_AMPLITUDE_TOKENS = (
    "_amplitude",
    "_rms",
    "_peak",
    "_energy",
    "_db",
)


def _is_amplitude_or_level(field: str) -> bool:
    return any(token in field for token in _AMPLITUDE_TOKENS)


def _common_complete_numeric_features(
    selected: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    labels = list(selected)
    if not labels or not selected[labels[0]]:
        return []
    all_records = [record for records in selected.values() for record in records.values()]
    common = set(all_records[0])
    for record in all_records[1:]:
        common &= set(record)
    fields: list[str] = []
    for field in sorted(common):
        if field in _IDENTITY_FIELDS:
            continue
        if all(_finite_float(record.get(field)) is not None for record in all_records):
            fields.append(field)
    return fields


def _policy_fields(all_fields: list[str]) -> dict[str, list[str]]:
    return {
        "all_common_numeric": list(all_fields),
        "without_amplitude": [field for field in all_fields if not _is_amplitude_or_level(field)],
        "without_technical_format": [field for field in all_fields if _is_usable_feature(field)],
        "without_technical_and_amplitude": [
            field
            for field in all_fields
            if _is_usable_feature(field) and not _is_amplitude_or_level(field)
        ],
    }


def _samples_for_fields(
    selected: dict[str, dict[str, dict[str, Any]]],
    labels: list[str],
    fields: list[str],
) -> list[tuple[str, str, list[float]]]:
    prompts = sorted(set(selected[labels[0]]) & set(selected[labels[1]]))
    return [
        (
            label,
            prompt_id,
            [float(selected[label][prompt_id][field]) for field in fields],
        )
        for prompt_id in prompts
        for label in labels
    ]


def _nearest_centroid_predictions(
    samples: list[tuple[str, str, list[float]]],
    labels: list[str],
) -> list[dict[str, Any]]:
    prompts = sorted({sample[1] for sample in samples})
    if len(prompts) < 2:
        raise ValueError("At least two prompt groups are required for leave-one-prompt-out prediction")
    predictions: list[dict[str, Any]] = []
    for held_out_prompt in prompts:
        training = [sample for sample in samples if sample[1] != held_out_prompt]
        test = [sample for sample in samples if sample[1] == held_out_prompt]
        if len(test) != 2:
            raise ValueError(f"Expected exactly two samples for prompt {held_out_prompt}")
        train_means = [_mean([sample[2][index] for sample in training]) for index in range(len(training[0][2]))]
        train_scales = [
            _population_sd([sample[2][index] for sample in training], train_means[index])
            for index in range(len(train_means))
        ]
        train_scales = [scale if scale > 0 else 1.0 for scale in train_scales]
        standardized_training = [
            (
                label,
                prompt_id,
                [(value - train_means[index]) / train_scales[index] for index, value in enumerate(values)],
            )
            for label, prompt_id, values in training
        ]
        feature_count = len(training[0][2])
        centroids = {
            label: [
                _mean([sample[2][index] for sample in standardized_training if sample[0] == label])
                for index in range(feature_count)
            ]
            for label in labels
        }
        for label, prompt_id, values in test:
            standardized_test = [
                (value - train_means[index]) / train_scales[index]
                for index, value in enumerate(values)
            ]
            distances = {
                candidate: math.sqrt(
                    sum(
                        (standardized_test[index] - centroids[candidate][index]) ** 2
                        for index in range(len(standardized_test))
                    )
                )
                for candidate in labels
            }
            predicted = min(labels, key=lambda candidate: (distances[candidate], candidate))
            predictions.append(
                {
                    "prompt_id": prompt_id,
                    "true_label": label,
                    "predicted_label": predicted,
                    "correct": predicted == label,
                    "held_out_prompt": prompt_id,
                }
            )
    return predictions


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile from an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_group_metrics(
    per_prompt: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        return {
            "replicates": 0,
            "seed": seed,
            "prompt_accuracy_ci95": None,
            "exact_prompt_accuracy_ci95": None,
        }
    rng = random.Random(seed)
    prompt_accuracy = [float(item["accuracy"]) for item in per_prompt]
    exact_accuracy = [float(item["all_correct"]) for item in per_prompt]
    prompt_means: list[float] = []
    exact_means: list[float] = []
    for _ in range(replicates):
        indices = [rng.randrange(len(per_prompt)) for _ in per_prompt]
        prompt_means.append(_mean([prompt_accuracy[index] for index in indices]))
        exact_means.append(_mean([exact_accuracy[index] for index in indices]))
    return {
        "replicates": replicates,
        "seed": seed,
        "prompt_accuracy_ci95": [
            _percentile(prompt_means, 0.025),
            _percentile(prompt_means, 0.975),
        ],
        "exact_prompt_accuracy_ci95": [
            _percentile(exact_means, 0.025),
            _percentile(exact_means, 0.975),
        ],
    }


def _evaluate_representation(
    samples: list[tuple[str, str, list[float]]],
    labels: list[str],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    predictions = _nearest_centroid_predictions(samples, labels)
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_prompt[str(prediction["prompt_id"])].append(prediction)
    per_prompt: list[dict[str, Any]] = []
    for prompt_id in sorted(by_prompt):
        prompt_predictions = by_prompt[prompt_id]
        correct = sum(bool(item["correct"]) for item in prompt_predictions)
        per_prompt.append(
            {
                "prompt_id": prompt_id,
                "correct": correct,
                "total": len(prompt_predictions),
                "accuracy": correct / len(prompt_predictions),
                "all_correct": correct == len(prompt_predictions),
            }
        )
    correct = sum(bool(item["correct"]) for item in predictions)
    prompt_accuracy = _mean([float(item["accuracy"]) for item in per_prompt])
    exact_prompt_accuracy = _mean([float(item["all_correct"]) for item in per_prompt])
    return {
        "sample_level": {
            "correct": correct,
            "total": len(predictions),
            "accuracy": correct / len(predictions),
        },
        "prompt_group": {
            "prompt_groups": len(per_prompt),
            "mean_accuracy": prompt_accuracy,
            "exact_accuracy": exact_prompt_accuracy,
            "bootstrap": _bootstrap_group_metrics(
                per_prompt,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            ),
        },
        "per_prompt": per_prompt,
    }


def _validate_pairing(
    sources: dict[str, list[dict[str, Any]]],
    *,
    selected_candidate_indices: dict[str, int],
    missing_candidate_indices: dict[str, int | None],
) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, int]]]:
    labels = list(sources)
    if len(labels) != 2:
        raise ValueError("Exactly two generator families are required")
    selected: dict[str, dict[str, dict[str, Any]]] = {}
    summaries: dict[str, dict[str, int]] = {}
    for label in labels:
        records = sources[label]
        selected_rows, missing_count = _select_candidate_rows(
            label,
            records,
            selected_candidate_index=int(selected_candidate_indices.get(label, 0)),
            missing_candidate_index=missing_candidate_indices.get(label),
        )
        selected[label] = selected_rows
        summaries[label] = {
            "input_records": len(records),
            "selected_records": len(selected_rows),
            "missing_candidate_index_records": missing_count,
        }
    prompt_sets = [set(rows) for rows in selected.values()]
    if prompt_sets[0] != prompt_sets[1]:
        raise ValueError(
            f"Prompt mismatch: missing_left={sorted(prompt_sets[1] - prompt_sets[0])[:10]}, "
            f"missing_right={sorted(prompt_sets[0] - prompt_sets[1])[:10]}"
        )
    if len(prompt_sets[0]) < 2:
        raise ValueError("At least two prompt groups are required")
    return labels, selected, summaries


def audit_feature_policies(
    sources: dict[str, list[dict[str, Any]]],
    *,
    selected_candidate_indices: dict[str, int] | None = None,
    missing_candidate_indices: dict[str, int | None] | None = None,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = 20260913,
) -> dict[str, Any]:
    """Audit generator-family predictability under explicit feature policies.

    All representations use the same prompt-held-out nearest-centroid
    predictor.  Bootstrap intervals resample whole prompt groups from the
    fixed cross-validated predictions, so they do not retrain on resampled
    observations or treat the two members of a prompt pair as independent.
    The output is an automatic proxy diagnostic only.
    """
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be nonnegative")
    selected_candidate_indices = selected_candidate_indices or {}
    missing_candidate_indices = missing_candidate_indices or {}
    labels, selected, input_summaries = _validate_pairing(
        sources,
        selected_candidate_indices=selected_candidate_indices,
        missing_candidate_indices=missing_candidate_indices,
    )
    all_fields = _common_complete_numeric_features(selected)
    if not all_fields:
        raise ValueError("No common complete numeric features are available")
    policies = _policy_fields(all_fields)
    if any(not fields for fields in policies.values()):
        empty = [name for name, fields in policies.items() if not fields]
        raise ValueError(f"Feature policies have no usable fields: {empty}")

    policy_reports: dict[str, Any] = {}
    for policy_index, (policy_name, fields) in enumerate(policies.items()):
        samples = _samples_for_fields(selected, labels, fields)
        representations = {
            "raw": samples,
            "pair_centered": _pair_centered_vectors(samples),
        }
        representation_reports: dict[str, Any] = {}
        for representation_index, (representation_name, representation_samples) in enumerate(
            representations.items()
        ):
            representation_reports[representation_name] = _evaluate_representation(
                representation_samples,
                labels,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=seed + policy_index * 100 + representation_index,
            )
        policy_reports[policy_name] = {
            "fields": fields,
            "field_count": len(fields),
            "excluded_amplitude_or_level_fields": [
                field for field in all_fields if field not in fields and _is_amplitude_or_level(field)
            ],
            "excluded_technical_or_format_fields": [
                field for field in all_fields if field not in fields and _is_usable_feature(field) is False
            ],
            "representations": representation_reports,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "claim_scope": CLAIM_SCOPE,
        "generator_labels": labels,
        "inputs": input_summaries,
        "pairing": {
            "prompts": len(selected[labels[0]]),
            "selected_records": {label: len(selected[label]) for label in labels},
            "candidate_indices": {
                label: int(selected_candidate_indices.get(label, 0)) for label in labels
            },
            "prompt_disjoint_leave_one_out": True,
            "groups_resampled_as_units": True,
        },
        "feature_policy": {
            "all_common_numeric_fields": all_fields,
            "identity_fields_excluded_from_all_policies": sorted(_IDENTITY_FIELDS),
            "amplitude_or_level_tokens": list(_AMPLITUDE_TOKENS),
            "technical_fields_excluded_by_existing_policy": {
                "exact": sorted(_TECHNICAL_EXACT_FIELDS),
                "suffixes": list(_TECHNICAL_SUFFIXES),
            },
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": seed,
        },
        "policies": policy_reports,
        "interpretation": (
            "Family predictability is an automatic-feature distribution-shift diagnostic. Comparing "
            "policies identifies sensitivity to captured format and level features, but it does not "
            "establish listener preference, perceptual quality, evaluator-human agreement, or deployment utility."
        ),
    }
