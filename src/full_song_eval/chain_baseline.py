from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from full_song_eval.annotation_summary import RATING_DIMENSIONS
from full_song_eval.jsonl import read_jsonl
from full_song_eval.pairwise_baseline import (
    PREFERENCE_TO_B_LABEL,
    PairwiseExample,
    build_examples,
    cross_validate_logistic_regression,
    delta_key,
    evaluate_model,
    load_prompt_fold_map,
    prompt_group_folds,
    select_feature_names,
    standardize,
    train_logistic_regression,
)

DEFAULT_CHAIN_DIMENSIONS = [dimension for dimension in RATING_DIMENSIONS if dimension != "overall_preference"]
CHAIN_MODE_PREDICTED = "predicted_crossfit"
CHAIN_MODE_ORACLE = "oracle_teacher_forced"
CHAIN_MODES = {CHAIN_MODE_PREDICTED, CHAIN_MODE_ORACLE}


@dataclass(frozen=True)
class LinearModel:
    feature_names: list[str]
    means: list[float]
    scales: list[float]
    weights: list[float]
    bias: float


@dataclass(frozen=True)
class ChainTrainingRow:
    task_id: str
    prompt_id: str
    label_b_preferred: int
    base_features: list[float]
    aspect_targets: dict[str, float]


def chain_feature_name(dimension: str) -> str:
    return f"chain_delta_{dimension}"


def augment_pairwise_deltas_with_chain_context(
    annotations: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    *,
    chain_dimensions: list[str] | None = None,
) -> list[dict[str, Any]]:
    dimensions = chain_dimensions or DEFAULT_CHAIN_DIMENSIONS
    annotations_by_key = {delta_key_from_annotation(annotation): annotation for annotation in annotations}
    augmented = []
    for row in delta_rows:
        output_row = dict(row)
        annotation = annotations_by_key.get(delta_key(row))
        if annotation is not None:
            output_row.update(_chain_features(annotation, dimensions))
        augmented.append(output_row)
    return augmented


def build_chain_examples(
    annotations: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    *,
    base_feature_group: str = "all",
    chain_dimensions: list[str] | None = None,
):
    dimensions = chain_dimensions or DEFAULT_CHAIN_DIMENSIONS
    augmented_rows = augment_pairwise_deltas_with_chain_context(annotations, delta_rows, chain_dimensions=dimensions)
    base_feature_names = select_feature_names(delta_rows, base_feature_group)
    chain_feature_names = [chain_feature_name(dimension) for dimension in dimensions]
    return build_examples(annotations, augmented_rows, feature_names=base_feature_names + chain_feature_names)


def build_predicted_chain_examples(
    annotations: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    *,
    base_feature_group: str = "all",
    chain_dimensions: list[str] | None = None,
    epochs: int = 500,
    learning_rate: float = 0.05,
    l2: float = 0.01,
    seed: int = 0,
) -> tuple[list[PairwiseExample], list[str], dict[str, LinearModel], list[ChainTrainingRow]]:
    dimensions = chain_dimensions or DEFAULT_CHAIN_DIMENSIONS
    base_feature_names = select_feature_names(delta_rows, base_feature_group)
    rows = _chain_training_rows(annotations, delta_rows, base_feature_names=base_feature_names, dimensions=dimensions)
    feature_names = base_feature_names + [chain_feature_name(dimension) for dimension in dimensions]
    if not rows:
        return [], feature_names, {}, rows
    aspect_models = _train_aspect_models(
        rows,
        base_feature_names=base_feature_names,
        dimensions=dimensions,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
    )
    examples = _predicted_preference_examples(rows, aspect_models=aspect_models, dimensions=dimensions)
    return examples, feature_names, aspect_models, rows


def train_chain_baseline(
    *,
    annotations_path: Path,
    pairwise_deltas_path: Path,
    output_json: Path,
    base_feature_group: str = "all",
    chain_dimensions: list[str] | None = None,
    folds: int = 5,
    epochs: int = 500,
    learning_rate: float = 0.05,
    l2: float = 0.01,
    seed: int = 0,
    split_plan_path: Path | None = None,
    training_guard: dict[str, object] | None = None,
    chain_mode: str = CHAIN_MODE_PREDICTED,
) -> dict[str, Any]:
    annotations = read_jsonl(annotations_path)
    delta_rows = read_jsonl(pairwise_deltas_path)
    split_plan_error = None
    try:
        prompt_to_fold = load_prompt_fold_map(split_plan_path) if split_plan_path is not None else None
    except ValueError as exc:
        prompt_to_fold = None
        split_plan_error = str(exc)
    if chain_mode not in CHAIN_MODES:
        raise ValueError(f"unknown chain mode: {chain_mode}; expected one of {', '.join(sorted(CHAIN_MODES))}")
    dimensions = chain_dimensions or DEFAULT_CHAIN_DIMENSIONS
    if chain_mode == CHAIN_MODE_ORACLE:
        examples, feature_names = build_chain_examples(
            annotations,
            delta_rows,
            base_feature_group=base_feature_group,
            chain_dimensions=dimensions,
        )
        chain_rows: list[ChainTrainingRow] = []
        aspect_models: dict[str, LinearModel] = {}
    else:
        examples, feature_names, aspect_models, chain_rows = build_predicted_chain_examples(
            annotations,
            delta_rows,
            base_feature_group=base_feature_group,
            chain_dimensions=dimensions,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
        )
    summary: dict[str, Any] = {
        "annotations": str(annotations_path),
        "pairwise_deltas": str(pairwise_deltas_path),
        "base_feature_group": base_feature_group,
        "chain_mode": chain_mode,
        "chain_feature_source": "fold-held-out predicted aspect deltas" if chain_mode == CHAIN_MODE_PREDICTED else "gold annotation rating deltas",
        "leakage_safe": chain_mode == CHAIN_MODE_PREDICTED,
        "claim_boundary": "Use predicted_crossfit for deployable evaluator claims; oracle_teacher_forced is a diagnostic upper bound only.",
        "chain_dimensions": dimensions,
        "chain_feature_count": len(dimensions),
        "split_plan": str(split_plan_path) if split_plan_path is not None else None,
        "split_plan_prompt_count": len(prompt_to_fold) if prompt_to_fold is not None else 0,
        "usable_examples": len(examples),
        "features": len(feature_names),
        "feature_names": feature_names,
        "status": "ok",
    }
    if chain_mode == CHAIN_MODE_PREDICTED:
        summary["aspect_models"] = {dimension: _model_dict(model) for dimension, model in sorted(aspect_models.items())}
        summary["aspect_training_examples"] = len(chain_rows)
    if training_guard is not None:
        summary["training_guard"] = training_guard
    if split_plan_error is not None:
        summary["status"] = "invalid_split_plan"
        summary["reason"] = split_plan_error
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
    try:
        model = train_logistic_regression(
            examples,
            feature_names,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
        )
        summary["train_metrics"] = evaluate_model(model, examples)
        if chain_mode == CHAIN_MODE_PREDICTED:
            summary["cross_validation"] = _cross_validate_predicted_chain(
                chain_rows,
                base_feature_names=select_feature_names(delta_rows, base_feature_group),
                dimensions=dimensions,
                feature_names=feature_names,
                folds=folds,
                epochs=epochs,
                learning_rate=learning_rate,
                l2=l2,
                seed=seed,
                prompt_to_fold=prompt_to_fold,
            )
        else:
            summary["cross_validation"] = cross_validate_logistic_regression(
                examples,
                feature_names,
                folds=folds,
                epochs=epochs,
                learning_rate=learning_rate,
                l2=l2,
                seed=seed,
                prompt_to_fold=prompt_to_fold,
            )
        summary["model"] = {
            "feature_names": model.feature_names,
            "means": model.means,
            "scales": model.scales,
            "weights": model.weights,
            "bias": model.bias,
        }
    except ValueError as exc:
        summary["status"] = "insufficient_labels"
        summary["reason"] = str(exc)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def delta_key_from_annotation(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("prompt_id", "")),
        str(record.get("candidate_a", {}).get("generation_id", "")),
        str(record.get("candidate_b", {}).get("generation_id", "")),
    )


def _chain_features(annotation: dict[str, Any], dimensions: list[str]) -> dict[str, float]:
    ratings = annotation.get("ratings", {}) if isinstance(annotation.get("ratings", {}), dict) else {}
    candidate_a = ratings.get("candidate_a", {}) if isinstance(ratings.get("candidate_a", {}), dict) else {}
    candidate_b = ratings.get("candidate_b", {}) if isinstance(ratings.get("candidate_b", {}), dict) else {}
    features = {}
    for dimension in dimensions:
        a_value = candidate_a.get(dimension)
        b_value = candidate_b.get(dimension)
        if isinstance(a_value, int | float) and isinstance(b_value, int | float):
            features[chain_feature_name(dimension)] = float(b_value) - float(a_value)
    return features


def _chain_training_rows(
    annotations: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    *,
    base_feature_names: list[str],
    dimensions: list[str],
) -> list[ChainTrainingRow]:
    deltas_by_key = {delta_key(row): row for row in delta_rows}
    rows = []
    for annotation in annotations:
        preference = annotation.get("pairwise_preference")
        if preference not in PREFERENCE_TO_B_LABEL:
            continue
        delta_row = deltas_by_key.get(delta_key_from_annotation(annotation))
        if delta_row is None:
            continue
        base_features = _feature_values(delta_row, base_feature_names)
        if base_features is None:
            continue
        aspect_targets: dict[str, float] = {}
        for dimension in dimensions:
            target = _rating_delta(annotation, dimension)
            if target is None:
                break
            aspect_targets[dimension] = target
        if len(aspect_targets) != len(dimensions):
            continue
        rows.append(
            ChainTrainingRow(
                task_id=str(delta_row.get("task_id", annotation.get("annotation_id", ""))),
                prompt_id=str(annotation.get("prompt_id", "")),
                label_b_preferred=PREFERENCE_TO_B_LABEL[str(preference)],
                base_features=base_features,
                aspect_targets=aspect_targets,
            )
        )
    return rows


def _feature_values(row: dict[str, Any], feature_names: list[str]) -> list[float] | None:
    values = []
    for feature_name in feature_names:
        value = row.get(feature_name)
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            return None
        values.append(float(value))
    return values


def _rating_delta(annotation: dict[str, Any], dimension: str) -> float | None:
    ratings = annotation.get("ratings", {}) if isinstance(annotation.get("ratings", {}), dict) else {}
    candidate_a = ratings.get("candidate_a", {}) if isinstance(ratings.get("candidate_a", {}), dict) else {}
    candidate_b = ratings.get("candidate_b", {}) if isinstance(ratings.get("candidate_b", {}), dict) else {}
    candidate_a_value = candidate_a.get(dimension)
    candidate_b_value = candidate_b.get(dimension)
    if isinstance(candidate_a_value, int | float) and isinstance(candidate_b_value, int | float):
        return float(candidate_b_value) - float(candidate_a_value)
    return None


def _train_aspect_models(
    rows: list[ChainTrainingRow],
    *,
    base_feature_names: list[str],
    dimensions: list[str],
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> dict[str, LinearModel]:
    if not rows:
        raise ValueError("Need at least one labeled row with aspect ratings to train predicted-chain features")
    return {
        dimension: _train_linear_regression(
            [row.base_features for row in rows],
            [row.aspect_targets[dimension] for row in rows],
            base_feature_names,
            epochs=epochs,
            learning_rate=min(learning_rate, 0.05),
            l2=l2,
            seed=seed + dimension_index,
        )
        for dimension_index, dimension in enumerate(dimensions)
    }


def _train_linear_regression(
    feature_rows: list[list[float]],
    targets: list[float],
    feature_names: list[str],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> LinearModel:
    if not feature_rows:
        raise ValueError("Need at least one row to train an aspect model")
    examples = [PairwiseExample(task_id=str(index), prompt_id=str(index), label_b_preferred=0, features=features) for index, features in enumerate(feature_rows)]
    standardized, means, scales = standardize(examples)
    weights = [0.0 for _ in feature_names]
    bias = statistics.mean(targets)
    order = list(range(len(feature_rows)))
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(order)
        for index in order:
            prediction = bias + sum(weight * value for weight, value in zip(weights, standardized[index], strict=True))
            error = prediction - targets[index]
            bias -= learning_rate * error
            for feature_index, value in enumerate(standardized[index]):
                gradient = error * value + l2 * weights[feature_index]
                weights[feature_index] -= learning_rate * gradient
    return LinearModel(feature_names=feature_names, means=means, scales=scales, weights=weights, bias=bias)


def _predict_linear(model: LinearModel, features: list[float]) -> float:
    standardized = [(value - mean) / scale for value, mean, scale in zip(features, model.means, model.scales, strict=True)]
    return model.bias + sum(weight * value for weight, value in zip(model.weights, standardized, strict=True))


def _predicted_preference_examples(
    rows: list[ChainTrainingRow],
    *,
    aspect_models: dict[str, LinearModel],
    dimensions: list[str],
) -> list[PairwiseExample]:
    examples = []
    for row in rows:
        predicted_chain_features = [_predict_linear(aspect_models[dimension], row.base_features) for dimension in dimensions]
        examples.append(
            PairwiseExample(
                task_id=row.task_id,
                prompt_id=row.prompt_id,
                label_b_preferred=row.label_b_preferred,
                features=row.base_features + predicted_chain_features,
            )
        )
    return examples


def _cross_validate_predicted_chain(
    rows: list[ChainTrainingRow],
    *,
    base_feature_names: list[str],
    dimensions: list[str],
    feature_names: list[str],
    folds: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
    prompt_to_fold: dict[str, int] | None,
) -> dict[str, Any]:
    if len(rows) < 4:
        raise ValueError("Need at least four examples for prompt-group cross-validation")
    base_examples = [PairwiseExample(row.task_id, row.prompt_id, row.label_b_preferred, row.base_features) for row in rows]
    fold_metrics = []
    skipped_folds = []
    splits = prompt_group_folds(base_examples, folds=folds, seed=seed, prompt_to_fold=prompt_to_fold)
    for fold_index, (train_indices, test_indices) in enumerate(splits):
        train_rows = [rows[index] for index in train_indices]
        test_rows = [rows[index] for index in test_indices]
        if not train_rows or not test_rows:
            skipped_folds.append({"fold_index": fold_index, "reason": "empty_train_or_test"})
            continue
        if len({row.label_b_preferred for row in train_rows}) < 2:
            skipped_folds.append({"fold_index": fold_index, "reason": "single_preference_class_in_training"})
            continue
        aspect_models = _train_aspect_models(
            train_rows,
            base_feature_names=base_feature_names,
            dimensions=dimensions,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed + fold_index,
        )
        train_examples = _predicted_preference_examples(train_rows, aspect_models=aspect_models, dimensions=dimensions)
        test_examples = _predicted_preference_examples(test_rows, aspect_models=aspect_models, dimensions=dimensions)
        model = train_logistic_regression(
            train_examples,
            feature_names,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
        )
        metrics = evaluate_model(model, test_examples)
        metrics.update(
            {
                "fold_index": fold_index,
                "train_examples": len(train_examples),
                "test_examples": len(test_examples),
                "aspect_models": len(aspect_models),
            }
        )
        fold_metrics.append(metrics)
    if not fold_metrics:
        raise ValueError("No cross-validation fold had both preference classes in training data")
    return {
        "folds": len(fold_metrics),
        "fold_source": "predefined_prompt_folds" if prompt_to_fold is not None else "seeded_prompt_shuffle",
        "chain_mode": CHAIN_MODE_PREDICTED,
        "test_chain_feature_source": "aspect models trained without held-out prompt labels",
        "mean_accuracy": statistics.mean(metric["accuracy"] for metric in fold_metrics),
        "mean_log_loss": statistics.mean(metric["log_loss"] for metric in fold_metrics),
        "fold_metrics": fold_metrics,
        "skipped_folds": skipped_folds,
    }


def _model_dict(model: LinearModel) -> dict[str, Any]:
    return {
        "feature_names": model.feature_names,
        "means": model.means,
        "scales": model.scales,
        "weights": model.weights,
        "bias": model.bias,
    }
