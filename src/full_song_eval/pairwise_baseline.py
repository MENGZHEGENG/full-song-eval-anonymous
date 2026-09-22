from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PREFERENCE_TO_B_LABEL = {
    "a_much_better": 0,
    "a_better": 0,
    "b_better": 1,
    "b_much_better": 1,
}

FEATURE_GROUP_PREFIXES = {
    "all": ("delta_",),
    "asr": ("delta_asr_",),
    "audio": ("delta_audio_", "delta_duration_", "delta_sample_rate", "delta_channels"),
    "caption": ("delta_caption_", "delta_audio_language_"),
    "codec": ("delta_codec_", "delta_embedding_"),
    "melody": ("delta_melody_",),
    "no_section": ("delta_",),
    "section": ("delta_section_",),
    "stem": ("delta_stem_", "delta_interaction_"),
    "no_stem": (
        "delta_asr_",
        "delta_audio_",
        "delta_caption_",
        "delta_codec_",
        "delta_duration_",
        "delta_embedding_",
        "delta_sample_rate",
        "delta_channels",
        "delta_melody_",
        "delta_section_",
    ),
}


@dataclass(frozen=True)
class PairwiseExample:
    task_id: str
    prompt_id: str
    label_b_preferred: int
    features: list[float]


@dataclass(frozen=True)
class LogisticModel:
    feature_names: list[str]
    means: list[float]
    scales: list[float]
    weights: list[float]
    bias: float


def annotation_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("prompt_id", "")),
        str(record.get("candidate_a", {}).get("generation_id", "")),
        str(record.get("candidate_b", {}).get("generation_id", "")),
    )


def delta_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("prompt_id", "")),
        str(record.get("candidate_a_generation_id", "")),
        str(record.get("candidate_b_generation_id", "")),
    )


def default_feature_names(delta_rows: list[dict[str, Any]]) -> list[str]:
    names = set()
    for row in delta_rows:
        for key, value in row.items():
            if key.startswith("delta_") and isinstance(value, int | float) and math.isfinite(float(value)):
                names.add(key)
    return sorted(names)


def select_feature_names(delta_rows: list[dict[str, Any]], feature_group: str = "all") -> list[str]:
    if feature_group not in FEATURE_GROUP_PREFIXES:
        valid = ", ".join(sorted(FEATURE_GROUP_PREFIXES))
        raise ValueError(f"unknown feature group: {feature_group}; expected one of {valid}")
    names = default_feature_names(delta_rows)
    prefixes = FEATURE_GROUP_PREFIXES[feature_group]
    if feature_group == "no_section":
        return [name for name in names if not name.startswith("delta_section_")]
    selected = [name for name in names if name.startswith(prefixes)]
    if feature_group == "all":
        return names
    return selected


def build_examples(
    annotations: list[dict[str, Any]], delta_rows: list[dict[str, Any]], feature_names: list[str] | None = None
) -> tuple[list[PairwiseExample], list[str]]:
    features = default_feature_names(delta_rows) if feature_names is None else feature_names
    deltas_by_key = {delta_key(row): row for row in delta_rows}
    examples = []
    for annotation in annotations:
        preference = annotation.get("pairwise_preference")
        if preference not in PREFERENCE_TO_B_LABEL:
            continue
        key = annotation_key(annotation)
        if key not in deltas_by_key:
            continue
        delta_row = deltas_by_key[key]
        values = []
        missing = False
        for feature_name in features:
            value = delta_row.get(feature_name)
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                missing = True
                break
            values.append(float(value))
        if missing:
            continue
        examples.append(
            PairwiseExample(
                task_id=str(delta_row.get("task_id", annotation.get("annotation_id", ""))),
                prompt_id=str(annotation.get("prompt_id", "")),
                label_b_preferred=PREFERENCE_TO_B_LABEL[str(preference)],
                features=values,
            )
        )
    return examples, features


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def standardize(examples: list[PairwiseExample]) -> tuple[list[list[float]], list[float], list[float]]:
    if not examples:
        return [], [], []
    columns = list(zip(*(example.features for example in examples), strict=True))
    means = [statistics.mean(column) for column in columns]
    scales = []
    for column in columns:
        scale = statistics.pstdev(column)
        scales.append(scale if scale > 0.0 else 1.0)
    standardized = [
        [(value - mean) / scale for value, mean, scale in zip(example.features, means, scales, strict=True)]
        for example in examples
    ]
    return standardized, means, scales


def train_logistic_regression(
    examples: list[PairwiseExample],
    feature_names: list[str],
    *,
    epochs: int = 500,
    learning_rate: float = 0.05,
    l2: float = 0.01,
    seed: int = 0,
) -> LogisticModel:
    if not feature_names:
        raise ValueError("Need at least one selected feature to train a baseline")
    if len(examples) < 2:
        raise ValueError("Need at least two labeled examples to train a baseline")
    labels = {example.label_b_preferred for example in examples}
    if len(labels) < 2:
        raise ValueError("Need both preference classes to train a baseline")
    standardized, means, scales = standardize(examples)
    weights = [0.0 for _ in feature_names]
    bias = 0.0
    rng = random.Random(seed)
    order = list(range(len(examples)))
    for _ in range(epochs):
        rng.shuffle(order)
        for index in order:
            features = standardized[index]
            label = examples[index].label_b_preferred
            logit = bias + sum(weight * value for weight, value in zip(weights, features, strict=True))
            error = _sigmoid(logit) - label
            bias -= learning_rate * error
            for feature_index, value in enumerate(features):
                gradient = error * value + l2 * weights[feature_index]
                weights[feature_index] -= learning_rate * gradient
    return LogisticModel(feature_names=feature_names, means=means, scales=scales, weights=weights, bias=bias)


def predict_probability(model: LogisticModel, features: list[float]) -> float:
    standardized = [
        (value - mean) / scale for value, mean, scale in zip(features, model.means, model.scales, strict=True)
    ]
    logit = model.bias + sum(weight * value for weight, value in zip(model.weights, standardized, strict=True))
    return _sigmoid(logit)


def evaluate_model(model: LogisticModel, examples: list[PairwiseExample]) -> dict[str, Any]:
    if not examples:
        return {"examples": 0, "accuracy": None, "log_loss": None}
    correct = 0
    losses = []
    for example in examples:
        probability = min(max(predict_probability(model, example.features), 1e-8), 1.0 - 1e-8)
        prediction = int(probability >= 0.5)
        correct += int(prediction == example.label_b_preferred)
        label = example.label_b_preferred
        losses.append(-(label * math.log(probability) + (1 - label) * math.log(1.0 - probability)))
    return {"examples": len(examples), "accuracy": correct / len(examples), "log_loss": statistics.mean(losses)}


def load_prompt_fold_map(split_plan_path: Path) -> dict[str, int]:
    try:
        report = json.loads(split_plan_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"split plan not found: {split_plan_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid split plan JSON: {split_plan_path}: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"split plan must be a JSON object: {split_plan_path}")
    if report.get("split_plan_ready") is not True:
        raise ValueError(f"split plan is not ready: {split_plan_path}")
    rows = report.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"split plan has no rows: {split_plan_path}")
    prompt_to_fold: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"split plan row is not an object: {split_plan_path}")
        prompt_id = str(row.get("prompt_id", "")).strip()
        if not prompt_id:
            raise ValueError(f"split plan row is missing prompt_id: {split_plan_path}")
        try:
            fold = int(row["fold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"split plan row has invalid fold for prompt_id={prompt_id}: {split_plan_path}") from exc
        if prompt_id in prompt_to_fold and prompt_to_fold[prompt_id] != fold:
            raise ValueError(f"split plan assigns prompt_id={prompt_id} to multiple folds")
        prompt_to_fold[prompt_id] = fold
    if len(set(prompt_to_fold.values())) < 2:
        raise ValueError(f"split plan must contain at least two folds: {split_plan_path}")
    return prompt_to_fold


def prompt_group_folds(
    examples: list[PairwiseExample], folds: int, seed: int = 0, prompt_to_fold: dict[str, int] | None = None
) -> list[tuple[list[int], list[int]]]:
    prompts = sorted({example.prompt_id for example in examples})
    if prompt_to_fold is not None:
        missing_prompts = sorted(prompt for prompt in prompts if prompt not in prompt_to_fold)
        if missing_prompts:
            preview = ", ".join(missing_prompts[:5])
            raise ValueError(f"split plan missing prompt_id values: {preview}")
        fold_ids = sorted({int(prompt_to_fold[prompt]) for prompt in prompts})
        if len(fold_ids) < 2:
            raise ValueError("split plan must assign examples to at least two folds")
        splits = []
        for fold in fold_ids:
            train = []
            test = []
            for index, example in enumerate(examples):
                if int(prompt_to_fold[example.prompt_id]) == fold:
                    test.append(index)
                else:
                    train.append(index)
            splits.append((train, test))
        return splits
    rng = random.Random(seed)
    rng.shuffle(prompts)
    fold_count = max(2, min(folds, len(prompts)))
    prompt_to_fold = {prompt: index % fold_count for index, prompt in enumerate(prompts)}
    splits = []
    for fold in range(fold_count):
        train = []
        test = []
        for index, example in enumerate(examples):
            if prompt_to_fold[example.prompt_id] == fold:
                test.append(index)
            else:
                train.append(index)
        splits.append((train, test))
    return splits


def cross_validate_logistic_regression(
    examples: list[PairwiseExample],
    feature_names: list[str],
    *,
    folds: int = 5,
    epochs: int = 500,
    learning_rate: float = 0.05,
    l2: float = 0.01,
    seed: int = 0,
    prompt_to_fold: dict[str, int] | None = None,
) -> dict[str, Any]:
    if len(examples) < 4:
        raise ValueError("Need at least four examples for prompt-group cross-validation")
    fold_metrics = []
    for train_indices, test_indices in prompt_group_folds(examples, folds=folds, seed=seed, prompt_to_fold=prompt_to_fold):
        train_examples = [examples[index] for index in train_indices]
        test_examples = [examples[index] for index in test_indices]
        if len({example.label_b_preferred for example in train_examples}) < 2:
            continue
        model = train_logistic_regression(
            train_examples,
            feature_names,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
        )
        fold_metrics.append(evaluate_model(model, test_examples))
    if not fold_metrics:
        raise ValueError("No cross-validation fold had both preference classes in training data")
    return {
        "folds": len(fold_metrics),
        "fold_source": "predefined_prompt_folds" if prompt_to_fold is not None else "seeded_prompt_shuffle",
        "mean_accuracy": statistics.mean(metric["accuracy"] for metric in fold_metrics),
        "mean_log_loss": statistics.mean(metric["log_loss"] for metric in fold_metrics),
        "fold_metrics": fold_metrics,
    }
