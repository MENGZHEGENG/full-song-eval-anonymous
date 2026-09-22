from __future__ import annotations

import itertools
import statistics
from collections import Counter, defaultdict
from typing import Any

from full_song_eval.annotation_qc import annotation_task_id
from full_song_eval.annotation_summary import PREFERENCE_TO_SCORE, RATING_DIMENSIONS, preference_winner


def summarize_annotation_reliability(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = _records_by_task(records)
    overlapping_groups = {task_id: task_records for task_id, task_records in groups.items() if _unique_annotators(task_records) >= 2}
    preference_pairs = _preference_pairs(overlapping_groups)
    rating_pairs = _rating_pairs(overlapping_groups)
    return {
        "annotations": len(records),
        "unique_annotators": len({str(record.get("annotator_id", "")) for record in records if record.get("annotator_id")}),
        "tasks": len(groups),
        "overlap_tasks": len(overlapping_groups),
        "annotator_pairs": len(preference_pairs),
        "annotations_per_task": dict(sorted(Counter(len(task_records) for task_records in groups.values()).items())),
        "preference": _categorical_summary(preference_pairs, lambda record: str(record.get("pairwise_preference", ""))),
        "winner": _categorical_summary(preference_pairs, lambda record: preference_winner(str(record.get("pairwise_preference", "")))),
        "preference_score": _score_summary(preference_pairs),
        "ratings": _rating_summary(rating_pairs),
        "status": "pass" if preference_pairs else "insufficient_overlap",
    }


def _records_by_task(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[annotation_task_id(record)].append(record)
    return dict(groups)


def _unique_annotators(records: list[dict[str, Any]]) -> int:
    return len({str(record.get("annotator_id", "")) for record in records})


def _preference_pairs(groups: dict[str, list[dict[str, Any]]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for task_records in groups.values():
        for first, second in itertools.combinations(task_records, 2):
            if str(first.get("annotator_id", "")) == str(second.get("annotator_id", "")):
                continue
            if first.get("pairwise_preference") in PREFERENCE_TO_SCORE and second.get("pairwise_preference") in PREFERENCE_TO_SCORE:
                pairs.append((first, second))
    return pairs


def _categorical_summary(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], label_fn: Any
) -> dict[str, Any]:
    labels: list[str] = []
    exact = []
    for first, second in pairs:
        first_label = str(label_fn(first))
        second_label = str(label_fn(second))
        labels.extend([first_label, second_label])
        exact.append(first_label == second_label)
    observed = _mean([1.0 if value else 0.0 for value in exact])
    expected = _expected_agreement(labels)
    return {
        "pairs": len(pairs),
        "exact_agreement": observed,
        "chance_expected_agreement": expected,
        "kappa": _kappa(observed, expected),
        "label_counts": dict(sorted(Counter(labels).items())),
    }


def _score_summary(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    absolute_differences = [
        abs(PREFERENCE_TO_SCORE[str(first["pairwise_preference"])] - PREFERENCE_TO_SCORE[str(second["pairwise_preference"])])
        for first, second in pairs
    ]
    return {
        "pairs": len(absolute_differences),
        "mean_absolute_difference": _mean(absolute_differences),
        "max_absolute_difference": max(absolute_differences) if absolute_differences else None,
    }


def _rating_pairs(groups: dict[str, list[dict[str, Any]]]) -> dict[str, list[tuple[int, int]]]:
    pairs: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for task_records in groups.values():
        for first, second in itertools.combinations(task_records, 2):
            if str(first.get("annotator_id", "")) == str(second.get("annotator_id", "")):
                continue
            for candidate_key in ["candidate_a", "candidate_b"]:
                for dimension in RATING_DIMENSIONS:
                    first_value = _rating(first, candidate_key, dimension)
                    second_value = _rating(second, candidate_key, dimension)
                    if first_value is not None and second_value is not None:
                        pairs[dimension].append((first_value, second_value))
    return dict(pairs)


def _rating(record: dict[str, Any], candidate_key: str, dimension: str) -> int | None:
    value = record.get("ratings", {}).get(candidate_key, {}).get(dimension)
    if isinstance(value, int):
        return value
    return None


def _rating_summary(pairs_by_dimension: dict[str, list[tuple[int, int]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dimension in RATING_DIMENSIONS:
        pairs = pairs_by_dimension.get(dimension, [])
        absolute_differences = [abs(first - second) for first, second in pairs]
        exact = [first == second for first, second in pairs]
        summary[dimension] = {
            "pairs": len(pairs),
            "exact_agreement": _mean([1.0 if value else 0.0 for value in exact]),
            "mean_absolute_difference": _mean([float(value) for value in absolute_differences]),
            "max_absolute_difference": max(absolute_differences) if absolute_differences else None,
        }
    return summary


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _expected_agreement(labels: list[str]) -> float | None:
    if not labels:
        return None
    counts = Counter(labels)
    total = sum(counts.values())
    return sum((count / total) ** 2 for count in counts.values())


def _kappa(observed: float | None, expected: float | None) -> float | None:
    if observed is None or expected is None or expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)
