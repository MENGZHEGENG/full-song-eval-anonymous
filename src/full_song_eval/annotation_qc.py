from __future__ import annotations

from collections import Counter
from typing import Any

from full_song_eval.annotation_summary import PREFERENCE_TO_SCORE, RATING_DIMENSIONS

VALID_PREFERENCES = set(PREFERENCE_TO_SCORE) | {"invalid"}


def task_id(record: dict[str, Any]) -> str:
    value = str(record.get("task_id", ""))
    if not value:
        raise ValueError("task is missing task_id")
    return value


def annotation_task_id(record: dict[str, Any]) -> str:
    annotation_id = str(record.get("annotation_id", ""))
    annotator_id = str(record.get("annotator_id", ""))
    if annotator_id and annotation_id.startswith(f"{annotator_id}_"):
        return annotation_id[len(annotator_id) + 1 :]
    return str(record.get("task_id", "")) or annotation_id


def task_candidate_ids(task: dict[str, Any]) -> tuple[str, str]:
    return str(task["candidate_a"]["generation_id"]), str(task["candidate_b"]["generation_id"])


def annotation_candidate_ids(record: dict[str, Any]) -> tuple[str, str]:
    return str(record["candidate_a"]["generation_id"]), str(record["candidate_b"]["generation_id"])


def task_model_pair(task: dict[str, Any]) -> str:
    return f"{task['candidate_a']['model_id']} -> {task['candidate_b']['model_id']}"


def annotation_model_pair(record: dict[str, Any]) -> str:
    return f"{record['candidate_a']['model_id']} -> {record['candidate_b']['model_id']}"


def audit_annotations(tasks: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> dict[str, Any]:
    task_by_id = {task_id(task): task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("source packet contains duplicate task IDs")
    annotation_ids = [annotation_task_id(record) for record in annotations]
    duplicate_annotation_task_ids = sorted(task_id for task_id, count in Counter(annotation_ids).items() if count > 1)
    covered_task_ids = set(annotation_ids) & set(task_by_id)
    missing_task_ids = sorted(set(task_by_id) - covered_task_ids)
    unknown_task_ids = sorted(set(annotation_ids) - set(task_by_id))
    errors: list[dict[str, Any]] = []
    preference_counts: Counter[str] = Counter()
    model_pair_counts: Counter[str] = Counter(task_model_pair(task) for task in tasks)
    annotated_model_pair_counts: Counter[str] = Counter()
    for record in annotations:
        current_task_id = annotation_task_id(record)
        preference = str(record.get("pairwise_preference", ""))
        preference_counts[preference] += 1
        if preference not in VALID_PREFERENCES:
            errors.append({"task_id": current_task_id, "error": f"invalid preference {preference!r}"})
        task = task_by_id.get(current_task_id)
        if task is None:
            continue
        if annotation_candidate_ids(record) != task_candidate_ids(task):
            errors.append({"task_id": current_task_id, "error": "candidate IDs do not match source task"})
        annotated_model_pair_counts[annotation_model_pair(record)] += 1
        _append_rating_errors(record, current_task_id, errors)
    return {
        "source_tasks": len(tasks),
        "annotations": len(annotations),
        "covered_tasks": len(covered_task_ids),
        "missing_tasks": len(missing_task_ids),
        "unknown_tasks": len(unknown_task_ids),
        "duplicate_annotation_task_ids": duplicate_annotation_task_ids,
        "missing_task_ids": missing_task_ids,
        "unknown_task_ids": unknown_task_ids,
        "preference_counts": dict(sorted(preference_counts.items())),
        "source_model_pair_counts": dict(sorted(model_pair_counts.items())),
        "annotated_model_pair_counts": dict(sorted(annotated_model_pair_counts.items())),
        "errors": errors,
        "status": "pass" if not errors and not duplicate_annotation_task_ids and not unknown_task_ids else "fail",
    }


def _append_rating_errors(record: dict[str, Any], current_task_id: str, errors: list[dict[str, Any]]) -> None:
    ratings = record.get("ratings", {})
    for candidate_key in ["candidate_a", "candidate_b"]:
        candidate_ratings = ratings.get(candidate_key, {})
        for dimension in RATING_DIMENSIONS:
            value = candidate_ratings.get(dimension)
            if not isinstance(value, int) or value < 1 or value > 5:
                errors.append(
                    {
                        "task_id": current_task_id,
                        "error": f"{candidate_key}.{dimension} rating must be an integer from 1 to 5",
                    }
                )
