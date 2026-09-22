from __future__ import annotations

import random
from collections import Counter
from typing import Any


def task_id(task: dict[str, Any]) -> str:
    value = str(task.get("task_id", ""))
    if not value:
        raise ValueError("Annotation task is missing task_id")
    return value


def task_language(task: dict[str, Any]) -> str:
    return str(task.get("metadata", {}).get("language") or "unknown")


def task_genre(task: dict[str, Any]) -> str:
    return str(task.get("metadata", {}).get("genre") or "unknown")


def proxy_score(delta_row: dict[str, Any]) -> float:
    value = delta_row.get("proxy_contrast_score")
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def index_deltas_by_task_id(delta_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in delta_rows:
        current_task_id = str(row.get("task_id", ""))
        if not current_task_id:
            raise ValueError("Pairwise delta row is missing task_id")
        if current_task_id in indexed:
            raise ValueError(f"Duplicate task_id in pairwise deltas: {current_task_id}")
        indexed[current_task_id] = row
    return indexed


def sort_by_proxy_contrast(tasks: list[dict[str, Any]], deltas_by_task_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        tasks,
        key=lambda task: (-proxy_score(deltas_by_task_id.get(task_id(task), {})), task_id(task)),
    )


def select_high_contrast(
    tasks: list[dict[str, Any]], deltas_by_task_id: dict[str, dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("count must be non-negative")
    return sort_by_proxy_contrast(tasks, deltas_by_task_id)[:count]


def select_balanced(
    tasks: list[dict[str, Any]],
    deltas_by_task_id: dict[str, dict[str, Any]],
    count: int,
    *,
    seed: int = 0,
    initial_tasks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("count must be non-negative")
    selected: list[dict[str, Any]] = []
    selected_ids = {task_id(task) for task in initial_tasks or []}
    language_counts = Counter(task_language(task) for task in initial_tasks or [])
    genre_counts = Counter(task_genre(task) for task in initial_tasks or [])
    rng = random.Random(seed)
    remaining = [task for task in tasks if task_id(task) not in selected_ids]
    rng.shuffle(remaining)
    while remaining and len(selected) < count:
        remaining.sort(
            key=lambda task: (
                language_counts[task_language(task)],
                genre_counts[task_genre(task)],
                -proxy_score(deltas_by_task_id.get(task_id(task), {})),
                task_id(task),
            )
        )
        task = remaining.pop(0)
        selected.append(task)
        selected_ids.add(task_id(task))
        language_counts[task_language(task)] += 1
        genre_counts[task_genre(task)] += 1
    return selected


def plan_annotation_batch(
    tasks: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    *,
    high_contrast_count: int,
    balanced_count: int,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if high_contrast_count < 0 or balanced_count < 0:
        raise ValueError("batch counts must be non-negative")
    deltas_by_task_id = index_deltas_by_task_id(delta_rows)
    high_contrast = select_high_contrast(tasks, deltas_by_task_id, high_contrast_count)
    balanced = select_balanced(
        tasks,
        deltas_by_task_id,
        balanced_count,
        seed=seed,
        initial_tasks=high_contrast,
    )
    selected = [*high_contrast, *balanced]
    selected_ids = [task_id(task) for task in selected]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Annotation batch selection produced duplicate task IDs")
    summary = {
        "tasks": len(selected),
        "high_contrast_tasks": len(high_contrast),
        "balanced_tasks": len(balanced),
        "languages": dict(sorted(Counter(task_language(task) for task in selected).items())),
        "genres": dict(sorted(Counter(task_genre(task) for task in selected).items())),
        "task_ids": selected_ids,
        "high_contrast_task_ids": [task_id(task) for task in high_contrast],
        "balanced_task_ids": [task_id(task) for task in balanced],
    }
    return selected, summary
