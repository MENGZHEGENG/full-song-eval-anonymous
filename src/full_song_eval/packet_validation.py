from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def parse_expectation(text: str) -> tuple[str, int]:
    if "=" not in text:
        raise ValueError(f"Expectation must use key=count format: {text!r}")
    key, count_text = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Expectation key cannot be empty: {text!r}")
    try:
        count = int(count_text)
    except ValueError as exc:
        raise ValueError(f"Expectation count must be an integer: {text!r}") from exc
    if count < 0:
        raise ValueError(f"Expectation count must be non-negative: {text!r}")
    return key, count


def parse_expectations(items: list[str] | None) -> dict[str, int]:
    expectations: dict[str, int] = {}
    for item in items or []:
        key, count = parse_expectation(item)
        if key in expectations:
            raise ValueError(f"Duplicate expectation key: {key!r}")
        expectations[key] = count
    return expectations


def validate_packet(
    tasks: list[dict[str, Any]],
    *,
    packet_dir: Path | None = None,
    require_audio: bool = False,
    expected_tasks: int | None = None,
    expected_candidate_a_source_roles: dict[str, int] | None = None,
    expected_candidate_a_source_indices: dict[str, int] | None = None,
    expected_languages: dict[str, int] | None = None,
    expected_genres: dict[str, int] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "status": "pass",
        "tasks": len(tasks),
        "expected_tasks": expected_tasks,
        "duplicate_task_ids": _duplicates(str(task.get("task_id", "")) for task in tasks),
        "candidate_a_source_roles": _counter_from_tasks(tasks, "source_generator_role"),
        "candidate_a_source_indices": _counter_from_tasks(tasks, "source_candidate_index"),
        "languages": _counter_from_tasks(tasks, "language"),
        "genres": _counter_from_tasks(tasks, "genre"),
        "missing_audio": 0,
        "missing_audio_paths": [],
        "expectation_errors": [],
    }

    errors: list[str] = []
    if expected_tasks is not None and len(tasks) != expected_tasks:
        errors.append(f"expected {expected_tasks} tasks, found {len(tasks)}")
    if summary["duplicate_task_ids"]:
        errors.append("duplicate task IDs found")

    for label, observed, expected in [
        ("candidate_a_source_roles", summary["candidate_a_source_roles"], expected_candidate_a_source_roles),
        ("candidate_a_source_indices", summary["candidate_a_source_indices"], expected_candidate_a_source_indices),
        ("languages", summary["languages"], expected_languages),
        ("genres", summary["genres"], expected_genres),
    ]:
        if expected is None or expected == {}:
            continue
        if observed != expected:
            errors.append(f"{label} mismatch: expected {expected}, found {observed}")

    if require_audio:
        if packet_dir is None:
            raise ValueError("packet_dir is required when require_audio=True")
        missing_audio = _missing_audio_paths(tasks, packet_dir)
        summary["missing_audio"] = len(missing_audio)
        summary["missing_audio_paths"] = missing_audio[:20]
        if missing_audio:
            errors.append(f"{len(missing_audio)} local audio references are missing")

    summary["expectation_errors"] = errors
    if errors:
        summary["status"] = "fail"
    return summary


def _counter_from_tasks(tasks: list[dict[str, Any]], metadata_key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for task in tasks:
        value = task.get("metadata", {}).get(metadata_key)
        if value is None:
            value = "missing"
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _duplicates(values: Any) -> list[str]:
    counts = Counter(value for value in values if value)
    return sorted(value for value, count in counts.items() if count > 1)


def _missing_audio_paths(tasks: list[dict[str, Any]], packet_dir: Path) -> list[str]:
    missing: list[str] = []
    for task in tasks:
        for side in ("candidate", "candidate_a", "candidate_b"):
            candidate = task.get(side)
            if not isinstance(candidate, dict):
                continue
            path = _candidate_audio_path(candidate, packet_dir)
            if path is not None and not path.exists():
                missing.append(str(path))
    return missing


def _candidate_audio_path(candidate: dict[str, Any], packet_dir: Path) -> Path | None:
    local_audio_uri = candidate.get("local_audio_uri")
    if local_audio_uri:
        return (packet_dir / str(local_audio_uri)).resolve()
    audio_uri = candidate.get("audio_uri")
    if audio_uri:
        return Path(str(audio_uri)).resolve()
    return None
