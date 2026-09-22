from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from full_song_eval.jsonl import read_jsonl

FORBIDDEN_SHEET_COLUMNS = {
    "generation_id",
    "model_id",
    "audio_uri",
    "candidate_a_generation_id",
    "candidate_a_model_id",
    "candidate_a_audio_uri",
    "candidate_b_generation_id",
    "candidate_b_model_id",
    "candidate_b_audio_uri",
}


def validate_blind_packet(packet_dir: Path) -> dict[str, Any]:
    tasks_path = packet_dir / "annotation_tasks.jsonl"
    sheet_path = packet_dir / "annotation_sheet.csv"
    key_path = packet_dir / "annotation_key.csv"
    html_path = packet_dir / "index.html"
    summary_path = packet_dir / "packet_summary.json"
    errors: list[str] = []
    for required_path in [tasks_path, sheet_path, key_path, html_path, summary_path]:
        if not required_path.exists():
            errors.append(f"missing required file: {required_path.name}")
    tasks = read_jsonl(tasks_path) if tasks_path.exists() else []
    sheet_columns = _sheet_columns(sheet_path) if sheet_path.exists() else []
    leaked_columns = sorted(set(sheet_columns) & FORBIDDEN_SHEET_COLUMNS)
    if leaked_columns:
        errors.append(f"annotation_sheet.csv exposes identity columns: {', '.join(leaked_columns)}")
    key_summary = _validate_key(key_path, tasks) if key_path.exists() else {"rows": 0, "errors": []}
    errors.extend(key_summary["errors"])
    html_leaks = _html_leaks(html_path, tasks) if html_path.exists() else []
    if html_leaks:
        errors.append(f"index.html exposes candidate identifiers: {', '.join(html_leaks[:10])}")
    summary = _summary(summary_path) if summary_path.exists() else {}
    if summary.get("blind") is not True:
        errors.append("packet_summary.json does not mark blind=true")
    return {
        "packet_dir": str(packet_dir),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "tasks": len(tasks),
        "sheet_columns": sheet_columns,
        "key_rows": key_summary["rows"],
        "html_leaks": html_leaks,
        "summary_blind": summary.get("blind"),
    }


def validate_blind_packets(packet_dirs: list[Path]) -> dict[str, Any]:
    packets = [validate_blind_packet(packet_dir) for packet_dir in packet_dirs]
    return {
        "status": "pass" if all(packet["status"] == "pass" for packet in packets) else "fail",
        "packets": packets,
        "packets_total": len(packets),
        "packets_failed": sum(packet["status"] != "pass" for packet in packets),
    }


def _sheet_columns(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def _validate_key(path: Path, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    expected = []
    for task in tasks:
        if task.get("task_type") == "pairwise_preference_rating":
            expected.extend([(task["task_id"], "candidate_a"), (task["task_id"], "candidate_b")])
        else:
            expected.append((task["task_id"], "candidate"))
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observed = [(row.get("task_id", ""), row.get("candidate_label", "")) for row in rows]
    errors = []
    if sorted(observed) != sorted(expected):
        errors.append("annotation_key.csv rows do not match annotation_tasks.jsonl candidates")
    required_columns = {"task_id", "candidate_label", "generation_id", "model_id", "audio_uri", "local_audio_uri"}
    if rows and not required_columns.issubset(rows[0]):
        missing = sorted(required_columns - set(rows[0]))
        errors.append(f"annotation_key.csv missing columns: {', '.join(missing)}")
    return {"rows": len(rows), "errors": errors}


def _html_leaks(path: Path, tasks: list[dict[str, Any]]) -> list[str]:
    html = path.read_text(encoding="utf-8")
    candidate_values = set()
    for task in tasks:
        candidates = [task.get("candidate", {})]
        if task.get("task_type") == "pairwise_preference_rating":
            candidates = [task.get("candidate_a", {}), task.get("candidate_b", {})]
        for candidate in candidates:
            for key in ["generation_id", "model_id", "audio_uri", "local_audio_uri"]:
                value = str(candidate.get(key, ""))
                if value:
                    candidate_values.add(value)
    return sorted(value for value in candidate_values if value in html)


def _summary(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
