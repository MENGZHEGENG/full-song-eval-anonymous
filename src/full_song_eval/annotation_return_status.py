from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from full_song_eval.annotation_qc import audit_annotations
from full_song_eval.jsonl import read_jsonl


def parse_source_task_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Source task spec must be packet_id=annotation_tasks, got: {value}")
    packet_id, path = value.split("=", 1)
    packet_id = packet_id.strip()
    if not packet_id:
        raise ValueError(f"Source task spec has empty packet_id: {value}")
    return packet_id, Path(path)


def audit_annotation_returns(
    assignment_plan_path: Path,
    *,
    source_tasks_by_packet: dict[str, Path] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    plan = _load_json(assignment_plan_path)
    assignments = plan.get("assignments", []) if isinstance(plan, dict) else []
    if not isinstance(assignments, list):
        assignments = []
    source_tasks_by_packet = source_tasks_by_packet or {}
    rows = []
    packet_counts: dict[str, dict[str, int]] = {}
    annotator_counts: dict[str, dict[str, int]] = {}
    for row in assignments:
        if not isinstance(row, dict):
            continue
        assignment = _audit_assignment(row, source_tasks_by_packet, require_complete=require_complete)
        rows.append(assignment)
        _increment(packet_counts, assignment["packet_id"], assignment["status"])
        _increment(annotator_counts, assignment["annotator_id"], assignment["status"])
    returned = sum(row["status"] in {"complete", "incomplete", "invalid"} for row in rows)
    complete = sum(row["status"] == "complete" for row in rows)
    invalid = sum(row["status"] == "invalid" for row in rows)
    pending = sum(row["status"] == "pending" for row in rows)
    if invalid:
        status = "fail"
    elif pending:
        status = "needs_returns"
    elif require_complete and any(row["status"] != "complete" for row in rows):
        status = "needs_completion"
    else:
        status = "pass"
    return {
        "status": status,
        "assignment_plan": str(assignment_plan_path),
        "assignments_total": len(rows),
        "returned_assignments": returned,
        "complete_assignments": complete,
        "pending_assignments": pending,
        "invalid_assignments": invalid,
        "require_complete": require_complete,
        "packet_status_counts": packet_counts,
        "annotator_status_counts": annotator_counts,
        "assignments": rows,
    }


def write_annotation_return_status(
    assignment_plan_path: Path,
    output_json: Path,
    *,
    source_tasks_by_packet: dict[str, Path] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    report = audit_annotation_returns(
        assignment_plan_path,
        source_tasks_by_packet=source_tasks_by_packet,
        require_complete=require_complete,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _audit_assignment(row: dict[str, Any], source_tasks_by_packet: dict[str, Path], *, require_complete: bool) -> dict[str, Any]:
    packet_id = str(row.get("packet_id", ""))
    annotator_id = str(row.get("annotator_id", ""))
    expected_path = Path(str(row.get("returned_annotation_jsonl") or row.get("expected_annotation_jsonl") or ""))
    result: dict[str, Any] = {
        "assignment_id": str(row.get("assignment_id", "")),
        "annotator_id": annotator_id,
        "packet_id": packet_id,
        "expected_annotation_jsonl": str(expected_path),
        "status": "pending",
        "qc": None,
    }
    if not str(expected_path) or not expected_path.exists():
        return result
    source_tasks = source_tasks_by_packet.get(packet_id)
    if source_tasks is None or not source_tasks.exists():
        result["status"] = "returned"
        result["qc"] = {"status": "not_audited", "reason": "missing source annotation_tasks path"}
        return result
    try:
        qc = audit_annotations(read_jsonl(source_tasks), read_jsonl(expected_path))
    except (OSError, ValueError, KeyError) as error:
        result["status"] = "invalid"
        result["qc"] = {"status": "fail", "error": str(error)}
        return result
    result["qc"] = qc
    if qc["status"] != "pass":
        result["status"] = "invalid"
    elif require_complete and qc.get("missing_tasks", 0):
        result["status"] = "incomplete"
    else:
        result["status"] = "complete"
    return result


def _increment(counts: dict[str, dict[str, int]], key: str, status: str) -> None:
    bucket = counts.setdefault(key, {"complete": 0, "incomplete": 0, "invalid": 0, "pending": 0, "returned": 0})
    bucket[status] = bucket.get(status, 0) + 1


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
