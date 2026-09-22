from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from full_song_eval.annotation_reliability import summarize_annotation_reliability
from full_song_eval.jsonl import read_jsonl


def audit_batch_annotation_reliability(
    *,
    process_report_path: Path,
    output_json: Path,
    output_md: Path,
    min_overlap_tasks: int = 1,
    min_preference_pairs: int = 1,
) -> dict[str, Any]:
    process_report = _load_json(process_report_path)
    rows = [row for row in process_report.get("assignments", []) if isinstance(row, dict)]
    processed_rows = [row for row in rows if row.get("status") == "processed"]
    records, annotation_files, errors = _load_processed_records(processed_rows)
    reliability = summarize_annotation_reliability(records) if records else _empty_reliability_summary()
    report = _build_report(
        process_report=process_report,
        process_report_path=process_report_path,
        records=records,
        annotation_files=annotation_files,
        errors=errors,
        reliability=reliability,
        min_overlap_tasks=min_overlap_tasks,
        min_preference_pairs=min_preference_pairs,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(format_annotation_reliability_audit(report), encoding="utf-8")
    return report


def format_annotation_reliability_audit(report: dict[str, Any]) -> str:
    reliability = report.get("reliability", {}) if isinstance(report.get("reliability", {}), dict) else {}
    preference = reliability.get("preference", {}) if isinstance(reliability.get("preference", {}), dict) else {}
    winner = reliability.get("winner", {}) if isinstance(reliability.get("winner", {}), dict) else {}
    preference_score = reliability.get("preference_score", {}) if isinstance(reliability.get("preference_score", {}), dict) else {}
    lines = [
        "# Batch 01 Annotation Reliability Audit",
        "",
        "## Verdict",
        "",
        f"- Status: `{report.get('status', 'unknown')}`",
        f"- Reliability ready: `{str(bool(report.get('reliability_ready', False))).lower()}`",
        f"- Process report status: `{report.get('process_status', 'missing')}`",
        f"- Known blockers: {_inline_list(report.get('known_blockers', []))}",
        "",
        "## Batch Coverage",
        "",
        f"- Assignments total: {report.get('assignments_total', 0)}",
        f"- Processed assignments: {report.get('processed_assignments', 0)}",
        f"- Pending assignments: {report.get('pending_assignments', 0)}",
        f"- Invalid assignments: {report.get('invalid_assignments', 0)}",
        f"- Annotation records loaded: {report.get('annotation_record_count', 0)}",
        f"- Annotation files loaded: {len(report.get('annotation_files', []) or [])}",
        "",
        "## Inter-Annotator Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Tasks | {_fmt(reliability.get('tasks'))} |",
        f"| Overlap tasks | {_fmt(reliability.get('overlap_tasks'))} |",
        f"| Annotator pairs | {_fmt(reliability.get('annotator_pairs'))} |",
        f"| Unique annotators | {_fmt(reliability.get('unique_annotators'))} |",
        f"| Preference exact agreement | {_fmt(preference.get('exact_agreement'))} |",
        f"| Preference kappa | {_fmt(preference.get('kappa'))} |",
        f"| Winner exact agreement | {_fmt(winner.get('exact_agreement'))} |",
        f"| Winner kappa | {_fmt(winner.get('kappa'))} |",
        f"| Preference score mean absolute difference | {_fmt(preference_score.get('mean_absolute_difference'))} |",
        "",
        "## Gate Thresholds",
        "",
        f"- Minimum overlap tasks: {report.get('min_overlap_tasks', 0)}",
        f"- Minimum preference pairs: {report.get('min_preference_pairs', 0)}",
        "",
        "## Input Files",
        "",
        f"- Process report: `{report.get('process_report', '')}`",
    ]
    for path in report.get("annotation_files", []) or []:
        lines.append(f"- Annotation JSONL: `{path}`")
    if report.get("errors"):
        lines.extend(["", "## Errors", ""])
        for error in report.get("errors", []) or []:
            lines.append(f"- {error}")
    lines.append("")
    return "\n".join(lines)


def _build_report(
    *,
    process_report: dict[str, Any],
    process_report_path: Path,
    records: list[dict[str, Any]],
    annotation_files: list[str],
    errors: list[str],
    reliability: dict[str, Any],
    min_overlap_tasks: int,
    min_preference_pairs: int,
) -> dict[str, Any]:
    process_status = str(process_report.get("status", "missing")) if process_report else "missing"
    blocker_status = _blocker_status(
        process_status=process_status,
        records=records,
        errors=errors,
        reliability=reliability,
        min_overlap_tasks=min_overlap_tasks,
        min_preference_pairs=min_preference_pairs,
    )
    preference = reliability.get("preference", {}) if isinstance(reliability.get("preference", {}), dict) else {}
    winner = reliability.get("winner", {}) if isinstance(reliability.get("winner", {}), dict) else {}
    preference_score = reliability.get("preference_score", {}) if isinstance(reliability.get("preference_score", {}), dict) else {}
    report = {
        "status": blocker_status["status"],
        "reliability_ready": blocker_status["status"] == "pass",
        "known_blockers": blocker_status["known_blockers"],
        "process_report": str(process_report_path),
        "process_status": process_status,
        "assignments_total": int(process_report.get("assignments_total", 0) or 0),
        "processed_assignments": int(process_report.get("processed_assignments", 0) or 0),
        "pending_assignments": int(process_report.get("pending_assignments", 0) or 0),
        "invalid_assignments": int(process_report.get("invalid_assignments", 0) or 0),
        "annotation_files": annotation_files,
        "annotation_record_count": len(records),
        "tasks": reliability.get("tasks", 0),
        "overlap_tasks": reliability.get("overlap_tasks", 0),
        "annotator_pairs": reliability.get("annotator_pairs", 0),
        "unique_annotators": reliability.get("unique_annotators", 0),
        "min_overlap_tasks": min_overlap_tasks,
        "min_preference_pairs": min_preference_pairs,
        "preference_exact_agreement": preference.get("exact_agreement"),
        "preference_kappa": preference.get("kappa"),
        "winner_exact_agreement": winner.get("exact_agreement"),
        "winner_kappa": winner.get("kappa"),
        "preference_score_mean_absolute_difference": preference_score.get("mean_absolute_difference"),
        "reliability": reliability,
        "errors": errors,
    }
    return report


def _blocker_status(
    *,
    process_status: str,
    records: list[dict[str, Any]],
    errors: list[str],
    reliability: dict[str, Any],
    min_overlap_tasks: int,
    min_preference_pairs: int,
) -> dict[str, Any]:
    if process_status == "missing":
        return {"status": "needs_returns", "known_blockers": ["annotation_returns", "human_annotations"]}
    if process_status == "fail" or errors:
        return {"status": "fail", "known_blockers": ["annotation_return_processing"]}
    if not records:
        return {"status": "needs_returns", "known_blockers": ["annotation_returns", "human_annotations"]}
    if int(reliability.get("overlap_tasks", 0) or 0) < min_overlap_tasks:
        return {"status": "needs_human_annotations", "known_blockers": ["annotation_overlap", "human_annotations"]}
    if int(reliability.get("annotator_pairs", 0) or 0) < min_preference_pairs:
        return {"status": "needs_human_annotations", "known_blockers": ["annotation_reliability", "human_annotations"]}
    if reliability.get("status") != "pass":
        return {"status": "needs_human_annotations", "known_blockers": ["annotation_reliability", "human_annotations"]}
    return {"status": "pass", "known_blockers": []}


def _load_processed_records(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    records: list[dict[str, Any]] = []
    annotation_files: list[str] = []
    errors: list[str] = []
    for row in rows:
        output_jsonl = str(row.get("output_jsonl", ""))
        if not output_jsonl:
            errors.append(f"processed assignment {row.get('assignment_id', 'unknown')} is missing output_jsonl")
            continue
        path = Path(output_jsonl)
        if not path.exists():
            errors.append(f"processed assignment {row.get('assignment_id', 'unknown')} output JSONL is missing: {path}")
            continue
        try:
            loaded = read_jsonl(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        records.extend(loaded)
        annotation_files.append(str(path))
    return records, annotation_files, errors


def _empty_reliability_summary() -> dict[str, Any]:
    return {
        "annotations": 0,
        "unique_annotators": 0,
        "tasks": 0,
        "overlap_tasks": 0,
        "annotator_pairs": 0,
        "annotations_per_task": {},
        "preference": {"pairs": 0, "exact_agreement": None, "chance_expected_agreement": None, "kappa": None, "label_counts": {}},
        "winner": {"pairs": 0, "exact_agreement": None, "chance_expected_agreement": None, "kappa": None, "label_counts": {}},
        "preference_score": {"pairs": 0, "mean_absolute_difference": None, "max_absolute_difference": None},
        "ratings": {},
        "status": "insufficient_overlap",
    }


def _inline_list(values: Any) -> str:
    if not values:
        return "none"
    if not isinstance(values, list):
        return f"`{values}`"
    return ", ".join(f"`{value}`" for value in values)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}
