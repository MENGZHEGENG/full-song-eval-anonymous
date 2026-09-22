from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from full_song_eval.musiccaps_hf_extended_sweep import (
    TERMINAL_RESULT_STATUSES,
    load_tasks,
    output_json_for_task,
    result_matches_task,
)

DEFAULT_TASKS_JSON = Path("configs/musiccaps_hf_top1296_min5_replication8_tasks.json")
DEFAULT_OUTPUT_DIR = Path("outputs/tables/musiccaps_top1296_min5_replication8_hf")
DEFAULT_OUTPUT_JSON = Path("outputs/tables/musiccaps_hf_top1296_min5_replication8_result_gate.json")
DEFAULT_OUTPUT_MD = Path("docs/musiccaps_hf_top1296_min5_replication8_result_gate.md")


def write_musiccaps_hf_result_gate(
    *,
    tasks_json: Path = DEFAULT_TASKS_JSON,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_md: Path = DEFAULT_OUTPUT_MD,
) -> dict[str, Any]:
    report = check_musiccaps_hf_result_gate(tasks_json=tasks_json, output_dir=output_dir)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(report), encoding="utf-8")
    return report


def check_musiccaps_hf_result_gate(*, tasks_json: Path, output_dir: Path) -> dict[str, Any]:
    tasks = load_tasks(tasks_json)
    expected_paths = {output_json_for_task(task, output_dir): task for task in tasks}
    expected_names = {path.name for path in expected_paths}
    passing_rows: list[dict[str, Any]] = []
    terminal_skip_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    unreadable_rows: list[dict[str, Any]] = []

    for path, task in expected_paths.items():
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            missing_rows.append(_task_row(task, path))
            continue
        except (json.JSONDecodeError, OSError) as error:
            unreadable_rows.append({**_task_row(task, path), "error": f"{type(error).__name__}: {error}"})
            continue
        if not isinstance(report, dict):
            unreadable_rows.append({**_task_row(task, path), "error": "result file must contain a JSON object"})
            continue
        status = str(report.get("status", "missing"))
        if not result_matches_task(path, task):
            mismatch_rows.append(
                {
                    **_task_row(task, path),
                    "status": status,
                    "found_model_id": report.get("model_id"),
                    "found_seed": report.get("seed"),
                    "found_top_label_count": report.get("top_label_count"),
                    "found_fold_count": report.get("fold_count"),
                    "found_min_label_count": report.get("min_label_count"),
                }
            )
        elif status == "pass":
            passing_rows.append(_task_row(task, path))
        elif status in TERMINAL_RESULT_STATUSES:
            terminal_skip_rows.append({**_task_row(task, path), "status": status})
        else:
            unresolved_rows.append({**_task_row(task, path), "status": status})

    unexpected_files = sorted(str(path) for path in output_dir.glob("*.json") if path.name not in expected_names)
    all_accounted = not (missing_rows or unresolved_rows or mismatch_rows or unreadable_rows or unexpected_files)
    if all_accounted and len(passing_rows) == len(tasks):
        status = "pass"
    elif all_accounted and len(passing_rows) + len(terminal_skip_rows) == len(tasks):
        status = "pass_with_backend_skips"
    else:
        status = "needs_results"

    blockers: list[str] = []
    if missing_rows:
        blockers.append("missing_result_files")
    if unresolved_rows:
        blockers.append("unresolved_result_statuses")
    if mismatch_rows:
        blockers.append("task_identity_or_setting_mismatch")
    if unreadable_rows:
        blockers.append("unreadable_result_files")
    if unexpected_files:
        blockers.append("unexpected_result_files")

    return {
        "status": status,
        "tasks_json": str(tasks_json),
        "output_dir": str(output_dir),
        "task_count": len(tasks),
        "passing_task_count": len(passing_rows),
        "terminal_skip_task_count": len(terminal_skip_rows),
        "missing_task_count": len(missing_rows),
        "unresolved_task_count": len(unresolved_rows),
        "mismatched_task_count": len(mismatch_rows),
        "unreadable_task_count": len(unreadable_rows),
        "unexpected_file_count": len(unexpected_files),
        "ready_for_final_summary": status in {"pass", "pass_with_backend_skips"},
        "known_blockers": blockers,
        "claim_scope": "MusicCaps public-label frozen-encoder result completeness only; not listener preference or perceptual-quality evidence.",
        "passing_rows": passing_rows,
        "terminal_skip_rows": terminal_skip_rows,
        "missing_rows": missing_rows[:50],
        "unresolved_rows": unresolved_rows[:50],
        "mismatch_rows": mismatch_rows[:50],
        "unreadable_rows": unreadable_rows[:50],
        "unexpected_files": unexpected_files[:50],
        "recommended_next_action": _recommended_next_action(status, blockers),
    }


def _task_row(task: Any, path: Path) -> dict[str, Any]:
    return {
        "index": task.index,
        "model_id": task.model_id,
        "seed": task.seed,
        "top_label_count": task.top_label_count,
        "fold_count": task.fold_count,
        "min_label_count": task.min_label_count,
        "path": str(path),
    }


def _recommended_next_action(status: str, blockers: list[str]) -> str:
    if status == "pass":
        return "Run the final sweep summary and family analysis, then refresh paper tables."
    if status == "pass_with_backend_skips":
        return "Run the final summary with backend skips disclosed, then refresh paper tables without listener claims."
    if "task_identity_or_setting_mismatch" in blockers:
        return "Inspect mismatched result files before dispatching retries or admitting summaries."
    if "unreadable_result_files" in blockers:
        return "Repair or remove unreadable result files, then rerun this gate."
    return "Continue dispatching missing tasks, then rerun this gate before final summary admission."


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MusicCaps HF Task Result Gate",
        "",
        f"- Status: `{report['status']}`",
        f"- Ready for final summary: `{str(report['ready_for_final_summary']).lower()}`",
        f"- Passing tasks: `{report['passing_task_count']}` / `{report['task_count']}`",
        f"- Terminal backend skips: `{report['terminal_skip_task_count']}`",
        f"- Missing tasks: `{report['missing_task_count']}`",
        f"- Unresolved tasks: `{report['unresolved_task_count']}`",
        f"- Mismatched tasks: `{report['mismatched_task_count']}`",
        f"- Unreadable tasks: `{report['unreadable_task_count']}`",
        f"- Unexpected files: `{report['unexpected_file_count']}`",
        f"- Claim scope: {report['claim_scope']}",
        f"- Next action: {report['recommended_next_action']}",
    ]
    if report["known_blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in report["known_blockers"])
    for label, key in [
        ("Missing Task Sample", "missing_rows"),
        ("Unresolved Task Sample", "unresolved_rows"),
        ("Mismatch Task Sample", "mismatch_rows"),
        ("Unreadable Task Sample", "unreadable_rows"),
    ]:
        if report[key]:
            lines.extend(["", f"## {label}", "", "| Index | Model | Seed | Status |", "|---:|---|---|---|"])
            for row in report[key][:20]:
                lines.append(f"| {row['index']} | `{row['model_id']}` | `{row['seed']}` | `{row.get('status', '-')}` |")
    return "\n".join(lines) + "\n"
