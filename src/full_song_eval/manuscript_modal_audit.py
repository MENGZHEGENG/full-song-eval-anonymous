from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_TEXT_PATHS = (
    Path("paper/main.tex"),
    Path("paper/appendix_tables.tex"),
    Path("docs/manuscript_completion_matrix.md"),
)
MODAL_RE = re.compile(r"\b(cannot|can|should)\b", re.IGNORECASE)
SUPPORT_MARKERS = (
    "ablations",
    "annotation",
    "annotations",
    "asr",
    "automatic",
    "boundary",
    "caption",
    "claim",
    "code",
    "diagnostic",
    "evidence",
    "feature",
    "gate",
    "held-out",
    "human",
    "intelligibility",
    "label",
    "labels",
    "metadata",
    "metrics",
    "pYIN",
    "pipeline",
    "preference",
    "processing errors",
    "prompts",
    "proxy",
    "public",
    "reproducibly",
    "results",
    "rubric",
    "smoke",
    "source separation",
    "uncertainty",
)
FAIL_STATUSES = {"needs_precise_rewrite", "unsupported_capability"}


def audit_manuscript_modals(*, root: Path, text_paths: list[Path] | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing_paths: list[str] = []
    paths = text_paths or list(DEFAULT_TEXT_PATHS)
    for relative_path in paths:
        path = root / relative_path
        if not path.exists():
            missing_paths.append(str(relative_path))
            continue
        rows.extend(_modal_rows(path, relative_path))

    findings = [row for row in rows if row["status"] in FAIL_STATUSES]
    status_counts: dict[str, int] = {}
    modal_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        modal_counts[row["modal"]] = modal_counts.get(row["modal"], 0) + 1

    return {
        "status": "pass" if not findings and not missing_paths else "fail",
        "checked_paths": [str(path) for path in paths],
        "missing_paths": missing_paths,
        "modal_count": len(rows),
        "modal_counts": dict(sorted(modal_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "review_rows": rows,
        "findings": findings,
    }


def write_manuscript_modal_audit(*, root: Path, output_json: Path, output_md: Path) -> dict[str, Any]:
    report = audit_manuscript_modals(root=root)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(report), encoding="utf-8")
    return report


def _modal_rows(path: Path, relative_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = _strip_latex_comments(path.read_text(encoding="utf-8"))
    for match in MODAL_RE.finditer(text):
        modal = match.group(1).lower()
        sentence = _sentence_window(text, match.start()).strip()
        status, rationale = _classify_modal(modal, sentence)
        rows.append(
            {
                "path": str(relative_path),
                "line": _line_number(text, match.start()),
                "modal": modal,
                "status": status,
                "rationale": rationale,
                "sentence": _normalize_space(sentence),
            }
        )
    return rows


def _classify_modal(modal: str, sentence: str) -> tuple[str, str]:
    lowered = sentence.lower()
    if modal == "should":
        return "needs_precise_rewrite", "replace recommendation-style wording with an evidence-backed requirement, condition, or claim boundary"
    if modal == "cannot":
        return "negative_boundary", "states a blocked or unsupported claim boundary"
    if "?" in sentence or "\\item[rq" in lowered:
        return "research_question", "appears inside a research question rather than a result claim"
    if any(marker.lower() in lowered for marker in SUPPORT_MARKERS):
        return "supported_capability", "capability wording is tied to a concrete evidence source, method component, or claim boundary"
    return "unsupported_capability", "capability wording lacks an obvious evidence marker in the same sentence"


def _sentence_window(text: str, offset: int) -> str:
    left_candidates = [text.rfind(delimiter, 0, offset) for delimiter in (".", "?", "!", "\n\n")]
    left = max(left_candidates) + 1
    right_candidates = [index for delimiter in (".", "?", "!", "\n\n") if (index := text.find(delimiter, offset)) != -1]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return text[left:right]


def _strip_latex_comments(text: str) -> str:
    return "\n".join(_strip_latex_comment_line(line) for line in text.split("\n"))


def _strip_latex_comment_line(line: str) -> str:
    for index, char in enumerate(line):
        if char == "%" and _is_comment_marker(line, index):
            return line[:index] + " " * (len(line) - index)
    return line


def _is_comment_marker(line: str, index: int) -> bool:
    backslash_count = 0
    cursor = index - 1
    while cursor >= 0 and line[cursor] == "\\":
        backslash_count += 1
        cursor -= 1
    return backslash_count % 2 == 0


def _line_number(text: str, offset: int) -> int:
    return text[:offset].count("\n") + 1


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Manuscript Modal Audit",
        "",
        "## Status",
        "",
        f"- Status: `{report.get('status', 'unknown')}`",
        f"- Checked paths: {len(report.get('checked_paths', []))}",
        f"- Modal uses: {report.get('modal_count', 0)}",
        f"- Modal counts: `{json.dumps(report.get('modal_counts', {}), sort_keys=True)}`",
        f"- Status counts: `{json.dumps(report.get('status_counts', {}), sort_keys=True)}`",
        "",
        "## Review Rows",
        "",
        "| Path | Line | Modal | Status | Rationale | Sentence |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for row in report.get("review_rows", []) or []:
        lines.append(
            "| `{}` | {} | `{}` | `{}` | {} | {} |".format(
                row.get("path", ""),
                row.get("line", 0),
                row.get("modal", ""),
                row.get("status", ""),
                _cell(row.get("rationale", "")),
                _cell(row.get("sentence", "")),
            )
        )
    if report.get("findings"):
        lines.extend(["", "## Findings", ""])
        for row in report.get("findings", []) or []:
            lines.append(
                "- `{}` line {}: `{}` uses `{}` ({})".format(
                    row.get("path", ""),
                    row.get("line", 0),
                    row.get("modal", ""),
                    row.get("status", ""),
                    row.get("rationale", ""),
                )
            )
    return "\n".join(lines) + "\n"


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
