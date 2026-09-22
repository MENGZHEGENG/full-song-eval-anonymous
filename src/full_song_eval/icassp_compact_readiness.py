from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REQUIRED_SECTIONS = [
    "Motivation and Positioning",
    "Protocol and Evaluator",
    "Current Reproducible Pipeline",
    "ICASSP Submission Plan",
    "Limitations and Ethics",
    "Conclusion",
]
REQUIRED_TERMS = [
    r"\fullsongeval",
    r"\songdrame",
    r"\songchain",
    "human preference results are intentionally deferred",
    "not human preference or perceptual-quality results",
]
TODO_RE = re.compile(r"\\todo\s*\{|\b(?:TODO|FIXME|TBD)\b")
DRAFT_TODO_MACRO_RE = re.compile(r"\\(?:newcommand|renewcommand|providecommand)\s*\{?\\todo\b")
CITATION_RE = re.compile(r"\\cite(?:[tp])?\s*\{")
TABLE_RE = re.compile(r"\\begin\s*\{table\}")


def write_icassp_compact_readiness(
    output_json: Path,
    *,
    paper_path: Path,
    paper_claim_audit_path: Path | None = None,
    target_words: int = 3200,
    min_words: int = 900,
    min_citations: int = 8,
    min_tables: int = 1,
) -> dict[str, Any]:
    report = audit_icassp_compact_readiness(
        paper_path=paper_path,
        paper_claim_audit_path=paper_claim_audit_path,
        target_words=target_words,
        min_words=min_words,
        min_citations=min_citations,
        min_tables=min_tables,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def audit_icassp_compact_readiness(
    *,
    paper_path: Path,
    paper_claim_audit_path: Path | None = None,
    target_words: int = 3200,
    min_words: int = 900,
    min_citations: int = 8,
    min_tables: int = 1,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    text = paper_path.read_text(encoding="utf-8") if paper_path.exists() else ""
    if not paper_path.exists():
        findings.append({"severity": "error", "message": "missing compact manuscript file", "path": str(paper_path), "line": 0})
    sections = _sections(text)
    for required in REQUIRED_SECTIONS:
        if required not in sections:
            findings.append({"severity": "error", "message": f"missing compact section: {required}", "path": str(paper_path), "line": 0})
    body_text = _without_preamble(text)
    for line_number, line in enumerate(text.splitlines(), start=1):
        if DRAFT_TODO_MACRO_RE.search(line):
            findings.append({"severity": "error", "message": "draft-only todo macro definition", "path": str(paper_path), "line": line_number, "text": line.strip()})
    for line_number, line in enumerate(body_text.splitlines(), start=_body_start_line(text)):
        if TODO_RE.search(line):
            findings.append({"severity": "error", "message": "active TODO/FIXME/TBD marker", "path": str(paper_path), "line": line_number, "text": line.strip()})
    word_count = _word_count(body_text)
    citation_count = len(CITATION_RE.findall(text))
    table_count = len(TABLE_RE.findall(text))
    if word_count < min_words:
        findings.append({"severity": "error", "message": f"word count below compact minimum {min_words}", "path": str(paper_path), "line": 0, "value": word_count})
    if word_count > target_words:
        findings.append({"severity": "error", "message": f"word count above compact target {target_words}", "path": str(paper_path), "line": 0, "value": word_count})
    if citation_count < min_citations:
        findings.append({"severity": "error", "message": f"citation count below compact minimum {min_citations}", "path": str(paper_path), "line": 0, "value": citation_count})
    if table_count < min_tables:
        findings.append({"severity": "error", "message": f"table count below compact minimum {min_tables}", "path": str(paper_path), "line": 0, "value": table_count})
    for term in REQUIRED_TERMS:
        if term.lower() not in text.lower():
            findings.append({"severity": "error", "message": f"missing compact term/guard: {term}", "path": str(paper_path), "line": 0})
    claim_audit = _load_json(paper_claim_audit_path) if paper_claim_audit_path else {}
    claim_status = str(claim_audit.get("status", "missing")) if paper_claim_audit_path else "not_checked"
    audited_paths = claim_audit.get("paper_paths", []) if isinstance(claim_audit.get("paper_paths", []), list) else []
    if paper_claim_audit_path and claim_status != "pass":
        findings.append({"severity": "error", "message": "paper claim audit is not passing", "path": str(paper_claim_audit_path), "line": 0, "value": claim_status})
    if paper_claim_audit_path and str(paper_path) not in audited_paths:
        findings.append({"severity": "error", "message": "compact manuscript is missing from paper claim audit", "path": str(paper_claim_audit_path), "line": 0, "value": str(paper_path)})
    errors = [finding for finding in findings if finding["severity"] == "error"]
    return {
        "status": "pass" if not errors else "fail",
        "compact_ready": not errors,
        "paper_path": str(paper_path),
        "paper_claim_audit": str(paper_claim_audit_path) if paper_claim_audit_path else None,
        "paper_claim_audit_status": claim_status,
        "paper_claims_allowed": bool(claim_audit.get("paper_claims_allowed", False)),
        "known_blockers": [] if not errors else ["compact_source_readiness"],
        "metrics": {
            "word_count": word_count,
            "target_words": target_words,
            "excess_words": max(0, word_count - target_words),
            "citation_count": citation_count,
            "table_count": table_count,
            "sections": sections,
        },
        "requirements": {
            "required_sections": REQUIRED_SECTIONS,
            "required_terms": REQUIRED_TERMS,
            "target_words": target_words,
            "min_words": min_words,
            "min_citations": min_citations,
            "min_tables": min_tables,
            "claim_audit_must_include_compact": bool(paper_claim_audit_path),
        },
        "findings": findings,
        "errors": errors,
    }


def _sections(text: str) -> list[str]:
    sections = []
    marker = r"\section{"
    start = 0
    while True:
        index = text.find(marker, start)
        if index < 0:
            break
        brace_start = index + len(r"\section")
        section, end = _balanced_brace_content(text, brace_start)
        if section is not None:
            sections.append(section.strip())
            start = end
        else:
            start = index + len(marker)
    return sections


def _balanced_brace_content(text: str, brace_start: int) -> tuple[str | None, int]:
    while brace_start < len(text) and text[brace_start].isspace():
        brace_start += 1
    if brace_start >= len(text) or text[brace_start] != "{":
        return None, brace_start
    depth = 0
    content_start = brace_start + 1
    for index in range(brace_start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index], index + 1
    return None, len(text)


def _without_preamble(text: str) -> str:
    marker = "\\begin{document}"
    index = text.find(marker)
    return text[index + len(marker) :] if index >= 0 else text


def _body_start_line(text: str) -> int:
    marker = "\\begin{document}"
    index = text.find(marker)
    if index < 0:
        return 1
    return text[:index].count("\n") + 1


def _word_count(text: str) -> int:
    cleaned = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^]]*\])?(?:\{[^}]*\})?", " ", text)
    cleaned = re.sub(r"[^A-Za-z0-9'-]+", " ", cleaned)
    return len([word for word in cleaned.split() if word])


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}
