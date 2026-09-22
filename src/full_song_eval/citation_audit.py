from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

CITATION_COMMAND_RE = re.compile(r"\\cite[a-zA-Z]*\*?")
TEX_INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]*)\}")
BIB_ENTRY_RE = re.compile(r"@\s*([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
IGNORED_BIB_TYPES = {"comment", "preamble", "string"}


def audit_citations(paper_path: Path | list[Path], bib_path: Path) -> dict[str, Any]:
    paper_paths = [paper_path] if isinstance(paper_path, Path) else list(paper_path)
    findings: list[dict[str, Any]] = []
    bib_text = ""
    tex_sources: list[dict[str, Any]] = []
    for current_paper in paper_paths:
        if not current_paper.exists():
            findings.append({"severity": "error", "message": "missing manuscript file", "path": str(current_paper), "line": 0})
        else:
            tex_sources.extend(_tex_sources(current_paper))
    if not bib_path.exists():
        findings.append({"severity": "error", "message": "missing bibliography file", "path": str(bib_path), "line": 0})
    else:
        bib_text = bib_path.read_text(encoding="utf-8")

    missing_tex_inputs = [source for source in tex_sources if not source["exists"]]
    for source in missing_tex_inputs:
        findings.append(
            {
                "severity": "error",
                "message": f"included TeX file is missing: {source['path']}",
                "path": str(source.get("included_by", paper_paths[0] if paper_paths else "")),
                "line": int(source.get("line", 0)),
            }
        )
    citations = [occurrence for source in tex_sources if source["exists"] for occurrence in _citation_occurrences(source["text"], Path(str(source["path"])))]
    bib_entries = _bib_entries(bib_text, bib_path) if bib_text else []
    cited_keys = [occurrence["key"] for occurrence in citations]
    bib_keys = [entry["key"] for entry in bib_entries]
    cited_key_set = set(cited_keys)
    bib_key_set = set(bib_keys)

    if any(path.exists() for path in paper_paths) and not citations:
        findings.append({"severity": "error", "message": "no citation commands found", "path": str(paper_paths[0]), "line": 0})

    missing_keys = sorted(cited_key_set - bib_key_set)
    first_citation = _first_key_occurrences(citations)
    for key in missing_keys:
        occurrence = first_citation.get(key, {})
        findings.append(
            {
                "severity": "error",
                "message": f"cited key missing from bibliography: {key}",
                "path": str(occurrence.get("path", paper_paths[0] if paper_paths else "")),
                "line": int(occurrence.get("line", 0)),
                "key": key,
            }
        )

    uncited_keys = sorted(bib_key_set - cited_key_set)
    first_bib_line = _first_key_lines(bib_entries)
    for key in uncited_keys:
        findings.append(
            {
                "severity": "warning",
                "message": f"bibliography entry is not cited: {key}",
                "path": str(bib_path),
                "line": first_bib_line.get(key, 0),
                "key": key,
            }
        )

    duplicate_keys = sorted(key for key, count in Counter(bib_keys).items() if count > 1)
    for key in duplicate_keys:
        findings.append(
            {
                "severity": "warning",
                "message": f"duplicate bibliography key: {key}",
                "path": str(bib_path),
                "line": first_bib_line.get(key, 0),
                "key": key,
            }
        )

    errors = [finding for finding in findings if finding["severity"] == "error"]
    warnings = [finding for finding in findings if finding["severity"] == "warning"]
    return {
        "status": "pass" if not errors else "fail",
        "paper": str(paper_paths[0]) if paper_paths else "",
        "papers": [str(path) for path in paper_paths],
        "bib": str(bib_path),
        "metrics": {
            "citations_total": len(cited_keys),
            "citations_unique": len(cited_key_set),
            "bib_entries": len(bib_keys),
            "uncited_count": len(uncited_keys),
            "missing_count": len(missing_keys),
            "duplicate_bib_key_count": len(duplicate_keys),
            "tex_source_count": sum(1 for source in tex_sources if source["exists"]),
            "missing_tex_input_count": len(missing_tex_inputs),
        },
        "tex_sources": [str(source["path"]) for source in tex_sources if source["exists"]],
        "missing_tex_inputs": [str(source["path"]) for source in missing_tex_inputs],
        "cited_keys": sorted(cited_key_set),
        "bib_keys": sorted(bib_key_set),
        "missing_bib_keys": missing_keys,
        "uncited_bib_keys": uncited_keys,
        "duplicate_bib_keys": duplicate_keys,
        "findings": findings,
        "warnings": warnings,
        "errors": errors,
    }


def _citation_occurrences(text: str, paper_path: Path) -> list[dict[str, Any]]:
    cleaned_text = _strip_latex_comments(text)
    occurrences: list[dict[str, Any]] = []
    for command_match in CITATION_COMMAND_RE.finditer(cleaned_text):
        content, content_start = _citation_key_content(cleaned_text, command_match.end())
        if content is None:
            continue
        line_number = cleaned_text[: command_match.start()].count("\n") + 1
        for key in _split_citation_keys(content):
            occurrences.append({"key": key, "path": str(paper_path), "line": line_number, "offset": content_start})
    return occurrences


def _tex_sources(paper_path: Path) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[Path] = set()

    def visit(path: Path, *, included_by: Path | None = None, line: int = 0) -> None:
        normalized = path.resolve() if path.exists() else path
        if normalized in seen:
            return
        seen.add(normalized)
        if not path.exists():
            sources.append({"path": path, "exists": False, "text": "", "included_by": included_by or paper_path, "line": line})
            return
        text = path.read_text(encoding="utf-8")
        sources.append({"path": path, "exists": True, "text": text, "included_by": included_by or path, "line": line})
        cleaned_text = _strip_latex_comments(text)
        for match in TEX_INPUT_RE.finditer(cleaned_text):
            raw_path = match.group(1).strip()
            if not raw_path:
                continue
            child = _tex_input_path(path.parent, raw_path)
            child_line = cleaned_text[: match.start()].count("\n") + 1
            visit(child, included_by=path, line=child_line)

    visit(paper_path)
    return sources


def _tex_input_path(base: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = base / path
    return path if path.suffix else path.with_suffix(".tex")


def _citation_key_content(text: str, start_index: int) -> tuple[str | None, int]:
    cursor = _skip_whitespace(text, start_index)
    while cursor < len(text) and text[cursor] == "[":
        _, next_cursor = _balanced_delimited_content(text, cursor, "[", "]")
        if next_cursor <= cursor:
            return None, cursor
        cursor = _skip_whitespace(text, next_cursor)
    content, next_cursor = _balanced_delimited_content(text, cursor, "{", "}")
    if next_cursor <= cursor:
        return None, cursor
    return content, cursor + 1


def _split_citation_keys(content: str) -> list[str]:
    return [key.strip() for key in content.replace("\n", " ").split(",") if key.strip()]


def _bib_entries(text: str, bib_path: Path) -> list[dict[str, Any]]:
    cleaned_text = _strip_latex_comments(text)
    entries: list[dict[str, Any]] = []
    for entry_match in BIB_ENTRY_RE.finditer(cleaned_text):
        entry_type = entry_match.group(1).lower()
        if entry_type in IGNORED_BIB_TYPES:
            continue
        key = entry_match.group(2).strip()
        entries.append({"key": key, "path": str(bib_path), "line": cleaned_text[: entry_match.start()].count("\n") + 1})
    return entries


def _balanced_delimited_content(text: str, start_index: int, open_char: str, close_char: str) -> tuple[str | None, int]:
    if start_index >= len(text) or text[start_index] != open_char:
        return None, start_index
    depth = 0
    content_start = start_index + 1
    for index in range(start_index, len(text)):
        char = text[index]
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[content_start:index], index + 1
    return None, start_index


def _skip_whitespace(text: str, start_index: int) -> int:
    cursor = start_index
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor


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


def _first_key_lines(items: list[dict[str, Any]]) -> dict[str, int]:
    lines: dict[str, int] = {}
    for item in items:
        lines.setdefault(str(item["key"]), int(item.get("line", 0)))
    return lines


def _first_key_occurrences(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    occurrences: dict[str, dict[str, Any]] = {}
    for item in items:
        occurrences.setdefault(str(item["key"]), item)
    return occurrences
