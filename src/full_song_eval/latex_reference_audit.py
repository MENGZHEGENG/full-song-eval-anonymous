from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

LABEL_RE = re.compile(r"\\label\s*\{([^}]*)\}")
REF_RE = re.compile(r"\\(?:ref|autoref|cref|Cref|eqref|pageref|nameref)\*?\s*\{([^}]*)\}")
FLOAT_RE = re.compile(r"\\begin\s*\{(table|figure)\}(?:\[[^]]*\])?(.*?)\\end\s*\{\1\}", re.DOTALL)
CAPTION_RE = re.compile(r"\\caption(?:\[[^]]*\])?\s*\{")
INCLUDE_GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\s*\{([^}]*)\}")
INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]*)\}")
GRAPHIC_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".eps")
FLOAT_LABEL_PREFIXES = ("tab:", "fig:")


def audit_latex_references(paper_path: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    if not paper_path.exists():
        findings.append({"severity": "error", "message": "missing manuscript file", "path": str(paper_path), "line": 0})
        return _report(paper_path, [], [], [], [], [], findings)

    fragments, inputs = _collect_tex_fragments(paper_path, findings)
    labels = [label for path, text in fragments for label in _label_occurrences(text, path)]
    refs = [ref for path, text in fragments for ref in _ref_occurrences(text, path)]
    floats = [item for path, text in fragments for item in _float_occurrences(text, path)]
    graphics = [item for path, text in fragments for item in _includegraphics_occurrences(text, path)]

    label_keys = [label["key"] for label in labels]
    ref_keys = [ref["key"] for ref in refs]
    label_key_set = set(label_keys)
    ref_key_set = set(ref_keys)
    first_label_line = _first_key_lines(labels)
    first_ref_line = _first_key_lines(refs)
    unreferenced_float_labels = _unreferenced_float_labels(floats, ref_key_set)
    float_environments = _float_label_environments(floats)

    for key in sorted(ref_key_set - label_key_set):
        findings.append(
            {
                "severity": "error",
                "message": f"reference points to missing label: {key}",
                "path": first_ref_line.get(key, {}).get("path", str(paper_path)),
                "line": first_ref_line.get(key, {}).get("line", 0),
                "key": key,
            }
        )

    for key in sorted(key for key, count in Counter(label_keys).items() if count > 1):
        findings.append(
            {
                "severity": "error",
                "message": f"duplicate label: {key}",
                "path": first_label_line.get(key, {}).get("path", str(paper_path)),
                "line": first_label_line.get(key, {}).get("line", 0),
                "key": key,
            }
        )

    for key in unreferenced_float_labels:
        environment = float_environments.get(key, "table/figure")
        findings.append(
            {
                "severity": "error",
                "message": f"{environment} label is not referenced in manuscript text: {key}",
                "path": first_label_line.get(key, {}).get("path", str(paper_path)),
                "line": first_label_line.get(key, {}).get("line", 0),
                "key": key,
            }
        )

    for key in sorted((label_key_set - ref_key_set) - set(unreferenced_float_labels)):
        findings.append(
            {
                "severity": "warning",
                "message": f"label is not referenced: {key}",
                "path": first_label_line.get(key, {}).get("path", str(paper_path)),
                "line": first_label_line.get(key, {}).get("line", 0),
                "key": key,
            }
        )

    for item in floats:
        if not item["has_caption"]:
            findings.append(
                {
                    "severity": "error",
                    "message": f"{item['environment']} environment is missing a caption",
                    "path": str(paper_path),
                    "line": item["line"],
                    "environment": item["environment"],
                }
            )
        if not item["has_label"]:
            findings.append(
                {
                    "severity": "error",
                    "message": f"{item['environment']} environment is missing a label",
                    "path": str(paper_path),
                    "line": item["line"],
                    "environment": item["environment"],
                }
            )

    for item in graphics:
        resolved_path = _resolve_graphic_path(Path(item["path"]).parent, item["target"])
        item["resolved_path"] = str(resolved_path) if resolved_path else None
        item["exists"] = resolved_path is not None
        if resolved_path is None:
            findings.append(
                {
                    "severity": "error",
                    "message": f"included graphics file is missing: {item['target']}",
                    "path": item["path"],
                    "line": item["line"],
                    "target": item["target"],
                }
            )

    return _report(paper_path, labels, refs, floats, graphics, inputs, findings)


def _collect_tex_fragments(paper_path: Path, findings: list[dict[str, Any]]) -> tuple[list[tuple[Path, str]], list[dict[str, Any]]]:
    fragments: list[tuple[Path, str]] = []
    inputs: list[dict[str, Any]] = []
    pending = [paper_path]
    seen: set[Path] = set()
    while pending:
        current_path = pending.pop()
        resolved_current = current_path.resolve()
        if resolved_current in seen:
            continue
        seen.add(resolved_current)
        text = _strip_latex_comments(current_path.read_text(encoding="utf-8"))
        fragments.append((current_path, text))
        for item in _input_occurrences(text, current_path):
            resolved_path = _resolve_tex_path(current_path.parent, item["target"])
            item["resolved_path"] = str(resolved_path) if resolved_path else None
            item["exists"] = resolved_path is not None
            inputs.append(item)
            if resolved_path is None:
                findings.append(
                    {
                        "severity": "error",
                        "message": f"included TeX file is missing: {item['target']}",
                        "path": item["path"],
                        "line": item["line"],
                        "target": item["target"],
                    }
                )
            else:
                pending.append(resolved_path)
    return fragments, inputs


def _report(
    paper_path: Path,
    labels: list[dict[str, Any]],
    refs: list[dict[str, Any]],
    floats: list[dict[str, Any]],
    graphics: list[dict[str, Any]],
    inputs: list[dict[str, Any]],
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    errors = [finding for finding in findings if finding["severity"] == "error"]
    warnings = [finding for finding in findings if finding["severity"] == "warning"]
    labels_unique = sorted({label["key"] for label in labels})
    refs_unique = sorted({ref["key"] for ref in refs})
    missing_references = sorted({ref["key"] for ref in refs} - {label["key"] for label in labels})
    unused_labels = sorted({label["key"] for label in labels} - {ref["key"] for ref in refs})
    unreferenced_float_labels = _unreferenced_float_labels(floats, set(refs_unique))
    return {
        "status": "pass" if not errors else "fail",
        "paper": str(paper_path),
        "metrics": {
            "labels_total": len(labels),
            "labels_unique": len(labels_unique),
            "references_total": len(refs),
            "references_unique": len(refs_unique),
            "tables": sum(1 for item in floats if item["environment"] == "table"),
            "figures": sum(1 for item in floats if item["environment"] == "figure"),
            "graphics_includes": len(graphics),
            "tex_includes": len(inputs),
            "missing_reference_count": len(missing_references),
            "unused_label_count": len(unused_labels),
            "unreferenced_float_label_count": len(unreferenced_float_labels),
        },
        "labels": labels,
        "references": refs,
        "floats": floats,
        "graphics": graphics,
        "tex_includes": inputs,
        "missing_references": missing_references,
        "unused_labels": unused_labels,
        "unreferenced_float_labels": unreferenced_float_labels,
        "findings": findings,
        "warnings": warnings,
        "errors": errors,
    }


def _label_occurrences(text: str, paper_path: Path) -> list[dict[str, Any]]:
    return [
        {"key": match.group(1).strip(), "path": str(paper_path), "line": _line_number(text, match.start())}
        for match in LABEL_RE.finditer(text)
        if match.group(1).strip()
    ]


def _ref_occurrences(text: str, paper_path: Path) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for match in REF_RE.finditer(text):
        line = _line_number(text, match.start())
        for key in _split_keys(match.group(1)):
            if _is_latex_macro_parameter_reference(key):
                continue
            refs.append({"key": key, "path": str(paper_path), "line": line})
    return refs


def _is_latex_macro_parameter_reference(key: str) -> bool:
    return "#" in key


def _float_occurrences(text: str, paper_path: Path) -> list[dict[str, Any]]:
    floats: list[dict[str, Any]] = []
    for match in FLOAT_RE.finditer(text):
        body = match.group(2)
        labels = [label.group(1).strip() for label in LABEL_RE.finditer(body) if label.group(1).strip()]
        floats.append(
            {
                "environment": match.group(1),
                "path": str(paper_path),
                "line": _line_number(text, match.start()),
                "has_caption": CAPTION_RE.search(body) is not None,
                "has_label": bool(labels),
                "labels": labels,
            }
        )
    return floats


def _includegraphics_occurrences(text: str, paper_path: Path) -> list[dict[str, Any]]:
    return [
        {"target": match.group(1).strip(), "path": str(paper_path), "line": _line_number(text, match.start())}
        for match in INCLUDE_GRAPHICS_RE.finditer(text)
        if match.group(1).strip()
    ]


def _input_occurrences(text: str, paper_path: Path) -> list[dict[str, Any]]:
    return [
        {"target": match.group(1).strip(), "path": str(paper_path), "line": _line_number(text, match.start())}
        for match in INPUT_RE.finditer(text)
        if match.group(1).strip()
    ]


def _split_keys(content: str) -> list[str]:
    return [key.strip() for key in content.replace("\n", " ").split(",") if key.strip()]


def _resolve_graphic_path(base_dir: Path, target: str) -> Path | None:
    path = (base_dir / target).resolve()
    if path.exists():
        return path
    if path.suffix:
        return None
    for extension in GRAPHIC_EXTENSIONS:
        candidate = path.with_suffix(extension)
        if candidate.exists():
            return candidate
    return None


def _resolve_tex_path(base_dir: Path, target: str) -> Path | None:
    path = (base_dir / target).resolve()
    if path.exists():
        return path
    if not path.suffix:
        candidate = path.with_suffix(".tex")
        if candidate.exists():
            return candidate
    return None


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


def _first_key_lines(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lines: dict[str, dict[str, Any]] = {}
    for item in items:
        lines.setdefault(str(item["key"]), {"path": str(item.get("path", "")), "line": int(item.get("line", 0))})
    return lines


def _unreferenced_float_labels(floats: list[dict[str, Any]], ref_key_set: set[str]) -> list[str]:
    float_labels = {
        label
        for item in floats
        for label in item["labels"]
        if any(label.startswith(prefix) for prefix in FLOAT_LABEL_PREFIXES)
    }
    return sorted(float_labels - ref_key_set)


def _float_label_environments(floats: list[dict[str, Any]]) -> dict[str, str]:
    environments: dict[str, str] = {}
    for item in floats:
        for label in item["labels"]:
            environments.setdefault(label, str(item["environment"]))
    return environments
