from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PAPERS = [Path("paper/main.tex"), Path("paper/icassp_compact.tex")]
DEFAULT_BIBLIOGRAPHY = Path("paper/references.bib")
DEFAULT_THEMES = [
    {
        "id": "svs_foundation",
        "name": "SVS foundation and uncertainty",
        "required_any": ["muskits2022", "muskits_espnet2024", "slm_svs2025", "robust_svs_uncertainty2025", "mmgenre2026", "singmospro2025", "liu2025bridging_svc"],
        "minimum_keys": 3,
    },
    {
        "id": "human_music_preference",
        "name": "Human music preference and full-song benchmarks",
        "required_any": ["musicprefs2025", "songeval2025", "songbench2026", "asae_challenge2026"],
        "minimum_keys": 3,
    },
    {
        "id": "stem_temporal_song_evaluation",
        "name": "Stem-aware and temporal song evaluation",
        "required_any": ["song_aesthetics_msaf2026", "songsqa2026"],
        "minimum_keys": 2,
    },
    {
        "id": "shi_evaluation_toolkits",
        "name": "Shi-line evaluation discipline and metric toolkits",
        "required_any": ["speechdrame2025", "versa2025", "versav22025"],
        "minimum_keys": 3,
    },
    {
        "id": "shi_pressure_set_2026",
        "name": "2026 Shi and Anuttacon pressure set",
        "required_any": ["han2026singingsds", "bagpiper2026", "espnet32026", "speechdrame2025", "songbench2026"],
        "minimum_keys": 4,
    },
    {
        "id": "chain_evaluator",
        "name": "Chain-structured evaluator hypothesis",
        "required_any": ["arecho2025"],
        "minimum_keys": 1,
    },
    {
        "id": "open_song_generators",
        "name": "Open and current full-song generators under study",
        "required_any": ["acestep2025", "yue2025", "qwenmusic2026"],
        "minimum_keys": 2,
    },
]


@dataclass(frozen=True)
class ThemeCoverage:
    theme_id: str
    name: str
    paper: str
    required_any: list[str]
    present_keys: list[str]
    missing_keys: list[str]
    minimum_keys: int
    status: str


def write_literature_coverage_audit(
    *,
    output_json: Path,
    output_md: Path,
    paper_paths: list[Path] | None = None,
    bibliography_path: Path = DEFAULT_BIBLIOGRAPHY,
    themes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    report = build_literature_coverage_audit(
        paper_paths=paper_paths or DEFAULT_PAPERS,
        bibliography_path=bibliography_path,
        themes=themes or DEFAULT_THEMES,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(report), encoding="utf-8")
    return report


def build_literature_coverage_audit(
    *,
    paper_paths: list[Path] | None = None,
    bibliography_path: Path = DEFAULT_BIBLIOGRAPHY,
    themes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    paper_paths = paper_paths or DEFAULT_PAPERS
    themes = themes or DEFAULT_THEMES
    bib_keys = _bib_keys(bibliography_path)
    findings: list[dict[str, Any]] = []
    if not bib_keys:
        findings.append({"severity": "error", "id": "missing_bibliography", "message": "Bibliography is missing or contains no BibTeX keys.", "path": str(bibliography_path)})
    papers = []
    coverage_rows: list[ThemeCoverage] = []
    for paper_path in paper_paths:
        text = _read_text(paper_path)
        cite_keys = _cite_keys(text)
        if not text:
            findings.append({"severity": "error", "id": "missing_paper", "message": "Paper source is missing or empty.", "path": str(paper_path)})
        if text and not cite_keys:
            findings.append({"severity": "error", "id": "missing_citations", "message": "Paper source contains no citation commands.", "path": str(paper_path)})
        missing_bib = sorted(key for key in cite_keys if key not in bib_keys)
        if missing_bib:
            findings.append(
                {
                    "severity": "error",
                    "id": "missing_bib_keys",
                    "message": "Paper cites keys that are not present in the bibliography.",
                    "path": str(paper_path),
                    "missing_keys": missing_bib,
                }
            )
        paper_coverage = []
        for theme in themes:
            row = _theme_coverage(theme, cite_keys=cite_keys, paper=str(paper_path))
            coverage_rows.append(row)
            paper_coverage.append(_coverage_dict(row))
            if row.status != "pass":
                findings.append(
                    {
                        "severity": "error",
                        "id": "literature_theme_missing",
                        "message": f"{paper_path} lacks required coverage for {row.name}.",
                        "path": str(paper_path),
                        "theme": row.theme_id,
                        "present_keys": row.present_keys,
                        "missing_keys": row.missing_keys,
                        "minimum_keys": row.minimum_keys,
                    }
                )
        papers.append(
            {
                "path": str(paper_path),
                "citation_count": len(cite_keys),
                "unique_citation_count": len(set(cite_keys)),
                "theme_count": len(paper_coverage),
                "themes_passed": sum(1 for row in paper_coverage if row["status"] == "pass"),
                "coverage": paper_coverage,
            }
        )
    errors = [finding for finding in findings if finding["severity"] == "error"]
    required_rows = len(paper_paths) * len(themes)
    passed_rows = sum(1 for row in coverage_rows if row.status == "pass")
    return {
        "status": "pass" if not errors else "fail",
        "bibliography": str(bibliography_path),
        "bibliography_key_count": len(bib_keys),
        "papers": papers,
        "theme_count": len(themes),
        "paper_count": len(paper_paths),
        "required_theme_rows": required_rows,
        "passed_theme_rows": passed_rows,
        "missing_theme_rows": required_rows - passed_rows,
        "themes": themes,
        "findings": findings,
        "errors": errors,
        "known_blockers": ["literature_coverage"] if errors else [],
    }


def _theme_coverage(theme: dict[str, Any], *, cite_keys: list[str], paper: str) -> ThemeCoverage:
    required = [str(key) for key in theme.get("required_any", [])]
    minimum = int(theme.get("minimum_keys", len(required)))
    cited = set(cite_keys)
    present = sorted(key for key in required if key in cited)
    missing = sorted(key for key in required if key not in cited)
    status = "pass" if len(present) >= minimum else "missing_coverage"
    return ThemeCoverage(
        theme_id=str(theme["id"]),
        name=str(theme.get("name", theme["id"])),
        paper=paper,
        required_any=required,
        present_keys=present,
        missing_keys=missing,
        minimum_keys=minimum,
        status=status,
    )


def _coverage_dict(row: ThemeCoverage) -> dict[str, Any]:
    return {
        "theme_id": row.theme_id,
        "name": row.name,
        "paper": row.paper,
        "required_any": row.required_any,
        "present_keys": row.present_keys,
        "missing_keys": row.missing_keys,
        "minimum_keys": row.minimum_keys,
        "status": row.status,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Literature Coverage Audit",
        "",
        "This audit checks that the full and compact manuscripts cite the required literature themes used to position FullSongEval.",
        "",
        "## Summary",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Papers: {report.get('paper_count')}",
        f"- Themes: {report.get('theme_count')}",
        f"- Passed theme rows: {report.get('passed_theme_rows')}/{report.get('required_theme_rows')}",
        f"- Missing theme rows: {report.get('missing_theme_rows')}",
        f"- Bibliography keys: {report.get('bibliography_key_count')}",
        "",
        "## Per-Paper Coverage",
        "",
    ]
    for paper in report.get("papers", []) or []:
        lines.append(f"### `{paper.get('path')}`")
        lines.append("")
        lines.append(f"- Unique citations: {paper.get('unique_citation_count')}")
        lines.append(f"- Themes passed: {paper.get('themes_passed')}/{paper.get('theme_count')}")
        for row in paper.get("coverage", []) or []:
            lines.append(f"- `{row.get('theme_id')}`: `{row.get('status')}`; present: {_inline_keys(row.get('present_keys', []))}; missing: {_inline_keys(row.get('missing_keys', []))}")
        lines.append("")
    lines.extend(["## Findings", ""])
    findings = report.get("findings", []) if isinstance(report.get("findings", []), list) else []
    if findings:
        for finding in findings:
            lines.append(f"- `{finding.get('severity')}` `{finding.get('id')}`: {finding.get('message')}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _inline_keys(values: Any) -> str:
    if not values:
        return "none"
    if not isinstance(values, list):
        return f"`{values}`"
    return ", ".join(f"`{value}`" for value in values)


def _bib_keys(path: Path) -> set[str]:
    text = _read_text(path)
    return set(re.findall(r"@\w+\s*\{\s*([^,]+),", text))


def _cite_keys(text: str) -> list[str]:
    keys: list[str] = []
    for match in re.finditer(r"\\cite(?:[tp])?(?:\[[^\]]*\])*\{([^}]+)\}", text):
        keys.extend(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""
