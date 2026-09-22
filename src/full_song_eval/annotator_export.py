from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from full_song_eval.blind_packet import validate_blind_packet

REQUIRED_EXPORT_FILES = ["index.html", "annotation_sheet.csv", "README.md", "packet_summary.json"]
PRIVATE_PACKET_FILES = {"annotation_key.csv", "annotation_tasks.jsonl", "batch_plan.json"}
PRIVATE_TEXT_TOKENS = ["annotation_key", "annotation_tasks", "generation_id", "model_id", "audio_uri", "local_audio_uri"]
AUDIO_SRC_RE = re.compile(r"<audio\s+[^>]*src=\"([^\"]+)\"", re.IGNORECASE)


def export_annotator_packet(
    packet_dir: Path,
    output_dir: Path,
    *,
    copy_mode: str = "hardlink",
    overwrite: bool = False,
) -> dict[str, Any]:
    if copy_mode not in {"hardlink", "copy"}:
        raise ValueError("copy_mode must be 'hardlink' or 'copy'")
    packet_report = validate_blind_packet(packet_dir)
    if packet_report["status"] != "pass":
        errors = "; ".join(packet_report["errors"])
        raise ValueError(f"Cannot export packet that fails blind validation: {errors}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        _clear_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in ["index.html", "annotation_sheet.csv", "packet_summary.json"]:
        shutil.copy2(packet_dir / filename, output_dir / filename)
    summary = _load_json(packet_dir / "packet_summary.json")
    (output_dir / "README.md").write_text(_export_readme(summary), encoding="utf-8")
    audio_count = _materialize_audio_dir(packet_dir / "audio", output_dir / "audio", copy_mode=copy_mode)
    report = validate_annotator_export(output_dir)
    return {
        "output_dir": str(output_dir),
        "source_packet_dir": str(packet_dir),
        "copy_mode": copy_mode,
        "audio_files": audio_count,
        "validation": report,
        "status": report["status"],
    }


def validate_annotator_export(export_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    for filename in REQUIRED_EXPORT_FILES:
        if not (export_dir / filename).is_file():
            errors.append(f"missing required export file: {filename}")
    leaked_private_files = sorted(filename for filename in PRIVATE_PACKET_FILES if (export_dir / filename).exists())
    if leaked_private_files:
        errors.append(f"export includes private files: {', '.join(leaked_private_files)}")
    symlinks = sorted(str(path.relative_to(export_dir)) for path in export_dir.rglob("*") if path.is_symlink()) if export_dir.exists() else []
    if symlinks:
        errors.append(f"export contains symlinks: {', '.join(symlinks[:10])}")
    text_leaks = _text_leaks(export_dir)
    if text_leaks:
        errors.append(f"export text references private fields/files: {', '.join(text_leaks)}")
    audio_checks = _audio_checks(export_dir)
    errors.extend(audio_checks["errors"])
    return {
        "export_dir": str(export_dir),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "required_files": REQUIRED_EXPORT_FILES,
        "private_files_present": leaked_private_files,
        "symlinks": symlinks,
        "audio_files": audio_checks["audio_files"],
        "audio_sources": audio_checks["audio_sources"],
        "text_leaks": text_leaks,
    }


def _clear_directory(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _export_readme(summary: dict[str, Any]) -> str:
    dimensions = summary.get("rating_dimensions", [])
    dimension_lines = "\n".join(f"- `{dimension}`" for dimension in dimensions)
    task_count = summary.get("tasks", "unknown")
    task_type = str(summary.get("task_type", "listening"))
    return f"""# Blind Listening Packet

This annotator package contains {task_count} {task_type} tasks.

Files to use:

- `index.html`: browser listening page with anonymized audio controls.
- `annotation_sheet.csv`: response sheet to fill and return.
- `audio/`: anonymized local audio files used by `index.html`.

Candidate identities are intentionally hidden. Please do not try to identify systems or file origins while rating. Use `invalid_reason` only if audio is missing, silent, corrupted, or impossible to judge.

Rating dimensions:

{dimension_lines}
"""


def _materialize_audio_dir(source_audio_dir: Path, output_audio_dir: Path, *, copy_mode: str) -> int:
    if not source_audio_dir.is_dir():
        raise FileNotFoundError(f"Missing packet audio directory: {source_audio_dir}")
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for source_entry in sorted(source_audio_dir.iterdir()):
        if source_entry.is_dir() and not source_entry.is_symlink():
            continue
        resolved_source = source_entry.resolve(strict=True)
        if not resolved_source.is_file():
            raise FileNotFoundError(f"Packet audio entry does not resolve to a file: {source_entry}")
        destination = output_audio_dir / source_entry.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        if copy_mode == "hardlink":
            try:
                os.link(resolved_source, destination)
            except OSError:
                shutil.copy2(resolved_source, destination)
        else:
            shutil.copy2(resolved_source, destination)
        count += 1
    return count


def _text_leaks(export_dir: Path) -> list[str]:
    leaks = set()
    for filename in REQUIRED_EXPORT_FILES:
        path = export_dir / filename
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for token in PRIVATE_TEXT_TOKENS:
            if token in text:
                leaks.add(token)
    return sorted(leaks)


def _audio_checks(export_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    html_path = export_dir / "index.html"
    audio_sources: list[str] = []
    if html_path.is_file():
        audio_sources = AUDIO_SRC_RE.findall(html_path.read_text(encoding="utf-8"))
    audio_dir = export_dir / "audio"
    audio_files = sorted(str(path.relative_to(export_dir)) for path in audio_dir.iterdir() if path.is_file()) if audio_dir.is_dir() else []
    if not audio_sources:
        errors.append("index.html has no audio sources")
    for source in audio_sources:
        source_path = Path(source)
        if source_path.is_absolute() or ".." in source_path.parts:
            errors.append(f"index.html has unsafe audio source: {source}")
            continue
        resolved = export_dir / source_path
        if not resolved.is_file():
            errors.append(f"index.html audio source is missing: {source}")
    if len(audio_files) < len(audio_sources):
        errors.append("audio directory has fewer files than index.html audio sources")
    return {"errors": errors, "audio_files": audio_files, "audio_sources": audio_sources}
