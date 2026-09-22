from __future__ import annotations

import csv
import json
import os
from itertools import combinations
from html import escape
from pathlib import Path
from typing import Any

from full_song_eval.jsonl import write_jsonl


RATING_DIMENSIONS = [
    "lyric_intelligibility",
    "lyric_melody_alignment",
    "melody_memorability",
    "vocal_naturalness",
    "breath_and_phrasing",
    "genre_fit",
    "arrangement_fit",
    "structure_coherence",
    "mix_quality",
    "overall_preference",
]


def build_single_candidate_tasks(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    audio_root: Path | None = None,
    require_audio: bool = False,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for index, record in enumerate(sorted(records, key=_generation_sort_key), start=1):
        task = {
            "task_id": f"single_{index:04d}",
            "task_type": "single_candidate_rating",
            "prompt_id": record["prompt_id"],
            "prompt": record.get("prompt", ""),
            "lyrics": record.get("lyrics", ""),
            "candidate": _candidate_from_record(record, output_dir, audio_root, require_audio),
            "metadata": record.get("metadata", {}),
            "duration_seconds": record.get("duration_seconds"),
            "sample_rate": record.get("sample_rate"),
            "channels": record.get("channels"),
            "rating_dimensions": RATING_DIMENSIONS,
        }
        tasks.append(task)
    return tasks


def build_pairwise_tasks(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    audio_root: Path | None = None,
    require_audio: bool = False,
    max_pairs_per_prompt: int | None = None,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record["prompt_id"]), []).append(record)

    tasks: list[dict[str, Any]] = []
    for prompt_id in sorted(groups):
        prompt_records = sorted(groups[prompt_id], key=_generation_sort_key)
        pair_count = 0
        for candidate_a, candidate_b in combinations(prompt_records, 2):
            if max_pairs_per_prompt is not None and pair_count >= max_pairs_per_prompt:
                break
            pair_count += 1
            task = {
                "task_id": f"pair_{len(tasks) + 1:04d}_{prompt_id}_{pair_count:02d}",
                "task_type": "pairwise_preference_rating",
                "prompt_id": prompt_id,
                "prompt": candidate_a.get("prompt", ""),
                "lyrics": candidate_a.get("lyrics", ""),
                "candidate_a": _candidate_from_record(candidate_a, output_dir, audio_root, require_audio),
                "candidate_b": _candidate_from_record(candidate_b, output_dir, audio_root, require_audio),
                "metadata": candidate_a.get("metadata", {}),
                "pairwise_preference_options": [
                    "a_much_better",
                    "a_better",
                    "tie",
                    "b_better",
                    "b_much_better",
                    "invalid",
                ],
                "rating_dimensions": RATING_DIMENSIONS,
            }
            tasks.append(task)
    return tasks


def write_annotation_packet(tasks: list[dict[str, Any]], output_dir: Path, *, blind: bool = True) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "tasks_jsonl": output_dir / "annotation_tasks.jsonl",
        "sheet_csv": output_dir / "annotation_sheet.csv",
        "key_csv": output_dir / "annotation_key.csv",
        "html": output_dir / "index.html",
        "readme": output_dir / "README.md",
        "summary_json": output_dir / "packet_summary.json",
    }
    write_jsonl(paths["tasks_jsonl"], tasks)
    _write_sheet(paths["sheet_csv"], tasks, blind=blind)
    blind_audio_uris = _write_blind_audio_links(output_dir, tasks) if blind else {}
    _write_key(paths["key_csv"], tasks)
    paths["html"].write_text(_render_html(tasks, blind=blind, blind_audio_uris=blind_audio_uris), encoding="utf-8")
    paths["readme"].write_text(_render_readme(tasks, blind=blind), encoding="utf-8")
    paths["summary_json"].write_text(json.dumps(_summary(tasks, blind=blind), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def _generation_sort_key(record: dict[str, Any]) -> tuple[str, str]:
    candidate_index = record.get("metadata", {}).get("candidate_index")
    candidate_sort = -1 if candidate_index is None else int(candidate_index)
    return str(record.get("prompt_id", "")), f"{candidate_sort:08d}_{record.get('generation_id', '')}"


def _candidate_from_record(
    record: dict[str, Any],
    output_dir: Path,
    audio_root: Path | None,
    require_audio: bool,
) -> dict[str, Any]:
    audio_uri = str(record["audio_uri"])
    return {
        "generation_id": record["generation_id"],
        "model_id": record["model_id"],
        "audio_uri": audio_uri,
        "local_audio_uri": _local_audio_uri(
            audio_uri=audio_uri,
            output_dir=output_dir,
            audio_root=audio_root,
            require_audio=require_audio,
        ),
    }


def _local_audio_uri(
    *,
    audio_uri: str,
    output_dir: Path,
    audio_root: Path | None,
    require_audio: bool,
) -> str:
    if audio_root is None:
        return audio_uri
    audio_path = audio_root / audio_uri
    if require_audio and not audio_path.exists():
        raise FileNotFoundError(f"Missing audio for packet: {audio_path}")
    relative = os.path.relpath(audio_path, output_dir)
    return Path(relative).as_posix()


def _write_sheet(path: Path, tasks: list[dict[str, Any]], *, blind: bool) -> None:
    task_type = _task_type(tasks)
    if task_type == "pairwise_preference_rating":
        _write_pairwise_sheet(path, tasks, blind=blind)
        return
    _write_single_sheet(path, tasks, blind=blind)


def _write_single_sheet(path: Path, tasks: list[dict[str, Any]], *, blind: bool) -> None:
    identity_fields = [] if blind else ["generation_id", "model_id"]
    fieldnames = [
        "task_id",
        "prompt_id",
        *identity_fields,
        "genre",
        "language",
        "duration_seconds",
        "audio_uri",
        *RATING_DIMENSIONS,
        "invalid_reason",
        "free_text_notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for task in tasks:
            metadata = task.get("metadata", {})
            row = {
                "task_id": task["task_id"],
                "prompt_id": task["prompt_id"],
                "genre": metadata.get("genre", ""),
                "language": metadata.get("language", ""),
                "duration_seconds": task.get("duration_seconds"),
                "audio_uri": task["candidate"]["local_audio_uri"],
                "invalid_reason": "",
                "free_text_notes": "",
            }
            if not blind:
                row["generation_id"] = task["candidate"]["generation_id"]
                row["model_id"] = task["candidate"]["model_id"]
            for dimension in RATING_DIMENSIONS:
                row[dimension] = ""
            writer.writerow(row)


def _write_pairwise_sheet(path: Path, tasks: list[dict[str, Any]], *, blind: bool) -> None:
    candidate_identity_fields = [] if blind else [
        "candidate_a_generation_id",
        "candidate_a_model_id",
        "candidate_a_audio_uri",
        "candidate_b_generation_id",
        "candidate_b_model_id",
        "candidate_b_audio_uri",
    ]
    fieldnames = [
        "task_id",
        "prompt_id",
        *candidate_identity_fields,
        "genre",
        "language",
        "pairwise_preference",
        *[f"candidate_a_{dimension}" for dimension in RATING_DIMENSIONS],
        *[f"candidate_b_{dimension}" for dimension in RATING_DIMENSIONS],
        "invalid_reason",
        "free_text_notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for task in tasks:
            metadata = task.get("metadata", {})
            row = {
                "task_id": task["task_id"],
                "prompt_id": task["prompt_id"],
                "genre": metadata.get("genre", ""),
                "language": metadata.get("language", ""),
                "pairwise_preference": "",
                "invalid_reason": "",
                "free_text_notes": "",
            }
            if not blind:
                row.update(
                    {
                        "candidate_a_generation_id": task["candidate_a"]["generation_id"],
                        "candidate_a_model_id": task["candidate_a"]["model_id"],
                        "candidate_a_audio_uri": task["candidate_a"]["local_audio_uri"],
                        "candidate_b_generation_id": task["candidate_b"]["generation_id"],
                        "candidate_b_model_id": task["candidate_b"]["model_id"],
                        "candidate_b_audio_uri": task["candidate_b"]["local_audio_uri"],
                    }
                )
            for dimension in RATING_DIMENSIONS:
                row[f"candidate_a_{dimension}"] = ""
                row[f"candidate_b_{dimension}"] = ""
            writer.writerow(row)


def _write_key(path: Path, tasks: list[dict[str, Any]]) -> None:
    task_type = _task_type(tasks)
    fieldnames = [
        "task_id",
        "candidate_label",
        "generation_id",
        "model_id",
        "audio_uri",
        "local_audio_uri",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for task in tasks:
            if task_type == "pairwise_preference_rating":
                candidates = [("candidate_a", task["candidate_a"]), ("candidate_b", task["candidate_b"])]
            else:
                candidates = [("candidate", task["candidate"])]
            for label, candidate in candidates:
                writer.writerow(
                    {
                        "task_id": task["task_id"],
                        "candidate_label": label,
                        "generation_id": candidate["generation_id"],
                        "model_id": candidate["model_id"],
                        "audio_uri": candidate["audio_uri"],
                        "local_audio_uri": candidate["local_audio_uri"],
                    }
                )


def _write_blind_audio_links(output_dir: Path, tasks: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    task_type = _task_type(tasks)
    uris: dict[tuple[str, str], str] = {}
    for task in tasks:
        if task_type == "pairwise_preference_rating":
            candidates = [("candidate_a", task["candidate_a"]), ("candidate_b", task["candidate_b"])]
        else:
            candidates = [("candidate", task["candidate"])]
        for label, candidate in candidates:
            source_uri = Path(str(candidate["local_audio_uri"]))
            suffix = source_uri.suffix or ".wav"
            link_name = f"{_safe_filename(str(task['task_id']))}_{label}{suffix}"
            link_path = audio_dir / link_name
            target_path = output_dir / source_uri
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            relative_target = os.path.relpath(target_path, audio_dir)
            link_path.symlink_to(relative_target)
            uris[(str(task["task_id"]), label)] = f"audio/{link_name}"
    return uris


def _safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)


def _render_html(tasks: list[dict[str, Any]], *, blind: bool, blind_audio_uris: dict[tuple[str, str], str]) -> str:
    task_type = _task_type(tasks)
    rows = "\n".join(_render_pairwise_task_html(task, blind=blind, blind_audio_uris=blind_audio_uris) if task_type == "pairwise_preference_rating" else _render_task_html(task, blind=blind, blind_audio_uris=blind_audio_uris) for task in tasks)
    dimensions = "".join(f"<li>{escape(dimension)}</li>" for dimension in RATING_DIMENSIONS)
    title = "SongDRAME Pairwise Pilot Packet" if task_type == "pairwise_preference_rating" else "SongDRAME Single-Candidate Pilot Packet"
    guidance = (
        "Choose a pairwise preference, then rate each candidate from 1 to 5 on every dimension."
        if task_type == "pairwise_preference_rating"
        else "Rate each generated song from 1 to 5 on every dimension."
    )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>SongDRAME Annotation Packet</title>
  <style>
    body {{ font-family: system-ui, sans-serif; line-height: 1.45; margin: 2rem; max-width: 1100px; }}
    article {{ border: 1px solid #ddd; border-radius: 0.75rem; padding: 1rem; margin: 1rem 0; }}
    audio {{ width: 100%; }}
    pre {{ background: #f7f7f7; padding: 0.75rem; white-space: pre-wrap; }}
    .meta {{ color: #555; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>{guidance} Use <strong>invalid_reason</strong> only if audio is missing, silent, corrupted, or impossible to judge.</p>
  <p class="meta">This listening page is {'blind: model and generation identifiers are hidden from annotators.' if blind else 'unblinded: model and generation identifiers are shown.'}</p>
  <h2>Rating dimensions</h2>
  <ol>{dimensions}</ol>
  <h2>Items</h2>
  {rows}
</body>
</html>
"""


def _render_task_html(task: dict[str, Any], *, blind: bool, blind_audio_uris: dict[tuple[str, str], str]) -> str:
    candidate = task["candidate"]
    metadata = task.get("metadata", {})
    genre = metadata.get("genre", "unknown")
    language = metadata.get("language", "unknown")
    identity = "" if blind else f" | Generation: {escape(str(candidate['generation_id']))} | Model: {escape(str(candidate['model_id']))}"
    audio_uri = blind_audio_uris.get((str(task["task_id"]), "candidate"), str(candidate["local_audio_uri"]))
    return f"""<article id=\"{escape(task['task_id'])}\">
  <h3>{escape(task['task_id'])}</h3>
  <p class=\"meta\">Prompt: {escape(str(task['prompt_id']))}{identity} | Genre: {escape(str(genre))} | Language: {escape(str(language))}</p>
  <audio controls src=\"{escape(str(audio_uri))}\"></audio>
  <h4>Prompt</h4>
  <pre>{escape(str(task.get('prompt', '')))}</pre>
  <h4>Lyrics</h4>
  <pre>{escape(str(task.get('lyrics', '')))}</pre>
</article>"""


def _render_pairwise_task_html(task: dict[str, Any], *, blind: bool, blind_audio_uris: dict[tuple[str, str], str]) -> str:
    candidate_a = task["candidate_a"]
    candidate_b = task["candidate_b"]
    metadata = task.get("metadata", {})
    genre = metadata.get("genre", "unknown")
    language = metadata.get("language", "unknown")
    candidate_a_title = "Candidate A" if blind else f"Candidate A: {candidate_a['generation_id']}"
    candidate_b_title = "Candidate B" if blind else f"Candidate B: {candidate_b['generation_id']}"
    candidate_a_audio_uri = blind_audio_uris.get((str(task["task_id"]), "candidate_a"), str(candidate_a["local_audio_uri"]))
    candidate_b_audio_uri = blind_audio_uris.get((str(task["task_id"]), "candidate_b"), str(candidate_b["local_audio_uri"]))
    return f"""<article id=\"{escape(task['task_id'])}\">
  <h3>{escape(task['task_id'])}</h3>
  <p class=\"meta\">Prompt: {escape(str(task['prompt_id']))} | Genre: {escape(str(genre))} | Language: {escape(str(language))}</p>
  <h4>{escape(str(candidate_a_title))}</h4>
  <audio controls src=\"{escape(str(candidate_a_audio_uri))}\"></audio>
  <h4>{escape(str(candidate_b_title))}</h4>
  <audio controls src=\"{escape(str(candidate_b_audio_uri))}\"></audio>
  <h4>Prompt</h4>
  <pre>{escape(str(task.get('prompt', '')))}</pre>
  <h4>Lyrics</h4>
  <pre>{escape(str(task.get('lyrics', '')))}</pre>
</article>"""


def _render_readme(tasks: list[dict[str, Any]], *, blind: bool) -> str:
    task_type = _task_type(tasks)
    if task_type == "pairwise_preference_rating":
        description = "pairwise preference and candidate rating tasks"
        caveat = "Use the pairwise preference field plus 1--5 ratings for each candidate."
    else:
        description = "single-candidate rating tasks"
        caveat = "Use 1--5 integer ratings for each dimension. Do not treat this packet as a preference study; pairwise comparison requires at least two candidates per prompt."
    return f"""# SongDRAME Annotation Packet

This packet contains {len(tasks)} {description} for pilot inspection.

Files:

- `index.html`: browser-friendly listening sheet with audio controls.
- `annotation_tasks.jsonl`: machine-readable task records.
- `annotation_sheet.csv`: spreadsheet-friendly rating sheet.
- `annotation_key.csv`: private task-to-generation key for auditing; do not show it to annotators in blinded studies.
- `packet_summary.json`: compact count summary.

Blind mode: {'enabled; annotator-facing HTML and response CSV hide model and generation identifiers.' if blind else 'disabled; annotator-facing HTML and response CSV include model and generation identifiers.'}

{caveat}
"""


def _summary(tasks: list[dict[str, Any]], *, blind: bool) -> dict[str, Any]:
    genres = sorted({str(task.get("metadata", {}).get("genre", "")) for task in tasks})
    languages = sorted({str(task.get("metadata", {}).get("language", "")) for task in tasks})
    task_type = _task_type(tasks)
    return {
        "tasks": len(tasks),
        "task_type": task_type,
        "genres": genres,
        "languages": languages,
        "rating_dimensions": RATING_DIMENSIONS,
        "blind": blind,
    }


def _task_type(tasks: list[dict[str, Any]]) -> str:
    if not tasks:
        return "empty"
    task_type = str(tasks[0].get("task_type", "single_candidate_rating"))
    if any(task.get("task_type") != task_type for task in tasks):
        raise ValueError("Cannot write mixed task types in one annotation packet")
    return task_type
