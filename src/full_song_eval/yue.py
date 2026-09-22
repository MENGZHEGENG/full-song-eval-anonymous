from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

DEFAULT_STAGE1_MODEL = "m-a-p/YuE-s1-7B-anneal-en-cot"
DEFAULT_STAGE2_MODEL = "m-a-p/YuE-s2-1B-general"


def yue_genre_text(record: dict[str, Any]) -> str:
    genre = str(record.get("genre") or record.get("metadata", {}).get("genre") or "pop").replace("_", " ")
    vocal_style = str(record.get("vocal_style", "expressive lead vocal with natural phrasing"))
    instrumentation = str(record.get("instrumentation", "genre-appropriate accompaniment"))
    language = str(record.get("language") or record.get("metadata", {}).get("language") or "en")
    return f"{genre}, {language} lyrics, {vocal_style}, {instrumentation}, clean mix, coherent song structure"


def write_yue_prompt_files(record: dict[str, Any], prompt_dir: Path) -> tuple[Path, Path]:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_id = str(record["prompt_id"])
    genre_path = prompt_dir / f"{prompt_id}_genre.txt"
    lyrics_path = prompt_dir / f"{prompt_id}_lyrics.txt"
    genre_path.write_text(yue_genre_text(record) + "\n", encoding="utf-8")
    lyrics_path.write_text(str(record.get("lyrics", "")).strip() + "\n", encoding="utf-8")
    return genre_path, lyrics_path


def yue_inference_command(
    *,
    genre_path: Path,
    lyrics_path: Path,
    output_dir: Path,
    python_executable: str = "python",
    stage1_model: str = DEFAULT_STAGE1_MODEL,
    stage2_model: str = DEFAULT_STAGE2_MODEL,
    run_n_segments: int = 2,
    stage2_batch_size: int = 4,
    max_new_tokens: int | None = None,
    seed: int | None = None,
) -> list[str]:
    if run_n_segments < 1:
        raise ValueError("run_n_segments must be positive")
    if stage2_batch_size < 1:
        raise ValueError("stage2_batch_size must be positive")
    command = [
        python_executable,
        "infer.py",
        "--stage1_model",
        stage1_model,
        "--stage2_model",
        stage2_model,
        "--genre_txt",
        str(genre_path),
        "--lyrics_txt",
        str(lyrics_path),
        "--run_n_segments",
        str(run_n_segments),
        "--stage2_batch_size",
        str(stage2_batch_size),
        "--output_dir",
        str(output_dir),
    ]
    if max_new_tokens is not None:
        command.extend(["--max_new_tokens", str(max_new_tokens)])
    if seed is not None:
        command.extend(["--seed", str(seed)])
    return command


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)
