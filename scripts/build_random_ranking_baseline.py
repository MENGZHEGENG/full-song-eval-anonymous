#!/usr/bin/env python3
"""Build strict-ranking and tie-aware random-label calibration diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from full_song_eval.random_ranking_baseline import build_random_ranking_baseline


DEFAULT_MUSICCAPS_MANIFEST = Path("data/musiccaps_manifest.jsonl")
DEFAULT_MUSICCAPS_REPORT = Path("reports/musiccaps_gate_calibration.json")
DEFAULT_MTG_REPORT = Path("reports/mtg_jamendo_metadata_calibration.json")
DEFAULT_MTG_POINTER = Path("data/mtg_manifest_dir.txt")
DEFAULT_OUTPUT = Path("reports/random_ranking_baseline_calibration.json")


def resolve_mtg_manifest(pointer_path: Path) -> Path:
    """Resolve the verified temporary manifest directory pointer."""

    try:
        directory_text = pointer_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"unreadable MTG manifest pointer: {type(exc).__name__}") from exc
    if not directory_text:
        raise ValueError("MTG manifest pointer is empty")
    manifest = Path(directory_text) / "mtg_manifest.jsonl"
    if not manifest.is_file():
        raise ValueError(f"verified MTG manifest is missing: {manifest}")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musiccaps-manifest", type=Path, default=DEFAULT_MUSICCAPS_MANIFEST)
    parser.add_argument("--mtg-manifest", type=Path)
    parser.add_argument("--mtg-manifest-pointer", type=Path, default=DEFAULT_MTG_POINTER)
    parser.add_argument("--musiccaps-report", type=Path, default=DEFAULT_MUSICCAPS_REPORT)
    parser.add_argument("--mtg-report", type=Path, default=DEFAULT_MTG_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _reproduction_command(_args: argparse.Namespace) -> str:
    return "PYTHONHASHSEED=0 PYTHONPATH=src python3 scripts/build_random_ranking_baseline.py"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mtg_manifest = args.mtg_manifest or resolve_mtg_manifest(args.mtg_manifest_pointer)
    report = build_random_ranking_baseline(
        musiccaps_manifest_path=args.musiccaps_manifest,
        mtg_manifest_path=mtg_manifest,
        musiccaps_report_path=args.musiccaps_report,
        mtg_report_path=args.mtg_report,
        reproduction_command=_reproduction_command(args),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
