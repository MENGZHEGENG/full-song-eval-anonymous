#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from full_song_eval.musiccaps_gate_calibration import _difference_interval


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _summary(
    *,
    source: str,
    condition: str,
    condition_values: dict[str, float],
    reference_values: dict[str, float],
    reference_condition: str,
) -> dict[str, Any]:
    differences = {
        group: condition_values[group] - reference_values[group]
        for group in sorted(condition_values)
    }
    return {
        "source": source,
        "condition": condition,
        "reference_condition": reference_condition,
        "text_view": "target_label_masked",
        "condition_macro_auprc_mean": round(_mean(list(condition_values.values())), 6),
        "reference_macro_auprc_mean": round(_mean(list(reference_values.values())), 6),
        "paired_summary": _difference_interval(differences),
        "positive_group_differences": sum(value > 0 for value in differences.values()),
        "group_count": len(differences),
    }


def build_summary(musiccaps_path: Path, mtg_path: Path) -> dict[str, Any]:
    musiccaps = json.loads(musiccaps_path.read_text(encoding="utf-8"))
    mtg = json.loads(mtg_path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = []

    musiccaps_rows = musiccaps["per_author_metrics"]
    musiccaps_prior = {
        str(row["author_id"]): float(row["prior_macro_auprc"])
        for row in musiccaps_rows
    }
    musiccaps_observed = {
        str(row["author_id"]): float(row["condition_metrics"]["observed"]["target_label_masked"]["macro_auprc"])
        for row in musiccaps_rows
    }
    musiccaps_conditions = {
        "Observed metadata": musiccaps_observed,
        "Paraphrase addition": {
            str(row["author_id"]): float(row["condition_metrics"]["paraphrase_evidence_injection"]["1.0"]["target_label_masked"]["macro_auprc"])
            for row in musiccaps_rows
        },
        "Label permutation": {
            str(row["author_id"]): float(row["condition_metrics"]["label_set_permutation"]["target_label_masked"]["macro_auprc"])
            for row in musiccaps_rows
        },
        "Full corruption": {
            str(row["author_id"]): float(row["condition_metrics"]["label_corruption"]["1.0"]["target_label_masked"]["macro_auprc"])
            for row in musiccaps_rows
        },
    }
    for condition, values in musiccaps_conditions.items():
        reference = musiccaps_observed if condition == "Paraphrase addition" else musiccaps_prior
        reference_condition = "Observed metadata" if condition == "Paraphrase addition" else "Frequency baseline"
        entries.append(
            _summary(
                source="MusicCaps",
                condition=condition,
                condition_values=values,
                reference_values=reference,
                reference_condition=reference_condition,
            )
        )
    musiccaps_matched_deletion = {
        str(row["author_id"]): float(row["condition_metrics"]["observed"]["target_matched_random_deletion"]["macro_auprc"])
        for row in musiccaps_rows
    }
    entries.append(
        _summary(
            source="MusicCaps",
            condition="Target masking",
            condition_values=musiccaps_observed,
            reference_values=musiccaps_matched_deletion,
            reference_condition="Matched deletion",
        )
    )

    mtg_rows = mtg["folds"]
    mtg_prior = {
        str(row["heldout_artist_fold"]): float(row["prior_macro_auprc"])
        for row in mtg_rows
    }
    mtg_observed = {
        str(row["heldout_artist_fold"]): float(row["condition_metrics"]["observed"]["target_label_masked"]["macro_auprc"])
        for row in mtg_rows
    }
    mtg_conditions = {
        "Observed metadata": mtg_observed,
        "Codeword addition": {
            str(row["heldout_artist_fold"]): float(row["condition_metrics"]["codeword_injection"]["target_label_masked"]["macro_auprc"])
            for row in mtg_rows
        },
        "Label permutation": {
            str(row["heldout_artist_fold"]): float(row["condition_metrics"]["label_set_permutation"]["target_label_masked"]["macro_auprc"])
            for row in mtg_rows
        },
        "Full corruption": {
            str(row["heldout_artist_fold"]): float(row["condition_metrics"]["full_label_corruption"]["target_label_masked"]["macro_auprc"])
            for row in mtg_rows
        },
    }
    for condition, values in mtg_conditions.items():
        reference = mtg_observed if condition == "Codeword addition" else mtg_prior
        reference_condition = "Observed metadata" if condition == "Codeword addition" else "Frequency baseline"
        entries.append(
            _summary(
                source="MTG-Jamendo",
                condition=condition,
                condition_values=values,
                reference_values=reference,
                reference_condition=reference_condition,
            )
        )
    mtg_matched_deletion = {
        str(row["heldout_artist_fold"]): float(row["condition_metrics"]["observed"]["target_matched_random_deletion"]["macro_auprc"])
        for row in mtg_rows
    }
    entries.append(
        _summary(
            source="MTG-Jamendo",
            condition="Target masking",
            condition_values=mtg_observed,
            reference_values=mtg_matched_deletion,
            reference_condition="Matched deletion",
        )
    )

    return {
        "schema_version": 1,
        "status": "complete",
        "scope": "Target-masked table values and matched-deletion comparisons used in the current manuscript.",
        "inference_note": "Student-t summaries describe ten fixed held-out group differences; training sets overlap.",
        "inputs": {
            "musiccaps": {"path": str(musiccaps_path), "sha256": _sha256(musiccaps_path)},
            "mtg_jamendo": {"path": str(mtg_path), "sha256": _sha256(mtg_path)},
        },
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--musiccaps", type=Path, required=True)
    parser.add_argument("--mtg-jamendo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_summary(args.musiccaps, args.mtg_jamendo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"], "entries": len(report["entries"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
