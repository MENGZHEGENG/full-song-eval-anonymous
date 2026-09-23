#!/usr/bin/env python3
"""Aggregate ten verified audio-feature folds under the frozen support rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from full_song_eval.musiccaps_gate_calibration import _difference_interval


CONDITIONS = ("observed", "label_set_permutation", "full_label_corruption")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(blocks: list[dict], groups: list[str]) -> dict:
    fold_scores: dict[str, dict[str, float]] = {}
    support_by_fold: dict[str, dict[str, int]] = {}
    for group in groups:
        taxonomy_scores: dict[str, dict[str, float]] = {}
        for taxonomy in sorted({block["taxonomy"] for block in blocks if block["fold"] == group}):
            items = [block for block in blocks if block["fold"] == group and block["taxonomy"] == taxonomy]
            if not items:
                continue
            taxonomy_scores[taxonomy] = {
                condition: sum(item["ap"][condition] for item in items) / len(items)
                for condition in CONDITIONS
            }
        if not taxonomy_scores:
            return {"status": "infeasible_missing_supported_fold", "missing_fold": group}
        fold_scores[group] = {
            condition: sum(value[condition] for value in taxonomy_scores.values()) / len(taxonomy_scores)
            for condition in CONDITIONS
        }
        support_by_fold[group] = {taxonomy: sum(item["fold"] == group and item["taxonomy"] == taxonomy for item in blocks)
                                  for taxonomy in taxonomy_scores}
    contrasts = {
        f"observed_minus_{condition}": _difference_interval({
            group: scores["observed"] - scores[condition] for group, scores in fold_scores.items()
        })
        for condition in CONDITIONS[1:]
    }
    return {"status": "complete", "block_count": len(blocks), "fold_scores": fold_scores,
            "support_by_fold": support_by_fold, "paired_contrasts": contrasts}


def aggregate(config_path: Path, fold_dir: Path) -> dict:
    config = json.loads(config_path.read_text())
    if config["schema"] != "fullsongeval-mtg-audio-feature-consequence/v2":
        raise ValueError("unexpected protocol")
    reports = []
    paths = [fold_dir / f"fold_{index:02d}.json" for index in range(10)]
    for index, path in enumerate(paths):
        value = json.loads(path.read_text())
        if value["schema"] != "fullsongeval-mtg-audio-feature-consequence-fold/v1" or value["status"] != "complete" or value["fold_index"] != index:
            raise ValueError(f"invalid fold report: {path}")
        if value["inputs_sha256"]["config"] != digest(config_path):
            raise ValueError("fold used a different protocol")
        prediction_path = fold_dir / f"predictions_{index:02d}.jsonl.gz"
        if digest(prediction_path) != value["predictions_sha256"]:
            raise ValueError(f"prediction replay mismatch: {prediction_path}")
        reports.append(value)
    for key in ("manifest", "annotations", "features", "feature_report", "gate_report"):
        if len({value["inputs_sha256"][key] for value in reports}) != 1:
            raise ValueError(f"fold input differs: {key}")
    if any(value["target_masked_admission_intervals"] != reports[0]["target_masked_admission_intervals"] for value in reports):
        raise ValueError("target-masked admission receipt differs across folds")
    groups = [value["heldout_artist_fold"] for value in reports]
    if len(set(groups)) != 10:
        raise ValueError("repeated held-out artist fold")
    blocks = [block for value in reports for block in value["blocks"]]
    for block in blocks:
        if set(block["ap"]) != set(CONDITIONS) or not 0 < block["k"] < block["n"]:
            raise ValueError("invalid paired block")
    primary = [block for block in blocks if block["k"] >= 5]
    secondary = blocks
    main = summarize(primary, groups)
    return {
        "schema": "fullsongeval-mtg-audio-feature-consequence/v2",
        "status": main["status"],
        "scope": config["interpretation"],
        "protocol_sha256": digest(config_path),
        "target_masked_admission_intervals": reports[0]["target_masked_admission_intervals"],
        "fold_reports_sha256": {str(path): digest(path) for path in paths},
        "shared_inputs_sha256": {key: reports[0]["inputs_sha256"][key] for key in ("manifest", "annotations", "features", "feature_report", "gate_report")},
        "feature_coverage": {value["heldout_artist_fold"]: {
            "training_tracks": value["training_tracks_with_features"],
            "heldout_tracks": value["heldout_tracks_with_features"],
            "heldout_consensus_tracks": value["heldout_tracks_with_consensus"],
        } for value in reports},
        "all_supported_block_count": len(blocks),
        "primary_five_positive": main,
        "secondary_any_positive": summarize(secondary, groups),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.config, args.fold_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "all_supported_blocks": result["all_supported_block_count"],
                      "primary_blocks": result["primary_five_positive"].get("block_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
