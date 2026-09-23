#!/usr/bin/env python3
"""Test the frozen reference comparison on new human-written music captions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

from full_song_eval.calibration_prediction_replay import write_mtg_jamendo_prediction_replay
from full_song_eval.mtg_jamendo_metadata_calibration import _fold_metrics
from full_song_eval.musiccaps_gate_calibration import _average_precision_pairs, _difference_interval, _prior_macro_auprc
from full_song_eval.random_ranking_baseline import expected_tie_aware_average_precision


CONDITIONS = ("observed", "label_set_permutation", "full_label_corruption")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(caption_path: Path, mtg_path: Path, output_path: Path) -> list[dict]:
    canonical = {row["row_id"]: row for row in (json.loads(line) for line in mtg_path.read_text().splitlines())}
    selected: dict[str, dict] = {}
    with caption_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["is_valid_subset"] != "True":
                continue
            track_id = f"track_{int(row['track_id']):07d}"
            if track_id not in canonical:
                continue
            if track_id not in selected or row["caption_id"] < selected[track_id]["caption_id"]:
                selected[track_id] = row
    manifest = []
    for track_id in sorted(selected):
        base = canonical[track_id]
        manifest.append({"row_id": track_id, "artist_id": base["artist_id"], "artist_fold": base["artist_fold"],
                         "caption": selected[track_id]["caption"], "genre_list": base["genre_list"]})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest))
    return manifest


def tie_sizes(scores: list[float]) -> list[int]:
    counts = Counter(scores)
    return [counts[score] for score in sorted(counts, reverse=True)]


def evaluate(config_path: Path, caption_path: Path, mtg_path: Path, manifest_path: Path, replay_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    if config["schema"] != "fullsongeval-song-describer-reference-check/v1":
        raise ValueError("unexpected frozen protocol")
    if digest(caption_path) != config["source"]["caption_csv_sha256"] or digest(mtg_path) != config["source"]["canonical_mtg_manifest_sha256"]:
        raise ValueError("frozen source checksum mismatch")
    rows = build_manifest(caption_path, mtg_path, manifest_path)
    groups = sorted({row["artist_fold"] for row in rows})
    if len(groups) != 10:
        raise ValueError("expected ten artist folds")
    top_count = config["protocol"]["top_label_count"]
    min_count = config["protocol"]["min_training_label_count"]
    seed = config["protocol"]["control_seed"]
    source_fold_metrics = {}
    fold_labels = {}
    for index, heldout in enumerate(groups):
        train = [row for row in rows if row["artist_fold"] != heldout]
        test = [row for row in rows if row["artist_fold"] == heldout]
        counts = Counter(label for row in train for label in row["genre_list"])
        labels = [label for label, count in sorted(counts.items(), key=lambda value: (-value[1], value[0])) if count >= min_count][:top_count]
        if len(labels) != top_count:
            raise ValueError(f"fold {heldout} lacks five supported training labels")
        metrics = _fold_metrics(train, test, labels=labels, seed=f"{seed}|fold:{index}")
        prior = _prior_macro_auprc(train, test, labels=labels, target_field="genre_list")
        source_fold_metrics[heldout] = {condition: metrics[condition]["target_label_masked"]["macro_auprc"] for condition in CONDITIONS}
        source_fold_metrics[heldout]["prior"] = prior
        fold_labels[heldout] = labels
    replay_receipt = write_mtg_jamendo_prediction_replay(manifest_path=manifest_path, output_path=replay_path,
                                                         top_label_count=top_count, min_label_count=min_count, seed=seed)
    if replay_receipt["manifest_sha256"] != digest(manifest_path) or replay_receipt["primary_view"] != "target_label_masked":
        raise ValueError("prediction replay source mismatch")
    pairs: dict[tuple[str, str, str], list[tuple[float, bool]]] = defaultdict(list)
    with gzip.open(replay_path, "rt") as handle:
        for line in handle:
            item = json.loads(line)
            if item["condition"] in CONDITIONS:
                key = (item["heldout_group"], item["condition"], item["label"])
                pairs[key].append((float(item["score"]), bool(item["target"])))
    comparison = {}
    supports = {}
    for heldout in groups:
        comparison[heldout] = {}
        supports[heldout] = {}
        for condition in CONDITIONS:
            actual = []
            reference = []
            label_support = {}
            for label in fold_labels[heldout]:
                values = pairs[(heldout, condition, label)]
                n = len(values)
                k = sum(target for _, target in values)
                if k == 0:
                    continue
                actual.append(_average_precision_pairs(values))
                reference.append(expected_tie_aware_average_precision(n=n, k=k, tie_group_sizes=tie_sizes([score for score, _ in values])))
                label_support[label] = {"n": n, "k": k}
            if not actual:
                raise ValueError(f"no positive held-out labels: {heldout}/{condition}")
            macro_actual = sum(actual) / len(actual)
            if abs(macro_actual - source_fold_metrics[heldout][condition]) > 5.1e-7:
                raise ValueError(f"replay disagrees with scorer report: {heldout}/{condition}")
            comparison[heldout][condition] = {"ap": macro_actual, "tie_aware_reference": sum(reference) / len(reference),
                                            "prevalence_reference": source_fold_metrics[heldout]["prior"]}
            supports[heldout][condition] = label_support
        if supports[heldout]["observed"] != supports[heldout]["label_set_permutation"] or supports[heldout]["observed"] != supports[heldout]["full_label_corruption"]:
            raise ValueError(f"control label support differs: {heldout}")
    intervals = {}
    for condition in CONDITIONS:
        intervals[condition] = {}
        for reference in ("prevalence_reference", "tie_aware_reference"):
            values = {group: comparison[group][condition]["ap"] - comparison[group][condition][reference] for group in groups}
            interval = _difference_interval(values)
            intervals[condition][reference] = {"summary": interval, "accepts": interval["ci95_low"] > 0}
    return {
        "schema": "fullsongeval-song-describer-reference-check/v1",
        "status": "complete",
        "scope": config["interpretation"],
        "inputs_sha256": {"config": digest(config_path), "captions": digest(caption_path), "mtg_manifest": digest(mtg_path),
                          "study_manifest": digest(manifest_path), "prediction_replay": digest(replay_path)},
        "caption_track_count": len(rows),
        "fold_labels": fold_labels,
        "fold_support": supports,
        "fold_comparisons": comparison,
        "decision_intervals": intervals,
        "replay_receipt": replay_receipt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--mtg-manifest", type=Path, required=True)
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.config, args.captions, args.mtg_manifest, args.study_manifest, args.replay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "tracks": result["caption_track_count"],
                      "decisions": {condition: {reference: value["accepts"] for reference, value in comparisons.items()}
                                    for condition, comparisons in result["decision_intervals"].items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
