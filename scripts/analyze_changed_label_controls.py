#!/usr/bin/env python3
"""Analyze primary control predictions only on changed selected-label records.

This is a post-audit sensitivity analysis, not a new prospective test. The
original scorers and rotations are unchanged; only the held-out evaluation
population is restricted to recipients whose selected-label vector changed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from full_song_eval.musiccaps_gate_calibration import _average_precision_pairs, _difference_interval, _row_key
from full_song_eval.random_ranking_baseline import (
    MUSICCAPS_SEED,
    MTG_JAMENDO_SEED,
    _condition_inputs,
    _normalized_label_set,
    _selected_labels,
    expected_tie_aware_average_precision,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analyze(source: str, manifest: Path, replay: Path, report_path: Path) -> dict[str, Any]:
    if source not in {"MusicCaps", "MTG-Jamendo"}:
        raise ValueError("source must be MusicCaps or MTG-Jamendo")
    rows = _read_jsonl(manifest)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_manifest_hash = (
        report["provenance"]["manifest_sha256"] if source == "MusicCaps"
        else report["dataset"]["manifest_sha256"]
    )
    if _sha256(manifest) != expected_manifest_hash:
        raise ValueError("manifest does not match canonical report")
    group_field, target_field, seed, fold_key, conditions = (
        ("author_id", "aspect_list", MUSICCAPS_SEED, "per_author_metrics", ("label_set_permutation", "full_label_corruption"))
        if source == "MusicCaps" else
        ("artist_fold", "genre_list", MTG_JAMENDO_SEED, "folds", ("label_set_permutation", "full_label_corruption"))
    )
    groups = sorted({str(row[group_field]) for row in rows})
    folds = report[fold_key]
    if len(groups) != 10 or len(folds) != 10:
        raise ValueError("expected ten canonical held-out groups")
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for index, group in enumerate(groups):
        fold = folds[index]
        if str(fold.get(group_field if source == "MusicCaps" else "heldout_artist_fold")) != group:
            raise ValueError(f"fold order mismatch at {group}")
        test = [row for row in rows if str(row[group_field]) == group]
        train = [row for row in rows if str(row[group_field]) != group]
        labels, _ = _selected_labels(
            train, target_field=target_field,
            top_label_count=int(report["design"]["top_label_count"]),
            min_label_count=int(report["design"]["min_label_count"]),
        )
        if len(labels) != 16:
            raise ValueError(f"fold {group} lacks 16 selected labels")
        original_ids = [_row_key(row, i) for i, row in enumerate(test)]
        if len(original_ids) != len(set(original_ids)):
            raise ValueError(f"duplicate record identifiers in {group}")
        original_vectors = [frozenset(_normalized_label_set(row, target_field).intersection(labels)) for row in test]
        for condition in conditions:
            normalized_condition = "permutation_a" if condition == "label_set_permutation" else "permutation_b"
            _, changed_test, _, _, _ = _condition_inputs(
                train, test, source="musiccaps" if source == "MusicCaps" else "mtg_jamendo",
                target_field=target_field, fold_seed=f"{seed}|fold:{index}", condition=normalized_condition,
            )
            changed_vectors = [frozenset(_normalized_label_set(row, target_field).intersection(labels)) for row in changed_test]
            changed_ids = {
                record_id for record_id, before, after in zip(original_ids, original_vectors, changed_vectors)
                if before != after
            }
            if not changed_ids or len(changed_test) != len(test):
                raise ValueError(f"empty or malformed changed subset for {group}/{condition}")
            metadata[(group, condition)] = {
                "ids": original_ids, "labels": labels, "changed_ids": changed_ids,
                "targets": {record_id: vector for record_id, vector in zip(original_ids, changed_vectors)},
                "label_results": [], "all_ap": [], "record_count": len(test),
            }

    current_key: tuple[str, str, str] | None = None
    block: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    def flush() -> None:
        nonlocal block
        if current_key is None:
            return
        group, condition, label = current_key
        info = metadata[(group, condition)]
        if label not in info["labels"] or len(block) != info["record_count"]:
            raise ValueError(f"incomplete replay block: {current_key}")
        ids = [str(item["record_id"]) for item in block]
        if ids != info["ids"]:
            raise ValueError(f"replay record order differs: {current_key}")
        for item in block:
            expected = label in info["targets"][str(item["record_id"])]
            if bool(item["target"]) != expected:
                raise ValueError(f"replay target differs: {current_key}")
        all_pairs = [(float(item["score"]), bool(item["target"])) for item in block]
        if any(target for _, target in all_pairs):
            info["all_ap"].append(_average_precision_pairs(all_pairs))
        subset = [item for item in block if str(item["record_id"]) in info["changed_ids"]]
        pairs = [(float(item["score"]), bool(item["target"])) for item in subset]
        positive_count = sum(target for _, target in pairs)
        if positive_count:
            ties = [count for _, count in sorted(Counter(score for score, _ in pairs).items(), reverse=True)]
            info["label_results"].append({
                "label": label, "n": len(pairs), "k": positive_count,
                "ap": _average_precision_pairs(pairs),
                "prevalence": positive_count / len(pairs),
                "tie_aware": expected_tie_aware_average_precision(n=len(pairs), k=positive_count, tie_group_sizes=ties),
            })
        block = []

    with gzip.open(replay, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("source") != source:
                raise ValueError("replay source mismatch")
            condition = str(item["condition"])
            if condition not in conditions:
                continue
            key = (str(item["heldout_group"]), condition, str(item["label"]))
            if key != current_key:
                flush()
                if key in seen:
                    raise ValueError(f"repeated replay block: {key}")
                seen.add(key)
                current_key = key
            block.append(item)
    flush()
    if len(seen) != 10 * len(conditions) * 16:
        raise ValueError(f"unexpected number of control-label replay blocks: {len(seen)}")

    summaries: list[dict[str, Any]] = []
    intervals: dict[str, Any] = {}
    for condition in conditions:
        prevalence_differences: dict[str, float] = {}
        tie_differences: dict[str, float] = {}
        for index, group in enumerate(groups):
            info = metadata[(group, condition)]
            label_results = info["label_results"]
            if not label_results:
                raise ValueError(f"no supported labels in {group}/{condition}")
            all_ap = sum(info["all_ap"]) / len(info["all_ap"])
            path = ("label_set_permutation",) if condition == "label_set_permutation" else (
                ("label_corruption", "1.0") if source == "MusicCaps" else ("full_label_corruption",)
            )
            node = folds[index]["condition_metrics"]
            for part in path:
                node = node[part]
            canonical_ap = float(node["target_label_masked"]["macro_auprc"])
            if abs(all_ap - canonical_ap) > 1e-4:
                raise ValueError(f"whole-fold replay differs from report: {group}/{condition}")
            ap = sum(entry["ap"] for entry in label_results) / len(label_results)
            prevalence = sum(entry["prevalence"] for entry in label_results) / len(label_results)
            tie = sum(entry["tie_aware"] for entry in label_results) / len(label_results)
            prevalence_differences[group] = ap - prevalence
            tie_differences[group] = ap - tie
            summaries.append({
                "group": group, "condition": condition, "record_count": info["record_count"],
                "changed_record_count": len(info["changed_ids"]),
                "changed_fraction": len(info["changed_ids"]) / info["record_count"],
                "supported_labels": len(label_results), "macro_ap": ap,
                "macro_prevalence": prevalence, "macro_tie_aware": tie,
                "ap_minus_prevalence": ap - prevalence, "ap_minus_tie_aware": ap - tie,
                "whole_fold_ap_replay": all_ap,
            })
        intervals[condition] = {
            "ap_minus_prevalence": _difference_interval(prevalence_differences),
            "ap_minus_tie_aware": _difference_interval(tie_differences),
        }
    return {
        "schema": "fullsongeval-changed-selected-label-control-analysis/v1",
        "status": "complete", "source": source,
        "analysis_status": "post_audit_sensitivity_not_prospective",
        "scope": "Only held-out recipients whose selected-label vector changed; original refitted scorers and rotations are unchanged.",
        "inputs_sha256": {"manifest": _sha256(manifest), "replay": _sha256(replay), "report": _sha256(report_path)},
        "folds": summaries, "intervals": intervals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("MusicCaps", "MTG-Jamendo"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.source, args.manifest, args.replay, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "source": result["source"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
