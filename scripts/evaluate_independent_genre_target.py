#!/usr/bin/env python3
"""Evaluate frozen MTG text scores against independent consensus genre labels."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from full_song_eval.musiccaps_gate_calibration import _average_precision_pairs, _difference_interval


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _read_annotations(path: Path, taxonomies: set[str]) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, set[str]]]:
    outcomes: dict[str, dict[str, str]] = {}
    artists: dict[str, str] = {}
    classes: dict[str, set[str]] = {taxonomy: set() for taxonomy in taxonomies}
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:5] != ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION"]:
            raise ValueError("unexpected independent annotation header")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 6:
                raise ValueError("annotation line lacks fields")
            track_id, artist_id = fields[:2]
            if track_id in outcomes:
                raise ValueError(f"duplicate annotation track: {track_id}")
            target: dict[str, str] = {}
            for field in fields[5:]:
                if "---" not in field:
                    continue
                taxonomy, answer_text = field.split("---", 1)
                if taxonomy not in taxonomies:
                    continue
                answers = [value.strip().casefold() for value in answer_text.split(",")]
                if len(answers) != 3 or len(set(answers)) != 1 or not answers[0]:
                    raise ValueError(f"non-consensus entry in clean source: {track_id}/{taxonomy}")
                if taxonomy in target:
                    raise ValueError(f"duplicate taxonomy: {track_id}/{taxonomy}")
                target[taxonomy] = answers[0]
                classes[taxonomy].add(_norm(answers[0]))
            outcomes[track_id] = target
            artists[track_id] = artist_id
    return outcomes, artists, classes


def evaluate(config_path: Path, annotation_path: Path, manifest_path: Path, replay_path: Path, report_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    taxonomies = tuple(config["outcome_taxonomies"])
    conditions = tuple(config["conditions"])
    outcomes, outcome_artists, classes = _read_annotations(annotation_path, set(taxonomies))
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if _sha256(manifest_path) != report["dataset"]["manifest_sha256"]:
        raise ValueError("canonical manifest hash mismatch")
    if not all(report["false_admission"].values()):
        raise ValueError("canonical gate did not admit both invalid controls")
    manifest_by_id = {str(row["row_id"]): row for row in manifest}
    if len(manifest_by_id) != len(manifest):
        raise ValueError("canonical manifest has duplicate tracks")
    joined = 0
    for track_id, artist_id in outcome_artists.items():
        row = manifest_by_id.get(track_id)
        if row is None:
            continue
        joined += 1
        if row["artist_id"] != artist_id:
            raise ValueError(f"artist identifier mismatch: {track_id}")
    if joined < 1000:
        raise ValueError("too few independent annotations join to the canonical manifest")

    results: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    key: tuple[str, str, str] | None = None
    block: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    def flush() -> None:
        nonlocal block
        if key is None:
            return
        group, condition, label = key
        normalized_label = _norm(label)
        for taxonomy in taxonomies:
            if normalized_label not in classes[taxonomy]:
                continue
            pairs: list[tuple[float, bool]] = []
            track_ids: list[str] = []
            for item in block:
                track_id = str(item["record_id"])
                human_label = outcomes.get(track_id, {}).get(taxonomy)
                if human_label is None:
                    continue
                pairs.append((float(item["score"]), _norm(human_label) == normalized_label))
                track_ids.append(track_id)
            positive_count = sum(target for _, target in pairs)
            if 0 < positive_count < len(pairs):
                result_key = (group, taxonomy, normalized_label, condition)
                if result_key in results:
                    raise ValueError(f"duplicate result: {result_key}")
                results[result_key] = {
                    "ap": _average_precision_pairs(pairs),
                    "prevalence": positive_count / len(pairs),
                    "n": len(pairs), "k": positive_count,
                    "record_ids_sha256": hashlib.sha256("\n".join(track_ids).encode()).hexdigest(),
                }
        block = []

    with gzip.open(replay_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("source") != "MTG-Jamendo":
                raise ValueError("score replay source mismatch")
            condition = str(item["condition"])
            if condition not in conditions:
                continue
            next_key = (str(item["heldout_group"]), condition, str(item["label"]))
            if next_key != key:
                flush()
                if next_key in seen:
                    raise ValueError(f"repeated replay block: {next_key}")
                seen.add(next_key)
                key = next_key
            track_id = str(item["record_id"])
            row = manifest_by_id.get(track_id)
            if row is None or row["artist_fold"] != next_key[0]:
                raise ValueError(f"replay track/fold mismatch: {track_id}")
            block.append(item)
    flush()
    if len(seen) != 10 * len(conditions) * 16:
        raise ValueError(f"unexpected number of replay blocks: {len(seen)}")

    groups = sorted({str(row["artist_fold"]) for row in manifest})
    comparisons: list[dict[str, Any]] = []
    fold_scores: dict[str, dict[str, float]] = {}
    for group in groups:
        taxonomy_scores: dict[str, dict[str, float]] = defaultdict(dict)
        for taxonomy in taxonomies:
            labels = sorted({label for g, t, label, _ in results if g == group and t == taxonomy})
            common_labels = [
                label for label in labels
                if all((group, taxonomy, label, condition) in results for condition in conditions)
            ]
            if not common_labels:
                continue
            for label in common_labels:
                items = [results[(group, taxonomy, label, condition)] for condition in conditions]
                if len({(item["n"], item["k"], item["record_ids_sha256"]) for item in items}) != 1:
                    raise ValueError(f"conditions have unmatched human outcome sets: {group}/{taxonomy}/{label}")
                comparisons.append({
                    "group": group, "taxonomy": taxonomy, "label": label,
                    "n": items[0]["n"], "k": items[0]["k"],
                    "prevalence": items[0]["prevalence"],
                    "ap": {condition: item["ap"] for condition, item in zip(conditions, items)},
                })
            for condition in conditions:
                taxonomy_scores[taxonomy][condition] = sum(
                    results[(group, taxonomy, label, condition)]["ap"] for label in common_labels
                ) / len(common_labels)
        if not taxonomy_scores:
            raise ValueError(f"no supported consensus-genre comparison in {group}")
        fold_scores[group] = {
            condition: sum(values[condition] for values in taxonomy_scores.values()) / len(taxonomy_scores)
            for condition in conditions
        }
    contrasts: dict[str, Any] = {}
    for condition in conditions[1:]:
        contrasts[f"observed_minus_{condition}"] = _difference_interval({
            group: values["observed"] - values[condition] for group, values in fold_scores.items()
        })
    sensitivity_blocks = [item for item in comparisons if item["k"] >= 5]
    sensitivity_fold_scores: dict[str, dict[str, float]] = {}
    for group in groups:
        taxonomy_scores = []
        for taxonomy in taxonomies:
            items = [
                item for item in sensitivity_blocks
                if item["group"] == group and item["taxonomy"] == taxonomy
            ]
            if items:
                taxonomy_scores.append({
                    condition: sum(item["ap"][condition] for item in items) / len(items)
                    for condition in conditions
                })
        if len(taxonomy_scores) == 0:
            raise ValueError(f"five-positive sensitivity has no supported taxonomy in {group}")
        sensitivity_fold_scores[group] = {
            condition: sum(item[condition] for item in taxonomy_scores) / len(taxonomy_scores)
            for condition in conditions
        }
    sensitivity_contrasts = {
        f"observed_minus_{condition}": _difference_interval({
            group: values["observed"] - values[condition]
            for group, values in sensitivity_fold_scores.items()
        })
        for condition in conditions[1:]
    }
    return {
        "schema": "fullsongeval-independent-consensus-genre-target/v1",
        "status": "complete",
        "analysis_status": "retrospective_independent_human_target_same_track_source",
        "scope": "Frozen text scorers predict independently collected consensus genre labels on the same MTG track collection; no audio waveform is processed.",
        "inputs_sha256": {
            "config": _sha256(config_path), "annotations": _sha256(annotation_path),
            "manifest": _sha256(manifest_path), "replay": _sha256(replay_path), "report": _sha256(report_path),
        },
        "canonical_gate_false_admission": report["false_admission"],
        "annotation_track_count": len(outcomes), "joined_track_count": joined,
        "taxonomy_consensus_counts": {taxonomy: sum(taxonomy in value for value in outcomes.values()) for taxonomy in taxonomies},
        "class_values": {taxonomy: sorted(values) for taxonomy, values in classes.items()},
        "eligible_block_count": len(comparisons), "blocks": comparisons,
        "fold_scores": fold_scores, "paired_contrasts": contrasts,
        "post_audit_five_positive_sensitivity": {
            "minimum_positive_annotations_per_block": 5,
            "eligible_block_count": len(sensitivity_blocks),
            "fold_scores": sensitivity_fold_scores,
            "paired_contrasts": sensitivity_contrasts,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.config, args.annotations, args.manifest, args.replay, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "joined_tracks": result["joined_track_count"], "eligible_blocks": result["eligible_block_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
