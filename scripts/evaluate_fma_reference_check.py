#!/usr/bin/env python3
"""Run the frozen new-collection FMA text/genre reference check."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

from full_song_eval.calibration_prediction_replay import _scenario_records
from full_song_eval.musiccaps_gate_calibration import (
    _average_precision_pairs,
    _difference_interval,
    _label_set_permutation,
    _partial_label_set_permutation,
)
from full_song_eval.random_ranking_baseline import expected_tie_aware_average_precision


CONDITIONS = ("observed", "label_set_permutation", "full_label_corruption")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source(archive: Path, config: dict, manifest_path: Path) -> tuple[list[dict], dict]:
    expected = config["source"]["metadata_sha1"]
    sha1 = hashlib.sha1()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha1.update(chunk)
    if sha1.hexdigest() != expected:
        raise ValueError("official FMA metadata SHA-1 mismatch")
    with zipfile.ZipFile(archive) as source:
        with io.TextIOWrapper(source.open("fma_metadata/genres.csv"), encoding="utf-8", newline="") as handle:
            genres = {row["genre_id"]: row["title"].strip().lower() for row in csv.DictReader(handle)}
        genre_titles = set(genres.values())
        with io.TextIOWrapper(source.open("fma_metadata/tracks.csv"), encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            group_header, field_header, _index_header = next(reader), next(reader), next(reader)
            columns = {(group, field): idx for idx, (group, field) in enumerate(zip(group_header, field_header))}
            needed = (("set", "subset"), ("track", "title"), ("artist", "name"),
                      ("album", "title"), ("artist", "id"), ("track", "genre_top"))
            if any(key not in columns for key in needed):
                raise ValueError(f"FMA schema lacks fields: {[key for key in needed if key not in columns]}")
            def field(row: list[str], group: str, name: str) -> str:
                return row[columns[(group, name)]].strip()
            rows = []
            exclusions = Counter()
            for raw in reader:
                if field(raw, "set", "subset") != config["source"]["subset"]:
                    continue
                artist_id = field(raw, "artist", "id")
                genre_id = field(raw, "track", "genre_top")
                parts = [field(raw, *key.split(".", 1)) for key in config["protocol"]["input_fields"]]
                genre = genres.get(genre_id, genre_id.lower())
                if not artist_id or not genre_id or genre not in genre_titles or not any(parts):
                    exclusions["missing_required_field"] += 1
                    continue
                track_id = raw[0].strip()
                seed = config["protocol"]["seed"]
                fold = int(hashlib.sha256(f"{seed}|{artist_id}".encode()).hexdigest(), 16) % 10
                rows.append({"row_id": track_id, "artist_id": artist_id,
                             "artist_fold": f"artist_fold_{fold:02d}",
                             "caption": " ".join(part for part in parts if part),
                             "genre_list": [genre]})
    rows.sort(key=lambda row: int(row["row_id"]))
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate FMA track IDs")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    provenance = {"official_sha1": sha1.hexdigest(), "source_sha256": sha256(archive),
                  "manifest_sha256": sha256(manifest_path), "record_count": len(rows),
                  "artist_count": len({row["artist_id"] for row in rows}),
                  "genre_counts": dict(sorted(Counter(row["genre_list"][0] for row in rows).items())),
                  "exclusions": dict(exclusions)}
    return rows, provenance


def evaluate(config_path: Path, archive: Path, manifest_path: Path, replay_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    if config["schema"] != "fullsongeval-fma-reference-check/v1":
        raise ValueError("unexpected protocol")
    rows, provenance = read_source(archive, config, manifest_path)
    groups = sorted({row["artist_fold"] for row in rows})
    if len(groups) != config["protocol"]["fold_count"]:
        raise ValueError("FMA subset lacks ten folds")
    fold_results = {}
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    with replay_path.open("wb") as raw_out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_out, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as output:
                for fold_index, heldout in enumerate(groups):
                    train = [row for row in rows if row["artist_fold"] != heldout]
                    test = [row for row in rows if row["artist_fold"] == heldout]
                    counts = Counter(row["genre_list"][0] for row in train)
                    labels = [label for label, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
                              if count >= config["protocol"]["min_training_label_count"]][:config["protocol"]["top_label_count"]]
                    if len(labels) != config["protocol"]["top_label_count"]:
                        raise ValueError(f"fold {heldout} lacks frozen label count")
                    seed = f"{config['protocol']['seed']}|fold:{fold_index}"
                    scenarios = (
                        ("observed", train, test, f"{seed}|observed"),
                        ("label_set_permutation",
                         _label_set_permutation(train, target_field="genre_list", seed=f"{seed}|permuted-train"),
                         _label_set_permutation(test, target_field="genre_list", seed=f"{seed}|permuted-test"),
                         f"{seed}|permuted"),
                        ("full_label_corruption",
                         _partial_label_set_permutation(train, target_field="genre_list", level=1.0, seed=f"{seed}|corrupt-train"),
                         _partial_label_set_permutation(test, target_field="genre_list", level=1.0, seed=f"{seed}|corrupt-test"),
                         f"{seed}|corrupt"),
                    )
                    fold_results[heldout] = {"train_count": len(train), "test_count": len(test),
                                             "selected_labels": labels, "conditions": {}}
                    for condition, scenario_train, scenario_test, scenario_seed in scenarios:
                        pairs = defaultdict(list)
                        for item in _scenario_records(scenario_train, scenario_test, labels=labels,
                                                      target_field="genre_list", seed=scenario_seed,
                                                      source="FMA", heldout_group=heldout,
                                                      condition=condition):
                            output.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
                            pairs[item["label"]].append((float(item["score"]), bool(item["target"])))
                        actual, prevalence, tie_aware, support = [], [], [], {}
                        for label in labels:
                            values = pairs[label]
                            n = len(values)
                            k = sum(target for _, target in values)
                            support[label] = {"n": n, "k": k}
                            if k == 0:
                                continue
                            actual.append(_average_precision_pairs(values))
                            prevalence.append(k / n)
                            ties = Counter(score for score, _ in values)
                            sizes = [ties[score] for score in sorted(ties, reverse=True)]
                            tie_aware.append(expected_tie_aware_average_precision(n=n, k=k, tie_group_sizes=sizes))
                        if not actual:
                            raise ValueError(f"fold {heldout}/{condition} has no supported label")
                        fold_results[heldout]["conditions"][condition] = {
                            "ap": sum(actual) / len(actual),
                            "prevalence_reference": sum(prevalence) / len(prevalence),
                            "tie_aware_reference": sum(tie_aware) / len(tie_aware),
                            "support": support,
                        }
                    supports = [fold_results[heldout]["conditions"][condition]["support"] for condition in CONDITIONS]
                    if supports[0] != supports[1] or supports[0] != supports[2]:
                        raise ValueError(f"control support differs in {heldout}")
    intervals = {}
    for condition in CONDITIONS:
        intervals[condition] = {}
        for reference in ("prevalence_reference", "tie_aware_reference"):
            differences = {group: fold_results[group]["conditions"][condition]["ap"] -
                           fold_results[group]["conditions"][condition][reference] for group in groups}
            summary = _difference_interval(differences)
            intervals[condition][reference] = {"summary": summary, "accepts": summary["ci95_low"] > 0}
    return {"schema": config["schema"], "status": "complete", "scope": config["interpretation"],
            "protocol_sha256": sha256(config_path), "source": provenance,
            "manifest_sha256": sha256(manifest_path), "replay_sha256": sha256(replay_path),
            "folds": fold_results, "decision_intervals": intervals}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "archive", "manifest", "replay", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.config, args.archive, args.manifest, args.replay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "records": report["source"]["record_count"],
                      "decisions": {condition: {ref: value["accepts"] for ref, value in refs.items()}
                                    for condition, refs in report["decision_intervals"].items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
