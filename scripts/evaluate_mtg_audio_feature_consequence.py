#!/usr/bin/env python3
"""Run one frozen artist-fold audio-feature consequence comparison."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from evaluate_independent_genre_target import _norm, _read_annotations
from full_song_eval.musiccaps_gate_calibration import (
    _average_precision_pairs,
    _difference_interval,
    _label_set_permutation,
    _partial_label_set_permutation,
)


SEED = "full-song-eval-mtg-jamendo-metadata-v1"
CONDITIONS = ("observed", "label_set_permutation", "full_label_corruption")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_features(path: Path) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            record_id = str(row["record_id"])
            vector = np.asarray(row["vector"], dtype=np.float64)
            if record_id in features or vector.shape != (80,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"duplicate or invalid feature vector: {record_id}")
            features[record_id] = vector
    return features


def condition_targets(rows: list[dict], fold_index: int) -> dict[str, list[dict]]:
    seed = f"{SEED}|fold:{fold_index}"
    return {
        "observed": rows,
        "label_set_permutation": _label_set_permutation(rows, target_field="genre_list", seed=f"{seed}|permuted-train"),
        "full_label_corruption": _partial_label_set_permutation(rows, target_field="genre_list", level=1.0, seed=f"{seed}|corrupt-train"),
    }


def evaluate_fold(
    config_path: Path,
    manifest_path: Path,
    annotation_path: Path,
    feature_path: Path,
    feature_report_path: Path,
    gate_report_path: Path,
    fold_index: int,
    prediction_path: Path,
) -> dict:
    config = json.loads(config_path.read_text())
    if config["schema"] != "fullsongeval-mtg-audio-feature-consequence/v2":
        raise ValueError("unexpected frozen protocol")
    pinned = config["sources"]
    for source, key in ((manifest_path, "canonical_mtg_manifest_sha256"),
                        (annotation_path, "clean_annotations_sha256"),
                        (gate_report_path, "metadata_gate_report_sha256")):
        if digest(source) != pinned[key]:
            raise ValueError(f"frozen source mismatch: {key}")
    feature_report = json.loads(feature_report_path.read_text())
    if feature_report["status"] != "complete" or digest(feature_path) != feature_report["output_sha256"]:
        raise ValueError("unverified audio feature file")
    if feature_report["inputs_sha256"] != {
        "tar_manifest": pinned["feature_manifest_sha256"],
        "track_manifest": pinned["feature_track_manifest_sha256"],
    }:
        raise ValueError("feature archive manifests differ from frozen protocol")
    gate = json.loads(gate_report_path.read_text())
    if digest(manifest_path) != gate["dataset"]["manifest_sha256"]:
        raise ValueError("canonical MTG manifest mismatch")
    folds = gate["folds"]
    if len(folds) != 10 or not 0 <= fold_index < 10 or folds[fold_index]["fold"] != fold_index:
        raise ValueError("invalid canonical artist fold")
    admission_view = config["admission_and_training"]["admission_view"]
    if admission_view != "target_label_masked":
        raise ValueError("unexpected admission view")
    admission_intervals = {}
    for condition in CONDITIONS:
        differences = {
            item["heldout_artist_fold"]: item["condition_metrics"][condition][admission_view]["macro_auprc"] - item["prior_macro_auprc"]
            for item in folds
        }
        admission_intervals[condition] = _difference_interval(differences)
        if admission_intervals[condition]["ci95_low"] <= 0:
            raise ValueError(f"the target-masked metadata rule did not admit: {condition}")
    fold = folds[fold_index]
    heldout = fold["heldout_artist_fold"]
    labels = list(fold["labels"])
    if len(labels) != 16 or len(set(labels)) != 16:
        raise ValueError("invalid frozen selected-label inventory")
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    train_rows = [row for row in manifest if row["artist_fold"] != heldout]
    test_rows = [row for row in manifest if row["artist_fold"] == heldout]
    if len(train_rows) != fold["train_record_count"] or len(test_rows) != fold["test_record_count"]:
        raise ValueError("fold population mismatch")
    targets = condition_targets(train_rows, fold_index)
    if any([row["row_id"] for row in rows] != [row["row_id"] for row in train_rows] for rows in targets.values()):
        raise ValueError("condition row alignment mismatch")
    for condition in CONDITIONS[1:]:
        for label in labels:
            if sum(label in row["genre_list"] for row in targets[condition]) != sum(label in row["genre_list"] for row in train_rows):
                raise ValueError(f"training prevalence not preserved: {condition}/{label}")
    features = read_features(feature_path)
    train_positions = [index for index, row in enumerate(train_rows) if row["row_id"] in features]
    taxonomies = tuple(config["evaluation"]["human_target_taxonomies"])
    outcomes, _, classes = _read_annotations(annotation_path, set(taxonomies))
    test_with_features = [row for row in test_rows if row["row_id"] in features]
    x_train = np.stack([features[train_rows[index]["row_id"]] for index in train_positions])
    x_test = np.stack([features[row["row_id"]] for row in test_with_features])
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    blocks: list[dict] = []
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for label in labels:
                normalized = _norm(label)
                matched_taxonomies = [taxonomy for taxonomy in taxonomies if normalized in classes[taxonomy]]
                if not matched_taxonomies:
                    continue
                scores: dict[str, np.ndarray] = {}
                for condition in CONDITIONS:
                    y_train = np.asarray([label in targets[condition][index]["genre_list"] for index in train_positions], dtype=np.int8)
                    if np.unique(y_train).size != 2:
                        raise ValueError(f"one-class training target: {fold_index}/{condition}/{label}")
                    model = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=0)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model.fit(x_train, y_train)
                    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
                        raise ValueError(f"model did not converge: {fold_index}/{condition}/{label}")
                    scores[condition] = model.decision_function(x_test)
                for taxonomy in matched_taxonomies:
                    test_indices = [index for index, row in enumerate(test_with_features) if taxonomy in outcomes.get(row["row_id"], {})]
                    y_test = [_norm(outcomes[test_with_features[index]["row_id"]][taxonomy]) == normalized for index in test_indices]
                    n = len(y_test)
                    k = sum(y_test)
                    if k == 0 or k == n:
                        continue
                    ap: dict[str, float] = {}
                    for condition in CONDITIONS:
                        pairs = [(float(scores[condition][index]), target) for index, target in zip(test_indices, y_test)]
                        ap[condition] = _average_precision_pairs(pairs)
                        for index, target in zip(test_indices, y_test):
                            item = {"fold": heldout, "taxonomy": taxonomy, "label": normalized, "condition": condition,
                                    "record_id": test_with_features[index]["row_id"], "target": int(target),
                                    "score": float(scores[condition][index])}
                            compressed.write((json.dumps(item, separators=(",", ":")) + "\n").encode())
                    blocks.append({"fold": heldout, "taxonomy": taxonomy, "label": normalized, "n": n, "k": k, "ap": ap})
    return {
        "schema": "fullsongeval-mtg-audio-feature-consequence-fold/v1",
        "status": "complete",
        "fold_index": fold_index,
        "heldout_artist_fold": heldout,
        "selected_labels": labels,
        "target_masked_admission_intervals": admission_intervals,
        "training_tracks_with_features": len(train_positions),
        "heldout_tracks_with_features": len(test_with_features),
        "heldout_tracks_with_consensus": sum(bool(outcomes.get(row["row_id"])) for row in test_with_features),
        "blocks": blocks,
        "predictions_sha256": digest(prediction_path),
        "inputs_sha256": {"config": digest(config_path), "manifest": digest(manifest_path), "annotations": digest(annotation_path),
                          "features": digest(feature_path), "feature_report": digest(feature_report_path), "gate_report": digest(gate_report_path)},
        "software": {"numpy": np.__version__, "scikit_learn": __import__("sklearn").__version__},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_fold(args.config, args.manifest, args.annotations, args.features, args.feature_report,
                           args.gate_report, args.fold_index, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "fold": result["fold_index"], "blocks": len(result["blocks"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
