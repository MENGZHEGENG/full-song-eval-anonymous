"""Independent metadata-only calibration on MTG-Jamendo genre tags.

This module turns two small, official MTG-Jamendo metadata files into a
provenance-carrying manifest and reruns the admission-rule falsification used
in the MusicCaps study.  It intentionally does not download or consume audio.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from full_song_eval.musiccaps_gate_calibration import (
    _difference_interval,
    _evaluate_scenario,
    _label_set_permutation,
    _partial_label_set_permutation,
    _prior_macro_auprc,
    _score_interval,
    _sha256,
)


FOLD_COUNT = 10
SOURCE_NAME = "MTG-Jamendo"
SOURCE_URL = "https://github.com/MTG/mtg-jamendo-dataset"
METADATA_URL = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/raw.meta.tsv"
TAGS_URL = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/raw_30s_cleantags_50artists.tsv"


def _stable_fold(artist_id: str, *, seed: str) -> str:
    digest = hashlib.sha256(f"{seed}|{artist_id}".encode("utf-8")).hexdigest()
    return f"artist_fold_{int(digest, 16) % FOLD_COUNT:02d}"


def _read_tsv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"missing TSV columns: {','.join(missing)}")
        return [
            {
                key: value.strip() if isinstance(value, str) else ""
                for key, value in record.items()
                if isinstance(key, str)
            }
            for record in reader
        ]


def build_mtg_jamendo_manifest(
    *,
    metadata_tsv: Path,
    tags_tsv: Path,
    output_manifest: Path,
    seed: str,
    source_revision: str,
) -> dict[str, Any]:
    """Join official metadata and genre tags into a deterministic JSONL manifest."""

    metadata = {
        item["TRACK_ID"]: item
        for item in _read_tsv(
            metadata_tsv,
            {"TRACK_ID", "ARTIST_ID", "ALBUM_ID", "TRACK_NAME", "ARTIST_NAME", "ALBUM_NAME"},
        )
    }
    tags = _read_tsv(tags_tsv, {"TRACK_ID", "ARTIST_ID", "ALBUM_ID", "TAGS"})
    records: list[dict[str, Any]] = []
    missing_metadata = 0
    for item in tags:
        meta = metadata.get(item["TRACK_ID"])
        if meta is None:
            missing_metadata += 1
            continue
        tag_parts = item["TAGS"].split("---")
        labels = sorted(
            {
                label.replace("_", " ").strip().lower()
                for category, label in zip(tag_parts[0::2], tag_parts[1::2])
                if category == "genre" and label.strip()
            }
        )
        if not labels:
            continue
        artist_id = item["ARTIST_ID"]
        caption = " ".join(
            value
            for value in (meta["TRACK_NAME"], meta["ARTIST_NAME"], meta["ALBUM_NAME"])
            if value
        )
        records.append(
            {
                "row_id": item["TRACK_ID"],
                "artist_id": artist_id,
                "artist_fold": _stable_fold(artist_id, seed=seed),
                "caption": caption,
                "genre_list": labels,
            }
        )
    records.sort(key=lambda record: str(record["row_id"]))
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    return {
        "source_name": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "metadata_url": METADATA_URL,
        "tags_url": TAGS_URL,
        "source_revision": source_revision,
        "metadata_sha256": _sha256(metadata_tsv),
        "tags_sha256": _sha256(tags_tsv),
        "manifest_sha256": _sha256(output_manifest),
        "record_count": len(records),
        "missing_metadata_count": missing_metadata,
        "audio_used": False,
        "license_scope": "Official metadata is CC BY-NC-SA 4.0; this calibration uses metadata only.",
    }


def _codeword_injection(rows: Sequence[dict[str, Any]], *, labels: Sequence[str], seed: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        changed = dict(row)
        positive = set(str(value) for value in row.get("genre_list", []))
        tokens = [
            "signal" + hashlib.sha256(f"{seed}|{label}".encode("utf-8")).hexdigest()[:14]
            for label in labels
            if label in positive
        ]
        changed["caption"] = " ".join([str(row.get("caption", "")).strip(), *tokens]).strip()
        result.append(changed)
    return result


def _fold_metrics(
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    *,
    labels: Sequence[str],
    seed: str,
) -> dict[str, Any]:
    observed = _evaluate_scenario(
        train_rows, test_rows, labels=labels, target_field="genre_list", seed=f"{seed}|observed"
    )
    permuted = _evaluate_scenario(
        _label_set_permutation(train_rows, target_field="genre_list", seed=f"{seed}|permuted-train"),
        _label_set_permutation(test_rows, target_field="genre_list", seed=f"{seed}|permuted-test"),
        labels=labels,
        target_field="genre_list",
        seed=f"{seed}|permuted",
    )
    corrupted = _evaluate_scenario(
        _partial_label_set_permutation(train_rows, target_field="genre_list", level=1.0, seed=f"{seed}|corrupt-train"),
        _partial_label_set_permutation(test_rows, target_field="genre_list", level=1.0, seed=f"{seed}|corrupt-test"),
        labels=labels,
        target_field="genre_list",
        seed=f"{seed}|corrupt",
    )
    codeword = _evaluate_scenario(
        _codeword_injection(train_rows, labels=labels, seed=f"{seed}|codeword-map"),
        _codeword_injection(test_rows, labels=labels, seed=f"{seed}|codeword-map"),
        labels=labels,
        target_field="genre_list",
        seed=f"{seed}|codeword",
    )
    return {"observed": observed, "label_set_permutation": permuted, "full_label_corruption": corrupted, "codeword_injection": codeword}


def build_mtg_jamendo_metadata_calibration(
    *,
    manifest_path: Path,
    provenance: dict[str, Any],
    top_label_count: int = 16,
    min_label_count: int = 25,
    seed: str = "full-song-eval-mtg-jamendo-metadata-v1",
) -> dict[str, Any]:
    """Run a ten-way artist-disjoint metadata calibration with invalid controls."""

    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = sorted({str(record.get("artist_fold", "")) for record in records})
    if len(groups) != FOLD_COUNT or any(not group for group in groups):
        raise ValueError(f"expected {FOLD_COUNT} nonempty artist folds, found {len(groups)}")
    folds: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    for index, heldout in enumerate(groups):
        train = [record for record in records if record["artist_fold"] != heldout]
        test = [record for record in records if record["artist_fold"] == heldout]
        counts = Counter(label for record in train for label in record["genre_list"])
        labels = [label for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])) if count >= min_label_count][:top_label_count]
        if len(labels) != top_label_count:
            raise ValueError(f"fold {index} lacks {top_label_count} train-supported labels")
        metrics = _fold_metrics(train, test, labels=labels, seed=f"{seed}|fold:{index}")
        prior = _prior_macro_auprc(train, test, labels=labels, target_field="genre_list")
        folds.append(
            {
                "fold": index,
                "heldout_artist_fold": heldout,
                "train_artist_folds": [group for group in groups if group != heldout],
                "test_artist_folds": [heldout],
                "groups_disjoint": all(record["artist_fold"] != heldout for record in train),
                "train_record_count": len(train),
                "test_record_count": len(test),
                "labels": labels,
                "train_label_counts": {label: counts[label] for label in labels},
                "label_vocabulary_scope": "training_fold_only",
                "prior_macro_auprc": prior,
                "condition_metrics": metrics,
            }
        )
        measurements.append({"group": heldout, "prior": prior, "metrics": metrics})

    def contrast(condition: str, view: str, reference: str | None = None) -> dict[str, Any]:
        differences = {
            measurement["group"]: float(measurement["metrics"][condition][view]["macro_auprc"])
            - (float(measurement["prior"]) if reference is None else float(measurement["metrics"][reference][view]["macro_auprc"]))
            for measurement in measurements
        }
        return _difference_interval(differences)

    paired = {
        "observed_unmasked_minus_prior": contrast("observed", "unmasked"),
        "permuted_unmasked_minus_prior": contrast("label_set_permutation", "unmasked"),
        "corrupted_unmasked_minus_prior": contrast("full_label_corruption", "unmasked"),
        "codeword_target_masked_minus_observed_target_masked": contrast("codeword_injection", "target_label_masked", "observed"),
    }
    invalid_controls = {
        "prevalence_preserving_label_set_permutation": paired["permuted_unmasked_minus_prior"],
        "full_label_corruption": paired["corrupted_unmasked_minus_prior"],
    }
    false_admission = {
        name: float(interval["ci95_low"]) > 0.0 for name, interval in invalid_controls.items()
    }
    aggregate_observed = {
        view: _score_interval([float(item["metrics"]["observed"][view]["macro_auprc"]) for item in measurements])
        for view in ("unmasked", "target_label_masked", "target_matched_random_deletion")
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "scope": "Independent public metadata calibration; not audio quality, listener preference, or full-song evidence.",
        "audio_used": False,
        "dataset": provenance,
        "design": {
            "target_field": "genre_list",
            "caption_fields": ["TRACK_NAME", "ARTIST_NAME", "ALBUM_NAME"],
            "split_unit": "artist_id hashed into ten deterministic artist-disjoint folds",
            "fold_count": FOLD_COUNT,
            "top_label_count": top_label_count,
            "min_label_count": min_label_count,
            "label_vocabulary_scope": "training_fold_only",
            "primary_metric": "macro_auprc",
        },
        "folds": folds,
        "aggregate_observed": aggregate_observed,
        "paired_intervals": paired,
        "false_admission": false_admission,
        "admission_rule_valid": not any(false_admission.values()),
        "interpretation": "A positive codeword response tests sensitivity. Any true false_admission entry rejects the prevalence-only admission rule.",
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-tsv", type=Path, required=True)
    parser.add_argument("--tags-tsv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", default="full-song-eval-mtg-jamendo-metadata-v1")
    parser.add_argument("--top-label-count", type=int, default=16)
    parser.add_argument("--min-label-count", type=int, default=25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    provenance = build_mtg_jamendo_manifest(
        metadata_tsv=args.metadata_tsv,
        tags_tsv=args.tags_tsv,
        output_manifest=args.manifest,
        seed=args.seed,
        source_revision=args.source_revision,
    )
    report = build_mtg_jamendo_metadata_calibration(
        manifest_path=args.manifest,
        provenance=provenance,
        top_label_count=args.top_label_count,
        min_label_count=args.min_label_count,
        seed=args.seed,
    )
    _write_json(args.report, report)
    print(json.dumps({"report": str(args.report), "status": report["status"], "admission_rule_valid": report["admission_rule_valid"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
