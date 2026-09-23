"""Post-audit MARD robustness check with artist-overlapping training albums removed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from full_song_eval.musiccaps_gate_calibration import _label_set_permutation
from full_song_eval.mtg_jamendo_metadata_calibration import _codeword_injection
from run_mard_prospective_transfer import evaluate_condition, fold_id, interval


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/mard_transfer_protocol.json"
SOURCE = ROOT / "data/external/mard/dataset_classification.json"
MAPPING = ROOT / "data/external/mard/album_artist_mbid.json"
OUTPUT = ROOT / "reports/mard_artist_disjoint_robustness.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    source = json.loads(SOURCE.read_text())
    mapping = json.loads(MAPPING.read_text())
    if sha(SOURCE) != protocol["source_sha256"] or set(mapping) != set(source):
        raise ValueError("MARD source or artist map does not match audited inputs")
    labels = sorted({value["genre"] for value in source.values()} - {protocol["excluded_label"]})
    seed = protocol["fold_seed"]
    rows = [
        {
            "row_id": album,
            "caption": source[album]["all_text"],
            "genre_list": [source[album]["genre"]],
            "artist_mbid": mapping[album],
            "fold": fold_id(album, seed=seed, fold_count=10),
        }
        for album in sorted(source)
    ]
    folds = []
    for heldout in range(10):
        test = [row for row in rows if row["fold"] == heldout]
        test_artists = {row["artist_mbid"] for row in test}
        unfiltered_train = [row for row in rows if row["fold"] != heldout]
        train = [row for row in unfiltered_train if row["artist_mbid"] not in test_artists]
        if test_artists & {row["artist_mbid"] for row in train}:
            raise ValueError("artist leakage remains")
        if any(sum(label in row["genre_list"] for row in test) == 0 for label in labels):
            raise ValueError("held-out label support missing")
        conditions = {
            "observed": evaluate_condition(train, test, labels=labels, seed=f"{seed}|fold:{heldout}|observed")
        }
        for name in ("rotation_a", "rotation_b"):
            rotated_train = _label_set_permutation(
                train, target_field="genre_list", seed=f"{seed}|fold:{heldout}|{name}|train"
            )
            rotated_test = _label_set_permutation(
                test, target_field="genre_list", seed=f"{seed}|fold:{heldout}|{name}|test"
            )
            conditions[name] = evaluate_condition(
                rotated_train, rotated_test, labels=labels, seed=f"{seed}|fold:{heldout}|{name}"
            )
        conditions["codeword"] = evaluate_condition(
            _codeword_injection(train, labels=labels, seed=f"{seed}|codeword"),
            _codeword_injection(test, labels=labels, seed=f"{seed}|codeword"),
            labels=labels,
            seed=f"{seed}|fold:{heldout}|codeword",
        )
        folds.append({
            "fold": heldout,
            "train_count": len(train),
            "heldout_count": len(test),
            "removed_train_count": len(unfiltered_train) - len(train),
            "artist_overlap_count": 0,
            "conditions": conditions,
        })
    comparisons = {}
    for name in ("observed", "rotation_a", "rotation_b", "codeword"):
        comparisons[name] = {
            reference: interval([
                fold["conditions"][name]["macro_ap"] - fold["conditions"][name][reference]
                for fold in folds
            ])
            for reference in ("prevalence", "tie_aware_reference")
        }
    comparisons["codeword_minus_observed"] = interval([
        fold["conditions"]["codeword"]["macro_ap"] - fold["conditions"]["observed"]["macro_ap"]
        for fold in folds
    ])
    report = {
        "schema_version": "full-song-eval-mard-artist-disjoint-robustness/v1",
        "status": "complete",
        "analysis_status": "post-audit robustness check; not prospectively specified",
        "source_sha256": sha(SOURCE),
        "protocol_sha256": sha(PROTOCOL),
        "artist_mapping_sha256": sha(MAPPING),
        "implementation_sha256": sha(Path(__file__)),
        "album_count": len(rows),
        "fold_count": len(folds),
        "removed_train_count_range": [min(f["removed_train_count"] for f in folds), max(f["removed_train_count"] for f in folds)],
        "folds": folds,
        "comparisons": comparisons,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "removed_train_count_range": report["removed_train_count_range"], "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
