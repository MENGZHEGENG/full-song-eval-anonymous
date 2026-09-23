"""Check the joined artist split and the reported sensitivity decision."""

import json
from pathlib import Path

from run_mard_prospective_transfer import fold_id


ROOT = Path(__file__).resolve().parents[1]


def test_artist_overlap_and_filtered_fold_decisions():
    mapping = json.loads((ROOT / "data/external/mard/album_artist_mbid.json").read_text())
    protocol = json.loads((ROOT / "configs/mard_transfer_protocol.json").read_text())
    audit = json.loads((ROOT / "reports/mard_artist_overlap_audit.json").read_text())
    robust = json.loads((ROOT / "reports/mard_artist_disjoint_robustness.json").read_text())
    assert len(mapping) == audit["album_count"] == 1300
    for fold in robust["folds"]:
        heldout = fold["fold"]
        test_artists = {
            artist for album, artist in mapping.items()
            if fold_id(album, seed=protocol["fold_seed"], fold_count=10) == heldout
        }
        excluded = sum(
            fold_id(album, seed=protocol["fold_seed"], fold_count=10) != heldout
            and artist in test_artists
            for album, artist in mapping.items()
        )
        assert excluded == fold["removed_train_count"]
        assert fold["artist_overlap_count"] == 0
    assert audit["cross_fold_artist_mbid_count"] == 41
    assert audit["albums_in_cross_fold_artist_groups"] == 132
    assert robust["comparisons"]["observed"]["tie_aware_reference"]["accepted"]
    for rotation in ("rotation_a", "rotation_b"):
        assert robust["comparisons"][rotation]["prevalence"]["accepted"]
        assert not robust["comparisons"][rotation]["tie_aware_reference"]["accepted"]
