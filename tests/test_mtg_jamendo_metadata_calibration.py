import json
from pathlib import Path

from full_song_eval.mtg_jamendo_metadata_calibration import (
    FOLD_COUNT,
    _codeword_injection,
    build_mtg_jamendo_manifest,
    build_mtg_jamendo_metadata_calibration,
)


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.write_text("\t".join(header) + "\n" + "".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")


def test_codeword_mapping_is_shared_across_training_and_heldout_rows() -> None:
    train = _codeword_injection(
        [{"caption": "train", "genre_list": ["rock"]}], labels=["rock"], seed="shared-map"
    )
    heldout = _codeword_injection(
        [{"caption": "heldout", "genre_list": ["rock"]}], labels=["rock"], seed="shared-map"
    )

    assert train[0]["caption"].split()[-1] == heldout[0]["caption"].split()[-1]


def test_builds_artist_disjoint_metadata_only_calibration(tmp_path: Path) -> None:
    metadata_rows = []
    tag_rows = []
    for fold in range(100):
        artist = f"artist_{fold:02d}"
        for index in range(3):
            track = f"track_{fold:02d}_{index:02d}"
            metadata_rows.append([track, artist, "album", f"title {index}", f"name {fold}", "record"])
            tags = "genre---rock---genre---pop" if index % 2 == 0 else "genre---rock"
            tag_rows.append([track, artist, "album", tags])
    metadata = tmp_path / "raw.meta.tsv"
    tags = tmp_path / "tags.tsv"
    manifest = tmp_path / "manifest.jsonl"
    metadata_rows[0].append("unmodeled extra field")
    _write_tsv(metadata, ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "TRACK_NAME", "ARTIST_NAME", "ALBUM_NAME"], metadata_rows)
    _write_tsv(tags, ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "TAGS"], tag_rows)

    provenance = build_mtg_jamendo_manifest(
        metadata_tsv=metadata, tags_tsv=tags, output_manifest=manifest, seed="test-seed", source_revision="test-revision"
    )
    report = build_mtg_jamendo_metadata_calibration(
        manifest_path=manifest, provenance=provenance, top_label_count=1, min_label_count=2, seed="test-seed"
    )

    assert provenance["record_count"] == 300
    assert provenance["audio_used"] is False
    assert report["status"] == "complete"
    assert report["audio_used"] is False
    assert len(report["folds"]) == FOLD_COUNT
    assert all(fold["groups_disjoint"] for fold in report["folds"])
    assert "codeword_target_masked_minus_observed_target_masked" in report["paired_intervals"]
    assert len([json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]) == 300
