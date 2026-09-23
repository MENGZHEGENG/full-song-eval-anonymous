from __future__ import annotations

import gzip
import json
from pathlib import Path

from full_song_eval.calibration_prediction_replay import (
    main,
    summarize_prediction_replay,
    verify_prediction_replay_against_report,
    write_mtg_jamendo_prediction_replay,
    write_musiccaps_prediction_replay,
)
from full_song_eval.mtg_jamendo_metadata_calibration import build_mtg_jamendo_manifest


def _write_manifest(path: Path) -> Path:
    records: list[dict[str, object]] = []
    for author_number in range(10):
        for repeat in range(2):
            for label, cue in (("low quality", "weathered fidelity"), ("instrumental", "wordless arrangement")):
                record_number = len(records)
                records.append(
                    {
                        "row_id": f"record-{record_number}",
                        "author_id": f"author-{author_number}",
                        "caption": f"{cue} example {author_number} {repeat}",
                        "aspect_list": [label],
                    }
                )
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_musiccaps_prediction_replay_is_deterministic_and_has_complete_primary_controls(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "musiccaps.jsonl")
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"

    first_receipt = write_musiccaps_prediction_replay(
        manifest_path=manifest,
        output_path=first,
        top_label_count=2,
        min_label_count=2,
        seed="test-seed",
    )
    second_receipt = write_musiccaps_prediction_replay(
        manifest_path=manifest,
        output_path=second,
        top_label_count=2,
        min_label_count=2,
        seed="test-seed",
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_receipt["record_count"] == 10 * 4 * 2 * 4
    assert first_receipt["sha256"] == second_receipt["sha256"]
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        replay = [json.loads(line) for line in handle]
    assert {record["condition"] for record in replay} == {
        "observed",
        "label_set_permutation",
        "within_author_caption_permutation",
        "full_label_corruption",
    }
    assert {record["view"] for record in replay} == {"target_label_masked"}
    assert all(record["score_direction"] == "higher_is_more_positive" for record in replay)
    assert all(record["source"] == "MusicCaps" for record in replay)


def test_musiccaps_prediction_replay_applies_caption_permutation_control(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "musiccaps.jsonl")
    output = tmp_path / "replay.jsonl.gz"
    write_musiccaps_prediction_replay(
        manifest_path=manifest,
        output_path=output,
        top_label_count=2,
        min_label_count=2,
        seed="test-seed",
    )
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        replay = [json.loads(line) for line in handle]
    observed = {
        (record["heldout_group"], record["record_id"], record["label"]): record["score"]
        for record in replay
        if record["condition"] == "observed"
    }
    caption_permuted = {
        (record["heldout_group"], record["record_id"], record["label"]): record["score"]
        for record in replay
        if record["condition"] == "within_author_caption_permutation"
    }
    assert any(observed[key] != caption_permuted[key] for key in observed)


def test_prediction_replay_summary_recomputes_per_group_macro_auprc(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "musiccaps.jsonl")
    output = tmp_path / "replay.jsonl.gz"
    write_musiccaps_prediction_replay(
        manifest_path=manifest,
        output_path=output,
        top_label_count=2,
        min_label_count=2,
        seed="test-seed",
    )

    summary = summarize_prediction_replay(output)

    assert summary["source"] == "MusicCaps"
    assert len(summary["groups"]) == 10
    observed = summary["groups"]["author-0"]["observed"]
    assert observed["labels_with_positive_support"] == 2
    assert 0.0 <= observed["macro_auprc"] <= 1.0


def test_prediction_replay_verifier_matches_musiccaps_report_structure(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "musiccaps.jsonl")
    output = tmp_path / "replay.jsonl.gz"
    write_musiccaps_prediction_replay(
        manifest_path=manifest,
        output_path=output,
        top_label_count=2,
        min_label_count=2,
        seed="test-seed",
    )
    summary = summarize_prediction_replay(output)
    per_author_metrics = []
    for author, conditions in summary["groups"].items():
        report_conditions = {
            condition: {"target_label_masked": metrics}
            for condition, metrics in conditions.items()
            if condition != "full_label_corruption"
        }
        report_conditions["label_corruption"] = {
            "1.0": {"target_label_masked": conditions["full_label_corruption"]}
        }
        per_author_metrics.append({"author_id": author, "condition_metrics": report_conditions})
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"per_author_metrics": per_author_metrics}), encoding="utf-8")

    receipt = verify_prediction_replay_against_report(output, report)

    assert receipt["status"] == "pass"
    assert receipt["checked_group_count"] == 10


def test_prediction_replay_cli_writes_verification_receipt(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "musiccaps.jsonl")
    replay = tmp_path / "replay.jsonl.gz"
    write_musiccaps_prediction_replay(
        manifest_path=manifest,
        output_path=replay,
        top_label_count=2,
        min_label_count=2,
        seed="test-seed",
    )
    summary = summarize_prediction_replay(replay)
    report_metrics = []
    for author, conditions in summary["groups"].items():
        report_conditions = {
            condition: {"target_label_masked": metrics}
            for condition, metrics in conditions.items()
            if condition != "full_label_corruption"
        }
        report_conditions["label_corruption"] = {
            "1.0": {"target_label_masked": conditions["full_label_corruption"]}
        }
        report_metrics.append({"author_id": author, "condition_metrics": report_conditions})
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"per_author_metrics": report_metrics}), encoding="utf-8")
    receipt = tmp_path / "receipt.json"

    exit_code = main(
        [
            "--musiccaps-manifest", str(manifest),
            "--musiccaps-output", str(tmp_path / "replayed.jsonl.gz"),
            "--musiccaps-report", str(report),
            "--verification-output", str(receipt),
            "--top-label-count", "2",
            "--min-label-count", "2",
            "--seed", "test-seed",
        ]
    )

    assert exit_code == 0
    assert json.loads(receipt.read_text(encoding="utf-8"))["verification"]["musiccaps"]["status"] == "pass"


def test_mtg_prediction_replay_covers_primary_controls(tmp_path: Path) -> None:
    metadata = tmp_path / "raw.meta.tsv"
    tags = tmp_path / "tags.tsv"
    manifest = tmp_path / "mtg_manifest.jsonl"
    metadata_lines = ["TRACK_ID\tARTIST_ID\tALBUM_ID\tTRACK_NAME\tARTIST_NAME\tALBUM_NAME\n"]
    tag_lines = ["TRACK_ID\tARTIST_ID\tALBUM_ID\tTAGS\n"]
    for artist_number in range(100):
        artist = f"artist-{artist_number:03d}"
        for repeat in range(3):
            track = f"track-{artist_number:03d}-{repeat}"
            metadata_lines.append(f"{track}\t{artist}\talbum\ttitle {repeat}\tname {artist_number}\trecord\n")
            tag_lines.append(f"{track}\t{artist}\talbum\tgenre---rock\n")
    metadata.write_text("".join(metadata_lines), encoding="utf-8")
    tags.write_text("".join(tag_lines), encoding="utf-8")
    build_mtg_jamendo_manifest(
        metadata_tsv=metadata,
        tags_tsv=tags,
        output_manifest=manifest,
        seed="test-seed",
        source_revision="test-revision",
    )
    output = tmp_path / "mtg_replay.jsonl.gz"

    receipt = write_mtg_jamendo_prediction_replay(
        manifest_path=manifest,
        output_path=output,
        top_label_count=1,
        min_label_count=2,
        seed="test-seed",
    )

    assert receipt["record_count"] == 10 * 30 * 1 * 4
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        replay = [json.loads(line) for line in handle]
    assert {record["condition"] for record in replay} == {
        "observed",
        "label_set_permutation",
        "full_label_corruption",
        "codeword_injection",
    }
    assert all(record["source"] == "MTG-Jamendo" for record in replay)
