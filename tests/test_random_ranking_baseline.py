from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts.build_random_ranking_baseline import resolve_mtg_manifest
from full_song_eval.random_ranking_baseline import (
    analyze_rotation_integrity,
    build_random_ranking_baseline,
    expected_average_precision,
    expected_tie_aware_average_precision,
    replay_target_masked_condition,
    rotation_donor_indices,
)


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_ARTIFACT = next(
    (
        path
        for path in (
            ROOT / "reports/random_ranking_baseline_calibration.json",
            ROOT / "reports/random_ranking_baseline_calibration.json",
        )
        if path.is_file()
    ),
    ROOT / "reports/random_ranking_baseline_calibration.json",
)


def test_expected_average_precision_uses_the_exact_finite_ranking_formula() -> None:
    assert expected_average_precision(n=4, k=1) == pytest.approx(25.0 / 48.0)
    assert expected_average_precision(n=4, k=2) == pytest.approx(49.0 / 72.0)
    assert expected_average_precision(n=4, k=4) == pytest.approx(1.0)
    assert math.isfinite(expected_average_precision(n=5_688, k=137))


@pytest.mark.parametrize(("n", "k"), [(0, 0), (4, 0), (4, 5), (4, -1)])
def test_expected_average_precision_rejects_undefined_support(n: int, k: int) -> None:
    with pytest.raises(ValueError):
        expected_average_precision(n=n, k=k)


def test_tie_aware_expectation_respects_ordered_score_blocks() -> None:
    assert expected_tie_aware_average_precision(n=4, k=2, tie_group_sizes=[1, 1, 1, 1]) == pytest.approx(
        expected_average_precision(n=4, k=2)
    )
    assert expected_tie_aware_average_precision(n=4, k=2, tie_group_sizes=[4]) == pytest.approx(0.5)
    assert expected_tie_aware_average_precision(n=4, k=2, tie_group_sizes=[2, 2]) == pytest.approx(7.0 / 12.0)


@pytest.mark.parametrize("tie_group_sizes", [[], [1, 2], [0, 4], [5]])
def test_tie_aware_expectation_rejects_malformed_blocks(tie_group_sizes: list[int]) -> None:
    with pytest.raises(ValueError):
        expected_tie_aware_average_precision(n=4, k=2, tie_group_sizes=tie_group_sizes)


def test_rotation_assignment_matches_the_frozen_control_hashing() -> None:
    rows = [
        {"row_id": "a", "labels": ["rock"]},
        {"row_id": "b", "labels": ["pop"]},
        {"row_id": "c", "labels": ["rock", "pop"]},
        {"row_id": "d", "labels": []},
    ]

    assert rotation_donor_indices(rows, seed="fixture-seed", rotation_kind="label_set") == {
        0: 3,
        1: 0,
        2: 1,
        3: 2,
    }
    assert rotation_donor_indices(rows, seed="fixture-corruption", rotation_kind="full_corruption") == {
        0: 1,
        1: 3,
        2: 0,
        3: 2,
    }


@pytest.mark.parametrize(
    ("seed", "rotation_kind"),
    [("fixture-seed", "label_set"), ("fixture-corruption", "full_corruption")],
)
def test_rotation_integrity_is_deterministic_and_audits_selected_label_vectors(
    seed: str, rotation_kind: str
) -> None:
    rows = [
        {"row_id": "a", "labels": ["rock"]},
        {"row_id": "b", "labels": ["pop"]},
        {"row_id": "c", "labels": ["rock", "pop"]},
        {"row_id": "d", "labels": []},
    ]

    result = analyze_rotation_integrity(
        rows,
        target_field="labels",
        selected_labels=["rock", "pop"],
        seed=seed,
        rotation_kind=rotation_kind,
    )

    assert result == analyze_rotation_integrity(
        rows,
        target_field="labels",
        selected_labels=["rock", "pop"],
        seed=seed,
        rotation_kind=rotation_kind,
    )
    assert result == {
        "row_count": 4,
        "assigned_recipient_count": 4,
        "donor_self_matches": 0,
        "exact_selected_label_vector_transfers": 4,
        "changed_selected_label_vectors": 4,
        "percent_changed": 100.0,
        "mean_selected_label_jaccard": 0.125,
        "prevalence_preserved": True,
    }


def test_target_masked_replay_exposes_score_ties_and_matches_tied_ap() -> None:
    train_rows = [
        {"row_id": "train-a", "caption": "rock", "labels": ["rock"]},
        {"row_id": "train-b", "caption": "rock", "labels": ["rock"]},
        {"row_id": "train-c", "caption": "rock", "labels": []},
        {"row_id": "train-d", "caption": "rock", "labels": []},
    ]
    test_rows = [
        {"row_id": "test-a", "caption": "rock", "labels": ["rock"]},
        {"row_id": "test-b", "caption": "rock", "labels": []},
        {"row_id": "test-c", "caption": "rock", "labels": ["rock"]},
        {"row_id": "test-d", "caption": "rock", "labels": []},
    ]

    result = replay_target_masked_condition(
        train_rows,
        test_rows,
        labels=["rock"],
        target_field="labels",
        scenario_seed="fixture-scenario",
    )

    assert result["macro_auprc"] == pytest.approx(0.5)
    assert result["strict_random_ranking_expected_macro_auprc"] == pytest.approx(49.0 / 72.0)
    assert result["tie_aware_random_label_expected_macro_auprc"] == pytest.approx(0.5)
    assert result["labels"] == [
        {
            "label": "rock",
            "test_record_count": 4,
            "positive_count": 2,
            "prevalence": 0.5,
            "actual_average_precision": 0.5,
            "strict_random_ranking_expected_average_precision": pytest.approx(49.0 / 72.0),
            "tie_aware_random_label_expected_average_precision": 0.5,
            "ordered_score_tie_group_sizes": [4],
        }
    ]


def test_builder_fails_closed_on_a_malformed_manifest(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{not-json}\n", encoding="utf-8")
    placeholder_report = tmp_path / "report.json"
    placeholder_report.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="MusicCaps manifest"):
        build_random_ranking_baseline(
            musiccaps_manifest_path=malformed,
            mtg_manifest_path=malformed,
            musiccaps_report_path=placeholder_report,
            mtg_report_path=placeholder_report,
            reproduction_command="unit-test",
        )


def test_mtg_manifest_pointer_resolves_only_the_verified_manifest_name(tmp_path: Path) -> None:
    integrity_directory = tmp_path / "verified"
    integrity_directory.mkdir()
    manifest = integrity_directory / "mtg_manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    pointer = tmp_path / "pointer.txt"
    pointer.write_text(str(integrity_directory) + "\n", encoding="utf-8")

    assert resolve_mtg_manifest(pointer) == manifest
    manifest.rename(integrity_directory / "unexpected-name.jsonl")
    with pytest.raises(ValueError, match="mtg_manifest.jsonl"):
        resolve_mtg_manifest(pointer)


def test_checked_in_calibration_has_the_known_strict_random_ranking_aggregates() -> None:
    report = json.loads(CALIBRATION_ARTIFACT.read_text(encoding="utf-8"))

    assert report["status"] == "complete"
    musiccaps = report["sources"]["musiccaps"]["strict_random_ranking"]["score_minus_expected"]
    mtg = report["sources"]["mtg_jamendo"]["strict_random_ranking"]["score_minus_expected"]
    assert musiccaps["observed"]["mean_difference"] == pytest.approx(0.055952891)
    assert musiccaps["permutation_a"]["mean_difference"] == pytest.approx(-0.010546609)
    assert musiccaps["permutation_b"]["mean_difference"] == pytest.approx(-0.010441109)
    assert mtg["observed"]["mean_difference"] == pytest.approx(0.029055456)
    assert mtg["permutation_a"]["mean_difference"] == pytest.approx(0.000457856)
    assert mtg["permutation_b"]["mean_difference"] == pytest.approx(0.000103356)
    musiccaps_tie = report["sources"]["musiccaps"]["tie_aware_random_label"]["score_minus_expected"]
    mtg_tie = report["sources"]["mtg_jamendo"]["tie_aware_random_label"]["score_minus_expected"]
    assert musiccaps_tie["observed"]["mean_difference"] == pytest.approx(0.055952891)
    assert musiccaps_tie["permutation_a"]["mean_difference"] == pytest.approx(-0.010546609)
    assert musiccaps_tie["permutation_b"]["mean_difference"] == pytest.approx(-0.010441109)
    assert mtg_tie["observed"]["mean_difference"] == pytest.approx(0.029070791)
    assert mtg_tie["permutation_a"]["mean_difference"] == pytest.approx(0.000472968)
    assert mtg_tie["permutation_b"]["mean_difference"] == pytest.approx(0.000118268)
    for summaries in (musiccaps, mtg, musiccaps_tie, mtg_tie):
        assert summaries["observed"]["accepted_above_random"] is True
        assert summaries["permutation_a"]["accepted_above_random"] is False
        assert summaries["permutation_b"]["accepted_above_random"] is False
    assert report["sources"]["musiccaps"]["rotation_integrity"]["permutation_a"][
        "donor_self_matches"
    ] == 0
    assert report["sources"]["mtg_jamendo"]["rotation_integrity"]["permutation_b"][
        "exact_selected_label_vector_transfers"
    ] == 55_128


def test_checked_in_calibration_uses_stable_input_identifiers() -> None:
    text = CALIBRATION_ARTIFACT.read_text(encoding="utf-8")
    report = json.loads(text)

    assert "/private" + "/tmp" not in text
    assert f"/{'Users'}/" not in text
    assert report["inputs"]["musiccaps_manifest"]["identifier"] == "musiccaps_manifest.jsonl"
    assert report["inputs"]["mtg_jamendo_manifest"]["identifier"] == (
        "verified_mtg_manifest/mtg_manifest.jsonl"
    )
    assert report["inputs"]["musiccaps_report"]["identifier"] == "musiccaps_gate_calibration.json"
    assert report["inputs"]["mtg_jamendo_report"]["identifier"] == "mtg_jamendo_metadata_calibration.json"
