from __future__ import annotations

from run_mard_prospective_transfer import evaluate_condition, fold_id, interval


def test_constant_score_reference_matches_grouped_tie_ap() -> None:
    train = [
        {"row_id": str(i), "caption": "the same review text", "genre_list": ["jazz" if i % 2 else "rock"]}
        for i in range(8)
    ]
    test = [
        {"row_id": str(i), "caption": "the same review text", "genre_list": ["jazz" if i % 2 else "rock"]}
        for i in range(8, 12)
    ]
    result = evaluate_condition(train, test, labels=["jazz", "rock"], seed="constant-score")
    assert result["macro_ap"] == 0.5
    assert result["prevalence"] == 0.5
    assert result["tie_aware_reference"] == 0.5


def test_decision_interval_and_album_folds_are_deterministic() -> None:
    assert interval([0.01] * 10)["accepted"] is True
    assert interval([-0.01] * 10)["accepted"] is False
    assert fold_id("album-123", seed="fixed", fold_count=10) == fold_id("album-123", seed="fixed", fold_count=10)
