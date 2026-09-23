from itertools import combinations

from full_song_eval.musiccaps_gate_calibration import _average_precision_pairs
from full_song_eval.random_ranking_baseline import expected_tie_aware_average_precision
from scripts.run_frozen_score_label_null import _grouped_ap, _wilson_interval


def test_grouped_ap_matches_canonical_metric_and_exact_null() -> None:
    scores = [3.0, 3.0, 2.0, 1.0]
    aps = []
    for selected in combinations(range(4), 2):
        targets = [index in selected for index in range(4)]
        block_counts = [sum(targets[:2]), int(targets[2]), int(targets[3])]
        ap = _grouped_ap(block_counts, [2, 1, 1], 2)
        assert abs(ap - _average_precision_pairs(zip(scores, targets))) < 1e-12
        aps.append(ap)
    expected = expected_tie_aware_average_precision(n=4, k=2, tie_group_sizes=[2, 1, 1])
    assert abs(sum(aps) / len(aps) - expected) < 1e-12


def test_wilson_boundary_counts() -> None:
    no_admissions = _wilson_interval(0, 200)
    all_admissions = _wilson_interval(200, 200)
    assert 0.0 <= no_admissions[0] < 1e-12
    assert 0.018 < no_admissions[1] < 0.020
    assert 0.980 < all_admissions[0] < 0.982
    assert all_admissions[1] == 1.0
