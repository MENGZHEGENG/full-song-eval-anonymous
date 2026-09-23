from __future__ import annotations

from pathlib import Path

from scripts.derive_target_masked_table import build_summary


def test_target_masked_table_values_match_manuscript() -> None:
    report = build_summary(
        Path("reports/musiccaps_gate_calibration.json"),
        Path("reports/mtg_jamendo_metadata_calibration.json"),
    )
    entries = {(entry["source"], entry["condition"]): entry for entry in report["entries"]}

    assert entries[("MusicCaps", "Label permutation")]["paired_summary"]["mean_difference"] == 0.013707
    assert entries[("MusicCaps", "Label permutation")]["positive_group_differences"] == 9
    assert entries[("MusicCaps", "Full corruption")]["paired_summary"]["ci95_high"] == 0.022871
    assert entries[("MusicCaps", "Target masking")]["reference_condition"] == "Matched deletion"
    assert entries[("MusicCaps", "Target masking")]["paired_summary"]["mean_difference"] == -0.054668
    assert entries[("MTG-Jamendo", "Observed metadata")]["paired_summary"]["mean_difference"] == 0.030482
    assert entries[("MTG-Jamendo", "Codeword addition")]["paired_summary"]["mean_difference"] == 0.546829
    assert entries[("MTG-Jamendo", "Label permutation")]["positive_group_differences"] == 10
    assert entries[("MTG-Jamendo", "Full corruption")]["paired_summary"]["ci95_low"] == 0.00099
    assert entries[("MTG-Jamendo", "Target masking")]["paired_summary"]["ci95_low"] == -0.00157
