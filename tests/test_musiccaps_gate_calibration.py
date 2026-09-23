from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from full_song_eval.musiccaps_gate_calibration import (
    CORRUPTION_LEVELS,
    EVIDENCE_STRENGTHS,
    PARAPHRASE_EVIDENCE,
    TEXT_VIEWS,
    _inject_caption,
    _label_set_permutation,
    _normalized_tokens,
    _target_view_tokens,
    _within_author_caption_permutation,
    build_musiccaps_gate_calibration,
    main,
    validate_gate_calibration_report,
)


def _write_manifest(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _balanced_rows() -> list[dict]:
    rows: list[dict] = []
    for author_index in range(10):
        author = f"author-{author_index}"
        for repetition in range(4):
            for label, cue in [
                ("low quality", "weathered fidelity"),
                ("instrumental", "wordless arrangement"),
            ]:
                row_index = len(rows)
                rows.append(
                    {
                        "row_id": f"row-{row_index}",
                        "youtube_id": f"video-{row_index}",
                        "author_id": author,
                        "caption": f"A {cue} example {author_index} {repetition}",
                        "aspect_list": [label],
                    }
                )
    return rows


def test_frozen_paraphrase_registry_has_no_target_token_overlap() -> None:
    assert len(PARAPHRASE_EVIDENCE) >= 16
    assert {"moderate tempo", "groovy bass"}.issubset(PARAPHRASE_EVIDENCE)
    for label, phrase in PARAPHRASE_EVIDENCE.items():
        assert phrase.strip()
        assert set(_normalized_tokens(label)).isdisjoint(_normalized_tokens(phrase))


def test_target_specific_random_deletion_matches_target_mask_count() -> None:
    caption = "low quality weathered fidelity broad stereo image"
    masked, masked_diagnostic = _target_view_tokens(
        caption,
        target_label="low quality",
        view="target_label_masked",
        seed="unit-test",
        row_key="row-1",
    )
    matched, matched_diagnostic = _target_view_tokens(
        caption,
        target_label="low quality",
        view="target_matched_random_deletion",
        seed="unit-test",
        row_key="row-1",
    )

    assert masked_diagnostic["removed_token_count"] == 2
    assert matched_diagnostic["removed_token_count"] == 2
    assert len(masked) == len(matched)
    assert "low" not in masked and "quality" not in masked
    assert "low" in matched and "quality" in matched
    assert masked != matched
    assert matched == _target_view_tokens(
        caption,
        target_label="low quality",
        view="target_matched_random_deletion",
        seed="unit-test",
        row_key="row-1",
    )[0]


def test_permutation_controls_preserve_the_quantities_they_are_meant_to_preserve() -> None:
    rows = _balanced_rows()
    labels_before = Counter(tuple(row["aspect_list"]) for row in rows)
    captions_by_author_before: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        captions_by_author_before[row["author_id"]][row["caption"]] += 1

    label_permuted = _label_set_permutation(rows, target_field="aspect_list", seed="unit-test")
    caption_permuted = _within_author_caption_permutation(rows, split_unit="author_id", seed="unit-test")

    assert Counter(tuple(row["aspect_list"]) for row in label_permuted) == labels_before
    assert any(left["aspect_list"] != right["aspect_list"] for left, right in zip(rows, label_permuted))
    captions_by_author_after: dict[str, Counter[str]] = defaultdict(Counter)
    for row in caption_permuted:
        captions_by_author_after[row["author_id"]][row["caption"]] += 1
    assert captions_by_author_after == captions_by_author_before
    assert any(left["caption"] != right["caption"] for left, right in zip(rows, caption_permuted))


def test_injection_ladders_have_auditable_exact_and_paraphrase_semantics() -> None:
    row = {"row_id": "row-1", "caption": "A plain excerpt", "aspect_list": ["low quality"]}
    exact = _inject_caption(
        row,
        labels=["low quality"],
        target_field="aspect_list",
        evidence_kind="exact_label",
        strength=1.0,
        seed="unit-test",
    )
    paraphrase = _inject_caption(
        row,
        labels=["low quality"],
        target_field="aspect_list",
        evidence_kind="paraphrase",
        strength=1.0,
        seed="unit-test",
    )

    assert exact["caption"].endswith("low quality")
    assert PARAPHRASE_EVIDENCE["low quality"] in paraphrase["caption"]
    assert set(_normalized_tokens("low quality")).isdisjoint(
        _normalized_tokens(PARAPHRASE_EVIDENCE["low quality"])
    )
    exact_masked, _ = _target_view_tokens(
        exact["caption"],
        target_label="low quality",
        view="target_label_masked",
        seed="unit-test",
        row_key="row-1",
    )
    paraphrase_masked, _ = _target_view_tokens(
        paraphrase["caption"],
        target_label="low quality",
        view="target_label_masked",
        seed="unit-test",
        row_key="row-1",
    )
    assert "low" not in exact_masked and "quality" not in exact_masked
    assert set(_normalized_tokens(PARAPHRASE_EVIDENCE["low quality"])).issubset(paraphrase_masked)


def test_report_uses_ten_heldout_authors_and_emits_all_calibration_outputs(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.jsonl", _balanced_rows())
    report = build_musiccaps_gate_calibration(
        manifest_path=manifest,
        top_label_count=2,
        min_label_count=2,
        code_commit="abc123",
        seed="unit-test",
    )

    assert report["status"] == "complete"
    assert report["paper_branch_status"] == "pending_combined_evidence"
    assert report["design"]["label_vocabulary_scope"] == "training_fold_only"
    assert report["design"]["effective_fold_count"] == 10
    assert report["design"]["primary_metric"] == "macro_auprc"
    assert report["text_views"] == list(TEXT_VIEWS)
    assert report["evidence_strengths"] == list(EVIDENCE_STRENGTHS)
    assert report["corruption_levels"] == list(CORRUPTION_LEVELS)
    assert len(report["per_author_metrics"]) == 10
    assert len(report["folds"]) == 10
    assert all(fold["groups_disjoint"] for fold in report["folds"])
    assert all(len(fold["labels"]) == 2 for fold in report["folds"])
    assert set(report["aggregate_metrics"]) == {
        "observed",
        "label_set_permutation",
        "within_author_caption_permutation",
        "exact_label_injection",
        "paraphrase_evidence_injection",
        "label_corruption",
    }
    assert report["aggregate_metrics"]["exact_label_injection"]["0.0"] == report["aggregate_metrics"]["observed"]
    assert report["aggregate_metrics"]["paraphrase_evidence_injection"]["0.0"] == report["aggregate_metrics"]["observed"]
    assert report["aggregate_metrics"]["label_corruption"]["0.0"] == report["aggregate_metrics"]["observed"]
    assert report["paired_intervals"]
    assert report["monotonicity_sensitivity"]
    assert report["false_admission_checks"]
    assert report["provenance"]["manifest_sha256"]
    assert report["provenance"]["implementation_sha256"]
    assert report["provenance"]["paraphrase_mapping_sha256"]
    assert validate_gate_calibration_report(report) == []


def test_validator_fails_closed_on_scientific_contract_tampering(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.jsonl", _balanced_rows())
    report = build_musiccaps_gate_calibration(
        manifest_path=manifest,
        top_label_count=2,
        min_label_count=2,
        code_commit="abc123",
        seed="unit-test",
    )
    report["paper_branch_status"] = "selected"
    report["folds"][0]["groups_disjoint"] = False
    report["evidence_strengths"] = [0.0, 1.0]
    report["aggregate_metrics"]["exact_label_injection"]["0.0"] = {}

    errors = validate_gate_calibration_report(report)

    assert "paper_branch_status_not_pending_combined_evidence" in errors
    assert "fold_group_overlap:0" in errors
    assert "invalid_evidence_strengths" in errors
    assert "zero_strength_not_equal_observed:exact_label_injection" in errors


def test_cli_writes_json_markdown_and_latex_then_validates(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.jsonl", _balanced_rows())
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    output_tex = tmp_path / "rows.tex"

    exit_code = main(
        [
            "--manifest",
            str(manifest),
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_md),
            "--output-latex",
            str(output_tex),
            "--top-label-count",
            "2",
            "--min-label-count",
            "2",
            "--seed",
            "unit-test",
            "--code-commit",
            "abc123",
        ]
    )

    assert exit_code == 0
    assert output_json.is_file() and output_md.is_file() and output_tex.is_file()
    assert "Gate-calibration" in output_md.read_text(encoding="utf-8")
    assert "Observed" in output_tex.read_text(encoding="utf-8")
    assert main(["--validate-report", str(output_json)]) == 0


def test_score_is_hash_seed_invariant() -> None:
    code = """
from collections import Counter
from full_song_eval.musiccaps_gate_calibration import _score_label_model
tokens = [f'token_{index}' for index in range(4000)]
model = {
    'alpha': 0.5,
    'positive_rows': 10007,
    'row_count': 30011,
    'all_token_rows': Counter({token: 10009 + (index * 7919) % 10000 for index, token in enumerate(tokens)}),
    'positive_tokens': Counter({token: 3 + (index * 1543) % 10000 for index, token in enumerate(tokens)}),
}
print(repr(_score_label_model(model, tokens)))
"""
    root = Path(__file__).resolve().parents[1]
    outputs = []
    for hash_seed in ("1", "2"):
        environment = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=str(root / "src"))
        outputs.append(
            subprocess.check_output([sys.executable, "-c", code], text=True, env=environment).strip()
        )

    assert outputs[0] == outputs[1]
