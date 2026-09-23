"""Calibrate metadata-evidence gates with deterministic MusicCaps controls.

The prespecified run is CPU-only and uses the ten MusicCaps caption authors as
leave-one-author-out inference units. Candidate labels are selected only from
each training fold. The experiment evaluates negative controls, controlled
evidence injection, and label corruption without using audio or a checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_MANIFEST = Path("data/musiccaps_manifest.jsonl")
DEFAULT_OUTPUT_JSON = Path("reports/musiccaps_gate_calibration.json")
DEFAULT_OUTPUT_MARKDOWN = Path("docs/musiccaps_gate_calibration.md")
DEFAULT_OUTPUT_LATEX = Path("paper/generated/musiccaps_gate_calibration_rows.tex")
DEFAULT_SEED = "full-song-eval-gate-calibration-v1"
TEXT_VIEWS = ("unmasked", "target_label_masked", "target_matched_random_deletion")
EVIDENCE_STRENGTHS = (0.0, 0.25, 0.5, 1.0)
CORRUPTION_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
PRIMARY_DESIGN = {
    "target_field": "aspect_list",
    "top_label_count": 16,
    "min_label_count": 25,
    "split_unit": "author_id",
    "fold_count": 10,
    "primary_metric": "macro_auprc",
}

# Frozen before the primary run. The registry covers every label that can
# enter a training-fold top-16 vocabulary in the frozen MusicCaps manifest.
# Each phrase is token-disjoint from its own target label.
PARAPHRASE_EVIDENCE = {
    "acoustic drums": "unamplified percussion",
    "acoustic guitar": "unplugged sixstring",
    "amateur recording": "homemade capture",
    "bass guitar": "lowregister strings",
    "electric guitar": "amplified sixstring",
    "emotional": "affective expression",
    "energetic": "high intensity",
    "fast tempo": "rapid pacing",
    "instrumental": "wordless arrangement",
    "instrumental music": "lyricfree arrangement",
    "live performance": "onstage rendition",
    "low quality": "degraded fidelity",
    "male voice": "masculine vocalization",
    "medium tempo": "moderate pacing",
    "moderate tempo": "balanced pacing",
    "mono": "single channel",
    "noisy": "cluttered sonics",
    "passionate": "fervent delivery",
    "poor audio quality": "degraded fidelity",
    "punchy kick": "forceful thump",
    "slow tempo": "leisurely pacing",
    "groovy": "rhythmic pocket",
    "groovy bass": "rhythmic lowend",
    "uptempo": "brisk pacing",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
}
T_CRITICAL_95_BY_DF = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _normalized_tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if len(token) > 1 and token not in STOPWORDS]


def _stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def _stable_fraction(value: str) -> float:
    return (_stable_hash(value) % 10**12) / float(10**12)


def _target_view_tokens(
    caption: str,
    *,
    target_label: str,
    view: str,
    seed: str,
    row_key: str,
) -> tuple[list[str], dict[str, int]]:
    """Apply one candidate-label view and return auditable deletion counts.

    The matched control removes the same number of positions as the target
    mask. It preferentially removes non-target positions, keeping the lexical
    evidence whenever the caption contains enough other tokens.
    """

    return _target_view_from_tokens(
        _normalized_tokens(caption),
        target_label=target_label,
        view=view,
        seed=seed,
        row_key=row_key,
    )


def _target_view_from_tokens(
    tokens: Sequence[str],
    *,
    target_label: str,
    view: str,
    seed: str,
    row_key: str,
) -> tuple[list[str], dict[str, int]]:
    if view not in TEXT_VIEWS:
        raise ValueError(f"unsupported text view: {view}")
    values = list(tokens)
    target_tokens = set(_normalized_tokens(target_label))
    target_positions = [index for index, token in enumerate(values) if token in target_tokens]
    removal_count = len(target_positions)
    if view == "unmasked":
        return values, {"reference_token_count": len(values), "removed_token_count": 0, "fallback_target_deletions": 0}
    if view == "target_label_masked":
        removed = set(target_positions)
        return (
            [token for index, token in enumerate(values) if index not in removed],
            {
                "reference_token_count": len(values),
                "removed_token_count": removal_count,
                "fallback_target_deletions": 0,
            },
        )

    target_position_set = set(target_positions)
    non_target_positions = [index for index in range(len(values)) if index not in target_position_set]
    # One cryptographic seed per candidate-row pair is enough to derive a
    # stable pseudo-random ordering; hashing every token dominates the full
    # 4,823-row calibration without improving the control semantics.
    ordering_seed = _stable_hash(f"{seed}|{row_key}|{target_label}|target-matched-deletion")
    ordering_mask = (1 << 256) - 1

    def ordering_key(index: int) -> tuple[int, int]:
        mixed = (ordering_seed ^ ((index + 1) * 0x9E3779B97F4A7C15)) & ordering_mask
        return mixed, index

    ranked_non_target = sorted(non_target_positions, key=ordering_key)
    ranked_target = sorted(target_positions, key=ordering_key)
    removed_positions = set((ranked_non_target + ranked_target)[:removal_count])
    fallback_count = sum(index in target_position_set for index in removed_positions)
    return (
        [token for index, token in enumerate(values) if index not in removed_positions],
        {
            "reference_token_count": len(values),
            "removed_token_count": len(removed_positions),
            "fallback_target_deletions": fallback_count,
        },
    )


def _label_set_permutation(
    rows: Sequence[dict[str, Any]], *, target_field: str, seed: str
) -> list[dict[str, Any]]:
    """Rotate complete label sets within a supplied split, preserving prevalence."""

    result = [dict(row) for row in rows]
    if len(rows) < 2:
        return result
    order = sorted(range(len(rows)), key=lambda index: (_stable_hash(f"{seed}|label-set|{_row_key(rows[index], index)}"), index))
    donors = order[1:] + order[:1]
    for recipient_index, donor_index in zip(order, donors):
        result[recipient_index][target_field] = copy.deepcopy(rows[donor_index].get(target_field, []))
    return result


def _within_author_caption_permutation(
    rows: Sequence[dict[str, Any]], *, split_unit: str, seed: str
) -> list[dict[str, Any]]:
    """Rotate captions only inside each author, preserving author-specific text."""

    result = [dict(row) for row in rows]
    by_unit: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_unit[_unit_value(row, split_unit)].append(index)
    for unit, indices in sorted(by_unit.items()):
        if len(indices) < 2:
            continue
        order = sorted(
            indices,
            key=lambda index: (_stable_hash(f"{seed}|caption|{unit}|{_row_key(rows[index], index)}"), index),
        )
        donors = order[1:] + order[:1]
        for recipient_index, donor_index in zip(order, donors):
            result[recipient_index]["caption"] = str(rows[donor_index].get("caption", ""))
    return result


def _partial_label_set_permutation(
    rows: Sequence[dict[str, Any]], *, target_field: str, level: float, seed: str
) -> list[dict[str, Any]]:
    """Corrupt a nested deterministic subset while preserving label prevalence."""

    if level not in CORRUPTION_LEVELS:
        raise ValueError(f"unsupported corruption level: {level}")
    result = [dict(row) for row in rows]
    if level == 0.0:
        return result
    selected = [
        index
        for index, row in enumerate(rows)
        if _stable_fraction(f"{seed}|corruption-select|{_row_key(row, index)}") < level
    ]
    if len(selected) < 2:
        return result
    order = sorted(
        selected,
        key=lambda index: (_stable_hash(f"{seed}|corruption-order|{_row_key(rows[index], index)}"), index),
    )
    donors = order[1:] + order[:1]
    for recipient_index, donor_index in zip(order, donors):
        result[recipient_index][target_field] = copy.deepcopy(rows[donor_index].get(target_field, []))
    return result


def _inject_caption(
    row: dict[str, Any],
    *,
    labels: Sequence[str],
    target_field: str,
    evidence_kind: str,
    strength: float,
    seed: str,
) -> dict[str, Any]:
    """Inject evidence into a nested subset of positive row-label pairs."""

    if evidence_kind not in {"exact_label", "paraphrase"}:
        raise ValueError(f"unsupported evidence kind: {evidence_kind}")
    if strength not in EVIDENCE_STRENGTHS:
        raise ValueError(f"unsupported evidence strength: {strength}")
    result = dict(row)
    if strength == 0.0:
        return result
    row_labels = set(_row_labels(row, target_field))
    phrases: list[str] = []
    for label in labels:
        if label not in row_labels:
            continue
        selected = _stable_fraction(f"{seed}|{evidence_kind}|{_row_key(row, 0)}|{label}") < strength
        if not selected:
            continue
        if evidence_kind == "paraphrase":
            if label not in PARAPHRASE_EVIDENCE:
                raise ValueError(f"missing frozen paraphrase evidence for label: {label}")
            phrases.append(PARAPHRASE_EVIDENCE[label])
        else:
            phrases.append(label)
    if phrases:
        original = str(row.get("caption", "")).strip()
        result["caption"] = " ".join([original, *phrases]).strip()
    return result


def build_musiccaps_gate_calibration(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    top_label_count: int = 16,
    min_label_count: int = 25,
    split_unit: str = "author_id",
    target_field: str = "aspect_list",
    seed: str = DEFAULT_SEED,
    code_commit: str | None = None,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_markdown: Path = DEFAULT_OUTPUT_MARKDOWN,
    output_latex: Path = DEFAULT_OUTPUT_LATEX,
) -> dict[str, Any]:
    """Build the calibration report without writing it."""

    if top_label_count < 1 or min_label_count < 1:
        raise ValueError("top_label_count and min_label_count must be positive")
    reproduction = _reproduction(
        manifest_path=manifest_path,
        output_json=output_json,
        output_markdown=output_markdown,
        output_latex=output_latex,
        top_label_count=top_label_count,
        min_label_count=min_label_count,
        split_unit=split_unit,
        target_field=target_field,
        seed=seed,
        code_commit=_resolve_code_commit(code_commit),
    )
    try:
        rows = _load_manifest(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _blocked_report(
            blocker=f"unreadable_manifest:{type(exc).__name__}",
            reproduction=reproduction,
            row_count=0,
            top_label_count=top_label_count,
            min_label_count=min_label_count,
            split_unit=split_unit,
            target_field=target_field,
            seed=seed,
        )
    authors = sorted({_unit_value(row, split_unit) for row in rows if _unit_value(row, split_unit)})
    missing_author_count = sum(not _unit_value(row, split_unit) for row in rows)
    if not rows:
        return _blocked_report(
            blocker="missing_or_empty_manifest",
            reproduction=reproduction,
            row_count=0,
            top_label_count=top_label_count,
            min_label_count=min_label_count,
            split_unit=split_unit,
            target_field=target_field,
            seed=seed,
        )
    if missing_author_count:
        return _blocked_report(
            blocker="missing_author_values",
            reproduction=reproduction,
            row_count=len(rows),
            top_label_count=top_label_count,
            min_label_count=min_label_count,
            split_unit=split_unit,
            target_field=target_field,
            seed=seed,
        )
    if len(authors) != 10:
        return _blocked_report(
            blocker=f"expected_exactly_10_authors:found_{len(authors)}",
            reproduction=reproduction,
            row_count=len(rows),
            top_label_count=top_label_count,
            min_label_count=min_label_count,
            split_unit=split_unit,
            target_field=target_field,
            seed=seed,
        )

    folds: list[dict[str, Any]] = []
    per_author_metrics: list[dict[str, Any]] = []
    blockers: set[str] = set()
    for fold_index, heldout_author in enumerate(authors):
        train_rows = [row for row in rows if _unit_value(row, split_unit) != heldout_author]
        test_rows = [row for row in rows if _unit_value(row, split_unit) == heldout_author]
        label_counts = Counter(label for row in train_rows for label in _row_labels(row, target_field))
        labels = [
            label
            for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
            if count >= min_label_count
        ][:top_label_count]
        fold_base = {
            "fold": fold_index,
            "heldout_author": heldout_author,
            "train_authors": [author for author in authors if author != heldout_author],
            "test_authors": [heldout_author],
            "train_row_count": len(train_rows),
            "test_row_count": len(test_rows),
            "groups_disjoint": all(_unit_value(row, split_unit) != heldout_author for row in train_rows),
            "labels": labels,
            "train_label_counts": {label: label_counts[label] for label in labels},
            "label_vocabulary_scope": "training_fold_only",
            "paraphrase_evidence_mapping": {label: PARAPHRASE_EVIDENCE.get(label) for label in labels},
        }
        if len(labels) != top_label_count:
            blockers.add(f"fold_{fold_index}:insufficient_train_label_frequency")
            folds.append({**fold_base, "blocker": "insufficient_train_label_frequency"})
            continue
        missing_mappings = [label for label in labels if label not in PARAPHRASE_EVIDENCE]
        overlapping_mappings = [
            label
            for label in labels
            if label in PARAPHRASE_EVIDENCE
            and not set(_normalized_tokens(label)).isdisjoint(_normalized_tokens(PARAPHRASE_EVIDENCE[label]))
        ]
        if missing_mappings or overlapping_mappings:
            detail = ",".join(sorted(missing_mappings + overlapping_mappings))
            blockers.add(f"fold_{fold_index}:invalid_paraphrase_mapping:{detail}")
            folds.append({**fold_base, "blocker": "invalid_paraphrase_mapping"})
            continue

        condition_metrics = _evaluate_fold_conditions(
            train_rows,
            test_rows,
            labels=labels,
            target_field=target_field,
            split_unit=split_unit,
            seed=f"{seed}|fold:{fold_index}",
        )
        prior_macro_auprc = _prior_macro_auprc(train_rows, test_rows, labels=labels, target_field=target_field)
        per_author_metrics.append(
            {
                "author_id": heldout_author,
                "prior_macro_auprc": prior_macro_auprc,
                "condition_metrics": condition_metrics,
            }
        )
        folds.append(
            {
                **fold_base,
                "prior_macro_auprc": prior_macro_auprc,
                "condition_metrics": condition_metrics,
                "blocker": None,
            }
        )

    if blockers or len(per_author_metrics) != 10:
        return {
            **_report_header(
                status="blocked",
                reproduction=reproduction,
                row_count=len(rows),
                top_label_count=top_label_count,
                min_label_count=min_label_count,
                split_unit=split_unit,
                target_field=target_field,
                seed=seed,
            ),
            "folds": folds,
            "per_author_metrics": per_author_metrics,
            "aggregate_metrics": {},
            "paired_intervals": {},
            "monotonicity_sensitivity": {},
            "false_admission_checks": [],
            "known_blockers": sorted(blockers),
        }

    aggregate_metrics = _aggregate_metrics(per_author_metrics)
    paired_intervals = _paired_intervals(per_author_metrics)
    monotonicity = _monotonicity_sensitivity(aggregate_metrics, paired_intervals)
    false_admission = _false_admission_checks(paired_intervals)
    report = {
        **_report_header(
            status="complete",
            reproduction=reproduction,
            row_count=len(rows),
            top_label_count=top_label_count,
            min_label_count=min_label_count,
            split_unit=split_unit,
            target_field=target_field,
            seed=seed,
        ),
        "folds": folds,
        "per_author_metrics": per_author_metrics,
        "aggregate_metrics": aggregate_metrics,
        "paired_intervals": paired_intervals,
        "monotonicity_sensitivity": monotonicity,
        "false_admission_checks": false_admission,
        "known_blockers": [],
    }
    report["validation_errors"] = validate_gate_calibration_report(report)
    if report["validation_errors"]:
        report["status"] = "blocked"
        report["known_blockers"] = ["self_validation_failed"]
    return report


def _evaluate_fold_conditions(
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    *,
    labels: Sequence[str],
    target_field: str,
    split_unit: str,
    seed: str,
) -> dict[str, Any]:
    observed = _evaluate_scenario(train_rows, test_rows, labels=labels, target_field=target_field, seed=f"{seed}|observed")
    label_permuted = _evaluate_scenario(
        _label_set_permutation(train_rows, target_field=target_field, seed=f"{seed}|label-permutation|train"),
        _label_set_permutation(test_rows, target_field=target_field, seed=f"{seed}|label-permutation|test"),
        labels=labels,
        target_field=target_field,
        seed=f"{seed}|label-permutation",
    )
    caption_permuted = _evaluate_scenario(
        _within_author_caption_permutation(train_rows, split_unit=split_unit, seed=f"{seed}|caption-permutation|train"),
        _within_author_caption_permutation(test_rows, split_unit=split_unit, seed=f"{seed}|caption-permutation|test"),
        labels=labels,
        target_field=target_field,
        seed=f"{seed}|caption-permutation",
    )
    exact: dict[str, Any] = {_level_key(0.0): copy.deepcopy(observed)}
    paraphrase: dict[str, Any] = {_level_key(0.0): copy.deepcopy(observed)}
    for strength in EVIDENCE_STRENGTHS[1:]:
        exact[_level_key(strength)] = _evaluate_scenario(
            [
                _inject_caption(
                    row,
                    labels=labels,
                    target_field=target_field,
                    evidence_kind="exact_label",
                    strength=strength,
                    seed=f"{seed}|exact",
                )
                for row in train_rows
            ],
            [
                _inject_caption(
                    row,
                    labels=labels,
                    target_field=target_field,
                    evidence_kind="exact_label",
                    strength=strength,
                    seed=f"{seed}|exact",
                )
                for row in test_rows
            ],
            labels=labels,
            target_field=target_field,
            seed=f"{seed}|exact|{strength}",
        )
        paraphrase[_level_key(strength)] = _evaluate_scenario(
            [
                _inject_caption(
                    row,
                    labels=labels,
                    target_field=target_field,
                    evidence_kind="paraphrase",
                    strength=strength,
                    seed=f"{seed}|paraphrase",
                )
                for row in train_rows
            ],
            [
                _inject_caption(
                    row,
                    labels=labels,
                    target_field=target_field,
                    evidence_kind="paraphrase",
                    strength=strength,
                    seed=f"{seed}|paraphrase",
                )
                for row in test_rows
            ],
            labels=labels,
            target_field=target_field,
            seed=f"{seed}|paraphrase|{strength}",
        )
    corruption: dict[str, Any] = {_level_key(0.0): copy.deepcopy(observed)}
    for level in CORRUPTION_LEVELS[1:]:
        corruption[_level_key(level)] = _evaluate_scenario(
            _partial_label_set_permutation(
                train_rows,
                target_field=target_field,
                level=level,
                seed=f"{seed}|corruption|train",
            ),
            _partial_label_set_permutation(
                test_rows,
                target_field=target_field,
                level=level,
                seed=f"{seed}|corruption|test",
            ),
            labels=labels,
            target_field=target_field,
            seed=f"{seed}|corruption|{level}",
        )
    return {
        "observed": observed,
        "label_set_permutation": label_permuted,
        "within_author_caption_permutation": caption_permuted,
        "exact_label_injection": exact,
        "paraphrase_evidence_injection": paraphrase,
        "label_corruption": corruption,
    }


def _evaluate_scenario(
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    *,
    labels: Sequence[str],
    target_field: str,
    seed: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    train_tokens = [(_row_key(row, index), _normalized_tokens(str(row.get("caption", "")))) for index, row in enumerate(train_rows)]
    test_tokens = [(_row_key(row, index), _normalized_tokens(str(row.get("caption", "")))) for index, row in enumerate(test_rows)]
    for view in TEXT_VIEWS:
        per_label_scores: dict[str, list[float]] = {}
        removed_total = fallback_total = reference_total = 0
        for label in labels:
            model = _fit_label_model(
                train_rows,
                train_tokens,
                label=label,
                target_field=target_field,
                view=view,
                seed=f"{seed}|train",
            )
            label_scores: list[float] = []
            for row, (row_key, tokens) in zip(test_rows, test_tokens):
                viewed, diagnostic = _target_view_from_tokens(
                    tokens,
                    target_label=label,
                    view=view,
                    seed=f"{seed}|test",
                    row_key=row_key,
                )
                label_scores.append(_score_label_model(model, viewed))
                reference_total += diagnostic["reference_token_count"]
                removed_total += diagnostic["removed_token_count"]
                fallback_total += diagnostic["fallback_target_deletions"]
            per_label_scores[label] = label_scores
        label_ap: list[float] = []
        label_support: dict[str, int] = {}
        positive_pair_count = 0
        for label in labels:
            actual = [label in set(_row_labels(row, target_field)) for row in test_rows]
            support = sum(actual)
            label_support[label] = support
            positive_pair_count += support
            if support:
                label_ap.append(_average_precision_pairs(zip(per_label_scores[label], actual)))
        result[view] = {
            "macro_auprc": round(sum(label_ap) / len(label_ap), 6) if label_ap else None,
            "labels_with_positive_support": len(label_ap),
            "positive_pair_count": positive_pair_count,
            "evaluated_pair_count": len(test_rows) * len(labels),
            "label_support": label_support,
            "deletion_diagnostics": {
                "reference_token_count": reference_total,
                "removed_token_count": removed_total,
                "fallback_target_deletions": fallback_total,
            },
        }
    return result


def _fit_label_model(
    rows: Sequence[dict[str, Any]],
    token_rows: Sequence[tuple[str, Sequence[str]]],
    *,
    label: str,
    target_field: str,
    view: str,
    seed: str,
) -> dict[str, Any]:
    all_token_rows: Counter[str] = Counter()
    positive_tokens: Counter[str] = Counter()
    positive_rows = 0
    for row, (row_key, tokens) in zip(rows, token_rows):
        viewed, _diagnostic = _target_view_from_tokens(
            tokens,
            target_label=label,
            view=view,
            seed=seed,
            row_key=row_key,
        )
        unique_tokens = set(viewed)
        all_token_rows.update(unique_tokens)
        is_positive = label in set(_row_labels(row, target_field))
        if is_positive:
            positive_rows += 1
            positive_tokens.update(unique_tokens)
    return {
        "row_count": len(rows),
        "positive_rows": positive_rows,
        "all_token_rows": all_token_rows,
        "positive_tokens": positive_tokens,
        "alpha": 0.5,
    }


def _score_label_model(model: dict[str, Any], tokens: Sequence[str]) -> float:
    alpha = float(model["alpha"])
    positive_rows = int(model["positive_rows"])
    negative_rows = int(model["row_count"]) - positive_rows
    score = math.log((positive_rows + alpha) / (negative_rows + alpha))
    for token in sorted(set(tokens)):
        positive_token_rows = int(model["positive_tokens"][token])
        negative_token_rows = int(model["all_token_rows"][token]) - positive_token_rows
        positive_rate = (positive_token_rows + alpha) / (positive_rows + 2.0 * alpha)
        negative_rate = (negative_token_rows + alpha) / (negative_rows + 2.0 * alpha)
        score += math.log(positive_rate / negative_rate)
    return score


def _prior_macro_auprc(
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    *,
    labels: Sequence[str],
    target_field: str,
) -> float | None:
    values: list[float] = []
    for label in labels:
        train_positive = sum(label in set(_row_labels(row, target_field)) for row in train_rows)
        score = (train_positive + 0.5) / (len(train_rows) + 1.0)
        actual = [label in set(_row_labels(row, target_field)) for row in test_rows]
        if any(actual):
            values.append(_average_precision_pairs((score, value) for value in actual))
    return round(sum(values) / len(values), 6) if values else None


def _average_precision_pairs(pairs: Iterable[tuple[float, bool]]) -> float:
    values = list(pairs)
    positive_count = sum(actual for _score, actual in values)
    if positive_count == 0:
        return 0.0
    grouped: dict[float, list[bool]] = defaultdict(list)
    for score, actual in values:
        grouped[float(score)].append(actual)
    true_positive = false_positive = 0
    previous_recall = area = 0.0
    for score in sorted(grouped, reverse=True):
        group = grouped[score]
        true_positive += sum(group)
        false_positive += sum(not value for value in group)
        recall = true_positive / positive_count
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def _aggregate_metrics(per_author_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def aggregate_scenario(path: Sequence[str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for view in TEXT_VIEWS:
            values = [
                _nested(author["condition_metrics"], [*path, view])["macro_auprc"]
                for author in per_author_metrics
            ]
            numeric = [float(value) for value in values if value is not None]
            result[view] = _score_interval(numeric)
        return result

    result = {
        "observed": aggregate_scenario(["observed"]),
        "label_set_permutation": aggregate_scenario(["label_set_permutation"]),
        "within_author_caption_permutation": aggregate_scenario(["within_author_caption_permutation"]),
        "exact_label_injection": {},
        "paraphrase_evidence_injection": {},
        "label_corruption": {},
    }
    for name, levels in (
        ("exact_label_injection", EVIDENCE_STRENGTHS),
        ("paraphrase_evidence_injection", EVIDENCE_STRENGTHS),
        ("label_corruption", CORRUPTION_LEVELS),
    ):
        result[name] = {_level_key(level): aggregate_scenario([name, _level_key(level)]) for level in levels}
    return result


def _paired_intervals(per_author_metrics: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    specs: dict[str, tuple[Sequence[str] | None, Sequence[str] | None]] = {
        "observed_unmasked_minus_prior": (["observed", "unmasked"], None),
        "label_set_permutation_unmasked_minus_prior": (["label_set_permutation", "unmasked"], None),
        "within_author_caption_permutation_unmasked_minus_prior": (["within_author_caption_permutation", "unmasked"], None),
        "full_label_corruption_unmasked_minus_prior": (["label_corruption", "1.0", "unmasked"], None),
        "observed_unmasked_minus_target_masked": (["observed", "unmasked"], ["observed", "target_label_masked"]),
        "observed_target_masked_minus_target_matched_deletion": (
            ["observed", "target_label_masked"],
            ["observed", "target_matched_random_deletion"],
        ),
        "exact_unmasked_1.0_minus_0.0": (
            ["exact_label_injection", "1.0", "unmasked"],
            ["exact_label_injection", "0.0", "unmasked"],
        ),
        "exact_target_masked_1.0_minus_0.0": (
            ["exact_label_injection", "1.0", "target_label_masked"],
            ["exact_label_injection", "0.0", "target_label_masked"],
        ),
        "exact_target_matched_deletion_1.0_minus_0.0": (
            ["exact_label_injection", "1.0", "target_matched_random_deletion"],
            ["exact_label_injection", "0.0", "target_matched_random_deletion"],
        ),
        "paraphrase_unmasked_1.0_minus_0.0": (
            ["paraphrase_evidence_injection", "1.0", "unmasked"],
            ["paraphrase_evidence_injection", "0.0", "unmasked"],
        ),
        "paraphrase_target_masked_1.0_minus_0.0": (
            ["paraphrase_evidence_injection", "1.0", "target_label_masked"],
            ["paraphrase_evidence_injection", "0.0", "target_label_masked"],
        ),
        "label_corruption_unmasked_1.0_minus_0.0": (
            ["label_corruption", "1.0", "unmasked"],
            ["label_corruption", "0.0", "unmasked"],
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, (left_path, right_path) in specs.items():
        differences: dict[str, float] = {}
        for author in per_author_metrics:
            left = _metric_for_author(author, left_path)
            right = float(author["prior_macro_auprc"]) if right_path is None else _metric_for_author(author, right_path)
            differences[str(author["author_id"])] = left - right
        result[name] = _difference_interval(differences)
    return result


def _metric_for_author(author: dict[str, Any], path: Sequence[str] | None) -> float:
    if path is None:
        return float(author["prior_macro_auprc"])
    value = _nested(author["condition_metrics"], path)["macro_auprc"]
    if value is None:
        raise ValueError(f"missing author metric at {'/'.join(path)}")
    return float(value)


def _score_interval(values: Sequence[float]) -> dict[str, Any]:
    interval = _raw_interval(values)
    return {
        "macro_auprc_mean": round(interval["mean"], 6),
        "macro_auprc_ci95_low": round(max(0.0, interval["ci95_low"]), 6),
        "macro_auprc_ci95_high": round(min(1.0, interval["ci95_high"]), 6),
        "ci95_method": "student_t_95",
        "n_independent_authors": len(values),
    }


def _difference_interval(differences: dict[str, float]) -> dict[str, Any]:
    interval = _raw_interval(list(differences.values()))
    return {
        "mean_difference": round(interval["mean"], 6),
        "ci95_low": round(interval["ci95_low"], 6),
        "ci95_high": round(interval["ci95_high"], 6),
        "sample_standard_deviation": round(interval["sample_standard_deviation"], 6),
        "standard_error": round(interval["standard_error"], 6),
        "ci95_method": "paired_student_t_95",
        "n_independent_authors": len(differences),
        "author_differences": {key: round(value, 6) for key, value in sorted(differences.items())},
    }


def _raw_interval(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("an interval requires at least one value")
    mean = sum(values) / len(values)
    if len(values) == 1:
        standard_deviation = standard_error = half_width = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        standard_deviation = math.sqrt(variance)
        standard_error = standard_deviation / math.sqrt(len(values))
        critical = T_CRITICAL_95_BY_DF.get(len(values) - 1, 1.96)
        half_width = critical * standard_error
    return {
        "mean": mean,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_error,
    }


def _monotonicity_sensitivity(
    aggregate_metrics: dict[str, Any], paired_intervals: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, levels, expected in (
        ("exact_label_injection", EVIDENCE_STRENGTHS, "nondecreasing"),
        ("paraphrase_evidence_injection", EVIDENCE_STRENGTHS, "nondecreasing"),
        ("label_corruption", CORRUPTION_LEVELS, "nonincreasing"),
    ):
        result[name] = {}
        for view in TEXT_VIEWS:
            means = [float(aggregate_metrics[name][_level_key(level)][view]["macro_auprc_mean"]) for level in levels]
            adjacent = [round(means[index + 1] - means[index], 6) for index in range(len(means) - 1)]
            monotone = all(value >= -1e-12 for value in adjacent) if expected == "nondecreasing" else all(value <= 1e-12 for value in adjacent)
            result[name][view] = {
                "expected_direction": expected,
                "levels": list(levels),
                "macro_auprc_means": means,
                "adjacent_differences": adjacent,
                "monotonic": monotone,
                "endpoint_sensitivity_per_unit_strength": round((means[-1] - means[0]) / (levels[-1] - levels[0]), 6),
            }
    result["endpoint_intervals"] = {
        key: value
        for key, value in paired_intervals.items()
        if key.startswith("exact_") or key.startswith("paraphrase_") or key.startswith("label_corruption_")
    }
    return result


def _false_admission_checks(paired_intervals: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for control, contrast_name in (
        ("prevalence_preserving_label_set_permutation", "label_set_permutation_unmasked_minus_prior"),
        ("within_author_caption_permutation", "within_author_caption_permutation_unmasked_minus_prior"),
        ("full_label_corruption", "full_label_corruption_unmasked_minus_prior"),
    ):
        interval = paired_intervals[contrast_name]
        admitted = float(interval["ci95_low"]) > 0.0
        checks.append(
            {
                "control": control,
                "admission_rule": "paired_95pct_ci_lower_bound_for_macro_auprc_minus_prior_above_zero",
                "contrast": contrast_name,
                "admitted": admitted,
                "false_admission": admitted,
                "passed": not admitted,
            }
        )
    return checks


def _report_header(
    *,
    status: str,
    reproduction: dict[str, Any],
    row_count: int,
    top_label_count: int,
    min_label_count: int,
    split_unit: str,
    target_field: str,
    seed: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "paper_branch_status": "pending_combined_evidence",
        "scope": "MusicCaps caption-to-label gate calibration; not audio quality, listener preference, or full-song evidence",
        "audio_used": False,
        "model_checkpoint_used": False,
        "gpu_used": False,
        "row_count": row_count,
        "design": {
            "target_field": target_field,
            "top_label_count": top_label_count,
            "min_label_count": min_label_count,
            "split_unit": split_unit,
            "requested_fold_count": 10,
            "effective_fold_count": 10,
            "label_vocabulary_scope": "training_fold_only",
            "primary_metric": "macro_auprc",
            "inference_unit": "heldout_author_id",
        },
        "prespecified_primary_design": dict(PRIMARY_DESIGN),
        "text_views": list(TEXT_VIEWS),
        "evidence_strengths": list(EVIDENCE_STRENGTHS),
        "corruption_levels": list(CORRUPTION_LEVELS),
        "control_semantics": {
            "label_set_permutation": "complete label sets rotated inside each train or test split; exact marginal prevalence preserved",
            "within_author_caption_permutation": "captions rotated only inside each author",
            "target_matched_random_deletion": "candidate-specific deletion count matched to the candidate target mask; non-target positions removed first",
            "label_corruption": "nested row subsets receive rotated complete label sets; split-level label prevalence preserved",
        },
        "paraphrase_evidence_mapping": dict(sorted(PARAPHRASE_EVIDENCE.items())),
        "seed": seed,
        "provenance": reproduction,
    }


def _blocked_report(
    *,
    blocker: str,
    reproduction: dict[str, Any],
    row_count: int,
    top_label_count: int,
    min_label_count: int,
    split_unit: str,
    target_field: str,
    seed: str,
) -> dict[str, Any]:
    return {
        **_report_header(
            status="blocked",
            reproduction=reproduction,
            row_count=row_count,
            top_label_count=top_label_count,
            min_label_count=min_label_count,
            split_unit=split_unit,
            target_field=target_field,
            seed=seed,
        ),
        "folds": [],
        "per_author_metrics": [],
        "aggregate_metrics": {},
        "paired_intervals": {},
        "monotonicity_sensitivity": {},
        "false_admission_checks": [],
        "known_blockers": [blocker],
    }


def validate_gate_calibration_report(report: dict[str, Any]) -> list[str]:
    """Return scientific-contract errors; an empty list means admissible."""

    errors: list[str] = []
    if report.get("status") != "complete":
        errors.append(f"status_not_complete:{report.get('status')}")
    if report.get("paper_branch_status") != "pending_combined_evidence":
        errors.append("paper_branch_status_not_pending_combined_evidence")
    if report.get("text_views") != list(TEXT_VIEWS):
        errors.append("invalid_text_views")
    if report.get("evidence_strengths") != list(EVIDENCE_STRENGTHS):
        errors.append("invalid_evidence_strengths")
    if report.get("corruption_levels") != list(CORRUPTION_LEVELS):
        errors.append("invalid_corruption_levels")
    if report.get("prespecified_primary_design") != PRIMARY_DESIGN:
        errors.append("invalid_prespecified_primary_design")
    design = report.get("design", {})
    if not isinstance(design, dict):
        errors.append("missing_design")
        design = {}
    if design.get("label_vocabulary_scope") != "training_fold_only":
        errors.append("label_vocabulary_not_training_only")
    if design.get("effective_fold_count") != 10 or design.get("inference_unit") != "heldout_author_id":
        errors.append("invalid_author_inference_design")
    if design.get("primary_metric") != "macro_auprc":
        errors.append("invalid_primary_metric")
    blockers = report.get("known_blockers")
    if blockers:
        errors.append("known_blockers_present:" + ",".join(sorted(str(value) for value in blockers)))

    mapping = report.get("paraphrase_evidence_mapping")
    if mapping != dict(sorted(PARAPHRASE_EVIDENCE.items())):
        errors.append("paraphrase_mapping_not_frozen_registry")
    else:
        for label, phrase in mapping.items():
            if not set(_normalized_tokens(label)).isdisjoint(_normalized_tokens(phrase)):
                errors.append(f"paraphrase_target_token_overlap:{label}")
    provenance = report.get("provenance", {})
    if not isinstance(provenance, dict):
        errors.append("missing_provenance")
        provenance = {}
    for field in ("manifest_sha256", "implementation_sha256", "paraphrase_mapping_sha256", "code_commit", "command"):
        if not provenance.get(field):
            errors.append(f"missing_provenance:{field}")
    expected_mapping_hash = hashlib.sha256(
        json.dumps(PARAPHRASE_EVIDENCE, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if provenance.get("paraphrase_mapping_sha256") != expected_mapping_hash:
        errors.append("invalid_paraphrase_mapping_sha256")

    folds = report.get("folds")
    if not isinstance(folds, list) or len(folds) != 10:
        errors.append("expected_exactly_10_folds")
    else:
        heldout = []
        for fold in folds:
            fold_id = fold.get("fold")
            heldout.append(fold.get("heldout_author"))
            if not fold.get("groups_disjoint"):
                errors.append(f"fold_group_overlap:{fold_id}")
            if fold.get("label_vocabulary_scope") != "training_fold_only":
                errors.append(f"fold_label_vocabulary_not_training_only:{fold_id}")
            labels = fold.get("labels", [])
            if len(labels) != design.get("top_label_count"):
                errors.append(f"fold_wrong_label_count:{fold_id}")
            train_counts = fold.get("train_label_counts", {})
            if any(int(train_counts.get(label, 0)) < int(design.get("min_label_count", 0)) for label in labels):
                errors.append(f"fold_label_below_minimum:{fold_id}")
            fold_mapping = fold.get("paraphrase_evidence_mapping", {})
            if set(fold_mapping) != set(labels) or any(not fold_mapping.get(label) for label in labels):
                errors.append(f"fold_invalid_paraphrase_mapping:{fold_id}")
        if len(set(heldout)) != 10:
            errors.append("heldout_authors_not_unique")

    per_author = report.get("per_author_metrics")
    if not isinstance(per_author, list) or len(per_author) != 10:
        errors.append("expected_exactly_10_author_metrics")
    elif len({row.get("author_id") for row in per_author}) != 10:
        errors.append("author_metrics_not_unique")
    aggregate = report.get("aggregate_metrics", {})
    expected_scenarios = {
        "observed",
        "label_set_permutation",
        "within_author_caption_permutation",
        "exact_label_injection",
        "paraphrase_evidence_injection",
        "label_corruption",
    }
    if not isinstance(aggregate, dict) or set(aggregate) != expected_scenarios:
        errors.append("missing_or_extra_aggregate_scenarios")
    else:
        for ladder in ("exact_label_injection", "paraphrase_evidence_injection", "label_corruption"):
            if aggregate[ladder].get("0.0") != aggregate["observed"]:
                errors.append(f"zero_strength_not_equal_observed:{ladder}")
        errors.extend(_metric_tree_errors(aggregate))
    intervals = report.get("paired_intervals")
    if not isinstance(intervals, dict) or not intervals:
        errors.append("missing_paired_intervals")
    else:
        for name, interval in intervals.items():
            if interval.get("n_independent_authors") != 10:
                errors.append(f"wrong_interval_unit_count:{name}")
            if interval.get("ci95_method") != "paired_student_t_95":
                errors.append(f"wrong_interval_method:{name}")
            if len(interval.get("author_differences", {})) != 10:
                errors.append(f"missing_author_differences:{name}")
    if not report.get("monotonicity_sensitivity"):
        errors.append("missing_monotonicity_sensitivity")
    checks = report.get("false_admission_checks")
    if not isinstance(checks, list) or len(checks) != 3:
        errors.append("missing_false_admission_checks")
    return sorted(set(errors))


def _metric_tree_errors(tree: Any, path: str = "aggregate_metrics") -> list[str]:
    errors: list[str] = []
    if not isinstance(tree, dict):
        return [f"invalid_metric_tree:{path}"]
    if "macro_auprc_mean" in tree:
        for field in ("macro_auprc_mean", "macro_auprc_ci95_low", "macro_auprc_ci95_high"):
            value = tree.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0:
                errors.append(f"invalid_metric:{path}:{field}")
        if tree.get("n_independent_authors") != 10:
            errors.append(f"wrong_metric_unit_count:{path}")
        return errors
    for key, value in tree.items():
        errors.extend(_metric_tree_errors(value, f"{path}/{key}"))
    return errors


def write_musiccaps_gate_calibration(
    *,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_markdown: Path = DEFAULT_OUTPUT_MARKDOWN,
    output_latex: Path = DEFAULT_OUTPUT_LATEX,
    **kwargs: Any,
) -> dict[str, Any]:
    report = build_musiccaps_gate_calibration(
        output_json=output_json,
        output_markdown=output_markdown,
        output_latex=output_latex,
        **kwargs,
    )
    for path in (output_json, output_markdown, output_latex):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_markdown.write_text(_markdown(report), encoding="utf-8")
    output_latex.write_text(_latex_rows(report), encoding="utf-8")
    return report


def validate_gate_calibration_report_path(path: Path) -> list[str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return [f"unreadable_report:{type(exc).__name__}"]
    if not isinstance(value, dict):
        return ["report_is_not_json_object"]
    return validate_gate_calibration_report(value)


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# MusicCaps Gate-calibration Results", ""]
    if report.get("status") != "complete":
        lines.extend([f"Status: blocked ({', '.join(report.get('known_blockers', []))}).", ""])
        return "\n".join(lines)
    lines.extend(
        [
            "The ten held-out caption authors are the inference units. Scores are macro AUPRC.",
            "The output remains pending combined evidence and does not select a paper position.",
            "",
            "| Condition | Text view | Mean | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for display, path in (
        ("Observed", ["observed"]),
        ("Label-set permutation", ["label_set_permutation"]),
        ("Within-author caption permutation", ["within_author_caption_permutation"]),
        ("Exact injection 1.0", ["exact_label_injection", "1.0"]),
        ("Paraphrase injection 1.0", ["paraphrase_evidence_injection", "1.0"]),
        ("Label corruption 1.0", ["label_corruption", "1.0"]),
    ):
        values = _nested(report["aggregate_metrics"], path)
        for view in TEXT_VIEWS:
            metric = values[view]
            lines.append(
                f"| {display} | {view.replace('_', ' ')} | {metric['macro_auprc_mean']:.3f} | "
                f"[{metric['macro_auprc_ci95_low']:.3f}, {metric['macro_auprc_ci95_high']:.3f}] |"
            )
    lines.extend(["", "## False-admission checks", ""])
    for check in report["false_admission_checks"]:
        lines.append(f"- {check['control']}: {'pass' if check['passed'] else 'fail'}")
    return "\n".join(lines) + "\n"


def _latex_rows(report: dict[str, Any]) -> str:
    if report.get("status") != "complete":
        return "% Gate-calibration report blocked; no result rows emitted.\n"
    lines = ["% Generated by musiccaps_gate_calibration.py; macro AUPRC over held-out authors."]
    for display, path in (
        ("Observed", ["observed"]),
        ("Label-set permutation", ["label_set_permutation"]),
        ("Within-author caption permutation", ["within_author_caption_permutation"]),
        ("Exact injection (1.0)", ["exact_label_injection", "1.0"]),
        ("Paraphrase injection (1.0)", ["paraphrase_evidence_injection", "1.0"]),
        ("Label corruption (1.0)", ["label_corruption", "1.0"]),
    ):
        values = _nested(report["aggregate_metrics"], path)
        metric = values["unmasked"]
        escaped = display.replace("-", "--")
        lines.append(
            f"{escaped} & {metric['macro_auprc_mean']:.3f} & "
            f"[{metric['macro_auprc_ci95_low']:.3f}, {metric['macro_auprc_ci95_high']:.3f}] \\\\"
        )
    return "\n".join(lines) + "\n"


def _reproduction(
    *,
    manifest_path: Path,
    output_json: Path,
    output_markdown: Path,
    output_latex: Path,
    top_label_count: int,
    min_label_count: int,
    split_unit: str,
    target_field: str,
    seed: str,
    code_commit: str | None,
) -> dict[str, Any]:
    arguments = [
        "python",
        "-m",
        "full_song_eval.musiccaps_gate_calibration",
        "--manifest",
        str(manifest_path),
        "--output-json",
        str(output_json),
        "--output-markdown",
        str(output_markdown),
        "--output-latex",
        str(output_latex),
        "--top-label-count",
        str(top_label_count),
        "--min-label-count",
        str(min_label_count),
        "--split-unit",
        split_unit,
        "--target-field",
        target_field,
        "--seed",
        seed,
    ]
    if code_commit:
        arguments.extend(["--code-commit", code_commit])
    mapping_hash = hashlib.sha256(
        json.dumps(PARAPHRASE_EVIDENCE, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "command": "PYTHONPATH=src " + " ".join(shlex.quote(argument) for argument in arguments),
        "validation_command": "PYTHONPATH=src python -m full_song_eval.musiccaps_gate_calibration --validate-report "
        + shlex.quote(str(output_json)),
        "manifest_sha256": _sha256(manifest_path),
        "implementation_sha256": _sha256(Path(__file__)),
        "paraphrase_mapping_sha256": mapping_hash,
        "code_commit": code_commit,
    }


def _resolve_code_commit(explicit: str | None) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip()
    environment_value = os.environ.get("FULL_SONG_EVAL_CODE_COMMIT", "").strip()
    if environment_value:
        return environment_value
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"manifest line {line_number} is not a JSON object")
        rows.append(value)
    return rows


def _row_labels(row: dict[str, Any], target_field: str) -> list[str]:
    values = row.get(target_field, [])
    if not isinstance(values, list):
        return []
    return sorted({str(value).strip().lower() for value in values if str(value).strip()})


def _row_key(row: dict[str, Any], index: int) -> str:
    return str(row.get("row_id") or row.get("youtube_id") or f"row:{index}")


def _unit_value(row: dict[str, Any], split_unit: str) -> str:
    value = row.get(split_unit)
    return str(value).strip() if value is not None else ""


def _level_key(level: float) -> str:
    return f"{level:.1f}" if level in {0.0, 0.5, 1.0} else f"{level:.2f}"


def _nested(value: dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        current = current[key]
    return current


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--output-latex", type=Path, default=DEFAULT_OUTPUT_LATEX)
    parser.add_argument("--top-label-count", type=int, default=16)
    parser.add_argument("--min-label-count", type=int, default=25)
    parser.add_argument("--split-unit", default="author_id")
    parser.add_argument("--target-field", default="aspect_list")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--code-commit")
    parser.add_argument("--validate-report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_report is not None:
        errors = validate_gate_calibration_report_path(args.validate_report)
        print(json.dumps({"errors": errors, "report": str(args.validate_report), "valid": not errors}, sort_keys=True))
        return 0 if not errors else 3
    report = write_musiccaps_gate_calibration(
        manifest_path=args.manifest,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        output_latex=args.output_latex,
        top_label_count=args.top_label_count,
        min_label_count=args.min_label_count,
        split_unit=args.split_unit,
        target_field=args.target_field,
        seed=args.seed,
        code_commit=args.code_commit,
    )
    print(json.dumps({"output": str(args.output_json), "status": report["status"]}, sort_keys=True))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
