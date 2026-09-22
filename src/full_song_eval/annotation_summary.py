from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

RATING_DIMENSIONS = [
    "lyric_intelligibility",
    "lyric_melody_alignment",
    "melody_memorability",
    "vocal_naturalness",
    "breath_and_phrasing",
    "genre_fit",
    "arrangement_fit",
    "structure_coherence",
    "mix_quality",
    "overall_preference",
]

PREFERENCE_TO_SCORE = {
    "a_much_better": 2.0,
    "a_better": 1.0,
    "tie": 0.0,
    "b_better": -1.0,
    "b_much_better": -2.0,
}


def preference_winner(preference: str) -> str:
    if preference in {"a_much_better", "a_better"}:
        return "candidate_a"
    if preference in {"b_better", "b_much_better"}:
        return "candidate_b"
    return preference


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _dimension_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dimension in RATING_DIMENSIONS:
        candidate_a = [float(record["ratings"]["candidate_a"][dimension]) for record in records]
        candidate_b = [float(record["ratings"]["candidate_b"][dimension]) for record in records]
        differences = [a_score - b_score for a_score, b_score in zip(candidate_a, candidate_b)]
        summary[dimension] = {
            "candidate_a_mean": _mean(candidate_a),
            "candidate_b_mean": _mean(candidate_b),
            "a_minus_b_mean": _mean(differences),
        }
    return summary


def summarize_annotations(records: list[dict[str, Any]]) -> dict[str, Any]:
    preference_counts = Counter(record.get("pairwise_preference", "missing") for record in records)
    valid_records = [record for record in records if record.get("pairwise_preference") != "invalid"]
    winner_counts = Counter(preference_winner(record.get("pairwise_preference", "missing")) for record in records)
    pairwise_scores = [
        PREFERENCE_TO_SCORE[record["pairwise_preference"]]
        for record in valid_records
        if record.get("pairwise_preference") in PREFERENCE_TO_SCORE
    ]
    annotator_counts = Counter(record.get("annotator_id", "missing") for record in records)
    prompt_counts = Counter(record.get("prompt_id", "missing") for record in records)
    return {
        "annotations": len(records),
        "valid_annotations": len(valid_records),
        "invalid_annotations": preference_counts.get("invalid", 0),
        "unique_annotators": len(annotator_counts),
        "unique_prompts": len(prompt_counts),
        "preference_counts": dict(sorted(preference_counts.items())),
        "winner_counts": dict(sorted(winner_counts.items())),
        "mean_pairwise_score_a_positive": _mean(pairwise_scores),
        "rating_dimensions": _dimension_summary(valid_records) if valid_records else {},
        "annotations_per_annotator": dict(sorted(annotator_counts.items())),
    }
