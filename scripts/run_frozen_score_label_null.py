"""Audit a second broken-link mechanism with frozen observed held-out scores.

The test reallocates each evaluated label independently within its held-out
group. It preserves label support and score ties, but does not refit a scorer
or supply an untouched cohort. The 200 draws and seed are fixed below.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from full_song_eval.calibration_prediction_replay import verify_prediction_replay_against_report
from full_song_eval.musiccaps_gate_calibration import _average_precision_pairs
from full_song_eval.random_ranking_baseline import (
    _paired_interval,
    expected_tie_aware_average_precision,
)


REPLICATES = 200
SEED_NAMESPACE = "full-song-eval-independent-heldout-label-null-v1"
INPUTS = {
    "musiccaps": (
        Path("reproduced/musiccaps_primary_prediction_replay.jsonl.gz"),
        Path("reports/musiccaps_gate_calibration.json"),
    ),
    "mtg_jamendo": (
        Path("reproduced/mtg_jamendo_primary_prediction_replay.jsonl.gz"),
        Path("reports/mtg_jamendo_metadata_calibration.json"),
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _grouped_ap(block_counts: list[int], block_sizes: list[int], k: int) -> float:
    positives = seen = 0
    area = 0.0
    for count, size in zip(block_counts, block_sizes):
        positives += count
        seen += size
        area += count * positives / (k * seen)
    return area


def _wilson_interval(successes: int, trials: int) -> list[float]:
    """Descriptive 95% interval for a conditional Monte Carlo admission rate."""
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return [center - half_width, center + half_width]


def _label_block(rows: list[dict], source_key: str, group: str, label: str) -> dict:
    n = len(rows)
    ids = [str(item["record_id"]) for item in rows]
    if len(set(ids)) != n:
        raise ValueError(f"duplicate record ID: {source_key}/{group}/{label}")
    scores = [float(item["score"]) for item in rows]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("nonfinite score")
    targets = [bool(item["target"]) for item in rows]
    k = sum(targets)
    if k == 0:
        return {"ids": ids, "aps": None, "support": 0, "n": n}
    order = sorted(range(n), key=lambda index: (-scores[index], ids[index]))
    block_ids = [0] * n
    block_sizes: list[int] = []
    previous_score: float | None = None
    for index in order:
        score = scores[index]
        if previous_score is None or score != previous_score:
            block_sizes.append(0)
            previous_score = score
        block_ids[index] = len(block_sizes) - 1
        block_sizes[-1] += 1
    observed_counts = [0] * len(block_sizes)
    for index, target in enumerate(targets):
        observed_counts[block_ids[index]] += target
    observed_ap = _grouped_ap(observed_counts, block_sizes, k)
    reference_ap = _average_precision_pairs(zip(scores, targets))
    if abs(observed_ap - reference_ap) > 1e-12:
        raise ValueError("score-block AP disagrees with canonical implementation")
    tie_expectation = expected_tie_aware_average_precision(
        n=n, k=k, tie_group_sizes=block_sizes
    )
    null_aps = []
    for replicate in range(REPLICATES):
        seed = f"{SEED_NAMESPACE}|{source_key}|{replicate}|{group}|{label}"
        rng = random.Random(int.from_bytes(hashlib.sha256(seed.encode()).digest(), "big"))
        sampled = rng.sample(range(n), k)
        counts = [0] * len(block_sizes)
        for index in sampled:
            counts[block_ids[index]] += 1
        null_aps.append(_grouped_ap(counts, block_sizes, k))
    return {
        "ids": ids,
        "aps": null_aps,
        "observed_ap": observed_ap,
        "prevalence": k / n,
        "tie_expectation": tie_expectation,
        "support": k,
        "n": n,
    }


def _source(source_key: str, replay_path: Path, report_path: Path, expected_hash: str) -> dict:
    digest = _sha256(replay_path)
    if digest != expected_hash:
        raise ValueError(f"replay digest changed: {source_key}")
    verified = verify_prediction_replay_against_report(replay_path, report_path)
    if verified.get("status") != "pass":
        raise ValueError(f"replay verification failed: {source_key}")
    group_values: dict[str, dict[str, list[float] | float | int]] = {}
    group_ids: dict[str, set[str]] = {}
    current: tuple[str, str] | None = None
    rows: list[dict] = []
    label_count = 0
    pair_count = 0

    def flush() -> None:
        nonlocal label_count, pair_count, rows
        if current is None:
            return
        group, label = current
        result = _label_block(rows, source_key, group, label)
        ids = set(result["ids"])
        if group in group_ids and group_ids[group] != ids:
            raise ValueError(f"different held-out record sets across labels: {group}")
        group_ids[group] = ids
        rows = []
        if result["support"] == 0:
            return
        slot = group_values.setdefault(
            group,
            {"null_ap": [0.0] * REPLICATES, "prevalence": 0.0,
             "tie_expectation": 0.0, "observed_ap": 0.0, "labels": 0},
        )
        slot["labels"] += 1
        slot["prevalence"] += result["prevalence"]
        slot["tie_expectation"] += result["tie_expectation"]
        slot["observed_ap"] += result["observed_ap"]
        for replicate, ap in enumerate(result["aps"]):
            slot["null_ap"][replicate] += ap
        label_count += 1
        pair_count += result["n"]

    with gzip.open(replay_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item["condition"] != "observed":
                continue
            if item["view"] != "target_label_masked":
                raise ValueError("unexpected replay view")
            key = (str(item["heldout_group"]), str(item["label"]))
            if key != current:
                flush()
                current = key
            rows.append(item)
    flush()
    if len(group_values) != 10:
        raise ValueError(f"expected ten groups for {source_key}")
    margins: dict[str, list[dict[str, float]]] = {
        "prevalence": [], "tie_expectation": []
    }
    for replicate in range(REPLICATES):
        for reference in margins:
            margins[reference].append({
                group: (slot["null_ap"][replicate] - slot[reference]) / slot["labels"]
                for group, slot in group_values.items()
            })
    summaries = {name: [_paired_interval(item) for item in values]
                 for name, values in margins.items()}
    decisions = {
        name: {
            "accepted_draws": sum(item["accepted_above_random"] for item in values),
            "draw_count": REPLICATES,
            "mean_margin_across_draws": sum(item["mean_difference"] for item in values) / REPLICATES,
            "mean_interval_low_across_draws": sum(item["ci95_low"] for item in values) / REPLICATES,
        }
        for name, values in summaries.items()
    }
    for item in decisions.values():
        item["admission_rate_wilson_95"] = _wilson_interval(
            item["accepted_draws"], REPLICATES
        )
    original = {
        name: _paired_interval({
            group: (slot["observed_ap"] - slot[name]) / slot["labels"]
            for group, slot in group_values.items()
        })
        for name in margins
    }
    return {
        "replay_sha256": digest,
        "report_sha256": _sha256(report_path),
        "evaluated_label_blocks": label_count,
        "evaluated_pairs": pair_count,
        "heldout_groups": len(group_values),
        "heldout_records": sum(len(ids) for ids in group_ids.values()),
        "original_observed": original,
        "independent_label_reallocation": decisions,
        "group_label_counts": {group: int(slot["labels"]) for group, slot in sorted(group_values.items())},
        "mean_sampled_ap_minus_exact_expectation": sum(
            summary["mean_difference"] for summary in summaries["tie_expectation"]
        ) / REPLICATES,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(
        "reproduced/frozen_score_independent_label_null.json"))
    args = parser.parse_args()
    verification = json.loads(Path(
        "reproduced/calibration_prediction_replay_verification.json").read_text())
    sources = {}
    for source_key, (replay_path, report_path) in INPUTS.items():
        expected = verification[source_key if source_key == "mtg_jamendo" else "musiccaps_export"]["sha256"]
        sources[source_key] = _source(source_key, replay_path, report_path, expected)
    output = {
        "schema": "fullsongeval-frozen-score-independent-label-null/v1",
        "status": "complete",
        "seed_namespace": SEED_NAMESPACE,
        "replicates": REPLICATES,
        "design": "For each held-out group and evaluated label, sample the observed positive count uniformly without replacement among records; keep observed scorer scores, score ties, training fits, and selected labels fixed.",
        "scope": "Post-audit conditional control on previously evaluated groups; not untouched-cohort validation or a refitted broken-label benchmark.",
        "sources": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: item["independent_label_reallocation"] for name, item in sources.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
