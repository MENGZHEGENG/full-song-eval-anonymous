#!/usr/bin/env python3
"""Post-audit FMA sensitivity excluding text identical to the prior MTG manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re

from full_song_eval.musiccaps_gate_calibration import _average_precision_pairs, _difference_interval
from full_song_eval.random_ranking_baseline import expected_tie_aware_average_precision


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def run(fma_manifest: Path, mtg_manifest: Path, replay: Path) -> dict:
    mtg_texts = {normalized(json.loads(line)["caption"]) for line in mtg_manifest.read_text().splitlines()}
    fma = [json.loads(line) for line in fma_manifest.read_text().splitlines()]
    excluded = {str(row["row_id"]) for row in fma if normalized(row["caption"]) in mtg_texts}
    blocks = defaultdict(list)
    with gzip.open(replay, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["record_id"] not in excluded:
                blocks[(row["heldout_group"], row["condition"], row["label"])].append(
                    (float(row["score"]), bool(row["target"]))
                )
    groups = sorted({group for group, _, _ in blocks})
    conditions = ("observed", "label_set_permutation", "full_label_corruption")
    if len(groups) != 10:
        raise ValueError("missing artist fold after overlap exclusion")
    comparisons = {}
    support = {}
    for group in groups:
        comparisons[group] = {}
        support[group] = {}
        labels = sorted({label for g, _, label in blocks if g == group})
        for condition in conditions:
            aps, prevs, ties = [], [], []
            label_support = {}
            for label in labels:
                values = blocks[(group, condition, label)]
                n = len(values)
                k = sum(target for _, target in values)
                label_support[label] = {"n": n, "k": k}
                if k == 0:
                    continue
                aps.append(_average_precision_pairs(values))
                prevs.append(k / n)
                counts = Counter(score for score, _ in values)
                ties.append(expected_tie_aware_average_precision(
                    n=n, k=k, tie_group_sizes=[counts[score] for score in sorted(counts, reverse=True)]))
            comparisons[group][condition] = {"ap": sum(aps) / len(aps),
                                              "prevalence_reference": sum(prevs) / len(prevs),
                                              "tie_aware_reference": sum(ties) / len(ties)}
            support[group][condition] = label_support
    decisions = {}
    for condition in conditions:
        decisions[condition] = {}
        for reference in ("prevalence_reference", "tie_aware_reference"):
            summary = _difference_interval({group: comparisons[group][condition]["ap"] - comparisons[group][condition][reference]
                                            for group in groups})
            decisions[condition][reference] = {"summary": summary, "accepts": summary["ci95_low"] > 0}
    return {"schema": "fullsongeval-fma-overlap-sensitivity/v1", "status": "complete",
            "interpretation": "Post-audit sensitivity only; the primary frozen FMA protocol includes these records. Excluding recipient records after reassignment can change label support across conditions.",
            "inputs_sha256": {"fma_manifest": digest(fma_manifest), "mtg_manifest": digest(mtg_manifest),
                              "replay": digest(replay)},
            "original_fma_records": len(fma), "excluded_exact_normalized_text_matches": len(excluded),
            "retained_records": len(fma) - len(excluded), "decision_intervals": decisions,
            "fold_comparisons": comparisons}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fma-manifest", "mtg-manifest", "replay", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    result = run(args.fma_manifest, args.mtg_manifest, args.replay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"excluded": result["excluded_exact_normalized_text_matches"],
                      "retained": result["retained_records"],
                      "decisions": {k: {r: x["accepts"] for r, x in v.items()}
                                    for k, v in result["decision_intervals"].items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
