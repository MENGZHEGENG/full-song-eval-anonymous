"""Recorded external-corpus check of the manuscript's text--label rule.

The source is the public MARD album-review genre-classification subset. Raw
reviews remain under ignored ``data/``; the report contains aggregate numbers
and source digests only. This local protocol lacks an independent pre-run seal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from full_song_eval.musiccaps_gate_calibration import (
    _average_precision_pairs,
    _fit_label_model,
    _label_set_permutation,
    _normalized_tokens,
    _row_key,
    _score_label_model,
    _target_view_from_tokens,
)
from full_song_eval.mtg_jamendo_metadata_calibration import _codeword_injection
from full_song_eval.random_ranking_baseline import expected_tie_aware_average_precision


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/mard_transfer_protocol.json"
DEFAULT_OUTPUT = ROOT / "reports/mard_prospective_transfer.json"
T_CRITICAL_DF9 = 2.2621571627409915
VIEW = "target_label_masked"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def interval(differences: list[float]) -> dict[str, float | int | bool]:
    if len(differences) != 10:
        raise ValueError("the frozen decision uses exactly ten held-out folds")
    mean = math.fsum(differences) / len(differences)
    sd = math.sqrt(math.fsum((value - mean) ** 2 for value in differences) / 9)
    half = T_CRITICAL_DF9 * sd / math.sqrt(10)
    return {
        "mean": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "accepted": mean - half > 0,
        "fold_count": 10,
    }


def fold_id(album_id: str, *, seed: str, fold_count: int) -> int:
    digest = hashlib.sha256(f"{seed}|{album_id}".encode("utf-8")).hexdigest()
    return int(digest, 16) % fold_count


def evaluate_condition(
    train: list[dict], test: list[dict], *, labels: list[str], seed: str
) -> dict[str, float | int | dict[str, int]]:
    train_tokens = [(_row_key(row, i), _normalized_tokens(row["caption"])) for i, row in enumerate(train)]
    test_tokens = [(_row_key(row, i), _normalized_tokens(row["caption"])) for i, row in enumerate(test)]
    aps: list[float] = []
    tie_references: list[float] = []
    supports: dict[str, int] = {}
    for label in labels:
        model = _fit_label_model(
            train, train_tokens, label=label, target_field="genre_list", view=VIEW, seed=f"{seed}|train"
        )
        scores: list[float] = []
        actual: list[bool] = []
        for row, (row_key, tokens) in zip(test, test_tokens):
            viewed, _diagnostic = _target_view_from_tokens(
                tokens, target_label=label, view=VIEW, seed=f"{seed}|test", row_key=row_key
            )
            scores.append(_score_label_model(model, viewed))
            actual.append(label in row["genre_list"])
        k = sum(actual)
        supports[label] = k
        if k == 0:
            raise ValueError(f"held-out fold has zero support for {label}")
        aps.append(_average_precision_pairs(zip(scores, actual)))
        ties = Counter(scores)
        ordered_ties = [ties[score] for score in sorted(ties, reverse=True)]
        tie_references.append(expected_tie_aware_average_precision(n=len(test), k=k, tie_group_sizes=ordered_ties))
    return {
        "macro_ap": math.fsum(aps) / len(aps),
        "prevalence": math.fsum(supports[label] / len(test) for label in labels) / len(labels),
        "tie_aware_reference": math.fsum(tie_references) / len(tie_references),
        "heldout_count": len(test),
        "label_support": supports,
    }


def run(*, protocol_path: Path, output_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source = ROOT / protocol["data_path"]
    if sha256(source) != protocol["source_sha256"]:
        raise ValueError("MARD source digest differs from frozen protocol")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if len(raw) != 1300 or len(raw) != len(set(raw)):
        raise ValueError("unexpected MARD album population")
    text_digests = [hashlib.sha256(item["all_text"].encode("utf-8")).hexdigest() for item in raw.values()]
    if len(set(text_digests)) != len(text_digests):
        raise ValueError("exact review-text duplicate across albums")
    labels = sorted({item["genre"] for item in raw.values()} - {protocol["excluded_label"]})
    if len(labels) != 12 or any(not _normalized_tokens(label) for label in labels):
        raise ValueError("the frozen twelve-label vocabulary is unavailable")
    seed = protocol["fold_seed"]
    count = protocol["fold_count"]
    if count != 10:
        raise ValueError("the frozen protocol specifies ten folds")
    rows = [
        {
            "row_id": album_id,
            "caption": raw[album_id]["all_text"],
            "genre_list": [raw[album_id]["genre"]],
            "fold": fold_id(album_id, seed=seed, fold_count=count),
        }
        for album_id in sorted(raw)
    ]
    folds: list[dict] = []
    for heldout in range(count):
        train = [row for row in rows if row["fold"] != heldout]
        test = [row for row in rows if row["fold"] == heldout]
        if not train or not test or any(sum(label in row["genre_list"] for row in test) == 0 for label in labels):
            raise ValueError(f"fold {heldout} lacks a frozen label or records")
        conditions: dict[str, dict] = {}
        conditions["observed"] = evaluate_condition(train, test, labels=labels, seed=f"{seed}|fold:{heldout}|observed")
        for name in ("rotation_a", "rotation_b"):
            rotated_train = _label_set_permutation(train, target_field="genre_list", seed=f"{seed}|fold:{heldout}|{name}|train")
            rotated_test = _label_set_permutation(test, target_field="genre_list", seed=f"{seed}|fold:{heldout}|{name}|test")
            result = evaluate_condition(
                rotated_train, rotated_test, labels=labels, seed=f"{seed}|fold:{heldout}|{name}"
            )
            result["changed_heldout_labels"] = sum(a["genre_list"] != b["genre_list"] for a, b in zip(test, rotated_test))
            conditions[name] = result
        conditions["codeword"] = evaluate_condition(
            _codeword_injection(train, labels=labels, seed=f"{seed}|codeword"),
            _codeword_injection(test, labels=labels, seed=f"{seed}|codeword"),
            labels=labels,
            seed=f"{seed}|fold:{heldout}|codeword",
        )
        folds.append({"fold": heldout, "conditions": conditions})
    comparisons: dict[str, dict] = {}
    for name in ("observed", "rotation_a", "rotation_b", "codeword"):
        comparisons[name] = {
            "prevalence": interval([
                fold["conditions"][name]["macro_ap"] - fold["conditions"][name]["prevalence"] for fold in folds
            ]),
            "tie_aware": interval([
                fold["conditions"][name]["macro_ap"] - fold["conditions"][name]["tie_aware_reference"] for fold in folds
            ]),
        }
    comparisons["sensitivity"] = {
        "codeword_minus_observed": interval([
            fold["conditions"]["codeword"]["macro_ap"] - fold["conditions"]["observed"]["macro_ap"]
            for fold in folds
        ])
    }
    report = {
        "schema_version": "full-song-eval-mard-prospective-transfer/v1",
        "status": "complete",
        "source_sha256": sha256(source),
        "protocol_sha256": sha256(protocol_path),
        "implementation_sha256": sha256(Path(__file__)),
        "source_commit": protocol["source_commit"],
        "source_scope": protocol["scope"],
        "album_count": len(rows),
        "fold_count": count,
        "labels": labels,
        "excluded_label": protocol["excluded_label"],
        "artist_disjointness": "not verified; source subset has no artist field",
        "folds": folds,
        "comparisons": comparisons,
        "candidate_audit_passes": bool(comparisons["sensitivity"]["codeword_minus_observed"]["accepted"])
        and not comparisons["rotation_a"]["prevalence"]["accepted"]
        and not comparisons["rotation_b"]["prevalence"]["accepted"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(protocol_path=args.protocol, output_path=args.output)
    print(json.dumps({"status": report["status"], "candidate_audit_passes": report["candidate_audit_passes"], "comparisons": report["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
