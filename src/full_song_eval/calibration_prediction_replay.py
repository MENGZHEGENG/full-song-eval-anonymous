"""Write deterministic per-example score replays for the metadata audit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from full_song_eval.musiccaps_gate_calibration import (
    _fit_label_model,
    _average_precision_pairs,
    _label_set_permutation,
    _load_manifest,
    _normalized_tokens,
    _partial_label_set_permutation,
    _row_key,
    _row_labels,
    _score_label_model,
    _target_view_from_tokens,
    _unit_value,
    _within_author_caption_permutation,
)
from full_song_eval.mtg_jamendo_metadata_calibration import _codeword_injection


DEFAULT_SEED = "full-song-eval-gate-calibration-v1"
PRIMARY_VIEW = "target_label_masked"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scenario_records(
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    *,
    labels: Sequence[str],
    target_field: str,
    seed: str,
    source: str,
    heldout_group: str,
    condition: str,
) -> Iterable[dict[str, Any]]:
    train_tokens = [
        (_row_key(item, index), _normalized_tokens(str(item.get("caption", ""))))
        for index, item in enumerate(train_rows)
    ]
    test_tokens = [
        (_row_key(item, index), _normalized_tokens(str(item.get("caption", ""))))
        for index, item in enumerate(test_rows)
    ]
    for label in labels:
        model = _fit_label_model(
            train_rows,
            train_tokens,
            label=label,
            target_field=target_field,
            view=PRIMARY_VIEW,
            seed=f"{seed}|train",
        )
        frequency_score = (int(model["positive_rows"]) + 0.5) / (int(model["row_count"]) + 1.0)
        for item, (record_id, tokens) in zip(test_rows, test_tokens):
            viewed, diagnostic = _target_view_from_tokens(
                tokens,
                target_label=label,
                view=PRIMARY_VIEW,
                seed=f"{seed}|test",
                row_key=record_id,
            )
            yield {
                "schema": "fullsongeval-calibration-prediction-replay/v1",
                "source": source,
                "heldout_group": heldout_group,
                "condition": condition,
                "view": PRIMARY_VIEW,
                "record_id": record_id,
                "label": label,
                "target": label in set(_row_labels(item, target_field)),
                "score": _score_label_model(model, viewed),
                "frequency_score": frequency_score,
                "score_direction": "higher_is_more_positive",
                "target_word_removals": int(diagnostic["removed_token_count"]),
            }


def _write_gzip_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                    count += 1
    return count, _sha256(path)


def summarize_prediction_replay(replay_path: Path) -> dict[str, Any]:
    """Recompute primary-view fold summaries from a sorted replay export.

    The exporter writes records grouped by held-out group, condition, and label.
    Streaming one label at a time keeps verification bounded by the largest
    label-by-fold block, not the full replay size.
    """

    source: str | None = None
    summaries: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    current_key: tuple[str, str, str] | None = None
    closed_keys: set[tuple[str, str, str]] = set()
    scores: list[tuple[float, bool]] = []
    removals = 0

    def flush_label() -> None:
        nonlocal scores, removals
        if current_key is None:
            return
        heldout_group, condition, _label = current_key
        summary = summaries[heldout_group].setdefault(
            condition,
            {
                "per_label_auprc": [],
                "labels_with_positive_support": 0,
                "positive_pair_count": 0,
                "evaluated_pair_count": 0,
                "target_word_removals": 0,
            },
        )
        positive_count = sum(target for _score, target in scores)
        summary["positive_pair_count"] += positive_count
        summary["evaluated_pair_count"] += len(scores)
        summary["target_word_removals"] += removals
        if positive_count:
            summary["per_label_auprc"].append(_average_precision_pairs(scores))
            summary["labels_with_positive_support"] += 1
        scores = []
        removals = 0

    with gzip.open(replay_path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if record.get("view") != PRIMARY_VIEW:
                raise ValueError(f"line {line_number} does not use {PRIMARY_VIEW}")
            record_source = str(record.get("source", ""))
            if not record_source:
                raise ValueError(f"line {line_number} has no source")
            if source is None:
                source = record_source
            elif source != record_source:
                raise ValueError("replay mixes sources")
            key = (str(record["heldout_group"]), str(record["condition"]), str(record["label"]))
            if key != current_key:
                flush_label()
                if key in closed_keys:
                    raise ValueError("replay repeats a completed label block")
                if current_key is not None:
                    closed_keys.add(current_key)
                current_key = key
            scores.append((float(record["score"]), bool(record["target"])))
            removals += int(record["target_word_removals"])
    flush_label()
    if source is None:
        raise ValueError("replay has no records")

    finalized: dict[str, dict[str, dict[str, Any]]] = {}
    for heldout_group, conditions in summaries.items():
        finalized[heldout_group] = {}
        for condition, summary in conditions.items():
            label_scores = summary.pop("per_label_auprc")
            summary["macro_auprc"] = round(sum(label_scores) / len(label_scores), 6) if label_scores else None
            finalized[heldout_group][condition] = summary
    return {
        "schema": "fullsongeval-calibration-prediction-replay-summary/v1",
        "source": source,
        "replay_sha256": _sha256(replay_path),
        "groups": finalized,
    }


def verify_prediction_replay_against_report(replay_path: Path, report_path: Path) -> dict[str, Any]:
    """Check replay-derived primary metrics against the canonical audit report."""

    summary = summarize_prediction_replay(replay_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source = summary["source"]
    if source == "MusicCaps":
        entries = report.get("per_author_metrics", [])
        group_key = "author_id"
        corruption_path = ("label_corruption", "1.0")
    elif source == "MTG-Jamendo":
        entries = report.get("folds", [])
        group_key = "heldout_artist_fold"
        corruption_path = ("full_label_corruption",)
    else:
        raise ValueError(f"unsupported replay source: {source}")

    canonical = {str(entry[group_key]): entry["condition_metrics"] for entry in entries}
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for group, conditions in summary["groups"].items():
        report_conditions = canonical.get(group)
        if report_conditions is None:
            mismatches.append({"group": group, "reason": "missing_group"})
            continue
        for condition, actual in conditions.items():
            path = corruption_path if condition == "full_label_corruption" else (condition,)
            expected: Any = report_conditions
            for component in path:
                if component not in expected:
                    mismatches.append({"group": group, "condition": condition, "reason": "missing_condition"})
                    expected = None
                    break
                expected = expected[component]
            if expected is None:
                continue
            expected = expected.get(PRIMARY_VIEW)
            if expected is None:
                mismatches.append({"group": group, "condition": condition, "reason": "missing_primary_view"})
                continue
            checked += 1
            expected_removals = expected.get("target_word_removals")
            if expected_removals is None:
                expected_removals = expected.get("deletion_diagnostics", {}).get("removed_token_count")
            fields = ("macro_auprc", "labels_with_positive_support", "positive_pair_count", "evaluated_pair_count")
            for field in fields:
                if actual.get(field) != expected.get(field):
                    mismatches.append(
                        {
                            "group": group,
                            "condition": condition,
                            "field": field,
                            "expected": expected.get(field),
                            "actual": actual.get(field),
                        }
                    )
            if expected_removals is not None and actual["target_word_removals"] != expected_removals:
                mismatches.append(
                    {
                        "group": group,
                        "condition": condition,
                        "field": "target_word_removals",
                        "expected": expected_removals,
                        "actual": actual["target_word_removals"],
                    }
                )
    return {
        "schema": "fullsongeval-calibration-prediction-replay-verification/v1",
        "status": "pass" if not mismatches else "fail",
        "source": source,
        "replay_sha256": summary["replay_sha256"],
        "report_sha256": _sha256(report_path),
        "checked_group_count": len(summary["groups"]),
        "checked_condition_count": checked,
        "mismatches": mismatches,
    }


def write_musiccaps_prediction_replay(
    *,
    manifest_path: Path,
    output_path: Path,
    top_label_count: int = 16,
    min_label_count: int = 25,
    split_unit: str = "author_id",
    target_field: str = "aspect_list",
    seed: str = DEFAULT_SEED,
) -> dict[str, Any]:
    """Replay primary MusicCaps controls as compact per-example score records."""

    rows = _load_manifest(manifest_path)
    groups = sorted({_unit_value(item, split_unit) for item in rows if _unit_value(item, split_unit)})
    if len(groups) != 10:
        raise ValueError(f"expected exactly ten nonempty groups, found {len(groups)}")

    def records() -> Iterable[dict[str, Any]]:
        for fold, heldout in enumerate(groups):
            train = [item for item in rows if _unit_value(item, split_unit) != heldout]
            test = [item for item in rows if _unit_value(item, split_unit) == heldout]
            counts = Counter(label for item in train for label in _row_labels(item, target_field))
            labels = [
                label
                for label, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
                if count >= min_label_count
            ][:top_label_count]
            if len(labels) != top_label_count:
                raise ValueError(f"fold {fold} lacks {top_label_count} train-supported labels")
            fold_seed = f"{seed}|fold:{fold}"
            scenarios = (
                ("observed", train, test, f"{fold_seed}|observed"),
                (
                    "label_set_permutation",
                    _label_set_permutation(train, target_field=target_field, seed=f"{fold_seed}|label-permutation|train"),
                    _label_set_permutation(test, target_field=target_field, seed=f"{fold_seed}|label-permutation|test"),
                    f"{fold_seed}|label-permutation",
                ),
                (
                    "within_author_caption_permutation",
                    _within_author_caption_permutation(train, split_unit=split_unit, seed=f"{fold_seed}|caption-permutation|train"),
                    _within_author_caption_permutation(test, split_unit=split_unit, seed=f"{fold_seed}|caption-permutation|test"),
                    f"{fold_seed}|caption-permutation",
                ),
                (
                    "full_label_corruption",
                    _partial_label_set_permutation(train, target_field=target_field, level=1.0, seed=f"{fold_seed}|corruption|train"),
                    _partial_label_set_permutation(test, target_field=target_field, level=1.0, seed=f"{fold_seed}|corruption|test"),
                    f"{fold_seed}|corruption|1.0",
                ),
            )
            for condition, scenario_train, scenario_test, scenario_seed in scenarios:
                yield from _scenario_records(
                    scenario_train,
                    scenario_test,
                    labels=labels,
                    target_field=target_field,
                    seed=scenario_seed,
                    source="MusicCaps",
                    heldout_group=heldout,
                    condition=condition,
                )

    count, digest = _write_gzip_jsonl(output_path, records())
    return {
        "schema": "fullsongeval-calibration-prediction-replay-receipt/v1",
        "source": "MusicCaps",
        "manifest_sha256": _sha256(manifest_path),
        "output": str(output_path),
        "record_count": count,
        "sha256": digest,
        "primary_view": PRIMARY_VIEW,
        "conditions": [
            "observed",
            "label_set_permutation",
            "within_author_caption_permutation",
            "full_label_corruption",
        ],
        "score_direction": "higher_is_more_positive",
    }


def write_mtg_jamendo_prediction_replay(
    *,
    manifest_path: Path,
    output_path: Path,
    top_label_count: int = 16,
    min_label_count: int = 25,
    seed: str = "full-song-eval-mtg-jamendo-metadata-v1",
) -> dict[str, Any]:
    """Replay primary MTG-Jamendo controls as compact per-example score records."""

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = sorted({str(item.get("artist_fold", "")) for item in rows})
    if len(groups) != 10 or any(not group for group in groups):
        raise ValueError(f"expected exactly ten nonempty artist folds, found {len(groups)}")

    def records() -> Iterable[dict[str, Any]]:
        for fold, heldout in enumerate(groups):
            train = [item for item in rows if str(item["artist_fold"]) != heldout]
            test = [item for item in rows if str(item["artist_fold"]) == heldout]
            counts = Counter(label for item in train for label in _row_labels(item, "genre_list"))
            labels = [
                label
                for label, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
                if count >= min_label_count
            ][:top_label_count]
            if len(labels) != top_label_count:
                raise ValueError(f"fold {fold} lacks {top_label_count} train-supported labels")
            fold_seed = f"{seed}|fold:{fold}"
            scenarios = (
                ("observed", train, test, f"{fold_seed}|observed"),
                (
                    "label_set_permutation",
                    _label_set_permutation(train, target_field="genre_list", seed=f"{fold_seed}|permuted-train"),
                    _label_set_permutation(test, target_field="genre_list", seed=f"{fold_seed}|permuted-test"),
                    f"{fold_seed}|permuted",
                ),
                (
                    "full_label_corruption",
                    _partial_label_set_permutation(train, target_field="genre_list", level=1.0, seed=f"{fold_seed}|corrupt-train"),
                    _partial_label_set_permutation(test, target_field="genre_list", level=1.0, seed=f"{fold_seed}|corrupt-test"),
                    f"{fold_seed}|corrupt",
                ),
                (
                    "codeword_injection",
                    _codeword_injection(train, labels=labels, seed=f"{fold_seed}|codeword-map"),
                    _codeword_injection(test, labels=labels, seed=f"{fold_seed}|codeword-map"),
                    f"{fold_seed}|codeword",
                ),
            )
            for condition, scenario_train, scenario_test, scenario_seed in scenarios:
                yield from _scenario_records(
                    scenario_train,
                    scenario_test,
                    labels=labels,
                    target_field="genre_list",
                    seed=scenario_seed,
                    source="MTG-Jamendo",
                    heldout_group=heldout,
                    condition=condition,
                )

    count, digest = _write_gzip_jsonl(output_path, records())
    return {
        "schema": "fullsongeval-calibration-prediction-replay-receipt/v1",
        "source": "MTG-Jamendo",
        "manifest_sha256": _sha256(manifest_path),
        "output": str(output_path),
        "record_count": count,
        "sha256": digest,
        "primary_view": PRIMARY_VIEW,
        "conditions": [
            "observed",
            "label_set_permutation",
            "full_label_corruption",
            "codeword_injection",
        ],
        "score_direction": "higher_is_more_positive",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musiccaps-manifest", type=Path, required=True)
    parser.add_argument("--musiccaps-output", type=Path, required=True)
    parser.add_argument("--musiccaps-report", type=Path)
    parser.add_argument("--mtg-manifest", type=Path)
    parser.add_argument("--mtg-output", type=Path)
    parser.add_argument("--mtg-report", type=Path)
    parser.add_argument("--top-label-count", type=int, default=16)
    parser.add_argument("--min-label-count", type=int, default=25)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--mtg-seed", default="full-song-eval-mtg-jamendo-metadata-v1")
    parser.add_argument("--verification-output", type=Path)
    args = parser.parse_args(argv)
    musiccaps_receipt = write_musiccaps_prediction_replay(
        manifest_path=args.musiccaps_manifest,
        output_path=args.musiccaps_output,
        top_label_count=args.top_label_count,
        min_label_count=args.min_label_count,
        seed=args.seed,
    )
    receipt: dict[str, Any] = {"musiccaps_export": musiccaps_receipt}
    if (args.mtg_manifest is None) != (args.mtg_output is None):
        parser.error("--mtg-manifest and --mtg-output must be supplied together")
    if args.mtg_manifest is not None and args.mtg_output is not None:
        receipt["mtg_jamendo"] = write_mtg_jamendo_prediction_replay(
            manifest_path=args.mtg_manifest,
            output_path=args.mtg_output,
            top_label_count=args.top_label_count,
            min_label_count=args.min_label_count,
            seed=args.mtg_seed,
        )
    if (args.mtg_manifest is None) != (args.mtg_report is None):
        parser.error("--mtg-manifest and --mtg-report must be supplied together for verification")
    if args.verification_output is not None and args.musiccaps_report is None:
        parser.error("--verification-output requires --musiccaps-report")
    verification: dict[str, Any] = {}
    if args.musiccaps_report is not None:
        verification["musiccaps"] = verify_prediction_replay_against_report(
            args.musiccaps_output, args.musiccaps_report
        )
    if args.mtg_report is not None:
        verification["mtg_jamendo"] = verify_prediction_replay_against_report(args.mtg_output, args.mtg_report)
    if verification:
        receipt["verification"] = verification
    if args.verification_output is not None:
        args.verification_output.parent.mkdir(parents=True, exist_ok=True)
        args.verification_output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if all(item["status"] == "pass" for item in verification.values()) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
