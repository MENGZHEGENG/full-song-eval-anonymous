"""Validate and summarize the four prespecified Stage-1 diagnostic reports.

This module is site-independent. It treats the ten held-out authors as the
paired inference units and never treats folds, models, configurations, or
random seeds as independent observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from full_song_eval.musiccaps_stage1_diagnostic import (
    CONDITION_NAMES,
    PREDICTOR_NAMES,
    T_CRITICAL_95_BY_DF,
    validate_stage1_report,
)

EXPECTED_DESIGNS = (
    ("aspect_list", 16, 25),
    ("aspect_list", 1296, 5),
    ("audioset_positive_labels", 16, 25),
    ("audioset_positive_labels", 211, 5),
)
EXPECTED_DESIGN_SET = frozenset(EXPECTED_DESIGNS)
CONTRAST_SPECS = {
    "target_mask_minus_prior": (
        ("target_label_masked", "prior_plus_text"),
        ("target_label_masked", "prior"),
    ),
    "global_mask_minus_prior": (
        ("global_union_masked", "prior_plus_text"),
        ("global_union_masked", "prior"),
    ),
    "global_mask_minus_matched_deletion": (
        ("global_union_masked", "prior_plus_text"),
        ("matched_random_deletion", "prior_plus_text"),
    ),
    "target_mask_minus_matched_deletion": (
        ("target_label_masked", "prior_plus_text"),
        ("matched_random_deletion", "prior_plus_text"),
    ),
}
DESCRIPTIVE_ONLY_CONTRASTS = {
    "target_mask_minus_matched_deletion": (
        "descriptive_only; random deletion is count-matched to the global mask, not the target-specific mask"
    )
}
PRIMARY_METRIC = "macro_auprc"


def build_stage1_summary(report_paths: Sequence[Path]) -> dict[str, Any]:
    loaded: list[tuple[Path, dict[str, Any]]] = []
    errors: list[str] = []
    if len(report_paths) != 4:
        errors.append("expected_exactly_four_reports")
    for path in report_paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            errors.append(f"unreadable_report:{path.name}:{type(exc).__name__}")
            continue
        if not isinstance(value, dict):
            errors.append(f"report_is_not_json_object:{path.name}")
            continue
        try:
            report_errors = validate_stage1_report(value)
        except (AttributeError, TypeError, ValueError):
            errors.append(f"invalid_report_schema:{path.name}")
            continue
        for error in report_errors:
            errors.append(f"invalid_report:{path.name}:{error}")
        if report_errors:
            continue
        loaded.append((path, value))

    design_to_report: dict[tuple[str, int, int], tuple[Path, dict[str, Any]]] = {}
    for path, report in loaded:
        try:
            design = _design_key(report)
        except (TypeError, ValueError):
            errors.append(f"invalid_design_fields:{path.name}")
            continue
        if design in design_to_report:
            errors.append(f"duplicate_design:{_design_slug(design)}")
        design_to_report[design] = (path, report)
        if not _valid_report_design(report):
            errors.append(f"invalid_design:{_design_slug(design)}")
        errors.extend(_report_integrity_errors(path, report, design))
    for design in EXPECTED_DESIGNS:
        if design not in design_to_report:
            errors.append(f"missing_design:{_design_slug(design)}")
    for design in design_to_report:
        if design not in EXPECTED_DESIGN_SET:
            errors.append(f"unexpected_design:{_design_slug(design)}")

    if loaded:
        errors.extend(_common_provenance_errors([report for _path, report in loaded]))
        errors.extend(_paired_unit_errors([report for _path, report in design_to_report.values()]))
    if errors:
        return {
            "schema_version": 1,
            "status": "blocked",
            "input_report_count": len(report_paths),
            "validation_errors": sorted(set(errors)),
            "summary_scope": "No scientific summary emitted because report validation failed.",
        }

    ordered = [design_to_report[design] for design in EXPECTED_DESIGNS]
    first_report = ordered[0][1]
    summarized_reports = [_summarize_report(path, report) for path, report in ordered]
    return {
        "schema_version": 1,
        "status": "complete",
        "input_report_count": 4,
        "code_commit": first_report["code_commit"],
        "implementation_sha256": first_report["implementation_sha256"],
        "manifest_sha256": first_report["reproduction"]["manifest_sha256"],
        "summary_implementation_sha256": _sha256(Path(__file__)),
        "independent_unit": "author_id",
        "independent_unit_count": 10,
        "inference_statement": (
            "Paired differences use macro-AUPRC for the same ten held-out authors. "
            "Models, configurations, folds, and seeds are not counted as independent observations."
        ),
        "interval_note": "Per-report paired Student-t 95% intervals; no multiplicity adjustment.",
        "paper_branch_status": "pending_stage2",
        "selected_paper_branches": [],
        "paper_branch_note": (
            "Stage 1 emits only CI-supported directional findings. It does not select a named paper-position "
            "branch; equivalence, calibration, annotation shortcuts, and downstream transfer require later stages."
        ),
        "reports": summarized_reports,
        "input_reports": [
            {
                "file_name": path.name,
                "sha256": _sha256(path),
                "design": _design_slug(_design_key(report)),
            }
            for path, report in ordered
        ],
        "validation_errors": [],
    }


def write_stage1_summary(
    report_paths: Sequence[Path],
    *,
    output_json: Path,
    output_md: Path,
    output_tex: Path,
) -> dict[str, Any]:
    summary = build_stage1_summary(report_paths)
    for path in (output_json, output_md, output_tex):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(_markdown(summary), encoding="utf-8")
    output_tex.write_text(_latex_rows(summary), encoding="utf-8")
    return summary


def _summarize_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    units = _unit_metric_map(report)
    contrasts: dict[str, dict[str, Any]] = {}
    outcomes: list[str] = []
    outcome_branches: dict[str, str] = {}
    inconclusive_contrasts: list[str] = []
    descriptive_only_contrasts: list[str] = []
    for name, (left, right) in CONTRAST_SPECS.items():
        differences = {
            unit_id: _unit_value_for(units[unit_id], *left) - _unit_value_for(units[unit_id], *right)
            for unit_id in sorted(units)
        }
        interval = _paired_t_interval(list(differences.values()))
        interval["metric"] = PRIMARY_METRIC
        interval["left"] = {"condition": left[0], "predictor": left[1]}
        interval["right"] = {"condition": right[0], "predictor": right[1]}
        interval["author_differences"] = {unit_id: round(value, 6) for unit_id, value in differences.items()}
        if name in DESCRIPTIVE_ONLY_CONTRASTS:
            interval["interpretation_scope"] = DESCRIPTIVE_ONLY_CONTRASTS[name]
            interval["outcome_branch"] = "descriptive_only_not_target_deletion_matched"
        else:
            interval["interpretation_scope"] = "prespecified_directional_contrast"
            interval["outcome_branch"] = _outcome_branch(name, interval)
        contrasts[name] = interval
        outcome_branches[name] = interval["outcome_branch"]
        if name in DESCRIPTIVE_ONLY_CONTRASTS:
            descriptive_only_contrasts.append(name)
        elif interval["outcome_branch"].endswith("_inconclusive"):
            inconclusive_contrasts.append(name)
        else:
            outcomes.append(interval["outcome_branch"])
    return {
        "design": _design_slug(_design_key(report)),
        "display_name": _display_name(report),
        "source_file": path.name,
        "target_field": report["target_field"],
        "top_label_count": report["top_label_count"],
        "min_label_count": report["min_label_count"],
        "row_count": report["row_count"],
        "independent_author_count": len(units),
        "key_metrics": {
            "prior": _aggregate_value(report, "normalized_unmasked", "prior"),
            "raw": _aggregate_value(report, "raw", "prior_plus_text"),
            "normalized_unmasked": _aggregate_value(report, "normalized_unmasked", "prior_plus_text"),
            "target_label_masked": _aggregate_value(report, "target_label_masked", "prior_plus_text"),
            "global_union_masked": _aggregate_value(report, "global_union_masked", "prior_plus_text"),
            "matched_random_deletion": _aggregate_value(report, "matched_random_deletion", "prior_plus_text"),
        },
        "paired_contrasts": contrasts,
        "supported_outcomes": sorted(outcomes),
        "outcome_branches": outcome_branches,
        "inconclusive_contrasts": sorted(inconclusive_contrasts),
        "descriptive_only_contrasts": sorted(descriptive_only_contrasts),
    }


def _paired_t_interval(differences: Sequence[float]) -> dict[str, Any]:
    if len(differences) != 10:
        raise ValueError(f"paired author inference requires exactly 10 values, received {len(differences)}")
    mean = sum(differences) / len(differences)
    variance = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
    standard_error = math.sqrt(variance / len(differences))
    critical = T_CRITICAL_95_BY_DF[len(differences) - 1]
    half_width = critical * standard_error
    return {
        "mean_difference": round(mean, 6),
        "ci95_low": round(mean - half_width, 6),
        "ci95_high": round(mean + half_width, 6),
        "sample_standard_deviation": round(math.sqrt(variance), 6),
        "standard_error": round(standard_error, 6),
        "ci95_method": "paired_student_t_95",
        "n_independent_authors": len(differences),
    }


def _outcome_branch(name: str, interval: dict[str, Any]) -> str:
    lower = float(interval["ci95_low"])
    upper = float(interval["ci95_high"])
    names = {
        "target_mask_minus_prior": ("target_mask_above_prior", "target_mask_below_prior", "target_mask_vs_prior_inconclusive"),
        "global_mask_minus_prior": ("global_mask_above_prior", "global_mask_below_prior", "global_mask_vs_prior_inconclusive"),
        "global_mask_minus_matched_deletion": (
            "global_mask_above_matched_deletion",
            "global_mask_below_matched_deletion",
            "global_mask_vs_matched_deletion_inconclusive",
        ),
        "target_mask_minus_matched_deletion": (
            "target_mask_above_matched_deletion",
            "target_mask_below_matched_deletion",
            "target_mask_vs_matched_deletion_inconclusive",
        ),
    }
    positive, negative, inconclusive = names[name]
    if lower > 0.0:
        return positive
    if upper < 0.0:
        return negative
    return inconclusive


def _common_provenance_errors(reports: Sequence[dict[str, Any]]) -> list[str]:
    reproductions = [report.get("reproduction") for report in reports]
    checks = {
        "code_commit": [report.get("code_commit") for report in reports],
        "implementation_sha256": [report.get("implementation_sha256") for report in reports],
        "manifest_sha256": [value.get("manifest_sha256") if isinstance(value, dict) else None for value in reproductions],
        "row_count": [report.get("row_count") for report in reports],
        "seed": [report.get("seed") for report in reports],
    }
    errors = []
    for name, values in checks.items():
        if any(value is None or value == "" for value in values):
            errors.append(f"missing_{name}")
        if any(not _valid_provenance_value(name, value) for value in values):
            errors.append(f"invalid_{name}")
        if values and any(value != values[0] for value in values[1:]):
            errors.append(f"mismatched_{name}")
    return errors


def _valid_provenance_value(name: str, value: Any) -> bool:
    if name == "row_count":
        return type(value) is int and value > 0
    return isinstance(value, str) and bool(value.strip())


def _report_integrity_errors(path: Path, report: dict[str, Any], design: tuple[str, int, int]) -> list[str]:
    errors: list[str] = []
    design_name = _design_slug(design)
    reproduction = report.get("reproduction")
    if not isinstance(reproduction, dict):
        errors.append(f"invalid_reproduction:{path.name}")
    else:
        nested_commit = reproduction.get("code_commit")
        nested_implementation = reproduction.get("implementation_sha256")
        if nested_commit != report.get("code_commit"):
            errors.append(f"nested_code_commit_mismatch:{design_name}")
        if nested_implementation != report.get("implementation_sha256"):
            errors.append(f"nested_implementation_sha256_mismatch:{design_name}")

    units: dict[str, dict[str, Any]] = {}
    folds = report.get("folds")
    if not isinstance(folds, list):
        return [*errors, f"invalid_folds:{design_name}"]
    if len(folds) != 10:
        errors.append(f"invalid_fold_count:{design_name}:{len(folds)}")
    fold_partitions: list[tuple[str, set[str], set[str]]] = []
    seen_fold_ids: set[int] = set()
    for fold_index, fold in enumerate(folds):
        if not isinstance(fold, dict):
            errors.append(f"invalid_fold:{design_name}:{fold_index}")
            continue
        raw_fold_id = fold.get("fold")
        fold_id = str(raw_fold_id) if type(raw_fold_id) is int else str(fold_index)
        if type(raw_fold_id) is not int:
            errors.append(f"invalid_fold_identifier:{design_name}:{fold_index}")
        elif raw_fold_id in seen_fold_ids:
            errors.append(f"duplicate_fold_identifier:{design_name}:{raw_fold_id}")
        else:
            seen_fold_ids.add(raw_fold_id)
        train_units = fold.get("train_units")
        test_units = fold.get("test_units")
        unit_rows = fold.get("unit_metrics")
        if not isinstance(train_units, list) or not isinstance(test_units, list) or not isinstance(unit_rows, list):
            errors.append(f"invalid_fold_units:{design_name}:{fold_id}")
            continue
        if fold.get("label_vocabulary_scope") != "training_fold_only":
            errors.append(f"fold_label_vocabulary_not_training_only:{design_name}:{fold_id}")
        if not all(_valid_unit_identifier(value) for value in [*train_units, *test_units]):
            errors.append(f"invalid_unit_identifier:{design_name}:{fold_id}")
        train_ids = [value for value in train_units if _valid_unit_identifier(value)]
        test_ids = [value for value in test_units if _valid_unit_identifier(value)]
        train_set = set(train_ids)
        test_set = set(test_ids)
        if train_set & test_set:
            errors.append(f"actual_fold_group_overlap:{design_name}:{fold_id}")
        if (
            len(train_units) != 9
            or len(test_units) != 1
            or len(train_set) != len(train_units)
            or len(test_set) != len(test_units)
        ):
            errors.append(f"invalid_leave_one_author_out_partition:{design_name}:{fold_id}")
        fold_partitions.append((fold_id, train_set, test_set))
        metric_ids: list[str] = []
        for unit in unit_rows:
            if not isinstance(unit, dict):
                errors.append(f"invalid_unit_metric:{design_name}:{fold_id}")
                continue
            raw_unit_id = unit.get("unit_id")
            if not _valid_unit_identifier(raw_unit_id):
                errors.append(f"invalid_unit_identifier:{design_name}:{fold_id}")
                continue
            unit_id = raw_unit_id
            metric_ids.append(unit_id)
            if unit_id in units:
                errors.append(f"duplicate_author_unit:{design_name}:{unit_id}")
            else:
                units[unit_id] = unit
        if set(metric_ids) != test_set or len(metric_ids) != len(test_units):
            errors.append(f"fold_unit_metrics_mismatch:{design_name}:{fold_id}")
    if len(units) != 10:
        errors.append(f"expected_ten_authors:{design_name}:{len(units)}")
    all_units = set(units)
    if len(all_units) == 10:
        for fold_id, train_set, test_set in fold_partitions:
            if len(test_set) != 1 or train_set != all_units - test_set:
                errors.append(f"invalid_leave_one_author_out_partition:{design_name}:{fold_id}")

    aggregate = report.get("aggregate_metrics")
    if not isinstance(aggregate, dict):
        return [*errors, f"invalid_aggregate_metrics:{design_name}"]
    for condition in CONDITION_NAMES:
        for predictor in PREDICTOR_NAMES:
            unit_values: list[float] = []
            for unit_id, unit in units.items():
                try:
                    value = unit["metrics"][condition][predictor][PRIMARY_METRIC]
                except (KeyError, TypeError):
                    errors.append(f"missing_unit_metric:{design_name}:{unit_id}:{condition}:{predictor}:{PRIMARY_METRIC}")
                    continue
                if not _valid_probability(value):
                    kind = "nonfinite" if isinstance(value, (int, float)) and not math.isfinite(float(value)) else "invalid"
                    errors.append(f"{kind}_unit_metric:{design_name}:{unit_id}:{condition}:{predictor}:{PRIMARY_METRIC}")
                    continue
                unit_values.append(float(value))
            try:
                aggregate_cell = aggregate[condition][predictor]
                aggregate_value = aggregate_cell[f"{PRIMARY_METRIC}_mean"]
            except (KeyError, TypeError):
                errors.append(f"missing_aggregate_metric:{design_name}:{condition}:{predictor}")
                continue
            if not _valid_probability(aggregate_value):
                errors.append(f"invalid_aggregate_metric:{design_name}:{condition}:{predictor}")
                continue
            if aggregate_cell.get("independent_unit_count") != 10:
                errors.append(f"invalid_aggregate_unit_count:{design_name}:{condition}:{predictor}")
            if aggregate_cell.get("uncertainty_unit") != "heldout_author_id_group":
                errors.append(f"invalid_aggregate_uncertainty_unit:{design_name}:{condition}:{predictor}")
            if len(unit_values) == 10:
                paired_mean = sum(unit_values) / len(unit_values)
                if not math.isclose(float(aggregate_value), paired_mean, abs_tol=1e-6):
                    errors.append(f"aggregate_unit_mean_mismatch:{design_name}:{condition}:{predictor}")
    return errors


def _valid_probability(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _valid_unit_identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _paired_unit_errors(reports: Sequence[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    expected_units: set[str] | None = None
    for report in reports:
        design = _design_slug(_design_key(report))
        units: dict[str, dict[str, Any]] = {}
        folds = report.get("folds")
        if not isinstance(folds, list):
            continue
        for fold in folds:
            if not isinstance(fold, dict):
                continue
            unit_rows = fold.get("unit_metrics")
            if not isinstance(unit_rows, list):
                continue
            for unit in unit_rows:
                if not isinstance(unit, dict) or not _valid_unit_identifier(unit.get("unit_id")):
                    continue
                unit_id = unit["unit_id"]
                if unit_id in units:
                    errors.append(f"duplicate_author_unit:{design}:{unit_id}")
                    continue
                units[unit_id] = unit
        if len(units) != 10:
            errors.append(f"expected_ten_authors:{design}:{len(units)}")
        unit_ids = set(units)
        if expected_units is None:
            expected_units = unit_ids
        elif unit_ids != expected_units:
            errors.append(f"mismatched_author_set:{design}")
        for unit_id, unit in units.items():
            for left, right in CONTRAST_SPECS.values():
                for condition, predictor in (left, right):
                    try:
                        _unit_value_for(unit, condition, predictor)
                    except (KeyError, TypeError, ValueError):
                        errors.append(f"missing_unit_metric:{design}:{unit_id}:{condition}:{predictor}:{PRIMARY_METRIC}")
    return errors


def _valid_report_design(report: dict[str, Any]) -> bool:
    return (
        report.get("schema_version") == 1
        and _design_key(report) in EXPECTED_DESIGN_SET
        and report.get("independent_unit") == "author_id"
        and report.get("independent_unit_count") == 10
        and report.get("requested_fold_count") == 10
        and report.get("effective_fold_count") == 10
        and report.get("primary_metric") == "macro_auprc_across_heldout_independent_units"
        and tuple(report.get("conditions", [])) == CONDITION_NAMES
        and tuple(report.get("predictors", [])) == PREDICTOR_NAMES
    )


def _unit_metric_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    units: dict[str, dict[str, Any]] = {}
    for fold in report.get("folds", []):
        for unit in fold.get("unit_metrics", []):
            unit_id = str(unit.get("unit_id"))
            if unit_id in units:
                raise ValueError(f"duplicate held-out author: {unit_id}")
            units[unit_id] = unit
    return units


def _unit_value_for(unit: dict[str, Any], condition: str, predictor: str) -> float:
    value = unit["metrics"][condition][predictor][PRIMARY_METRIC]
    if value is None:
        raise ValueError(f"missing {PRIMARY_METRIC} for unit {unit.get('unit_id')}")
    return float(value)


def _aggregate_value(report: dict[str, Any], condition: str, predictor: str) -> float:
    return float(report["aggregate_metrics"][condition][predictor][f"{PRIMARY_METRIC}_mean"])


def _design_key(report: dict[str, Any]) -> tuple[str, int, int]:
    target_field = report.get("target_field")
    top_label_count = report.get("top_label_count")
    min_label_count = report.get("min_label_count")
    if not isinstance(target_field, str) or type(top_label_count) is not int or type(min_label_count) is not int:
        raise ValueError("design fields must be a string and two integers")
    return (
        target_field,
        top_label_count,
        min_label_count,
    )


def _design_slug(design: tuple[str, int, int]) -> str:
    return f"{design[0]}:{design[1]}:{design[2]}"


def _display_name(report: dict[str, Any]) -> str:
    target = "Aspect labels" if report["target_field"] == "aspect_list" else "AudioSet labels"
    return f"{target} ({report['top_label_count']})"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _format(value: float) -> str:
    return f"{value:.4f}"


def _format_interval(contrast: dict[str, Any]) -> str:
    return f"{contrast['mean_difference']:.4f} [{contrast['ci95_low']:.4f}, {contrast['ci95_high']:.4f}]"


def _markdown(summary: dict[str, Any]) -> str:
    lines = ["# Stage-1 Existing-Data Diagnostic", ""]
    if summary["status"] != "complete":
        lines.extend(["Status: **BLOCKED**", "", *[f"- `{error}`" for error in summary["validation_errors"]], ""])
        return "\n".join(lines)
    lines.extend(
        [
            "Status: complete and validated.",
            "",
            "N = 10 held-out authors for every paired comparison. Models, settings, folds, and seeds are not independent observations.",
            "",
            "Named paper-position branch: pending Stage 2; none selected from Stage 1 alone.",
            "",
            "## Key macro-AUPRC results",
            "",
            "| Design | Prior | Raw | Normalized | Target mask | Global mask | Matched deletion |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for report in summary["reports"]:
        metrics = report["key_metrics"]
        lines.append(
            f"| {report['display_name']} | {_format(metrics['prior'])} | {_format(metrics['raw'])} | "
            f"{_format(metrics['normalized_unmasked'])} | {_format(metrics['target_label_masked'])} | "
            f"{_format(metrics['global_union_masked'])} | {_format(metrics['matched_random_deletion'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired author-level differences",
            "",
            "Each cell is mean macro-AUPRC difference with a paired Student-t 95% interval.",
            "",
            "| Design | Target - prior | Global - prior | Global - matched | Target - matched* | Supported branches | Inconclusive |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for report in summary["reports"]:
        contrasts = report["paired_contrasts"]
        lines.append(
            f"| {report['display_name']} | {_format_interval(contrasts['target_mask_minus_prior'])} | "
            f"{_format_interval(contrasts['global_mask_minus_prior'])} | "
            f"{_format_interval(contrasts['global_mask_minus_matched_deletion'])} | "
            f"{_format_interval(contrasts['target_mask_minus_matched_deletion'])} | "
            f"{', '.join(report['supported_outcomes']) or 'none'} | "
            f"{', '.join(report['inconclusive_contrasts']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "*Target - matched deletion is descriptive only: the random deletion control matches the global mask's deletion count, not the target-specific mask's count.",
            "",
            summary["interval_note"],
            "",
        ]
    )
    return "\n".join(lines)


def _latex_rows(summary: dict[str, Any]) -> str:
    if summary["status"] != "complete":
        return "% BLOCKED: Stage-1 summary validation failed.\n"
    lines = [r"% Design & Prior & Raw & Normalized & Target mask & Global mask & Matched deletion \\"]
    for report in summary["reports"]:
        metrics = report["key_metrics"]
        lines.append(
            f"{report['display_name']} & {_format(metrics['prior'])} & {_format(metrics['raw'])} & "
            f"{_format(metrics['normalized_unmasked'])} & {_format(metrics['target_label_masked'])} & "
            f"{_format(metrics['global_union_masked'])} & {_format(metrics['matched_random_deletion'])} " + r"\\"
        )
    lines.append(r"% Design contrast & mean [paired 95 percent CI] \\")
    for report in summary["reports"]:
        for name in CONTRAST_SPECS:
            marker = "*" if name in DESCRIPTIVE_ONLY_CONTRASTS else ""
            display_contrast = name.replace("_", " ") + marker
            lines.append(f"{report['display_name']}: {display_contrast} & {_format_interval(report['paired_contrasts'][name])} " + r"\\")
    lines.append(
        "% * descriptive only: random deletion is count-matched to the global mask, not the target-specific mask."
    )
    lines.append("% Named paper-position branch pending Stage 2; none selected from Stage 1 alone.")
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--output-tex", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = write_stage1_summary(
        args.reports,
        output_json=args.output_json,
        output_md=args.output_md,
        output_tex=args.output_tex,
    )
    print(json.dumps({"status": summary["status"], "validation_errors": summary["validation_errors"]}, sort_keys=True))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
