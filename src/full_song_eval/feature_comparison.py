from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TableSpec:
    label: str
    path: Path


@dataclass(frozen=True)
class FieldSummary:
    label: str
    field: str
    records: int
    numeric_records: int
    missing_records: int
    minimum: float | None
    mean: float | None
    maximum: float | None

    def as_row(self) -> dict[str, str | int | float | None]:
        return {
            "label": self.label,
            "field": self.field,
            "records": self.records,
            "numeric_records": self.numeric_records,
            "missing_records": self.missing_records,
            "min": self.minimum,
            "mean": self.mean,
            "max": self.maximum,
        }


def parse_table_spec(value: str) -> TableSpec:
    if "=" not in value:
        raise ValueError("input table specs must use LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise ValueError("table label must not be empty")
    if not path:
        raise ValueError("table path must not be empty")
    return TableSpec(label=label, path=Path(path))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def summarize_field(label: str, rows: list[dict[str, str]], field: str) -> FieldSummary:
    values = [parse_float(row.get(field)) for row in rows]
    numeric_values = [value for value in values if value is not None]
    if numeric_values:
        minimum: float | None = min(numeric_values)
        mean: float | None = sum(numeric_values) / len(numeric_values)
        maximum: float | None = max(numeric_values)
    else:
        minimum = mean = maximum = None
    return FieldSummary(
        label=label,
        field=field,
        records=len(rows),
        numeric_records=len(numeric_values),
        missing_records=len(rows) - len(numeric_values),
        minimum=minimum,
        mean=mean,
        maximum=maximum,
    )


def common_numeric_fields(tables: Iterable[list[dict[str, str]]]) -> list[str]:
    table_list = list(tables)
    if not table_list:
        return []
    common = set(table_list[0][0].keys()) if table_list[0] else set()
    for rows in table_list[1:]:
        common &= set(rows[0].keys()) if rows else set()
    numeric_fields = []
    for field in sorted(common):
        if all(any(parse_float(row.get(field)) is not None for row in rows) for rows in table_list):
            numeric_fields.append(field)
    return numeric_fields


def summarize_tables(specs: list[TableSpec], fields: list[str] | None = None) -> list[FieldSummary]:
    loaded = [(spec, read_csv_rows(spec.path)) for spec in specs]
    selected_fields = fields or common_numeric_fields(rows for _, rows in loaded)
    summaries: list[FieldSummary] = []
    for spec, rows in loaded:
        for field in selected_fields:
            summaries.append(summarize_field(spec.label, rows, field))
    return summaries
