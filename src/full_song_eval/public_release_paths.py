from __future__ import annotations

from pathlib import Path


PAPER_MATERIAL_PATH_PARTS = frozenset(
    {"paper", "papers", "table", "tables", "figure", "figures", "plot", "plots"}
)
PAPER_MATERIAL_SUFFIXES = frozenset({".bib", ".tex"})


def is_paper_material_path(path: Path | str) -> bool:
    """Return whether a relative release path belongs to paper or visual material."""
    candidate = Path(path)
    return candidate.suffix.lower() in PAPER_MATERIAL_SUFFIXES or any(
        part.lower() in PAPER_MATERIAL_PATH_PARTS for part in candidate.parts
    )
