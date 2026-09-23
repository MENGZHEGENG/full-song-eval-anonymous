from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evaluate_independent_genre_target import _norm, _read_annotations


def test_consensus_parser_keeps_only_unanimous_target(tmp_path: Path) -> None:
    source = tmp_path / "annotations.tsv"
    source.write_text(
        "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tANNOTATIONS\n"
        "track1\tartist1\talbum1\tpath1\t30\tgenre_dortmund---Hip-Hop,Hip-Hop,Hip-Hop\n",
        encoding="utf-8",
    )
    outcomes, artists, classes = _read_annotations(source, {"genre_dortmund"})
    assert outcomes == {"track1": {"genre_dortmund": "hip-hop"}}
    assert artists == {"track1": "artist1"}
    assert classes == {"genre_dortmund": {"hiphop"}}
    assert _norm("Hip-Hop") == _norm("hip hop")


def test_consensus_parser_rejects_disagreement(tmp_path: Path) -> None:
    source = tmp_path / "annotations.tsv"
    source.write_text(
        "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tANNOTATIONS\n"
        "track1\tartist1\talbum1\tpath1\t30\tgenre_dortmund---rock,rock,pop\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-consensus"):
        _read_annotations(source, {"genre_dortmund"})
