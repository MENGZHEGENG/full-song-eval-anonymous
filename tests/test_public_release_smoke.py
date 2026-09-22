from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_public_release_core_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]

    assert (root / "README.md").is_file()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src/full_song_eval/__init__.py").is_file()


def test_public_release_package_imports() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import full_song_eval"],
        cwd=root,
        env={"PYTHONPATH": str(root / "src")},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
