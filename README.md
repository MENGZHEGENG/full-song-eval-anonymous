# FullSongEval Full-Song Evaluation

FullSongEval is a claim-gated research codebase for studying full-song generation under label scarcity. It focuses on prompt-matched song evaluation, public-label proxy diagnostics, and reproducible evidence gates for music, singing, lyrics, arrangement, and mix.

## What This Release Supports

- Re-running public-label proxy diagnostics from public metadata and aggregate reports.
- Inspecting the MusicCaps caption/tag proxy diagnostics and representation stress tests.
- Running local unit tests for schemas, claim gates, release policy, and evaluation utilities.
- Reproducing the documented public-label checks with the Python entry points below.

## Claim Boundary

The current evidence supports public-label proxy diagnostics only. It does not claim listener preference, perceptual quality, generator superiority, evaluator accuracy, reranking gains, or optimization improvements. Those claims require public or returned listener labels plus the corresponding claim gates.

## Public Release Contents

The release provides source code, configuration examples, tests, public-label indexes, hashes, and aggregate reports. Paper sources, table rows, figures, generated audio, model weights, raw annotation returns, private keys, site-specific launch scripts, queue commands, node names, account names, and local machine paths remain in the private project repository.

## Reproducible Entry Points

```bash
PYTHONPATH=src python3 -m pytest
PYTHONPATH=src python3 scripts/build_public_release.py --help
PYTHONPATH=src python3 scripts/audit_public_release_export.py --help
```

The commands operate on the source tree and its public-label data paths. The release does not require a LaTeX toolchain.

## Citation

Citation information will be added when a stable software release is available.
