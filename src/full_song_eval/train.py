from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="FullSongEval training entrypoint placeholder.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without training.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    if args.dry_run:
        print(f"Dry-run OK: found {config_path}")
        return 0

    raise SystemExit(
        "Training is intentionally not implemented yet. Define the concrete baseline/generator command before launching GPU jobs."
    )


if __name__ == "__main__":
    raise SystemExit(main())
