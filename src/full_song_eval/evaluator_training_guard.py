from __future__ import annotations

import os
from collections.abc import Mapping

TRAINING_GUARD_ENV = "FULL_SONG_EVAL_ALLOW_EVALUATOR_TRAINING"
TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def training_guard(*, smoke_test_only: bool = False, env: Mapping[str, str] | None = None) -> dict[str, object]:
    source = os.environ if env is None else env
    raw_value = source.get(TRAINING_GUARD_ENV, "")
    env_enabled = raw_value.strip().lower() in TRUE_VALUES
    return {
        "env": TRAINING_GUARD_ENV,
        "env_enabled": env_enabled,
        "env_value_present": bool(raw_value),
        "smoke_test_only": smoke_test_only,
        "allowed": env_enabled or smoke_test_only,
        "mode": "real_training" if env_enabled else "smoke_test" if smoke_test_only else "blocked",
    }
