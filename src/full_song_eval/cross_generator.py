from __future__ import annotations

from copy import deepcopy
import random
from typing import Any


ORDER_POLICIES = {"left_first", "alternate", "random"}


def candidate_index(record: dict[str, Any]) -> int | None:
    value = record.get("metadata", {}).get("candidate_index")
    if value is None:
        return None
    return int(value)


def select_prompt_candidates(
    records: list[dict[str, Any]],
    *,
    selected_candidate_index: int,
    missing_candidate_index: int | None = None,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for record in records:
        current_candidate_index = candidate_index(record)
        if current_candidate_index is None:
            current_candidate_index = missing_candidate_index
        if current_candidate_index != selected_candidate_index:
            continue
        prompt_id = str(record["prompt_id"])
        if prompt_id in selected:
            duplicates.append(prompt_id)
            continue
        selected[prompt_id] = record
    if duplicates:
        duplicate_list = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"Duplicate selected candidates for prompt IDs: {duplicate_list}")
    return selected


def cross_generator_pair_records(
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    *,
    left_candidate_index: int = 0,
    right_candidate_index: int = 0,
    left_missing_candidate_index: int | None = None,
    right_missing_candidate_index: int | None = None,
    order_policy: str = "left_first",
    seed: int = 0,
    require_same_prompts: bool = True,
) -> list[dict[str, Any]]:
    if order_policy not in ORDER_POLICIES:
        raise ValueError(f"order_policy must be one of {sorted(ORDER_POLICIES)}")
    left_by_prompt = select_prompt_candidates(
        left_records,
        selected_candidate_index=left_candidate_index,
        missing_candidate_index=left_missing_candidate_index,
    )
    right_by_prompt = select_prompt_candidates(
        right_records,
        selected_candidate_index=right_candidate_index,
        missing_candidate_index=right_missing_candidate_index,
    )
    left_prompts = set(left_by_prompt)
    right_prompts = set(right_by_prompt)
    if require_same_prompts and left_prompts != right_prompts:
        missing_left = sorted(right_prompts - left_prompts)[:10]
        missing_right = sorted(left_prompts - right_prompts)[:10]
        raise ValueError(f"Prompt mismatch: missing_left={missing_left}, missing_right={missing_right}")
    prompt_ids = sorted(left_prompts & right_prompts)
    rng = random.Random(seed)
    paired_records: list[dict[str, Any]] = []
    for prompt_index, prompt_id in enumerate(prompt_ids):
        left_record = left_by_prompt[prompt_id]
        right_record = right_by_prompt[prompt_id]
        left_first = _is_left_first(order_policy, prompt_index, rng)
        first_record, first_role = (left_record, "left") if left_first else (right_record, "right")
        second_record, second_role = (right_record, "right") if left_first else (left_record, "left")
        paired_records.append(
            _with_pair_metadata(
                first_record,
                pair_candidate_index=0,
                source_candidate_index=candidate_index(first_record),
                source_role=first_role,
            )
        )
        paired_records.append(
            _with_pair_metadata(
                second_record,
                pair_candidate_index=1,
                source_candidate_index=candidate_index(second_record),
                source_role=second_role,
            )
        )
    return paired_records


def _is_left_first(order_policy: str, prompt_index: int, rng: random.Random) -> bool:
    if order_policy == "left_first":
        return True
    if order_policy == "alternate":
        return prompt_index % 2 == 0
    return rng.random() < 0.5


def _with_pair_metadata(
    record: dict[str, Any],
    *,
    pair_candidate_index: int,
    source_candidate_index: int | None,
    source_role: str,
) -> dict[str, Any]:
    copied = deepcopy(record)
    metadata = dict(copied.get("metadata", {}))
    metadata["source_candidate_index"] = source_candidate_index
    metadata["source_generator_role"] = source_role
    metadata["candidate_index"] = pair_candidate_index
    copied["metadata"] = metadata
    return copied
