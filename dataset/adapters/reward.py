"""Reward-model input contract."""
from __future__ import annotations

from typing import Any

from .base import select_keys


class RewardInputAdapter:
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        return select_keys(
            batch,
            (
                "input_ids",
                "attention_mask",
                "labels",
                "prompt_length",
                "reward_label",
                "ranking_label",
            ),
        )


__all__ = ["RewardInputAdapter"]
