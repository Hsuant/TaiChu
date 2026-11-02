"""On-policy trajectory input contract for future PPO/GRPO trainers."""
from __future__ import annotations

from typing import Any


class TrajectoryInputAdapter:
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        names = (
            "prompt_input_ids",
            "prompt_attention_mask",
            "response_input_ids",
            "response_attention_mask",
            "advantages",
            "old_log_probs",
            "reference_log_probs",
            "values",
            "rewards",
            "group_ids",
            "candidate_index",
            "stop_reason",
        )
        return {name: batch[name] for name in names if name in batch}


__all__ = ["TrajectoryInputAdapter"]
