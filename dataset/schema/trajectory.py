"""Reward-model and on-policy trajectory schemas."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

from .common import (
    CommonSample,
    binary_mask,
    float_values,
    positive_identifier,
    register_sample,
    require_aligned,
    token_ids,
)


@register_sample
@dataclass(slots=True, kw_only=True)
class RewardSample(CommonSample):
    """A prompt-response pair with a sequence or token reward target."""

    prompt_input_ids: list[int]
    response_input_ids: list[int]
    reward_label: float
    reward_mask: list[int] | None = None
    ranking_label: int | None = None
    sample_type: ClassVar[str] = "reward"
    array_fields: ClassVar[tuple[str, ...]] = (
        "prompt_input_ids",
        "response_input_ids",
        "reward_mask",
    )

    def __post_init__(self) -> None:
        CommonSample.__post_init__(self)
        self.prompt_input_ids = token_ids(
            self.prompt_input_ids,
            name="prompt_input_ids",
        )
        self.response_input_ids = token_ids(
            self.response_input_ids,
            name="response_input_ids",
        )
        self.reward_label = float(self.reward_label)
        if not math.isfinite(self.reward_label):
            raise ValueError("reward_label must be finite")
        if self.reward_mask is not None:
            self.reward_mask = binary_mask(self.reward_mask, name="reward_mask")
            require_aligned(
                "reward sample",
                self.response_input_ids,
                self.reward_mask,
            )
        if self.ranking_label is not None and not isinstance(self.ranking_label, int):
            raise TypeError("ranking_label must be an integer or None")


@register_sample
@dataclass(slots=True, kw_only=True)
class TrajectorySample(CommonSample):
    """Rollout data for PPO/GRPO/RLHF and future critic training."""

    prompt_input_ids: list[int]
    response_input_ids: list[int]
    rewards: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    old_log_probs: list[float] = field(default_factory=list)
    reference_log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    response_mask: list[int] | None = None
    group_id: str | None = None
    candidate_index: int | None = None
    stop_reason: str | None = None
    sample_type: ClassVar[str] = "trajectory"
    array_fields: ClassVar[tuple[str, ...]] = (
        "prompt_input_ids",
        "response_input_ids",
        "rewards",
        "advantages",
        "old_log_probs",
        "reference_log_probs",
        "values",
        "response_mask",
    )

    def __post_init__(self) -> None:
        CommonSample.__post_init__(self)
        self.prompt_input_ids = token_ids(
            self.prompt_input_ids,
            name="prompt_input_ids",
        )
        self.response_input_ids = token_ids(
            self.response_input_ids,
            name="response_input_ids",
        )
        for name in ("rewards", "advantages", "old_log_probs", "reference_log_probs", "values"):
            setattr(self, name, float_values(getattr(self, name), name=name))
        if self.response_mask is not None:
            self.response_mask = binary_mask(self.response_mask, name="response_mask")
            require_aligned(
                "trajectory response",
                self.response_input_ids,
                self.response_mask,
            )
        response_length = len(self.response_input_ids)
        for name in ("old_log_probs", "reference_log_probs", "values"):
            values = getattr(self, name)
            if values and len(values) != response_length:
                raise ValueError(f"{name} must align with response_input_ids")
        if self.group_id is not None:
            positive_identifier(self.group_id, "group_id")
        if self.candidate_index is not None and self.candidate_index < 0:
            raise ValueError("candidate_index cannot be negative")
        if self.stop_reason is not None and not isinstance(self.stop_reason, str):
            raise TypeError("stop_reason must be a string or None")


__all__ = ["RewardSample", "TrajectorySample"]
