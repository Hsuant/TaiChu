"""Preference-pair schemas for DPO, IPO, KTO and related methods."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .common import (
    CommonSample,
    binary_mask,
    positive_identifier,
    register_sample,
    require_aligned,
    token_ids,
)


@register_sample
@dataclass(slots=True, kw_only=True)
class PreferenceSample(CommonSample):
    """One prompt with aligned chosen and rejected continuations."""

    pair_id: str
    prompt_input_ids: list[int]
    chosen_input_ids: list[int]
    rejected_input_ids: list[int]
    chosen_mask: list[int]
    rejected_mask: list[int]
    preference_label: int = 1
    prompt_mask: list[int] | None = None
    sample_type: ClassVar[str] = "preference"
    array_fields: ClassVar[tuple[str, ...]] = (
        "prompt_input_ids",
        "chosen_input_ids",
        "rejected_input_ids",
        "prompt_mask",
        "chosen_mask",
        "rejected_mask",
    )

    def __post_init__(self) -> None:
        CommonSample.__post_init__(self)
        positive_identifier(self.pair_id, "pair_id")
        self.prompt_input_ids = token_ids(
            self.prompt_input_ids,
            name="prompt_input_ids",
        )
        self.chosen_input_ids = token_ids(
            self.chosen_input_ids,
            name="chosen_input_ids",
        )
        self.rejected_input_ids = token_ids(
            self.rejected_input_ids,
            name="rejected_input_ids",
        )
        require_aligned(
            "chosen response",
            self.chosen_input_ids,
            binary_mask(self.chosen_mask, name="chosen_mask"),
        )
        require_aligned(
            "rejected response",
            self.rejected_input_ids,
            binary_mask(self.rejected_mask, name="rejected_mask"),
        )
        self.chosen_mask = binary_mask(self.chosen_mask, name="chosen_mask")
        self.rejected_mask = binary_mask(self.rejected_mask, name="rejected_mask")
        if self.prompt_mask is not None:
            require_aligned(
                "prompt",
                self.prompt_input_ids,
                binary_mask(self.prompt_mask, name="prompt_mask"),
            )
            self.prompt_mask = binary_mask(self.prompt_mask, name="prompt_mask")
        if self.preference_label not in {0, 1}:
            raise ValueError("preference_label must be 0 or 1")


__all__ = ["PreferenceSample"]
