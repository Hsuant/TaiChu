"""Preference model inputs; pair-level identity is never discarded."""
from __future__ import annotations

from typing import Any


class PreferenceInputAdapter:
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        result = {
            "chosen_input_ids": batch["chosen_input_ids"],
            "chosen_attention_mask": batch["chosen_attention_mask"],
            "chosen_labels": batch["chosen_labels"],
            "rejected_input_ids": batch["rejected_input_ids"],
            "rejected_attention_mask": batch["rejected_attention_mask"],
            "rejected_labels": batch["rejected_labels"],
            "preference_label": batch["preference_label"],
            "pair_ids": batch["pair_ids"],
        }
        for name in (
            "chosen_response_input_ids",
            "chosen_response_attention_mask",
            "rejected_response_input_ids",
            "rejected_response_attention_mask",
        ):
            if name in batch:
                result[name] = batch[name]
        return result


__all__ = ["PreferenceInputAdapter"]
