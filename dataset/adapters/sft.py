"""SFT model input contract with task-side masks retained."""
from __future__ import annotations

from typing import Any

from .base import select_keys


class SFTInputAdapter:
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        model_inputs = select_keys(batch, ("input_ids", "attention_mask", "labels"))
        for name in ("loss_mask", "segment_ids"):
            if name in batch:
                model_inputs[name] = batch[name]
        return model_inputs


__all__ = ["SFTInputAdapter"]
