"""Pretraining and continued-pretraining model input contract."""
from __future__ import annotations

from typing import Any

from .base import select_keys


class PretrainingInputAdapter:
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        return select_keys(
            batch,
            (
                "input_ids",
                "attention_mask",
                "segment_ids",
                "position_ids",
                "labels",
            ),
        )


__all__ = ["PretrainingInputAdapter"]
