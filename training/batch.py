"""Batch transfer rules shared by training and evaluation loops."""

from __future__ import annotations

from typing import Any

import torch


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Move model tensors to the device and drop causally redundant padding."""
    non_blocking = device.type == "cuda"
    moved = dict(batch)
    moved["input_ids"] = batch["input_ids"].to(
        device,
        non_blocking=non_blocking,
    )
    moved["labels"] = batch["labels"].to(
        device,
        non_blocking=non_blocking,
    )

    mask = batch.get("attention_mask")
    if mask is not None:
        moved["attention_mask"] = (
            None
            if batch.get("attention_mask_is_right_padded", False)
            else mask.to(device, non_blocking=non_blocking)
        )
    for name in ("segment_ids", "position_ids"):
        value = batch.get(name)
        if value is not None:
            moved[name] = value.to(device, non_blocking=non_blocking)
    return moved


__all__ = ["move_batch_to_device"]
