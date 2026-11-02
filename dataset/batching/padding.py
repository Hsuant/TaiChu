"""Dynamic right padding for task batch builders."""
from __future__ import annotations

from collections.abc import Sequence

import torch


def padded_tensor(
    rows: Sequence[Sequence[int]],
    *,
    pad_id: int,
    pad_to_multiple_of: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("cannot pad an empty batch")
    if pad_to_multiple_of <= 0:
        raise ValueError("pad_to_multiple_of must be positive")
    max_length = max(len(row) for row in rows)
    if max_length % pad_to_multiple_of:
        max_length += pad_to_multiple_of - max_length % pad_to_multiple_of
    values = torch.full((len(rows), max_length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), max_length), dtype=torch.long)
    for index, row in enumerate(rows):
        values[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[index, : len(row)] = 1
    return values, mask


def padded_float_tensor(
    rows: Sequence[Sequence[float]],
    *,
    pad_value: float = 0.0,
) -> torch.Tensor:
    if not rows:
        raise ValueError("cannot pad an empty batch")
    max_length = max(len(row) for row in rows)
    values = torch.full(
        (len(rows), max_length),
        float(pad_value),
        dtype=torch.float32,
    )
    for index, row in enumerate(rows):
        if row:
            values[index, : len(row)] = torch.tensor(row, dtype=torch.float32)
    return values


__all__ = ["padded_float_tensor", "padded_tensor"]
