"""Segment-aware alignment helpers for sample-isolated attention."""
from __future__ import annotations

import torch


def aligned_segment_ids(
    attention_mask: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    padding_segment_id: int = -1,
) -> torch.Tensor:
    return torch.where(
        attention_mask.bool(),
        segment_ids,
        torch.full_like(segment_ids, padding_segment_id),
    )


def segment_position_ids(segment_ids: torch.Tensor) -> torch.Tensor:
    """Restart positions for every segment while preserving causal order."""
    if segment_ids.ndim != 2:
        raise ValueError("segment_ids must be rank 2")
    positions = torch.zeros_like(segment_ids)
    for batch_index in range(segment_ids.size(0)):
        counter = 0
        previous = None
        for position in range(segment_ids.size(1)):
            current = int(segment_ids[batch_index, position])
            if current < 0:
                positions[batch_index, position] = 0
                continue
            if previous is None or current != previous:
                counter = 0
            positions[batch_index, position] = counter
            counter += 1
            previous = current
    return positions


__all__ = ["aligned_segment_ids", "segment_position_ids"]
