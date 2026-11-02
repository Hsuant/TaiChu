"""Sampler protocols for map-style indexed tokenized datasets."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class IndexSampler(Protocol):
    def set_epoch(self, epoch: int) -> None: ...

    def __iter__(self) -> Iterator[int]: ...

    def __len__(self) -> int: ...


def validate_epoch(epoch: int) -> int:
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    return epoch


def materialize_indices(
    sample_count: int,
    *,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> list[int]:
    import random

    if sample_count < 0:
        raise ValueError("sample_count cannot be negative")
    indices = list(range(sample_count))
    if shuffle:
        random.Random(seed + validate_epoch(epoch) * 1_000_003).shuffle(indices)
    return indices


def pad_indices(indices: list[int], length: int) -> list[int]:
    if length <= len(indices):
        return indices[:length]
    if not indices:
        return []
    repeats = (length + len(indices) - 1) // len(indices)
    return (indices * repeats)[:length]


__all__ = [
    "IndexSampler",
    "materialize_indices",
    "pad_indices",
    "validate_epoch",
]
