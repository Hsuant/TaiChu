"""Deterministic global-index assignment without duplicating source data."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IndexRange:
    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start

    def __iter__(self):
        return iter(range(self.start, self.end))


class ShardManager:
    """Split a logical dataset into contiguous, optionally balanced slots."""

    def __init__(self, sample_count: int) -> None:
        if sample_count < 0:
            raise ValueError("sample_count cannot be negative")
        self.sample_count = int(sample_count)

    def ranges(self, slots: int, *, drop_last: bool = False) -> list[IndexRange]:
        if slots <= 0:
            raise ValueError("slots must be positive")
        usable = (
            self.sample_count - self.sample_count % slots
            if drop_last
            else self.sample_count
        )
        if not usable:
            return []
        if drop_last:
            block = usable // slots
            return [
                IndexRange(block * slot, block * (slot + 1))
                for slot in range(slots)
            ]
        block, remainder = divmod(usable, slots)
        ranges = []
        cursor = 0
        for slot in range(slots):
            size = block + (1 if slot >= slots - remainder else 0)
            ranges.append(IndexRange(cursor, cursor + size))
            cursor += size
        return ranges


__all__ = ["IndexRange", "ShardManager"]
