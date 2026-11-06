"""Rank-strided sampler with explicit padding and pair-safe sample units."""
from __future__ import annotations

import math
from collections.abc import Iterator

from .base import materialize_indices, pad_indices, validate_epoch


class DistributedSampler:
    """Assign whole samples to ranks; a preference sample remains one unit."""

    def __init__(
        self,
        sample_count: int,
        *,
        rank: int,
        world_size: int,
        seed: int = 42,
        shuffle: bool = True,
        drop_last: bool = False,
        pad_to_world_size: bool = True,
        pair_aware: bool = False,
    ) -> None:
        if sample_count < 0:
            raise ValueError("sample_count cannot be negative")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.sample_count = int(sample_count)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.pad_to_world_size = bool(pad_to_world_size)
        self.pair_aware = bool(pair_aware)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = validate_epoch(epoch)

    def _global_indices(self) -> list[int]:
        indices = materialize_indices(
            self.sample_count,
            seed=self.seed,
            epoch=self.epoch,
            shuffle=self.shuffle,
        )
        if self.drop_last:
            usable = self.sample_count - self.sample_count % self.world_size
            indices = indices[:usable]
        return indices

    def _rank_indices(self) -> list[int]:
        global_indices = self._global_indices()
        local = global_indices[self.rank :: self.world_size]
        if self.drop_last or not self.pad_to_world_size or not global_indices:
            return local
        expected = math.ceil(len(global_indices) / self.world_size)
        return pad_indices(local, expected)

    def __iter__(self) -> Iterator[int]:
        return iter(self._rank_indices())

    def __len__(self) -> int:
        if self.drop_last:
            return self.sample_count // self.world_size
        if not self.pad_to_world_size:
            return len(range(self.rank, self.sample_count, self.world_size))
        return math.ceil(self.sample_count / self.world_size)


__all__ = ["DistributedSampler"]
