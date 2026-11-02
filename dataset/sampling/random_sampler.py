"""Deterministic single-process random sampler."""
from __future__ import annotations

from collections.abc import Iterator

from .base import materialize_indices, validate_epoch


class RandomSampler:
    def __init__(
        self,
        sample_count: int,
        *,
        seed: int = 42,
        shuffle: bool = True,
        drop_last: bool = False,
        num_samples: int | None = None,
    ) -> None:
        if sample_count < 0:
            raise ValueError("sample_count cannot be negative")
        if num_samples is not None and num_samples < 0:
            raise ValueError("num_samples cannot be negative")
        self.sample_count = int(sample_count)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.num_samples = num_samples
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = validate_epoch(epoch)

    def _indices(self) -> list[int]:
        indices = materialize_indices(
            self.sample_count,
            seed=self.seed,
            epoch=self.epoch,
            shuffle=self.shuffle,
        )
        if self.num_samples is not None:
            target = self.num_samples
            if self.drop_last and target > len(indices):
                target = len(indices)
            if target <= len(indices):
                return indices[:target]
            if not indices:
                return []
            repeats = (target + len(indices) - 1) // len(indices)
            return (indices * repeats)[:target]
        return indices

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices())

    def __len__(self) -> int:
        if self.num_samples is not None:
            return min(self.num_samples, self.sample_count) if self.drop_last else self.num_samples
        return self.sample_count


__all__ = ["RandomSampler"]
