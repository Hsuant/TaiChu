"""Weighted mixing of indexed offline datasets."""
from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

from ..schema import CommonSample
from ..storage.base import IndexedSampleReader
from .base import validate_epoch


@dataclass(frozen=True, slots=True)
class MixSource:
    name: str
    reader: IndexedSampleReader
    weight: float


class DatasetMixer:
    def __init__(
        self,
        sources: list[MixSource],
        *,
        seed: int = 42,
        sample_budget: int | None = None,
    ) -> None:
        if not sources:
            raise ValueError("at least one mix source is required")
        if any(source.weight <= 0 for source in sources):
            raise ValueError("mix source weights must be positive")
        self.sources = tuple(sources)
        self.seed = int(seed)
        self.epoch = 0
        self.sample_budget = (
            sum(len(source.reader) for source in sources)
            if sample_budget is None
            else int(sample_budget)
        )
        if self.sample_budget < 0:
            raise ValueError("sample_budget cannot be negative")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = validate_epoch(epoch)

    def __iter__(self) -> Iterator[CommonSample]:
        if not self.sample_budget:
            return
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        available = [source for source in self.sources if len(source.reader) > 0]
        if not available:
            return
        weights = [source.weight for source in available]
        for _ in range(self.sample_budget):
            source = rng.choices(available, weights=weights, k=1)[0]
            index = rng.randrange(len(source.reader))
            yield source.reader[index]

    def __len__(self) -> int:
        return self.sample_budget


__all__ = ["DatasetMixer", "MixSource"]
