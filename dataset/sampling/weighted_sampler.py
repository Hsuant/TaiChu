"""Deterministic weighted sampling without replacement."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sequence

from .base import validate_epoch


class WeightedSampler:
    """Efraimidis-Spirakis weighted sampling with stable epoch seeds."""

    def __init__(
        self,
        weights: Sequence[float],
        *,
        seed: int = 42,
        num_samples: int | None = None,
        replacement: bool = False,
    ) -> None:
        if any(not math.isfinite(float(weight)) or weight < 0 for weight in weights):
            raise ValueError("weights must be finite and non-negative")
        if not any(float(weight) > 0 for weight in weights):
            raise ValueError("at least one weight must be positive")
        self.weights = [float(weight) for weight in weights]
        self.seed = int(seed)
        self.num_samples = len(weights) if num_samples is None else int(num_samples)
        if self.num_samples < 0:
            raise ValueError("num_samples cannot be negative")
        self.replacement = bool(replacement)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = validate_epoch(epoch)

    @staticmethod
    def _uniform(seed: int, epoch: int, index: int) -> float:
        digest = hashlib.blake2b(
            f"{seed}:{epoch}:{index}".encode("ascii"),
            digest_size=8,
        ).digest()
        value = int.from_bytes(digest, "big") / 2**64
        return value or 1e-12

    def _without_replacement(self) -> list[int]:
        keys = []
        for index, weight in enumerate(self.weights):
            if weight <= 0:
                continue
            uniform = self._uniform(self.seed, self.epoch, index)
            keys.append((uniform ** (1.0 / weight), index))
        keys.sort(reverse=True)
        return [index for _, index in keys[: self.num_samples]]

    def _with_replacement(self) -> list[int]:
        positive = [(index, weight) for index, weight in enumerate(self.weights) if weight > 0]
        total = sum(weight for _, weight in positive)
        result = []
        for position in range(self.num_samples):
            target = self._uniform(self.seed, self.epoch, position) * total
            cursor = 0.0
            for index, weight in positive:
                cursor += weight
                if target <= cursor:
                    result.append(index)
                    break
        return result

    def __iter__(self) -> Iterator[int]:
        return iter(
            self._with_replacement()
            if self.replacement
            else self._without_replacement()
        )

    def __len__(self) -> int:
        return min(self.num_samples, sum(weight > 0 for weight in self.weights)) if not self.replacement else self.num_samples


__all__ = ["WeightedSampler"]
