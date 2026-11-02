"""Online rollouts and runtime experience buffers share the same schemas."""
from __future__ import annotations

import random
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

from ..schema import CommonSample


T = TypeVar("T", bound=CommonSample)


class ExperienceBuffer:
    def __init__(self, max_samples: int | None = None) -> None:
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive or None")
        self._samples: deque[CommonSample] = deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def append(self, sample: CommonSample) -> None:
        with self._lock:
            self._samples.append(sample)

    def extend(self, samples: Iterable[CommonSample]) -> None:
        with self._lock:
            self._samples.extend(samples)

    def snapshot(self) -> tuple[CommonSample, ...]:
        with self._lock:
            return tuple(self._samples)

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)

    def __iter__(self) -> Iterator[CommonSample]:
        return iter(self.snapshot())


class HybridSampleSource:
    """Interleave offline data, online generation and an experience buffer."""

    def __init__(
        self,
        *,
        offline: Iterable[CommonSample] | None = None,
        online: Callable[[], CommonSample] | None = None,
        experience_buffer: ExperienceBuffer | None = None,
        online_ratio: float = 0.0,
        seed: int = 42,
    ) -> None:
        if not 0 <= online_ratio <= 1:
            raise ValueError("online_ratio must be in [0, 1]")
        if online is not None and online_ratio == 0:
            raise ValueError("online_ratio must be positive when online generation is set")
        if online is None and online_ratio > 0:
            raise ValueError("online generation callable is required")
        self.offline = offline
        self.online = online
        self.experience_buffer = experience_buffer
        self.online_ratio = float(online_ratio)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[CommonSample]:
        rng = random.Random(self.seed)
        offline_iter = iter(self.offline) if self.offline is not None else iter(())
        buffer_items = (
            self.experience_buffer.snapshot()
            if self.experience_buffer is not None
            else ()
        )
        while True:
            if rng.random() < self.online_ratio:
                if buffer_items and (self.online is None or rng.random() < 0.5):
                    yield buffer_items[rng.randrange(len(buffer_items))]
                elif self.online is not None:
                    yield self.online()
                else:
                    continue
            else:
                try:
                    yield next(offline_iter)
                except StopIteration:
                    return


__all__ = ["ExperienceBuffer", "HybridSampleSource"]
