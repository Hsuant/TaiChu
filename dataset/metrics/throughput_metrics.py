"""Wall-clock sample and token throughput metrics."""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class ThroughputSnapshot:
    batches: int = 0
    samples: int = 0
    tokens: int = 0
    bytes_read: int = 0
    elapsed_seconds: float = 0.0

    @property
    def samples_per_second(self) -> float:
        return 0.0 if not self.elapsed_seconds else self.samples / self.elapsed_seconds

    @property
    def tokens_per_second(self) -> float:
        return 0.0 if not self.elapsed_seconds else self.tokens / self.elapsed_seconds

    @property
    def bytes_per_second(self) -> float:
        return 0.0 if not self.elapsed_seconds else self.bytes_read / self.elapsed_seconds


class ThroughputMetrics:
    def __init__(self, *, clock=time.perf_counter) -> None:
        self._clock = clock
        self._started = self._clock()
        self._snapshot = ThroughputSnapshot()
        self._lock = threading.Lock()

    def record(
        self,
        *,
        batches: int = 0,
        samples: int = 0,
        tokens: int = 0,
        bytes_read: int = 0,
    ) -> None:
        with self._lock:
            self._snapshot.batches += int(batches)
            self._snapshot.samples += int(samples)
            self._snapshot.tokens += int(tokens)
            self._snapshot.bytes_read += int(bytes_read)

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            self._snapshot.elapsed_seconds = max(0.0, self._clock() - self._started)
            result: dict[str, float | int] = asdict(self._snapshot)
            result["samples_per_second"] = self._snapshot.samples_per_second
            result["tokens_per_second"] = self._snapshot.tokens_per_second
            result["bytes_per_second"] = self._snapshot.bytes_per_second
            return result


__all__ = ["ThroughputMetrics", "ThroughputSnapshot"]
