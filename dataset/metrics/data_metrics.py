"""Thread-safe data quality and pipeline counters."""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class DataMetricsSnapshot:
    batches: int = 0
    samples: int = 0
    tokens: int = 0
    padding_tokens: int = 0
    skipped_samples: int = 0
    accepted_documents: int = 0
    rejected_documents: int = 0
    duplicate_documents: int = 0


class DataMetrics:
    def __init__(self) -> None:
        self._snapshot = DataMetricsSnapshot()
        self._lock = threading.Lock()

    def record_batch(
        self,
        *,
        samples: int,
        tokens: int,
        padding_tokens: int = 0,
    ) -> None:
        with self._lock:
            self._snapshot.batches += 1
            self._snapshot.samples += int(samples)
            self._snapshot.tokens += int(tokens)
            self._snapshot.padding_tokens += int(padding_tokens)

    def record_skipped(self, count: int = 1) -> None:
        with self._lock:
            self._snapshot.skipped_samples += int(count)

    def record_quality(
        self,
        *,
        accepted: int = 0,
        rejected: int = 0,
        duplicates: int = 0,
    ) -> None:
        with self._lock:
            self._snapshot.accepted_documents += int(accepted)
            self._snapshot.rejected_documents += int(rejected)
            self._snapshot.duplicate_documents += int(duplicates)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return asdict(self._snapshot)


__all__ = ["DataMetrics", "DataMetricsSnapshot"]
