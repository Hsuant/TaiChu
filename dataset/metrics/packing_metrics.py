"""Packing utilization and fragmentation metrics."""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class PackingMetricsSnapshot:
    source_sequences: int = 0
    packed_sequences: int = 0
    source_tokens: int = 0
    packed_tokens: int = 0
    padding_tokens: int = 0
    segments: int = 0

    @property
    def utilization(self) -> float:
        total = self.packed_tokens + self.padding_tokens
        return 0.0 if total == 0 else self.packed_tokens / total


class PackingMetrics:
    def __init__(self) -> None:
        self._snapshot = PackingMetricsSnapshot()
        self._lock = threading.Lock()

    def record(
        self,
        *,
        source_sequences: int,
        packed_sequences: int,
        source_tokens: int,
        packed_tokens: int,
        padding_tokens: int,
        segments: int,
    ) -> None:
        with self._lock:
            self._snapshot.source_sequences += int(source_sequences)
            self._snapshot.packed_sequences += int(packed_sequences)
            self._snapshot.source_tokens += int(source_tokens)
            self._snapshot.packed_tokens += int(packed_tokens)
            self._snapshot.padding_tokens += int(padding_tokens)
            self._snapshot.segments += int(segments)

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            result: dict[str, float | int] = asdict(self._snapshot)
            result["utilization"] = self._snapshot.utilization
            return result


__all__ = ["PackingMetrics", "PackingMetricsSnapshot"]
