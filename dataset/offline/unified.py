"""Offline preparation orchestration shared by every task adapter."""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from typing import Any

from ..exceptions import TaiChuDataFormatError
from ..schema import CommonSample
from ..storage.base import SampleWriter
from .adapters import BasePreprocessingAdapter


@dataclass(slots=True)
class PreparationStats:
    source_samples: int = 0
    written_samples: int = 0
    skipped_samples: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "source_samples": self.source_samples,
            "written_samples": self.written_samples,
            "skipped_samples": self.skipped_samples,
        }


class UnifiedPreparationPipeline:
    """Validate, format, tokenize and persist task-specific samples."""

    def __init__(
        self,
        adapter: BasePreprocessingAdapter,
        writer: SampleWriter,
        *,
        invalid_policy: str = "error",
        batch_size: int = 256,
    ) -> None:
        if invalid_policy not in {"error", "skip"}:
            raise ValueError("invalid_policy must be 'error' or 'skip'")
        self.adapter = adapter
        self.writer = writer
        self.invalid_policy = invalid_policy
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def run(self, records: Iterable[dict[str, Any]]) -> PreparationStats:
        stats = PreparationStats()
        for batch in self._batches(records):
            stats.source_samples += len(batch)
            try:
                samples = self.adapter.prepare_batch(batch)
            except (TaiChuDataFormatError, TypeError, ValueError):
                if self.invalid_policy == "error":
                    self.writer.abort()
                    raise
                for record in batch:
                    try:
                        sample: CommonSample = self.adapter.prepare(record)
                    except (TaiChuDataFormatError, TypeError, ValueError):
                        stats.skipped_samples += 1
                        continue
                    self.writer.write(sample)
                    stats.written_samples += 1
                continue
            for sample in samples:
                self.writer.write(sample)
                stats.written_samples += 1
        return stats

    def run_parallel(
        self,
        records: Iterable[dict[str, Any]],
        *,
        executor: Executor,
        batch_worker: Callable[[list[dict[str, Any]]], list],
        max_inflight: int,
    ) -> PreparationStats:
        if max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        stats = PreparationStats()
        batches = iter(self._batches(records))
        pending: deque[tuple[Future, list[dict[str, Any]]]] = deque()

        def submit_next() -> bool:
            try:
                batch = next(batches)
            except StopIteration:
                return False
            pending.append((executor.submit(batch_worker, batch), batch))
            return True

        for _ in range(max_inflight):
            if not submit_next():
                break
        while pending:
            future, batch = pending.popleft()
            stats.source_samples += len(batch)
            try:
                samples = future.result()
            except (TaiChuDataFormatError, TypeError, ValueError):
                if self.invalid_policy == "error":
                    self.writer.abort()
                    raise
                self._write_fallback_batch(batch, stats)
            else:
                for sample in samples:
                    self.writer.write(sample)
                    stats.written_samples += 1
            submit_next()
        return stats

    def _write_fallback_batch(
        self,
        batch: list[dict[str, Any]],
        stats: PreparationStats,
    ) -> None:
        for record in batch:
            try:
                sample: CommonSample = self.adapter.prepare(record)
            except (TaiChuDataFormatError, TypeError, ValueError):
                stats.skipped_samples += 1
                continue
            self.writer.write(sample)
            stats.written_samples += 1

    def _batches(
        self,
        records: Iterable[dict[str, Any]],
    ) -> Iterator[list[dict[str, Any]]]:
        batch = []
        for record in records:
            batch.append(record)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


__all__ = ["PreparationStats", "UnifiedPreparationPipeline"]
