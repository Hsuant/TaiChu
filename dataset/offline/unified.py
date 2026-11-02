"""Offline preparation orchestration shared by every task adapter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

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
    ) -> None:
        if invalid_policy not in {"error", "skip"}:
            raise ValueError("invalid_policy must be 'error' or 'skip'")
        self.adapter = adapter
        self.writer = writer
        self.invalid_policy = invalid_policy

    def run(self, records: Iterable[dict[str, Any]]) -> PreparationStats:
        stats = PreparationStats()
        for record in records:
            stats.source_samples += 1
            try:
                sample: CommonSample = self.adapter.prepare(record)
            except (TaiChuDataFormatError, TypeError, ValueError):
                if self.invalid_policy == "error":
                    self.writer.abort()
                    raise
                stats.skipped_samples += 1
                continue
            self.writer.write(sample)
            stats.written_samples += 1
        return stats


__all__ = ["PreparationStats", "UnifiedPreparationPipeline"]
