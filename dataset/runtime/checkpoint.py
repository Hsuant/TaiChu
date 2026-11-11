"""Serializable data-progress state for checkpoint and resume."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..exceptions import TaiChuDataFormatError


@dataclass(frozen=True, slots=True)
class DataState:
    dataset_id: str
    dataset_version: str
    epoch: int
    batch_index: int
    consumed_samples: int

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.dataset_version:
            raise ValueError("dataset identity must be non-empty")
        if self.epoch < 0 or self.batch_index < 0 or self.consumed_samples < 0:
            raise ValueError("data progress values cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "epoch": self.epoch,
            "batch_index": self.batch_index,
            "consumed_samples": self.consumed_samples,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DataState":
        try:
            state = cls(
                dataset_id=str(raw["dataset_id"]),
                dataset_version=str(raw["dataset_version"]),
                epoch=int(raw["epoch"]),
                batch_index=int(raw["batch_index"]),
                consumed_samples=int(raw["consumed_samples"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TaiChuDataFormatError(f"invalid data state: {exc}") from exc
        return state

    def compatible_with(self, dataset_id: str, dataset_version: str) -> bool:
        return (
            self.dataset_id == dataset_id
            and self.dataset_version == dataset_version
        )


def skip_batches(iterator, count: int):
    if count < 0:
        raise ValueError("resume batch count cannot be negative")
    for _ in range(count):
        try:
            next(iterator)
        except StopIteration:
            return iterator
    return iterator


class ResumableBatchIterator:
    """Wrap a DataLoader with deterministic epoch and batch-position resume."""

    def __init__(self, loader, state: DataState | None = None) -> None:
        self.loader = loader
        self.state = state

    def __iter__(self):
        epoch = 0 if self.state is None else self.state.epoch
        batch_index = 0 if self.state is None else self.state.batch_index
        consumed_samples = 0 if self.state is None else self.state.consumed_samples
        if hasattr(self.loader, "set_epoch"):
            self.loader.set_epoch(epoch)
        if batch_index and hasattr(self.loader, "set_resume_batch"):
            self.loader.set_resume_batch(batch_index)
            iterator = iter(self.loader)
        else:
            iterator = skip_batches(iter(self.loader), batch_index)
        for batch_index, batch in enumerate(iterator, start=batch_index):
            sample_ids = batch.get("sample_ids") if isinstance(batch, dict) else None
            sample_count = len(sample_ids) if sample_ids is not None else len(batch["input_ids"])
            consumed_samples += sample_count
            reader = getattr(getattr(self.loader, "dataset", None), "reader", None)
            if reader is not None:
                self.state = DataState(
                    dataset_id=reader.dataset_id,
                    dataset_version=reader.version,
                    epoch=epoch,
                    batch_index=batch_index + 1,
                    consumed_samples=consumed_samples,
                )
            yield batch


__all__ = ["DataState", "ResumableBatchIterator", "skip_batches"]
