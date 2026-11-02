"""Packing contracts shared by task-specific strategies."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterator
from typing import Any, Protocol

from ..exceptions import TaiChuDataFormatError
from ..schema import CommonSample


@dataclass(slots=True)
class PackedSequence:
    sample_ids: list[str]
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    loss_mask: list[int]
    segment_ids: list[int]
    document_ids: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lengths = {
            len(self.input_ids),
            len(self.labels),
            len(self.attention_mask),
            len(self.loss_mask),
            len(self.segment_ids),
            len(self.document_ids),
        }
        if len(lengths) != 1:
            raise TaiChuDataFormatError("all packed arrays must have identical lengths")


class SamplePacker(Protocol):
    def pack(self, samples) -> Iterator[PackedSequence]: ...


__all__ = ["PackedSequence", "SamplePacker"]
