"""Conversation-aware sample-isolated packing."""
from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..schema import SFTSequenceSample
from .base import PackedSequence


class SampleIsolatedPacker:
    """Pack whole conversations and carry segment IDs for isolated attention."""

    def __init__(
        self,
        max_length: int,
        *,
        pad_id: int = 0,
        pad_to_max: bool = True,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must be >= 2")
        self.max_length = int(max_length)
        self.pad_id = int(pad_id)
        self.pad_to_max = bool(pad_to_max)

    def pack(
        self,
        samples: Iterable[SFTSequenceSample],
    ) -> Iterator[PackedSequence]:
        current = PackedSequence([], [], [], [], [], [], [])
        segment_id = 0
        for sample in samples:
            if len(sample.input_ids) > self.max_length:
                raise ValueError(
                    f"conversation {sample.sample_id!r} exceeds max_length; "
                    "truncate before conversation-aware packing"
                )
            if current.input_ids and (
                len(current.input_ids) + len(sample.input_ids) > self.max_length
            ):
                yield self._finish(current)
                current = PackedSequence([], [], [], [], [], [], [])
            length = len(sample.input_ids)
            current.sample_ids.extend([sample.sample_id] * length)
            current.input_ids.extend(sample.input_ids)
            current.labels.extend(sample.labels)
            current.attention_mask.extend(
                sample.attention_mask or [1] * length
            )
            current.loss_mask.extend(sample.loss_mask)
            current.segment_ids.extend([segment_id] * length)
            current.document_ids.extend([segment_id] * length)
            segment_id += 1
        if current.input_ids:
            yield self._finish(current)

    def _finish(self, values: PackedSequence) -> PackedSequence:
        if self.pad_to_max and len(values.input_ids) < self.max_length:
            missing = self.max_length - len(values.input_ids)
            values.input_ids.extend([self.pad_id] * missing)
            values.labels.extend([-100] * missing)
            values.attention_mask.extend([0] * missing)
            values.loss_mask.extend([0] * missing)
            values.segment_ids.extend([-1] * missing)
            values.document_ids.extend([-1] * missing)
        return values


__all__ = ["SampleIsolatedPacker"]
