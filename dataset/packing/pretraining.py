"""Streaming pretraining and continued-pretraining packing."""
from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..schema import PretrainingSample
from .base import PackedSequence


class PretrainingPacker:
    """Concatenate token streams while retaining document and sample boundaries."""

    def __init__(
        self,
        max_length: int,
        *,
        pad_id: int = 0,
        pad_to_max: bool = False,
        drop_last_remainder: bool = True,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must be >= 2")
        self.max_length = int(max_length)
        self.pad_id = int(pad_id)
        self.pad_to_max = bool(pad_to_max)
        self.drop_last_remainder = bool(drop_last_remainder)

    def pack(
        self,
        samples: Iterable[PretrainingSample],
    ) -> Iterator[PackedSequence]:
        buffer = {
            "sample_ids": [],
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
            "loss_mask": [],
            "segment_ids": [],
            "document_ids": [],
        }
        segment_cursor = 0
        document_cursor = 0

        def append(values: list, name: str, chunk: list) -> None:
            buffer[name].extend(chunk)

        def emit(padded: bool) -> PackedSequence:
            nonlocal buffer, segment_cursor
            sequence = PackedSequence(**buffer)
            if padded and len(sequence.input_ids) < self.max_length:
                missing = self.max_length - len(sequence.input_ids)
                sequence.input_ids.extend([self.pad_id] * missing)
                sequence.labels.extend([-100] * missing)
                sequence.attention_mask.extend([0] * missing)
                sequence.loss_mask.extend([0] * missing)
                sequence.segment_ids.extend([-1] * missing)
                sequence.document_ids.extend([-1] * missing)
            buffer = {
                "sample_ids": [],
                "input_ids": [],
                "labels": [],
                "attention_mask": [],
                "loss_mask": [],
                "segment_ids": [],
                "document_ids": [],
            }
            return sequence

        for sample in samples:
            labels = sample.labels or sample.input_ids
            attention_mask = sample.attention_mask or [1] * len(sample.input_ids)
            local_documents = (
                sample.document_ids
                if sample.document_ids
                else [document_cursor] * len(sample.input_ids)
            )
            if sample.document_ids:
                local_documents = [
                    document_cursor + value for value in sample.document_ids
                ]
                document_cursor = max(local_documents) + 1
            else:
                document_cursor += 1
            offset = 0
            while offset < len(sample.input_ids):
                remaining = self.max_length - len(buffer["input_ids"])
                if remaining <= 0:
                    yield emit(padded=self.pad_to_max)
                    remaining = self.max_length
                take = min(remaining, len(sample.input_ids) - offset)
                chunk_labels = labels[offset : offset + take]
                chunk_documents = local_documents[offset : offset + take]
                if buffer["document_ids"] and chunk_documents[0] != buffer["document_ids"][-1]:
                    chunk_labels = list(chunk_labels)
                    chunk_labels[0] = -100
                append(buffer["sample_ids"], "sample_ids", [sample.sample_id] * take)
                append(buffer["input_ids"], "input_ids", sample.input_ids[offset : offset + take])
                append(buffer["labels"], "labels", chunk_labels)
                append(
                    buffer["attention_mask"],
                    "attention_mask",
                    attention_mask[offset : offset + take],
                )
                append(
                    buffer["loss_mask"],
                    "loss_mask",
                    [
                        int(label != -100)
                        for label in chunk_labels
                    ],
                )
                append(
                    buffer["segment_ids"],
                    "segment_ids",
                    [segment_cursor] * take,
                )
                append(
                    buffer["document_ids"],
                    "document_ids",
                    chunk_documents,
                )
                offset += take
                if len(buffer["input_ids"]) == self.max_length:
                    yield emit(padded=False)
            segment_cursor += 1

        if buffer["input_ids"]:
            if self.pad_to_max:
                yield emit(padded=True)
            elif not self.drop_last_remainder:
                yield emit(padded=False)


__all__ = ["PretrainingPacker"]
