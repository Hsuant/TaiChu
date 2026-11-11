"""Document and causal-language-model sample schemas."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from .common import (
    CommonSample,
    binary_mask,
    positive_identifier,
    register_sample,
    require_aligned,
    token_ids,
)


@dataclass(slots=True, kw_only=True)
class DocumentSample(CommonSample):
    """One source document before or after tokenization."""

    text: str
    document_id: str
    sample_type: ClassVar[str] = "document"
    array_fields: ClassVar[tuple[str, ...]] = ("input_ids",)

    input_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        CommonSample.__post_init__(self)
        positive_identifier(self.document_id, "document_id")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        self.input_ids = token_ids(self.input_ids, name="input_ids") if self.input_ids else []


@register_sample
@dataclass(slots=True, kw_only=True)
class PretrainingSample(CommonSample):
    """A packed or unpacked causal-LM sequence with document boundaries."""

    input_ids: list[int]
    labels: list[int] | None = None
    attention_mask: list[int] | None = None
    document_ids: list[int] = field(default_factory=list)
    sample_type: ClassVar[str] = "pretraining"
    array_fields: ClassVar[tuple[str, ...]] = (
        "input_ids",
        "labels",
        "attention_mask",
        "document_ids",
    )

    @classmethod
    def _from_trusted(
        cls,
        *,
        sample_id: str,
        dataset_id: str,
        source_id: str,
        input_ids: list[int],
        labels: list[int],
        attention_mask: list[int] | None,
        document_ids: list[int],
        metadata: dict,
    ) -> "PretrainingSample":
        """Build a sample from tokenizer-produced arrays without revalidation."""
        sample = object.__new__(cls)
        sample.sample_id = sample_id
        sample.dataset_id = dataset_id
        sample.source_id = source_id
        sample.metadata = metadata
        sample.input_ids = input_ids
        sample.labels = labels
        sample.attention_mask = attention_mask
        sample.document_ids = document_ids
        return sample

    def __post_init__(self) -> None:
        CommonSample.__post_init__(self)
        self.input_ids = token_ids(self.input_ids)
        self.labels = (
            None
            if self.labels is None
            else token_ids(self.labels, name="labels", allow_ignore_index=True)
        )
        self.attention_mask = (
            None
            if self.attention_mask is None
            else binary_mask(self.attention_mask, name="attention_mask")
        )
        self.document_ids = (
            token_ids(self.document_ids, name="document_ids")
            if self.document_ids
            else []
        )
        aligned = [self.input_ids]
        for value in (self.labels, self.attention_mask, self.document_ids or None):
            if value is not None:
                aligned.append(value)
        require_aligned("pretraining sample", *aligned)
        if not self.input_ids:
            raise ValueError("input_ids cannot be empty")

    @property
    def seq_length(self) -> int:
        return len(self.input_ids)


@dataclass(slots=True, kw_only=True)
class TextSequenceSample(CommonSample):
    """Tokenized text before task-specific formatting."""

    input_ids: list[int]
    labels: list[int] | None = None
    loss_mask: list[int] | None = None
    sample_type: ClassVar[str] = "text_sequence"
    array_fields: ClassVar[tuple[str, ...]] = ("input_ids", "labels", "loss_mask")

    def __post_init__(self) -> None:
        CommonSample.__post_init__(self)
        self.input_ids = token_ids(self.input_ids)
        self.labels = (
            None
            if self.labels is None
            else token_ids(self.labels, name="labels", allow_ignore_index=True)
        )
        self.loss_mask = (
            None if self.loss_mask is None else binary_mask(self.loss_mask)
        )
        aligned = [self.input_ids]
        if self.labels is not None:
            aligned.append(self.labels)
        if self.loss_mask is not None:
            aligned.append(self.loss_mask)
        require_aligned("text sequence", *aligned)


__all__ = ["DocumentSample", "PretrainingSample", "TextSequenceSample"]
