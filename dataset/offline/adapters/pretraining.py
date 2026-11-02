"""Raw document to pretraining or continued-pretraining adapter."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...exceptions import TaiChuDataFormatError
from ...schema import PretrainingSample
from .base import (
    BasePreprocessingAdapter,
    TextTokenizer,
    dataset_identifier,
    sample_identifier,
    source_identifier,
)


class PretrainingPreprocessor(BasePreprocessingAdapter):
    sample_type = "pretraining"

    def __init__(
        self,
        tokenizer: TextTokenizer,
        *,
        text_column: str = "text",
        add_bos: bool = False,
        add_eos: bool = True,
        max_length: int | None = None,
    ) -> None:
        if max_length is not None and max_length < 1:
            raise ValueError("max_length must be positive or None")
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.max_length = max_length
        if add_bos and tokenizer.bos_token_id is None:
            raise ValueError("BOS token could not be resolved")
        if add_eos and tokenizer.eos_token_id is None:
            raise ValueError("EOS token could not be resolved")

    def validate(self, record: Mapping[str, Any]) -> None:
        super().validate(record)
        text = record.get(self.text_column)
        if not isinstance(text, str) or not text.strip():
            raise TaiChuDataFormatError(
                f"record must contain a non-empty {self.text_column!r} string"
            )

    def format(self, record: Mapping[str, Any]) -> str:
        return str(record[self.text_column]).strip()

    def tokenize(
        self,
        record: Mapping[str, Any],
        formatted: str,
    ) -> PretrainingSample:
        ids = self.tokenizer.encode(formatted)
        if self.add_bos and self.tokenizer.bos_token_id is not None:
            ids.insert(0, self.tokenizer.bos_token_id)
        if self.add_eos and self.tokenizer.eos_token_id is not None:
            ids.append(self.tokenizer.eos_token_id)
        if self.max_length is not None:
            ids = ids[: self.max_length]
        return PretrainingSample(
            sample_id=sample_identifier(record, "pretraining"),
            dataset_id=dataset_identifier(record, "pretraining"),
            source_id=source_identifier(record),
            input_ids=ids,
            document_ids=[0] * len(ids),
            metadata={"text_characters": len(formatted)},
        )


__all__ = ["PretrainingPreprocessor"]
