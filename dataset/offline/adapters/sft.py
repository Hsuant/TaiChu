"""Conversation to SFT adapter with explicit assistant loss spans."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...exceptions import TaiChuDataFormatError
from ...schema import ConversationMessage, SFTSequenceSample
from .base import (
    BasePreprocessingAdapter,
    ConversationRenderer,
    RenderedTurn,
    TextTokenizer,
    apply_max_length,
    dataset_identifier,
    normalize_reasoning_tags,
    sample_identifier,
    source_identifier,
    validate_reasoning_protocol,
)


class SFTPreprocessor(BasePreprocessingAdapter):
    sample_type = "sft"

    def __init__(
        self,
        tokenizer: TextTokenizer,
        renderer: ConversationRenderer,
        *,
        max_length: int | None = None,
        truncation: str = "left",
        assistant_only_loss: bool = True,
        train_on_eos: bool = True,
    ) -> None:
        if truncation not in {"left", "right"}:
            raise ValueError("truncation must be 'left' or 'right'")
        if train_on_eos and tokenizer.eos_token_id is None:
            raise ValueError("EOS token could not be resolved")
        if not assistant_only_loss and train_on_eos:
            raise ValueError("train_on_eos requires assistant_only_loss")
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.max_length = max_length
        self.truncation = truncation
        self.assistant_only_loss = assistant_only_loss
        self.train_on_eos = train_on_eos

    def validate(self, record: Mapping[str, Any]) -> None:
        super().validate(record)
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise TaiChuDataFormatError("SFT record requires a non-empty messages list")
        for message in messages:
            if not isinstance(message, Mapping):
                raise TaiChuDataFormatError("each message must be a mapping")
            self.renderer.render(self._normalized_message(message))

    def format(self, record: Mapping[str, Any]) -> list[RenderedTurn]:
        return [
            self.renderer.render(self._normalized_message(message))
            for message in record["messages"]
        ]

    @staticmethod
    def _normalized_message(
        message: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(message)
        if normalized.get("role") == "assistant":
            normalized["content"] = normalize_reasoning_tags(
                str(normalized.get("content", ""))
            )
        return normalized

    def tokenize(
        self,
        record: Mapping[str, Any],
        formatted: list[RenderedTurn],
    ) -> SFTSequenceSample:
        input_ids: list[int] = []
        loss_mask: list[int] = []
        for turn in formatted:
            prefix_ids = self.tokenizer.encode(turn.prefix)
            if turn.role == "assistant":
                validate_reasoning_protocol(turn.content)
            content_ids = self.tokenizer.encode(turn.content)
            suffix_ids = self.tokenizer.encode(turn.suffix)
            input_ids.extend(prefix_ids)
            input_ids.extend(content_ids)
            input_ids.extend(suffix_ids)
            train_turn = (not self.assistant_only_loss) or turn.role == "assistant"
            loss_mask.extend([0] * len(prefix_ids))
            loss_mask.extend([1 if train_turn else 0] * len(content_ids))
            loss_mask.extend(
                [
                    1
                    if (
                        train_turn
                        and self.train_on_eos
                        and token_id == self.tokenizer.eos_token_id
                    )
                    else 0
                    for token_id in suffix_ids
                ]
            )

        labels = [
            token if mask else -100
            for token, mask in zip(input_ids, loss_mask, strict=True)
        ]
        input_ids, labels, loss_mask = apply_max_length(
            input_ids,
            labels,
            loss_mask,
            max_length=self.max_length,
            truncation=self.truncation,
        )
        if not any(loss_mask):
            raise TaiChuDataFormatError(
                "SFT record has no trainable assistant tokens after truncation"
            )
        messages = [
            ConversationMessage(
                role=str(message["role"]),
                content=(
                    normalize_reasoning_tags(str(message["content"]))
                    if message["role"] == "assistant"
                    else str(message["content"])
                ),
                name=message.get("name"),
                metadata=dict(message.get("metadata", {})),
            )
            for message in record["messages"]
        ]
        return SFTSequenceSample(
            sample_id=sample_identifier(record, "SFT"),
            dataset_id=dataset_identifier(record, "sft"),
            source_id=source_identifier(record),
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=[1] * len(input_ids),
            segment_ids=[0] * len(input_ids),
            messages=messages,
        )


__all__ = ["SFTPreprocessor"]
