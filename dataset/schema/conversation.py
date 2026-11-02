"""Conversation and supervised fine-tuning schemas."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

from .common import (
    CommonSample,
    binary_mask,
    positive_identifier,
    register_sample,
    require_aligned,
    token_ids,
)


VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


@dataclass(slots=True)
class ConversationMessage:
    role: str
    content: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"unsupported message role {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("message name must be a string or None")
        if not isinstance(self.metadata, dict):
            raise TypeError("message metadata must be a dictionary")


@dataclass(slots=True)
class Conversation:
    messages: list[ConversationMessage] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.messages = [
            item
            if isinstance(item, ConversationMessage)
            else ConversationMessage(**item)
            for item in self.messages
        ]
        if not self.messages:
            raise ValueError("conversation must contain at least one message")

    def append(self, message: ConversationMessage | Mapping[str, Any]) -> None:
        self.messages.append(
            message
            if isinstance(message, ConversationMessage)
            else ConversationMessage(**message)
        )


@register_sample
@dataclass(slots=True, kw_only=True)
class SFTSequenceSample(CommonSample):
    """A rendered multi-turn conversation with an assistant-only loss mask."""

    input_ids: list[int]
    labels: list[int]
    loss_mask: list[int]
    attention_mask: list[int] | None = None
    segment_ids: list[int] | None = None
    messages: list[ConversationMessage] = field(default_factory=list)
    sample_type: ClassVar[str] = "sft"
    array_fields: ClassVar[tuple[str, ...]] = (
        "input_ids",
        "labels",
        "loss_mask",
        "attention_mask",
        "segment_ids",
    )

    def __post_init__(self) -> None:
        CommonSample.__post_init__(self)
        self.input_ids = token_ids(self.input_ids)
        self.labels = token_ids(
            self.labels,
            name="labels",
            allow_ignore_index=True,
        )
        self.loss_mask = binary_mask(self.loss_mask)
        self.attention_mask = (
            None
            if self.attention_mask is None
            else binary_mask(self.attention_mask, name="attention_mask")
        )
        self.segment_ids = (
            None
            if self.segment_ids is None
            else token_ids(self.segment_ids, name="segment_ids")
        )
        self.messages = [
            item
            if isinstance(item, ConversationMessage)
            else ConversationMessage(**item)
            for item in self.messages
        ]
        aligned = [self.input_ids, self.labels, self.loss_mask]
        if self.attention_mask is not None:
            aligned.append(self.attention_mask)
        if self.segment_ids is not None:
            aligned.append(self.segment_ids)
        require_aligned("SFT sample", *aligned)
        for index, mask in enumerate(self.loss_mask):
            if not mask and self.labels[index] != -100:
                raise ValueError("positions outside loss_mask must use label -100")


__all__ = [
    "Conversation",
    "ConversationMessage",
    "SFTSequenceSample",
    "VALID_ROLES",
]
