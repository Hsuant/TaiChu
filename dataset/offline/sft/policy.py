"""Conversation content policy shared by SFT selection and preprocessing."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ...exceptions import TaiChuDataFormatError
from ..adapters.base import (
    CANONICAL_THINK_OPEN,
    normalize_reasoning_tags,
    validate_reasoning_protocol,
)


_REASONING_BLOCK = re.compile(
    r"<\|think\|>.*?<\|/think\|>",
    flags=re.DOTALL,
)


class SFTContentRejected(TaiChuDataFormatError):
    """A rejected SFT sample with a stable reason code."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class SFTContentResult:
    messages: list[dict[str, str]]
    dropped_system_messages: int
    has_reasoning: bool
    assistant_texts: tuple[str, ...]


class SFTContentPolicy:
    """Apply reasoning normalization and role constraints."""

    def __init__(
        self,
        *,
        drop_system: bool,
    ) -> None:
        self.drop_system = bool(drop_system)

    @staticmethod
    def has_reasoning(messages: list[dict[str, str]]) -> bool:
        return any(
            message["role"] == "assistant"
            and CANONICAL_THINK_OPEN
            in normalize_reasoning_tags(message["content"])
            for message in messages
        )

    def apply(
        self,
        messages: list[dict[str, str]],
        *,
        strip_reasoning: bool,
    ) -> SFTContentResult:
        cleaned: list[dict[str, str]] = []
        dropped_system = 0
        has_reasoning = False
        assistant_texts = []
        for message in messages:
            role = message["role"]
            if role == "system" and self.drop_system:
                dropped_system += 1
                continue
            content = message["content"].strip()
            if role == "assistant":
                content = normalize_reasoning_tags(content)
                try:
                    validate_reasoning_protocol(content)
                except TaiChuDataFormatError as exc:
                    raise SFTContentRejected(
                        "malformed_reasoning",
                        str(exc),
                    ) from exc
                has_reasoning = has_reasoning or (
                    CANONICAL_THINK_OPEN in content
                )
                if strip_reasoning:
                    content = _REASONING_BLOCK.sub("", content).strip()
                    content = re.sub(r"\n{3,}", "\n\n", content)
                if not content:
                    raise SFTContentRejected(
                        "empty_assistant",
                        "assistant content is empty after reasoning cleanup",
                    )
                assistant_texts.append(content)
            cleaned.append({"role": role, "content": content})

        roles = {message["role"] for message in cleaned}
        if "user" not in roles or "assistant" not in roles:
            raise SFTContentRejected(
                "missing_role",
                "SFT content requires both user and assistant messages",
            )
        return SFTContentResult(
            messages=cleaned,
            dropped_system_messages=dropped_system,
            has_reasoning=has_reasoning,
            assistant_texts=tuple(assistant_texts),
        )


__all__ = [
    "SFTContentPolicy",
    "SFTContentRejected",
    "SFTContentResult",
]
