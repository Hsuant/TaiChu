"""Reasoning mode policy and canonical reasoning-output parsing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence


class ReasoningMode(str, Enum):
    ON = "on"
    OFF = "off"
    AUTO = "auto"

    @classmethod
    def parse(cls, value: str) -> "ReasoningMode":
        try:
            return cls(value.strip().lower())
        except ValueError as exc:
            raise ValueError(
                "reasoning_mode must be one of: on, off, auto"
            ) from exc


@dataclass(frozen=True, slots=True)
class ReasoningOutput:
    mode: ReasoningMode
    thinking: bool
    reasoning_text: str
    answer_text: str


def parse_reasoning_tokens(
    token_ids: Sequence[int],
    *,
    mode: ReasoningMode,
    think_token_id: int,
    end_think_token_id: int,
    decode: Callable[[Sequence[int]], str],
) -> ReasoningOutput:
    """Split generated IDs into canonical reasoning and final answer spans."""
    if mode is ReasoningMode.OFF:
        return ReasoningOutput(
            mode=mode,
            thinking=False,
            reasoning_text="",
            answer_text=decode(token_ids).strip(),
        )

    forced = mode is ReasoningMode.ON
    started = forced or (bool(token_ids) and token_ids[0] == think_token_id)
    if not started:
        return ReasoningOutput(
            mode=mode,
            thinking=False,
            reasoning_text="",
            answer_text=decode(token_ids).strip(),
        )

    start = 1 if not forced else 0
    try:
        end = token_ids.index(end_think_token_id, start)
    except ValueError as exc:
        raise RuntimeError("thinking response did not emit <|/think|>") from exc
    reasoning_ids = token_ids[start:end]
    answer_ids = token_ids[end + 1 :]
    return ReasoningOutput(
        mode=mode,
        thinking=True,
        reasoning_text=decode(reasoning_ids).strip(),
        answer_text=decode(answer_ids).strip(),
    )


__all__ = ["ReasoningMode", "ReasoningOutput", "parse_reasoning_tokens"]
