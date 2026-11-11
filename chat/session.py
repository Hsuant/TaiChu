"""Conversation history and context-window management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from chat.tokenizer import ChatTokenizer


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    text: str
    token_ids: tuple[int, ...]
    messages: tuple[ChatMessage, ...]
    dropped_turns: int


class ContextLengthError(ValueError):
    """Raised when one user turn cannot fit inside the model context."""


class Conversation:
    """Owns mutable dialogue state and keeps complete recent turns."""

    def __init__(self, tokenizer: ChatTokenizer, system_prompt: str = "") -> None:
        self._tokenizer = tokenizer
        self._system_prompt = system_prompt.strip()
        self._messages: list[ChatMessage] = []

    @property
    def messages(self) -> tuple[ChatMessage, ...]:
        return tuple(self._messages)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt.strip()

    def clear(self) -> None:
        self._messages.clear()

    def add_user(self, content: str) -> None:
        self._append("user", content)

    def add_assistant(self, content: str) -> None:
        self._append("assistant", content)

    def remove_last_user(self) -> None:
        if self._messages and self._messages[-1].role == "user":
            self._messages.pop()

    def apply_prompt(self, prompt: PreparedPrompt) -> None:
        """Commit prompt trimming only after a reply has been generated."""

        self._messages = list(prompt.messages)

    def prepare(self, max_input_tokens: int) -> PreparedPrompt:
        """Render the newest complete exchanges that fit the context budget."""

        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        body = list(self._messages)
        dropped_turns = 0

        prompt = self._render_prompt(body)
        token_ids = self._tokenizer.encode(prompt)
        while len(token_ids) > max_input_tokens and len(body) > 1:
            if (
                len(body) >= 2
                and body[0].role == "user"
                and body[1].role == "assistant"
            ):
                del body[:2]
                dropped_turns += 1
            else:
                body.pop(0)
            prompt = self._render_prompt(body)
            token_ids = self._tokenizer.encode(prompt)

        if len(token_ids) > max_input_tokens:
            raise ContextLengthError(
                "system prompt and latest user turn require "
                f"{len(token_ids)} tokens, but only {max_input_tokens} are available"
            )
        if not body or body[-1].role != "user":
            raise ValueError("conversation must end with one user message")

        return PreparedPrompt(
            text=prompt,
            token_ids=tuple(token_ids),
            messages=tuple(body),
            dropped_turns=dropped_turns,
        )

    def _append(self, role: str, content: str) -> None:
        text = content.strip()
        if not text:
            raise ValueError(f"{role} message cannot be empty")
        self._messages.append(ChatMessage(role=role, content=text))

    def _render_prompt(self, body: Sequence[ChatMessage]) -> str:
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(message.as_dict() for message in body)
        return self._tokenizer.render(messages, add_generation_prompt=True)


__all__ = [
    "ChatMessage",
    "ContextLengthError",
    "Conversation",
    "PreparedPrompt",
]
