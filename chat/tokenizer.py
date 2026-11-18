"""Tokenizers-to-chat-template adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


class ChatTokenizer(Protocol):
    """Minimal tokenizer contract used by sessions and the chat engine."""

    eos_token_id: int
    think_token_id: int
    end_think_token_id: int

    def render(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
        force_reasoning: bool = False,
    ) -> str:
        ...

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        ...


class HFChatTokenizer:
    """Adapter around the project's Hugging Face-compatible tokenizer bundle."""

    def __init__(self, tokenizer: Any) -> None:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("tokenizer does not provide apply_chat_template")
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id")
        think_token_id = tokenizer.convert_tokens_to_ids("<|think|>")
        end_think_token_id = tokenizer.convert_tokens_to_ids("<|/think|>")
        if think_token_id is None or end_think_token_id is None:
            raise ValueError("tokenizer must define canonical reasoning tokens")
        self._tokenizer = tokenizer
        self.eos_token_id = int(eos_token_id)
        self.think_token_id = int(think_token_id)
        self.end_think_token_id = int(end_think_token_id)

    @property
    def vocab_size(self) -> int:
        vocab_size = getattr(self._tokenizer, "vocab_size", None)
        if vocab_size is None:
            vocab_size = len(self._tokenizer.get_vocab())
        return int(vocab_size)

    @classmethod
    def from_path(cls, path: str | Path) -> "HFChatTokenizer":
        from transformers import AutoTokenizer

        source = Path(path).expanduser().resolve()
        if source.is_file():
            source = source.parent
        if not source.is_dir():
            raise FileNotFoundError(f"tokenizer directory not found: {source}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(source),
            local_files_only=True,
            trust_remote_code=False,
        )
        return cls(tokenizer)

    def render(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
        force_reasoning: bool = False,
    ) -> str:
        rendered = self._tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            force_reasoning=force_reasoning,
        )
        if not isinstance(rendered, str):
            raise TypeError("chat template must render to a string")
        return rendered

    def encode(self, text: str) -> list[int]:
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        return [int(token_id) for token_id in token_ids]

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


__all__ = ["ChatTokenizer", "HFChatTokenizer"]
