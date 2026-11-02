"""Shared offline preprocessing adapter contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ...exceptions import TaiChuDataFormatError
from ...schema import token_ids


@dataclass(frozen=True, slots=True)
class RenderedTurn:
    role: str
    prefix: str
    content: str
    suffix: str


class ConversationRenderer(Protocol):
    def render(self, message: Mapping[str, Any]) -> RenderedTurn: ...

    def render_prompt(self, messages: list[Mapping[str, Any]]) -> str: ...


class RoleTagRenderer:
    """Minimal explicit chat format that keeps assistant spans machine-locatable."""

    def __init__(
        self,
        *,
        start_token: str = "<|taichu_start|>",
        end_token: str = "<|taichu_end|>",
    ) -> None:
        self.start_token = start_token
        self.end_token = end_token

    def render(self, message: Mapping[str, Any]) -> RenderedTurn:
        role = str(message.get("role", ""))
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise TaiChuDataFormatError(f"unsupported message role {role!r}")
        if not isinstance(content, str):
            raise TaiChuDataFormatError("message content must be a string")
        prefix = f"{self.start_token}{role}\n"
        suffix = f"{self.end_token}\n"
        return RenderedTurn(role=role, prefix=prefix, content=content, suffix=suffix)

    def render_prompt(self, messages: list[Mapping[str, Any]]) -> str:
        return "".join(
            turn.prefix + turn.content + turn.suffix
            for turn in (self.render(message) for message in messages)
        )


class TextTokenizer:
    """Task-independent bridge for native Tokenizers and HF-style tokenizers."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        pad_token: str | None = "<|pad|>",
        bos_token: str | None = None,
        eos_token: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = self._vocab_size(tokenizer)
        self.pad_token_id = self._special_id("pad", pad_token)
        self.bos_token_id = self._special_id("bos", bos_token)
        self.eos_token_id = self._special_id("eos", eos_token)
        if self.pad_token_id is None:
            raise ValueError("pad_token_id is required")
        if self.vocab_size is not None:
            for name, value in (
                ("pad", self.pad_token_id),
                ("bos", self.bos_token_id),
                ("eos", self.eos_token_id),
            ):
                if value is not None and not 0 <= value < self.vocab_size:
                    raise ValueError(
                        f"{name}_token_id is outside tokenizer vocabulary"
                    )

    @staticmethod
    def _vocab_size(tokenizer: Any) -> int | None:
        getter = getattr(tokenizer, "get_vocab_size", None)
        if callable(getter):
            try:
                return int(getter(with_added_tokens=True))
            except TypeError:
                return int(getter())
        try:
            return int(len(tokenizer))
        except (TypeError, AttributeError):
            value = getattr(tokenizer, "vocab_size", None)
            return None if value is None else int(value)

    def _special_id(self, kind: str, configured_token: str | None) -> int | None:
        direct = getattr(self.tokenizer, f"{kind}_token_id", None)
        if direct is not None:
            return int(direct)
        token = configured_token or getattr(self.tokenizer, f"{kind}_token", None)
        if token is None:
            return None
        lookup = getattr(self.tokenizer, "token_to_id", None)
        if callable(lookup):
            value = lookup(token)
        else:
            get_vocab = getattr(self.tokenizer, "get_vocab", None)
            if callable(get_vocab):
                value = get_vocab().get(token)
            else:
                convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
                value = convert(token) if callable(convert) else None
                unknown = getattr(self.tokenizer, "unk_token_id", None)
                if value == unknown and token != getattr(
                    self.tokenizer,
                    "unk_token",
                    None,
                ):
                    value = None
        return None if value is None else int(value)

    def encode(self, text: str) -> list[int]:
        rows = self.encode_batch([text])
        if len(rows) != 1:
            raise TaiChuDataFormatError("tokenizer did not return exactly one sequence")
        return list(rows[0])

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        if not texts:
            return []
        method = getattr(self.tokenizer, "encode_batch", None)
        if callable(method):
            try:
                encoded = method(texts, add_special_tokens=False)
                rows = [self._ids(item) for item in encoded]
                return self._validate_rows(rows, len(texts))
            except TypeError:
                pass
        if callable(self.tokenizer):
            encoded = self.tokenizer(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )
            rows = [self._ids(row) for row in encoded["input_ids"]]
            return self._validate_rows(rows, len(texts))
        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            rows = []
            for text in texts:
                try:
                    encoded = encode(text, add_special_tokens=False)
                except TypeError as exc:
                    raise TypeError(
                        "tokenizer.encode() must accept add_special_tokens=False"
                    ) from exc
                rows.append(self._ids(encoded))
            return self._validate_rows(rows, len(texts))
        raise TypeError("tokenizer must expose encode_batch(), __call__() or encode()")

    @staticmethod
    def _ids(encoded: Any) -> list[int]:
        values = getattr(encoded, "ids", None)
        if values is None and isinstance(encoded, Mapping):
            values = encoded.get("input_ids")
        if values is None:
            values = encoded
        return token_ids(values)

    def _validate_rows(self, rows: list[list[int]], expected: int) -> list[list[int]]:
        if len(rows) != expected or any(not row for row in rows):
            raise TaiChuDataFormatError(
                f"tokenizer returned {len(rows)} non-empty sequences for {expected} texts"
            )
        return [
            token_ids(row, vocab_size=self.vocab_size)
            for row in rows
        ]


class BasePreprocessingAdapter:
    """Template-method base for schema validation and formatting."""

    sample_type: str

    def prepare(self, record: Mapping[str, Any]):
        self.validate(record)
        formatted = self.format(record)
        sample = self.tokenize(record, formatted)
        sample.metadata.update(self.build_metadata(record, formatted))
        sample.validate()
        return sample

    def validate(self, record: Mapping[str, Any]) -> None:
        if not isinstance(record, Mapping):
            raise TaiChuDataFormatError("raw sample must be a mapping")

    def format(self, record: Mapping[str, Any]):
        return record

    def tokenize(self, record: Mapping[str, Any], formatted):
        raise NotImplementedError

    def build_metadata(self, record: Mapping[str, Any], formatted) -> dict[str, Any]:
        return {}


def sample_identifier(record: Mapping[str, Any], prefix: str) -> str:
    value = record.get("sample_id")
    if value is None:
        raise TaiChuDataFormatError(f"{prefix} preprocessing requires sample_id")
    if not isinstance(value, str) or not value:
        raise TaiChuDataFormatError("sample_id must be a non-empty string")
    return value


def dataset_identifier(record: Mapping[str, Any], default: str) -> str:
    value = record.get("dataset_id", default)
    if not isinstance(value, str) or not value:
        raise TaiChuDataFormatError("dataset_id must be a non-empty string")
    return value


def source_identifier(record: Mapping[str, Any], default: str = "") -> str:
    value = record.get("source_id", default)
    if not isinstance(value, str):
        raise TaiChuDataFormatError("source_id must be a string")
    return value


def apply_max_length(
    input_ids: list[int],
    labels: list[int],
    loss_mask: list[int],
    *,
    max_length: int | None,
    truncation: str,
) -> tuple[list[int], list[int], list[int]]:
    if max_length is None or len(input_ids) <= max_length:
        return input_ids, labels, loss_mask
    if truncation == "left":
        return input_ids[-max_length:], labels[-max_length:], loss_mask[-max_length:]
    if truncation == "right":
        return input_ids[:max_length], labels[:max_length], loss_mask[:max_length]
    raise ValueError("truncation must be 'left', 'right' or None")


__all__ = [
    "BasePreprocessingAdapter",
    "ConversationRenderer",
    "RenderedTurn",
    "RoleTagRenderer",
    "TextTokenizer",
    "apply_max_length",
    "dataset_identifier",
    "sample_identifier",
    "source_identifier",
]
