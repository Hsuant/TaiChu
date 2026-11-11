"""Tokenizer access shared by parallel scan and output workers."""
from __future__ import annotations

from tokenizers import Tokenizer

from ..adapters import TextTokenizer


_WORKER_TOKENIZER: TextTokenizer | None = None


def initialize_tokenizer_worker(
    tokenizer_path: str,
    pad_token: str,
    bos_token: str,
    eos_token: str,
) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = TextTokenizer(
        Tokenizer.from_file(tokenizer_path),
        pad_token=pad_token,
        bos_token=bos_token,
        eos_token=eos_token,
    )


def tokenize_batch(
    texts: list[str],
    *,
    tokenizer: TextTokenizer | None = None,
) -> list[list[int]]:
    active_tokenizer = tokenizer or _WORKER_TOKENIZER
    if active_tokenizer is None:
        raise RuntimeError("tokenizer worker was not initialized")
    return active_tokenizer.encode_batch_unchecked(texts)


def get_worker_tokenizer() -> TextTokenizer:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("tokenizer worker was not initialized")
    return _WORKER_TOKENIZER


__all__ = [
    "get_worker_tokenizer",
    "initialize_tokenizer_worker",
    "tokenize_batch",
]
