"""Process-pool workers for batch tokenization during offline preparation."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from .adapters import PretrainingPreprocessor, TextTokenizer

_PRETRAINING_ADAPTER: PretrainingPreprocessor | None = None


def initialize_pretraining_worker(
    tokenizer_path: str,
    pad_token: str,
    bos_token: str,
    eos_token: str,
    text_column: str,
    add_bos: bool,
    add_eos: bool,
    max_length: int | None,
) -> None:
    global _PRETRAINING_ADAPTER
    tokenizer = TextTokenizer(
        Tokenizer.from_file(str(Path(tokenizer_path))),
        pad_token=pad_token,
        bos_token=bos_token,
        eos_token=eos_token,
    )
    _PRETRAINING_ADAPTER = PretrainingPreprocessor(
        tokenizer,
        text_column=text_column,
        add_bos=add_bos,
        add_eos=add_eos,
        max_length=max_length,
    )


def prepare_pretraining_batch(records: Iterable[dict[str, Any]]) -> list:
    if _PRETRAINING_ADAPTER is None:
        raise RuntimeError("pretraining worker was not initialized")
    return _PRETRAINING_ADAPTER.prepare_batch(records)


__all__ = [
    "initialize_pretraining_worker",
    "prepare_pretraining_batch",
]
