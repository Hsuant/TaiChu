"""Special-token protocol helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SPECIAL_TOKENS: Tuple[str, ...] = (
    "<|pad|>",
    "<unk>",
    "<|endoftext|>",
    "<|taichu_start|>",
    "<|taichu_end|>",
    "<|think|>",
    "<|/think|>",
    "<|ghissue|>",
    "<|final|>",
)
BOS_TOKEN = "<|taichu_start|>"
EOS_TOKEN = "<|taichu_end|>"
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<unk>"


class SpecialTokenError(ValueError):
    """Raised when special-token files disagree with the training protocol."""


def expected_ids(tokens: Iterable[str]) -> Dict[str, int]:
    return {token: index for index, token in enumerate(tokens)}


def validate_special_token_files(directory: Path, ordered_tokens: Iterable[str]) -> None:
    expected = expected_ids(ordered_tokens)
    config_path = directory / "tokenizer_config.json"
    map_path = directory / "special_tokens_map.json"
    if not config_path.is_file() or not map_path.is_file():
        raise SpecialTokenError(f"缺少 Tokenizer 配置文件: {directory}")

    with config_path.open("r", encoding="utf-8") as handle:
        tokenizer_config = json.load(handle)
    with map_path.open("r", encoding="utf-8") as handle:
        special_map = json.load(handle)

    if isinstance(special_map.get("additional_special_tokens"), list):
        special_map = dict(special_map)
        special_map["additional_special_tokens"] = [
            item["content"] if isinstance(item, dict) else item
            for item in special_map["additional_special_tokens"]
        ]

    declared = _declared_tokens(tokenizer_config)
    if declared != set(expected):
        raise SpecialTokenError(
            "tokenizer_config.json 声明的 Token 与训练协议不一致: "
            f"extra={sorted(declared.difference(expected))}, "
            f"missing={sorted(set(expected).difference(declared))}"
        )

    decoder = tokenizer_config.get("added_tokens_decoder", {})
    actual = {
        item["content"]: int(token_id)
        for token_id, item in decoder.items()
        if isinstance(item, dict) and "content" in item
    }
    if actual != expected:
        raise SpecialTokenError(
            f"tokenizer_config.json 的特殊 Token ID 不一致: "
            f"expected={expected}, actual={actual}"
        )

    standard = {
        "bos_token": special_map.get("bos_token"),
        "eos_token": special_map.get("eos_token"),
        "unk_token": special_map.get("unk_token"),
        "pad_token": special_map.get("pad_token"),
    }
    for name, value in standard.items():
        config_value = tokenizer_config.get(name)
        if value != config_value:
            raise SpecialTokenError(
                f"{name} 在 tokenizer_config.json 与 special_tokens_map.json 中不一致"
            )


def _declared_tokens(config: Dict[str, object]) -> set[str]:
    tokens: List[str] = []
    for name in ("bos_token", "eos_token", "unk_token", "pad_token"):
        value = config.get(name)
        if isinstance(value, str):
            tokens.append(value)
    additional = config.get("additional_special_tokens")
    if isinstance(additional, list):
        tokens.extend(
            item["content"] if isinstance(item, dict) else item
            for item in additional
        )
    return set(tokens)
