"""Functional validation for a saved HuggingFace Tokenizer."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from tokenizer.special_tokens import validate_special_token_files
from tokenizer.template import CHAT_TEMPLATE, validate_template_tokens


class TokenizerValidationError(ValueError):
    """Raised when a saved Tokenizer fails functional validation."""


ROUNDTRIP_SAMPLES = (
    "人工智能改变了世界。",
    "Byte-level BPE keeps spaces and\nnewlines stable.",
    "中英混合 mixed text, numbers 1234567890 and punctuation!?",
    "def tokenize(text):\n    return text.encode('utf-8')\n",
    "Emoji: 😀🚀 中文标点：，。！？",
)


def validate_tokenizer_fast(
    directory: Path,
    ordered_tokens: Iterable[str],
    expected_vocab_size: int,
) -> Dict[str, object]:
    """Reload with Tokenizers only and validate files, IDs, and round trips."""
    from tokenizers import Tokenizer

    ordered = tuple(ordered_tokens)
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    expected = {token: index for index, token in enumerate(ordered)}
    actual = {token: tokenizer.token_to_id(token) for token in ordered}
    if actual != expected:
        raise TokenizerValidationError(
            f"特殊 Token ID 不一致: expected={expected}, actual={actual}"
        )

    for token, token_id in expected.items():
        encoded = tokenizer.encode(token, add_special_tokens=False).ids
        if encoded != [token_id]:
            raise TokenizerValidationError(
                f"特殊 Token 未作为独立 Token 编码: {token} -> {encoded}"
            )
        if tokenizer.decode([token_id], skip_special_tokens=False) != token:
            raise TokenizerValidationError(f"特殊 Token 解码不一致: {token}")

    roundtrip_tokens: Dict[str, int] = {}
    for text in ROUNDTRIP_SAMPLES:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded != text:
            raise TokenizerValidationError(
                f"编码解码不可逆: input={text!r}, decoded={decoded!r}"
            )
        roundtrip_tokens[text] = len(ids)

    validate_special_token_files(directory, ordered)
    template_path = directory / "chat_template.jinja"
    if not template_path.is_file():
        raise TokenizerValidationError(f"Chat Template 文件不存在: {template_path}")
    saved_template = template_path.read_text(encoding="utf-8")
    if saved_template != CHAT_TEMPLATE:
        raise TokenizerValidationError("保存的 Chat Template 与代码定义不一致")
    validate_template_tokens(CHAT_TEMPLATE, ordered)
    actual_vocab_size = tokenizer.get_vocab_size()
    if actual_vocab_size != expected_vocab_size:
        raise TokenizerValidationError(
            f"实际词表大小 {actual_vocab_size} 与目标 {expected_vocab_size} 不一致"
        )
    return {
        "mode": "tokenizers_fast",
        "vocab_size": actual_vocab_size,
        "special_token_ids": actual,
        "roundtrip_token_counts": roundtrip_tokens,
        "chat_template_checked": "static",
    }


def validate_tokenizer(
    directory: Path,
    ordered_tokens: Iterable[str],
    expected_vocab_size: int,
) -> Dict[str, object]:
    from transformers import PreTrainedTokenizerFast

    ordered = tuple(ordered_tokens)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        str(directory),
        use_fast=True,
        local_files_only=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise TokenizerValidationError("重新加载的 Tokenizer 不是 fast tokenizer")

    expected = {token: index for index, token in enumerate(ordered)}
    actual_added = tokenizer.get_added_vocab()
    if set(actual_added) != set(expected):
        raise TokenizerValidationError(
            "added vocabulary 与协议不一致: "
            f"extra={sorted(set(actual_added).difference(expected))}, "
            f"missing={sorted(set(expected).difference(actual_added))}"
        )
    actual_ids = {token: actual_added[token] for token in ordered}
    if actual_ids != expected:
        raise TokenizerValidationError(
            f"特殊 Token ID 不一致: expected={expected}, actual={actual_ids}"
        )

    for token, token_id in expected.items():
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            raise TokenizerValidationError(
                f"特殊 Token 未作为独立 Token 编码: {token} -> {encoded}"
            )
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if decoded != token:
            raise TokenizerValidationError(
                f"特殊 Token 解码不一致: {token!r} != {decoded!r}"
            )

    roundtrip_tokens: Dict[str, int] = {}
    for text in ROUNDTRIP_SAMPLES:
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded != text:
            raise TokenizerValidationError(
                f"编码解码不可逆: input={text!r}, decoded={decoded!r}"
            )
        roundtrip_tokens[text] = len(ids)

    if tokenizer.encode("", add_special_tokens=False):
        raise TokenizerValidationError("空字符串不应产生普通 Token")

    chat_template = tokenizer.chat_template
    if not chat_template:
        raise TokenizerValidationError("chat_template 未保存")
    if chat_template != CHAT_TEMPLATE:
        raise TokenizerValidationError("tokenizer.chat_template 与代码定义不一致")
    validate_template_tokens(chat_template, ordered)
    generation_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "你好"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not generation_text.endswith("assistant\n"):
        raise TokenizerValidationError(
            f"add_generation_prompt 输出不符合预期: {generation_text!r}"
        )
    if not generation_text.startswith(ordered[3] if len(ordered) > 3 else ordered[0]):
        raise TokenizerValidationError("Chat Template 未使用声明的消息起始 Token")

    multi_turn = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "系统提示"},
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答"},
            {"role": "user", "content": "第二问"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    for role in ("system", "user", "assistant"):
        if f"{role}\n" not in multi_turn:
            raise TokenizerValidationError(f"多轮模板缺少角色边界: {role}")
    if not multi_turn.endswith("<|taichu_end|>\n"):
        raise TokenizerValidationError("多轮模板的消息结束边界不一致")

    validate_special_token_files(directory, ordered)
    actual_vocab_size = len(tokenizer)
    if actual_vocab_size > expected_vocab_size:
        raise TokenizerValidationError(
            f"实际词表大小 {actual_vocab_size} 超过配置 {expected_vocab_size}"
        )
    if actual_vocab_size < expected_vocab_size:
        raise TokenizerValidationError(
            f"实际词表大小 {actual_vocab_size} 未达到目标 {expected_vocab_size}"
        )

    return {
        "mode": "transformers_full",
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": actual_vocab_size,
        "special_token_ids": actual_ids,
        "roundtrip_token_counts": roundtrip_tokens,
        "chat_template_generation_output": generation_text,
        "chat_template_multi_turn_output": multi_turn,
    }


def quality_snapshot(tokenizer, samples: List[str]) -> Dict[str, int]:
    return {
        text: len(tokenizer.encode(text, add_special_tokens=False))
        for text in samples
    }
