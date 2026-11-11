"""Strict configuration for the TaiChu conversational entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


_TOP_LEVEL_KEYS = {
    "checkpoint",
    "tokenizer",
    "system_prompt",
    "device",
    "dtype",
    "context_tokens",
    "sampling",
}
_SAMPLING_KEYS = {
    "max_new_tokens",
    "temperature",
    "top_k",
    "top_p",
    "repetition_penalty",
    "seed",
}
_DEFAULT_SYSTEM_PROMPT = (
    "你是 TaiChu（太初），一个诚实、清晰的中文对话助手。"
    "回答应直接、准确；不确定时明确说明。"
)


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Token-level decoding parameters."""

    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = 0.9
    repetition_penalty: float = 1.05
    seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.max_new_tokens, int) or isinstance(
            self.max_new_tokens,
            bool,
        ):
            raise TypeError("sampling.max_new_tokens must be an integer")
        if self.max_new_tokens <= 0:
            raise ValueError("sampling.max_new_tokens must be positive")
        if not isinstance(self.temperature, (int, float)) or isinstance(
            self.temperature,
            bool,
        ):
            raise TypeError("sampling.temperature must be numeric")
        if self.temperature < 0:
            raise ValueError("sampling.temperature cannot be negative")
        if self.top_k is not None and (
            not isinstance(self.top_k, int) or isinstance(self.top_k, bool)
        ):
            raise TypeError("sampling.top_k must be an integer or null")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("sampling.top_k must be positive or null")
        if self.top_p is not None and (
            not isinstance(self.top_p, (int, float))
            or isinstance(self.top_p, bool)
        ):
            raise TypeError("sampling.top_p must be numeric or null")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("sampling.top_p must be in (0, 1] or null")
        if not isinstance(self.repetition_penalty, (int, float)) or isinstance(
            self.repetition_penalty,
            bool,
        ):
            raise TypeError("sampling.repetition_penalty must be numeric")
        if self.repetition_penalty < 1.0:
            raise ValueError("sampling.repetition_penalty must be at least 1.0")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("sampling.seed must be an integer")
        if self.seed < 0:
            raise ValueError("sampling.seed cannot be negative")


@dataclass(frozen=True, slots=True)
class ChatConfig:
    """Inputs and runtime policy for one chat application."""

    checkpoint: Path
    tokenizer: Path
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    device: str = "auto"
    dtype: str = "auto"
    context_tokens: int | None = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, Path):
            raise TypeError("checkpoint must be a pathlib.Path")
        if not isinstance(self.tokenizer, Path):
            raise TypeError("tokenizer must be a pathlib.Path")
        if not isinstance(self.system_prompt, str):
            raise TypeError("system_prompt must be a string")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        if not isinstance(self.dtype, str):
            raise TypeError("dtype must be a string")
        if self.device not in {"auto", "cpu", "cuda"} and not self.device.startswith("cuda:"):
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError(
                "dtype must be 'auto', 'float32', 'float16', or 'bfloat16'"
            )
        if self.context_tokens is not None:
            if not isinstance(self.context_tokens, int) or isinstance(
                self.context_tokens,
                bool,
            ):
                raise TypeError("context_tokens must be an integer or null")
            if self.context_tokens < 2:
                raise ValueError("context_tokens must be at least 2 or null")
        if not isinstance(self.sampling, SamplingConfig):
            raise TypeError("sampling must be a SamplingConfig")


def _required_path(
    raw: Mapping[str, Any],
    key: str,
    *,
    config_dir: Path,
) -> Path:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def load_chat_config(path: str | Path) -> ChatConfig:
    """Load a strict YAML chat configuration relative to its own directory."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"chat config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("chat config must be a YAML mapping")

    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"unknown chat config keys: {sorted(unknown)}")

    sampling_raw = raw.get("sampling", {})
    if not isinstance(sampling_raw, Mapping):
        raise ValueError("sampling must be a mapping")
    unknown_sampling = set(sampling_raw) - _SAMPLING_KEYS
    if unknown_sampling:
        raise ValueError(
            f"unknown sampling config keys: {sorted(unknown_sampling)}"
        )

    config_dir = config_path.parent
    return ChatConfig(
        checkpoint=_required_path(raw, "checkpoint", config_dir=config_dir),
        tokenizer=_required_path(raw, "tokenizer", config_dir=config_dir),
        system_prompt=raw.get(
            "system_prompt",
            _DEFAULT_SYSTEM_PROMPT,
        ),
        device=raw.get("device", "auto"),
        dtype=raw.get("dtype", "auto"),
        context_tokens=raw.get("context_tokens"),
        sampling=SamplingConfig(**dict(sampling_raw)),
    )


__all__ = ["ChatConfig", "SamplingConfig", "load_chat_config"]
