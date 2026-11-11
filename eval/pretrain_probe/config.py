"""Strict configuration for pretrained-model continuation probes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from chat.config import SamplingConfig


_TOP_LEVEL_KEYS = {
    "checkpoint",
    "tokenizer",
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


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    checkpoint: Path
    tokenizer: Path
    device: str = "auto"
    dtype: str = "auto"
    context_tokens: int | None = None
    sampling: SamplingConfig = field(
        default_factory=lambda: SamplingConfig(max_new_tokens=128)
    )

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, Path):
            raise TypeError("checkpoint must be a pathlib.Path")
        if not isinstance(self.tokenizer, Path):
            raise TypeError("tokenizer must be a pathlib.Path")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        if not isinstance(self.dtype, str):
            raise TypeError("dtype must be a string")
        if self.device not in {"auto", "cpu", "cuda"} and not self.device.startswith(
            "cuda:"
        ):
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


def load_probe_config(path: str | Path) -> ProbeConfig:
    """Load a strict continuation-probe YAML relative to its directory."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"probe config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("probe config must be a YAML mapping")

    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"unknown probe config keys: {sorted(unknown)}")

    sampling_raw = raw.get("sampling", {})
    if not isinstance(sampling_raw, Mapping):
        raise ValueError("sampling must be a mapping")
    unknown_sampling = set(sampling_raw) - _SAMPLING_KEYS
    if unknown_sampling:
        raise ValueError(
            f"unknown sampling config keys: {sorted(unknown_sampling)}"
        )

    config_dir = config_path.parent
    return ProbeConfig(
        checkpoint=_required_path(raw, "checkpoint", config_dir=config_dir),
        tokenizer=_required_path(raw, "tokenizer", config_dir=config_dir),
        device=raw.get("device", "auto"),
        dtype=raw.get("dtype", "auto"),
        context_tokens=raw.get("context_tokens"),
        sampling=SamplingConfig(**dict(sampling_raw)),
    )


__all__ = ["ProbeConfig", "load_probe_config"]
