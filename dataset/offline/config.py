"""Typed configuration for unified offline sample preparation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CleaningConfig:
    unicode_form: str = "NFC"
    min_chars: int = 0
    max_chars: int = 200_000
    collapse_whitespace: bool = False
    remove_control_chars: bool = True
    oversize_policy: str = "skip"
    deduplicate: bool = True

    def __post_init__(self) -> None:
        if self.unicode_form not in {"NFC", "NFKC"}:
            raise ValueError("unicode_form must be NFC or NFKC")
        if self.min_chars < 0 or self.max_chars < self.min_chars:
            raise ValueError("invalid text length limits")
        if self.oversize_policy not in {"skip", "truncate"}:
            raise ValueError("oversize_policy must be skip or truncate")


@dataclass(frozen=True, slots=True)
class PreparationConfig:
    tokenizer_path: Path
    output_dir: Path
    splits: dict[str, tuple[Path, ...]]
    task_type: str = "pretraining"
    dataset_id: str = "taichu-dataset"
    version: str = "1"
    text_column: str = "text"
    pad_token: str = "<|pad|>"
    bos_token: str = "<|taichu_start|>"
    eos_token: str = "<|taichu_end|>"
    max_seq_length: int | None = 4096
    add_bos: bool = False
    add_eos: bool = True
    assistant_only_loss: bool = True
    train_on_eos: bool = True
    truncation: str = "left"
    invalid_policy: str = "error"
    shard_size: int = 50_000
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)

    def __post_init__(self) -> None:
        if not self.task_type:
            raise ValueError("task_type must be non-empty")
        if not self.dataset_id or not self.version:
            raise ValueError("dataset_id and version must be non-empty")
        if not self.splits:
            raise ValueError("at least one split is required")
        if any(name not in {"train", "validation", "test"} for name in self.splits):
            raise ValueError("splits must be named train, validation or test")
        if any(not paths for paths in self.splits.values()):
            raise ValueError("every split needs at least one source path")
        if self.max_seq_length is not None and self.max_seq_length < 1:
            raise ValueError("max_seq_length must be positive or None")
        if self.truncation not in {"left", "right"}:
            raise ValueError("truncation must be left or right")
        if self.invalid_policy not in {"error", "skip"}:
            raise ValueError("invalid_policy must be error or skip")
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")


def load_preparation_config(path: str | Path) -> PreparationConfig:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load preparation configs") from exc
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("preparation config root must be a mapping")
    values = dict(raw)
    raw_splits = values.pop("splits", None)
    if not isinstance(raw_splits, dict) or not raw_splits:
        raise ValueError("preparation config needs a non-empty splits mapping")
    splits: dict[str, tuple[Path, ...]] = {}
    for name, paths in raw_splits.items():
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"split {name!r} must be a non-empty list")
        if any(not isinstance(item, (str, Path)) for item in paths):
            raise ValueError(f"split {name!r} paths must be strings")
        splits[str(name)] = tuple(Path(item) for item in paths)
    cleaning = values.pop("cleaning", {})
    if not isinstance(cleaning, dict):
        raise ValueError("cleaning must be a mapping")
    allowed = set(PreparationConfig.__dataclass_fields__) - {
        "splits",
        "cleaning",
    }
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown preparation config keys: {sorted(unknown)}")
    for name in ("tokenizer_path", "output_dir"):
        if name not in values:
            raise ValueError(f"missing preparation config key: {name}")
        values[name] = Path(values[name])
    cleaning_allowed = set(CleaningConfig.__dataclass_fields__)
    unknown_cleaning = set(cleaning) - cleaning_allowed
    if unknown_cleaning:
        raise ValueError(
            f"unknown cleaning config keys: {sorted(unknown_cleaning)}"
        )
    return PreparationConfig(
        splits=splits,
        cleaning=CleaningConfig(**cleaning),
        **values,
    )


__all__ = [
    "CleaningConfig",
    "PreparationConfig",
    "load_preparation_config",
]
