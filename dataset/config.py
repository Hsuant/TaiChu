"""Strict unified data-infrastructure configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import TaiChuConfigurationError


@dataclass(frozen=True, slots=True)
class StorageConfig:
    format: str = "binary_mmap"
    train_path: str | Path = ""
    validation_path: str | Path | None = None
    verify_checksums: bool = False
    num_workers: int = 0
    prefetch_factor: int = 2

    def __post_init__(self) -> None:
        if self.format != "binary_mmap":
            raise TaiChuConfigurationError(
                "storage.format currently supports only 'binary_mmap'"
            )
        if not self.train_path:
            raise TaiChuConfigurationError("storage.train_path is required")
        if self.num_workers < 0:
            raise TaiChuConfigurationError("storage.num_workers cannot be negative")
        if self.prefetch_factor <= 0:
            raise TaiChuConfigurationError("storage.prefetch_factor must be positive")

    def path_for(self, split: str) -> Path:
        if split in {"train", "training"}:
            return Path(self.train_path)
        if split in {"validation", "valid", "eval", "test"}:
            if self.validation_path is None:
                raise TaiChuConfigurationError(
                    f"no storage path configured for split {split!r}"
                )
            return Path(self.validation_path)
        raise TaiChuConfigurationError(f"unsupported split {split!r}")


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    strategy: str = "distributed"
    shuffle: bool = True
    seed: int = 42
    drop_last: bool = False
    pad_to_world_size: bool = True

    def __post_init__(self) -> None:
        if self.strategy not in {"random", "distributed", "pair_aware"}:
            raise TaiChuConfigurationError(
                "sampling.strategy must be random, distributed or pair_aware"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TaiChuConfigurationError("sampling.seed must be an integer")


@dataclass(frozen=True, slots=True)
class UnifiedPackingConfig:
    enabled: bool = True
    strategy: str = "stream"
    max_seq_length: int = 4096
    pad_to_multiple_of: int = 1
    drop_last_remainder: bool = True

    def __post_init__(self) -> None:
        if self.strategy not in {"stream", "sample_isolated", "none"}:
            raise TaiChuConfigurationError(
                "packing.strategy must be stream, sample_isolated or none"
            )
        if self.max_seq_length < 2:
            raise TaiChuConfigurationError("packing.max_seq_length must be >= 2")
        if self.pad_to_multiple_of <= 0:
            raise TaiChuConfigurationError(
                "packing.pad_to_multiple_of must be positive"
            )


@dataclass(frozen=True, slots=True)
class BatchingConfig:
    strategy: str = "fixed_length"
    micro_batch_size: int = 1
    pad_to_multiple_of: int = 1
    padding_free: bool = False

    def __post_init__(self) -> None:
        if self.strategy not in {
            "fixed_length",
            "padding_free",
            "preference_pair",
            "reward_pair",
            "trajectory",
        }:
            raise TaiChuConfigurationError(
                "batching.strategy must be fixed_length, padding_free, "
                "preference_pair, reward_pair or trajectory"
            )
        if self.micro_batch_size <= 0:
            raise TaiChuConfigurationError("batching.micro_batch_size must be positive")
        if self.pad_to_multiple_of <= 0:
            raise TaiChuConfigurationError(
                "batching.pad_to_multiple_of must be positive"
            )
        if self.strategy == "padding_free" and not self.padding_free:
            object.__setattr__(self, "padding_free", True)


@dataclass(frozen=True, slots=True)
class UnifiedDataConfig:
    task_type: str
    dataset_id: str
    storage: StorageConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    packing: UnifiedPackingConfig = field(default_factory=UnifiedPackingConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    pad_token: str = "<|pad|>"
    bos_token: str = "<|taichu_start|>"
    eos_token: str = "<|taichu_end|>"
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.task_type, str) or not self.task_type:
            raise TaiChuConfigurationError("task_type must be a non-empty string")
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise TaiChuConfigurationError("dataset_id must be a non-empty string")
        if self.schema_version != 1:
            raise TaiChuConfigurationError("unsupported config schema_version")
        if not isinstance(self.metadata, dict):
            raise TaiChuConfigurationError("metadata must be a mapping")
        if any(
            not isinstance(value, str) or not value
            for value in (self.pad_token, self.bos_token, self.eos_token)
        ):
            raise TaiChuConfigurationError(
                "pad_token, bos_token and eos_token must be non-empty strings"
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "UnifiedDataConfig":
        if not isinstance(raw, dict):
            raise TaiChuConfigurationError("unified data config must be a mapping")
        data = dict(raw.get("data", raw))
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise TaiChuConfigurationError(
                f"unknown unified data config keys: {sorted(unknown)}"
            )
        for name, config_type in (
            ("storage", StorageConfig),
            ("sampling", SamplingConfig),
            ("packing", UnifiedPackingConfig),
            ("batching", BatchingConfig),
        ):
            value = data.get(name, {})
            if isinstance(value, config_type):
                continue
            if not isinstance(value, dict):
                raise TaiChuConfigurationError(f"{name} must be a mapping")
            known = set(config_type.__dataclass_fields__)
            unknown = set(value) - known
            if unknown:
                raise TaiChuConfigurationError(
                    f"unknown {name} config keys: {sorted(unknown)}"
                )
            data[name] = config_type(**value)
        for required in ("task_type", "dataset_id", "storage"):
            if required not in data:
                raise TaiChuConfigurationError(
                    f"missing unified data config key: {required}"
                )
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "UnifiedDataConfig":
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML configs") from exc
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["storage"]["train_path"] = str(self.storage.train_path)
        if self.storage.validation_path is not None:
            result["storage"]["validation_path"] = str(self.storage.validation_path)
        return result
__all__ = [
    "BatchingConfig",
    "SamplingConfig",
    "StorageConfig",
    "UnifiedDataConfig",
    "UnifiedPackingConfig",
]
