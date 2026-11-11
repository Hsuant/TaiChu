"""Configuration for the document-aware pretraining pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import CleaningConfig


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    max_sequence_length: int = 2048
    max_segment_content_tokens: int = 2046
    overlap_tokens: int = 128
    min_fragment_tokens: int = 32

    def __post_init__(self) -> None:
        if self.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be >= 2")
        if self.max_segment_content_tokens < 1:
            raise ValueError("max_segment_content_tokens must be positive")
        if not 0 <= self.overlap_tokens < self.max_segment_content_tokens:
            raise ValueError("overlap_tokens must be in [0, content tokens)")
        if self.min_fragment_tokens < 0:
            raise ValueError("min_fragment_tokens cannot be negative")
        if self.max_segment_content_tokens + 1 > self.max_sequence_length:
            raise ValueError(
                "max_segment_content_tokens plus BOS or EOS exceeds "
                "max_sequence_length"
            )
        if self.stride_tokens <= 0:
            raise ValueError("derived stride_tokens must be positive")

    @property
    def stride_tokens(self) -> int:
        return self.max_segment_content_tokens - self.overlap_tokens


@dataclass(frozen=True, slots=True)
class DedupConfig:
    enabled: bool = True
    ngram_size: int = 13
    similarity_threshold: float = 0.8
    num_hashes: int = 128
    bands: int = 16
    exact_jaccard: bool = True

    def __post_init__(self) -> None:
        if self.ngram_size < 1:
            raise ValueError("dedup.ngram_size must be positive")
        if not 0 < self.similarity_threshold <= 1:
            raise ValueError("dedup.similarity_threshold must be in (0, 1]")
        if self.num_hashes < 1 or self.bands < 1:
            raise ValueError("dedup.num_hashes and dedup.bands must be positive")
        if self.num_hashes % self.bands:
            raise ValueError("dedup.num_hashes must be divisible by dedup.bands")
        if not self.exact_jaccard:
            raise ValueError(
                "dedup.exact_jaccard=false is unsupported because near-duplicate "
                "acceptance must be verified, not inferred from MinHash alone"
            )

    @property
    def rows(self) -> int:
        return self.num_hashes // self.bands


@dataclass(frozen=True, slots=True)
class SplitConfig:
    train_ratio: float = 0.98
    validation_ratio: float = 0.01
    diagnostic_ratio: float = 0.01
    seed: int = 42

    def __post_init__(self) -> None:
        ratios = (
            self.train_ratio,
            self.validation_ratio,
            self.diagnostic_ratio,
        )
        if any(value < 0 for value in ratios):
            raise ValueError("split ratios cannot be negative")
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("train/validation/diagnostic ratios must sum to 1")


@dataclass(frozen=True, slots=True)
class PretrainingSource:
    name: str
    paths: tuple[Path, ...]
    language: str = "unknown"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("source name must be non-empty")
        if not self.paths:
            raise ValueError(f"source {self.name!r} needs at least one path")


@dataclass(frozen=True, slots=True)
class PretrainingConfig:
    tokenizer_path: Path
    output_dir: Path
    sources: tuple[PretrainingSource, ...]
    task_type: str = "pretraining"
    dataset_id: str = "taichu-pretraining"
    version: str = "2"
    text_column: str = "text"
    pad_token: str = "<|pad|>"
    bos_token: str = "<|taichu_start|>"
    eos_token: str = "<|taichu_end|>"
    invalid_policy: str = "error"
    shard_size: int = 50_000
    processing_workers: int = 8
    output_workers: int | None = None
    fingerprint_batch_size: int = 64
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    cleaning: CleaningConfig = field(
        default_factory=lambda: CleaningConfig(
            min_chars=0,
            max_chars=10_000_000,
        )
    )

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.version:
            raise ValueError("dataset_id and version must be non-empty")
        if self.task_type not in {"pretraining", "cpt"}:
            raise ValueError("task_type must be pretraining or cpt")
        if not self.sources:
            raise ValueError("at least one pretraining source is required")
        names = [source.name for source in self.sources]
        if len(names) != len(set(names)):
            raise ValueError("pretraining source names must be unique")
        if self.invalid_policy not in {"error", "skip"}:
            raise ValueError("invalid_policy must be error or skip")
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if self.processing_workers <= 0:
            raise ValueError("processing_workers must be positive")
        if self.output_workers is not None and self.output_workers <= 0:
            raise ValueError("output_workers must be positive when configured")
        if self.fingerprint_batch_size <= 0:
            raise ValueError("fingerprint_batch_size must be positive")


def load_pretraining_config(path: str | Path) -> PretrainingConfig:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load pretraining configs") from exc
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("pretraining preparation config root must be a mapping")

    values = dict(raw)
    raw_sources = values.pop("sources", None)
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("pretraining preparation config needs a non-empty sources list")
    sources = []
    for item in raw_sources:
        if not isinstance(item, dict):
            raise ValueError("each source must be a mapping")
        unknown = set(item) - {"name", "paths", "language"}
        if unknown:
            raise ValueError(f"unknown source keys: {sorted(unknown)}")
        paths = item.get("paths")
        if (
            not isinstance(paths, list)
            or not paths
            or any(not isinstance(value, (str, Path)) for value in paths)
        ):
            raise ValueError("source paths must be a non-empty list")
        sources.append(
            PretrainingSource(
                name=str(item["name"]),
                language=str(item.get("language", "unknown")),
                paths=tuple(Path(value) for value in paths),
            )
        )

    for nested, cls in (
        ("segmentation", SegmentationConfig),
        ("dedup", DedupConfig),
        ("splits", SplitConfig),
        ("cleaning", CleaningConfig),
    ):
        value = values.get(nested, {})
        if isinstance(value, cls):
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{nested} must be a mapping")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown {nested} keys: {sorted(unknown)}")
        values[nested] = cls(**value)

    allowed = set(PretrainingConfig.__dataclass_fields__) - {"sources"}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown pretraining preparation keys: {sorted(unknown)}")
    for name in ("tokenizer_path", "output_dir"):
        if name not in values:
            raise ValueError(f"missing pretraining preparation key: {name}")
        values[name] = Path(values[name])
    return PretrainingConfig(sources=tuple(sources), **values)


__all__ = [
    "DedupConfig",
    "PretrainingSource",
    "PretrainingConfig",
    "SegmentationConfig",
    "SplitConfig",
    "load_pretraining_config",
]
