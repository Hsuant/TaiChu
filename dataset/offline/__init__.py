"""Canonical offline task preprocessing and binary dataset preparation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .adapters import (
    BasePreprocessingAdapter,
    PreferencePreprocessor,
    RewardPreprocessor,
    RoleTagRenderer,
    SFTPreprocessor,
    TextTokenizer,
    TrajectoryPreprocessor,
)
from .config import PreparationConfig, load_preparation_config
from .pretraining import (
    DedupConfig,
    PretrainingConfig,
    PretrainingDataPreparation,
    PretrainingSource,
    SegmentationConfig,
    SplitConfig,
    load_pretraining_config,
)
from .subset_validation import subset_validation_dataset

if TYPE_CHECKING:
    from .prepare import UnifiedDataPreparation


def __getattr__(name: str) -> Any:
    if name == "UnifiedDataPreparation":
        from .prepare import UnifiedDataPreparation

        return UnifiedDataPreparation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BasePreprocessingAdapter",
    "DedupConfig",
    "PreferencePreprocessor",
    "PreparationConfig",
    "PretrainingConfig",
    "PretrainingDataPreparation",
    "PretrainingSource",
    "RewardPreprocessor",
    "RoleTagRenderer",
    "SFTPreprocessor",
    "SegmentationConfig",
    "SplitConfig",
    "TextTokenizer",
    "TrajectoryPreprocessor",
    "UnifiedDataPreparation",
    "load_preparation_config",
    "load_pretraining_config",
    "subset_validation_dataset",
]
