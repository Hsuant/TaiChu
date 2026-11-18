"""Canonical offline task preprocessing and Parquet dataset preparation."""
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
    TokenSelectionConfig,
    load_pretraining_config,
    select_pretraining_split,
)
from .subset_validation import subset_validation_dataset

if TYPE_CHECKING:
    from .prepare import UnifiedDataPreparation


def __getattr__(name: str) -> Any:
    if name == "UnifiedDataPreparation":
        from .prepare import UnifiedDataPreparation

        return UnifiedDataPreparation
    if name in {"SFTDataPreparation", "SFTBuildConfig", "load_sft_build_config"}:
        from .sft import SFTBuildConfig, SFTDataPreparation, load_sft_build_config

        return {
            "SFTDataPreparation": SFTDataPreparation,
            "SFTBuildConfig": SFTBuildConfig,
            "load_sft_build_config": load_sft_build_config,
        }[name]
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
    "SFTBuildConfig",
    "SFTDataPreparation",
    "SegmentationConfig",
    "SplitConfig",
    "TextTokenizer",
    "TokenSelectionConfig",
    "TrajectoryPreprocessor",
    "UnifiedDataPreparation",
    "load_preparation_config",
    "load_pretraining_config",
    "load_sft_build_config",
    "select_pretraining_split",
    "subset_validation_dataset",
]
