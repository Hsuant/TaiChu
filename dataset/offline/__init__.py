"""Canonical offline task preprocessing and binary dataset preparation."""
from .adapters import (
    BasePreprocessingAdapter,
    PreferencePreprocessor,
    PretrainingPreprocessor,
    RewardPreprocessor,
    RoleTagRenderer,
    SFTPreprocessor,
    TextTokenizer,
    TrajectoryPreprocessor,
)
from .config import PreparationConfig, load_preparation_config
from .prepare import UnifiedDataPreparation

__all__ = [
    "BasePreprocessingAdapter",
    "PreferencePreprocessor",
    "PreparationConfig",
    "PretrainingPreprocessor",
    "RewardPreprocessor",
    "RoleTagRenderer",
    "SFTPreprocessor",
    "TextTokenizer",
    "TrajectoryPreprocessor",
    "UnifiedDataPreparation",
    "load_preparation_config",
]
