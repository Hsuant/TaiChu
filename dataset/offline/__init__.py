"""Canonical offline task preprocessing and binary dataset preparation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from .prepare import UnifiedDataPreparation


def __getattr__(name: str) -> Any:
    if name == "UnifiedDataPreparation":
        from .prepare import UnifiedDataPreparation

        return UnifiedDataPreparation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
