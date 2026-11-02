"""TaiChu unified LLM training-data infrastructure."""
from typing import Any

from .config import (
    BatchingConfig,
    SamplingConfig,
    StorageConfig,
    UnifiedDataConfig,
    UnifiedPackingConfig,
)
from .schema import (
    CommonSample,
    ConversationMessage,
    DocumentSample,
    MultimodalSample,
    PreferenceSample,
    PretrainingSample,
    RewardSample,
    SFTSequenceSample,
    TextSequenceSample,
    TrajectorySample,
)


def __getattr__(name: str) -> Any:
    if name == "UnifiedDataLoaderFactory":
        from .runtime import UnifiedDataLoaderFactory

        return UnifiedDataLoaderFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BatchingConfig",
    "CommonSample",
    "ConversationMessage",
    "DocumentSample",
    "MultimodalSample",
    "PreferenceSample",
    "PretrainingSample",
    "RewardSample",
    "SFTSequenceSample",
    "SamplingConfig",
    "StorageConfig",
    "TextSequenceSample",
    "TrajectorySample",
    "UnifiedDataConfig",
    "UnifiedDataLoaderFactory",
    "UnifiedPackingConfig",
]
