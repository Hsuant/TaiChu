"""Canonical document-aware pretraining data preparation."""

from .config import (
    DedupConfig,
    PretrainingConfig,
    PretrainingSource,
    SegmentationConfig,
    SplitConfig,
    load_pretraining_config,
)
from .deduplication import DocumentDeduplicator
from .pipeline import PROCESSING_VERSION, PretrainingDataPreparation
from .segmentation import build_segment_samples, sliding_window_bounds

__all__ = [
    "DedupConfig",
    "DocumentDeduplicator",
    "PROCESSING_VERSION",
    "PretrainingConfig",
    "PretrainingDataPreparation",
    "PretrainingSource",
    "SegmentationConfig",
    "SplitConfig",
    "build_segment_samples",
    "load_pretraining_config",
    "sliding_window_bounds",
]
