"""Canonical document-aware pretraining data preparation."""

from .config import (
    DedupConfig,
    PretrainingConfig,
    PretrainingSource,
    SegmentationConfig,
    SplitConfig,
    TokenSelectionConfig,
    load_pretraining_config,
)
from .deduplication import DocumentDeduplicator
from .diagnose import compare_dataset_overlap, diagnose_pretrain_dataset
from .output import select_pretraining_split
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
    "TokenSelectionConfig",
    "build_segment_samples",
    "compare_dataset_overlap",
    "diagnose_pretrain_dataset",
    "load_pretraining_config",
    "select_pretraining_split",
    "sliding_window_bounds",
]
