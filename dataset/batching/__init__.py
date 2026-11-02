"""Task-selectable batch construction and alignment APIs."""
from .alignment import aligned_segment_ids, segment_position_ids
from .batch_builder import (
    PreferenceBatchBuilder,
    PretrainingBatchBuilder,
    RewardBatchBuilder,
    SFTBatchBuilder,
    TrajectoryBatchBuilder,
)
from .base import BatchBuilder
from .padding import padded_float_tensor, padded_tensor

__all__ = [
    "BatchBuilder",
    "PreferenceBatchBuilder",
    "PretrainingBatchBuilder",
    "RewardBatchBuilder",
    "SFTBatchBuilder",
    "TrajectoryBatchBuilder",
    "aligned_segment_ids",
    "padded_float_tensor",
    "padded_tensor",
    "segment_position_ids",
]
