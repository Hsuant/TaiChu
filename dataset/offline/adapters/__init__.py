"""Task-specific preprocessing adapters over shared tokenizer contracts."""
from .base import (
    BasePreprocessingAdapter,
    ConversationRenderer,
    RoleTagRenderer,
    TextTokenizer,
)
from .preference import PreferencePreprocessor
from .pretraining import PretrainingPreprocessor
from .reward import RewardPreprocessor
from .sft import SFTPreprocessor
from .trajectory import TrajectoryPreprocessor

__all__ = [
    "BasePreprocessingAdapter",
    "ConversationRenderer",
    "PreferencePreprocessor",
    "PretrainingPreprocessor",
    "RewardPreprocessor",
    "RoleTagRenderer",
    "SFTPreprocessor",
    "TextTokenizer",
    "TrajectoryPreprocessor",
]
