"""Task-specific preprocessing adapters over shared tokenizer contracts."""
from .base import (
    BasePreprocessingAdapter,
    ConversationRenderer,
    RoleTagRenderer,
    TextTokenizer,
)
from .preference import PreferencePreprocessor
from .reward import RewardPreprocessor
from .sft import SFTPreprocessor
from .trajectory import TrajectoryPreprocessor

__all__ = [
    "BasePreprocessingAdapter",
    "ConversationRenderer",
    "PreferencePreprocessor",
    "RewardPreprocessor",
    "RoleTagRenderer",
    "SFTPreprocessor",
    "TextTokenizer",
    "TrajectoryPreprocessor",
]
