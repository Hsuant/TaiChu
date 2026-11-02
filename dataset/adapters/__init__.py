"""Task-specific model input adapters."""
from .base import ModelInputAdapter
from .preference import PreferenceInputAdapter
from .pretraining import PretrainingInputAdapter
from .reward import RewardInputAdapter
from .sft import SFTInputAdapter
from .trajectory import TrajectoryInputAdapter

__all__ = [
    "ModelInputAdapter",
    "PreferenceInputAdapter",
    "PretrainingInputAdapter",
    "RewardInputAdapter",
    "SFTInputAdapter",
    "TrajectoryInputAdapter",
]
