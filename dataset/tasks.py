"""Extensible task registry for runtime builders and model input adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .adapters import (
    ModelInputAdapter,
    PreferenceInputAdapter,
    PretrainingInputAdapter,
    RewardInputAdapter,
    SFTInputAdapter,
    TrajectoryInputAdapter,
)
from .batching import (
    BatchBuilder,
    PreferenceBatchBuilder,
    PretrainingBatchBuilder,
    RewardBatchBuilder,
    SFTBatchBuilder,
    TrajectoryBatchBuilder,
)
from .config import UnifiedDataConfig
from .exceptions import TaiChuConfigurationError
from .packing import PretrainingPacker, SampleIsolatedPacker


BatchBuilderFactory = Callable[[UnifiedDataConfig, int], BatchBuilder]
InputAdapterFactory = Callable[[], ModelInputAdapter]
PackerFactory = Callable[[UnifiedDataConfig, int], object | None]


@dataclass(frozen=True, slots=True)
class TaskSpec:
    sample_type: str
    batch_builder_factory: BatchBuilderFactory
    input_adapter_factory: InputAdapterFactory
    supports_packing: bool
    packer_factory: PackerFactory
    packing_strategies: frozenset[str]
    batching_strategies: frozenset[str]


TASK_REGISTRY: dict[str, TaskSpec] = {}


def register_task(name: str, spec: TaskSpec) -> None:
    if not name or not spec.sample_type:
        raise ValueError("task name and sample type must be non-empty")
    existing = TASK_REGISTRY.get(name)
    if existing is not None and existing != spec:
        raise ValueError(f"task {name!r} is already registered")
    TASK_REGISTRY[name] = spec


def get_task(name: str) -> TaskSpec:
    try:
        return TASK_REGISTRY[name]
    except KeyError as exc:
        raise TaiChuConfigurationError(
            f"unknown data task_type {name!r}; registered tasks: {sorted(TASK_REGISTRY)}"
        ) from exc


def _pretraining_builder(config: UnifiedDataConfig, pad_id: int):
    return PretrainingBatchBuilder(
        pad_id,
        pad_to_multiple_of=config.batching.pad_to_multiple_of,
    )


def _sft_builder(config: UnifiedDataConfig, pad_id: int):
    return SFTBatchBuilder(
        pad_id,
        padding_free=config.batching.padding_free,
        pad_to_multiple_of=config.batching.pad_to_multiple_of,
    )


def _preference_builder(config: UnifiedDataConfig, pad_id: int):
    if config.packing.enabled:
        raise TaiChuConfigurationError(
            "preference training requires packing.enabled=false to preserve pairs"
        )
    return PreferenceBatchBuilder(
        pad_id,
        pad_to_multiple_of=config.batching.pad_to_multiple_of,
        max_length=config.packing.max_seq_length,
    )


def _reward_builder(config: UnifiedDataConfig, pad_id: int):
    return RewardBatchBuilder(
        pad_id,
        pad_to_multiple_of=config.batching.pad_to_multiple_of,
    )


def _trajectory_builder(config: UnifiedDataConfig, pad_id: int):
    return TrajectoryBatchBuilder(
        pad_id,
        pad_to_multiple_of=config.batching.pad_to_multiple_of,
    )


def _pretraining_packer(config: UnifiedDataConfig, pad_id: int):
    return PretrainingPacker(
        config.packing.max_seq_length,
        pad_id=pad_id,
        pad_to_max=True,
        drop_last_remainder=config.packing.drop_last_remainder,
    )


def _sft_packer(config: UnifiedDataConfig, pad_id: int):
    return SampleIsolatedPacker(
        config.packing.max_seq_length,
        pad_id=pad_id,
        pad_to_max=not config.batching.padding_free,
    )


def _no_packer(config: UnifiedDataConfig, pad_id: int):
    return None


_STREAM_PACKING = frozenset({"stream"})
_SFT_PACKING = frozenset({"sample_isolated"})
_NO_PACKING = frozenset()
_PRETRAIN_BATCHING = frozenset({"fixed_length"})
_SFT_BATCHING = frozenset({"fixed_length", "padding_free"})
_PREFERENCE_BATCHING = frozenset({"preference_pair"})
_REWARD_BATCHING = frozenset({"reward_pair"})
_TRAJECTORY_BATCHING = frozenset({"trajectory"})


for _name, _sample_type, _builder, _adapter, _supports_packing, _packer, _packing_strategies, _batching_strategies in (
    ("pretraining", "pretraining", _pretraining_builder, PretrainingInputAdapter, True, _pretraining_packer, _STREAM_PACKING, _PRETRAIN_BATCHING),
    ("cpt", "pretraining", _pretraining_builder, PretrainingInputAdapter, True, _pretraining_packer, _STREAM_PACKING, _PRETRAIN_BATCHING),
    ("sft", "sft", _sft_builder, SFTInputAdapter, True, _sft_packer, _SFT_PACKING, _SFT_BATCHING),
    ("instruction_tuning", "sft", _sft_builder, SFTInputAdapter, True, _sft_packer, _SFT_PACKING, _SFT_BATCHING),
    ("preference", "preference", _preference_builder, PreferenceInputAdapter, False, _no_packer, _NO_PACKING, _PREFERENCE_BATCHING),
    ("dpo", "preference", _preference_builder, PreferenceInputAdapter, False, _no_packer, _NO_PACKING, _PREFERENCE_BATCHING),
    ("ipo", "preference", _preference_builder, PreferenceInputAdapter, False, _no_packer, _NO_PACKING, _PREFERENCE_BATCHING),
    ("kto", "preference", _preference_builder, PreferenceInputAdapter, False, _no_packer, _NO_PACKING, _PREFERENCE_BATCHING),
    ("reward", "reward", _reward_builder, RewardInputAdapter, False, _no_packer, _NO_PACKING, _REWARD_BATCHING),
    ("rlhf", "trajectory", _trajectory_builder, TrajectoryInputAdapter, False, _no_packer, _NO_PACKING, _TRAJECTORY_BATCHING),
    ("ppo", "trajectory", _trajectory_builder, TrajectoryInputAdapter, False, _no_packer, _NO_PACKING, _TRAJECTORY_BATCHING),
    ("grpo", "trajectory", _trajectory_builder, TrajectoryInputAdapter, False, _no_packer, _NO_PACKING, _TRAJECTORY_BATCHING),
):
    register_task(
        _name,
        TaskSpec(
            sample_type=_sample_type,
            batch_builder_factory=_builder,
            input_adapter_factory=_adapter,
            supports_packing=_supports_packing,
            packer_factory=_packer,
            packing_strategies=_packing_strategies,
            batching_strategies=_batching_strategies,
        ),
    )


__all__ = [
    "TASK_REGISTRY",
    "TaskSpec",
    "get_task",
    "register_task",
]
