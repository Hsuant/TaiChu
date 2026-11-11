"""Typed configuration for causal language-model training."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from dataset.config import UnifiedDataConfig
from model.config import ModelConfig, coerce_numeric_strings


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.name != "adamw":
            raise ValueError(f"unsupported optimizer {self.name!r}; expected 'adamw'")
        if self.learning_rate <= 0:
            raise ValueError("optimizer.learning_rate must be positive")
        if not 0.0 <= self.weight_decay <= 1.0:
            raise ValueError("optimizer.weight_decay must be in [0, 1]")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("optimizer beta values must be in [0, 1)")
        if self.epsilon <= 0:
            raise ValueError("optimizer.epsilon must be positive")


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    name: str = "warmup_stable_decay"
    warmup_ratio: float = 0.01
    stable_ratio: float = 0.8
    min_lr_ratio: float = 0.0
    decay_type: str = "cosine"

    def __post_init__(self) -> None:
        if self.name != "warmup_stable_decay":
            raise ValueError(
                f"unsupported scheduler {self.name!r}; expected 'warmup_stable_decay'"
            )
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("scheduler.warmup_ratio must be in [0, 1]")
        if not 0.0 <= self.stable_ratio <= 1.0:
            raise ValueError("scheduler.stable_ratio must be in [0, 1]")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("scheduler.min_lr_ratio must be in [0, 1]")
        if self.decay_type not in {"cosine", "linear"}:
            raise ValueError("scheduler.decay_type must be 'cosine' or 'linear'")


@dataclass(frozen=True, slots=True)
class DataLoaderRuntimeConfig:
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool | None = None


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    target_tokens: int = 2_000_000_000
    gradient_accumulation_steps: int = 1
    save_interval: int = 5_000
    log_interval: int = 10
    output_dir: str = "./experiments"
    experiment_name: str = ""
    use_mixed_precision: bool = True
    dtype: str = "bfloat16"
    seed: int = 42
    compile_model: bool = True
    compile_mode: str = "auto"
    compile_dynamic: bool = True
    distributed_timeout_minutes: int = 30
    max_grad_norm: float = 1.0
    dataloader: DataLoaderRuntimeConfig = field(
        default_factory=lambda: DataLoaderRuntimeConfig(drop_last=True)
    )

    def __post_init__(self) -> None:
        if self.target_tokens <= 0:
            raise ValueError("training.target_tokens must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("training.gradient_accumulation_steps must be positive")
        if self.save_interval < 0 or self.log_interval < 0:
            raise ValueError("training save/log intervals cannot be negative")
        if not self.output_dir:
            raise ValueError("training.output_dir must be non-empty")
        if self.dtype not in {"bfloat16", "float16"}:
            raise ValueError("training.dtype must be 'bfloat16' or 'float16'")
        if self.compile_mode not in {
            "auto",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError(f"unsupported compile mode {self.compile_mode!r}")
        if self.distributed_timeout_minutes <= 0:
            raise ValueError("training.distributed_timeout_minutes must be positive")
        if self.max_grad_norm < 0:
            raise ValueError("training.max_grad_norm cannot be negative")


@dataclass(frozen=True, slots=True)
class EvaluatingConfig:
    batch_size: int = 10
    eval_interval: int = 1_000
    log_interval: int = 10
    prompts: tuple[str, ...] | None = None
    num_generate_tokens: int = 50
    empty_prefix_eos_ppl: bool = False
    dataloader: DataLoaderRuntimeConfig = field(
        default_factory=lambda: DataLoaderRuntimeConfig(drop_last=False)
    )

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("evaluating.batch_size must be positive")
        if self.eval_interval < 0 or self.log_interval < 0:
            raise ValueError("evaluating intervals cannot be negative")
        if self.num_generate_tokens < 0:
            raise ValueError("evaluating.num_generate_tokens cannot be negative")
        if self.prompts is not None:
            if not all(isinstance(prompt, str) for prompt in self.prompts):
                raise ValueError("evaluating.prompts must contain strings")
            object.__setattr__(self, "prompts", tuple(self.prompts))


@dataclass(frozen=True, slots=True)
class EarlyStoppingConfig:
    enabled: bool = False
    monitor: str = "val_loss"
    patience: int = 5
    min_delta: float = 1e-4
    mode: str = "min"

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("early_stopping.patience must be positive")
        if self.min_delta < 0:
            raise ValueError("early_stopping.min_delta cannot be negative")
        if self.mode not in {"min", "max"}:
            raise ValueError("early_stopping.mode must be 'min' or 'max'")
        if self.enabled and self.monitor != "val_loss":
            raise ValueError("only val_loss early stopping is currently supported")


@dataclass(frozen=True, slots=True)
class SwanLabConfig:
    use_swanlab: bool = False
    swanlab_project: str = "TaiChu-Project"
    swanlab_experiment_name: str | None = "TaiChu_Experiment_Name"
    swanlab_log_dir: str = "./swanlogs"
    swanlab_mode: str | None = "cloud"
    swanlab_workspace: str | None = None
    swanlab_id: str | None = None
    swanlab_resume: str | None = None
    swanlab_init_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.swanlab_init_kwargs, Mapping):
            raise ValueError("swanlab.swanlab_init_kwargs must be a mapping")
        object.__setattr__(
            self,
            "swanlab_init_kwargs",
            dict(self.swanlab_init_kwargs),
        )


@dataclass(frozen=True, slots=True)
class PretrainConfig:
    tokenizer_path: str
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    data: UnifiedDataConfig
    training: TrainingConfig
    evaluating: EvaluatingConfig
    early_stopping: EarlyStoppingConfig
    swanlab: SwanLabConfig


_TOP_LEVEL_KEYS = {
    "tokenizer",
    "optimizer",
    "scheduler",
    "data",
    "training",
    "evaluating",
    "early_stopping",
    "swanlab",
}


def parse_token_count(value: int | float | str) -> int:
    """Parse token budgets such as ``4B``, ``100M``, or ``1.2T``."""
    if isinstance(value, bool):
        raise ValueError("token count cannot be boolean")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float):
        result = int(value)
    elif isinstance(value, str):
        text = value.strip().upper()
        multipliers = {
            "K": 1_000,
            "M": 1_000_000,
            "B": 1_000_000_000,
            "T": 1_000_000_000_000,
        }
        suffix = text[-1:] if text else ""
        if suffix in multipliers:
            result = int(float(text[:-1]) * multipliers[suffix])
        else:
            result = int(float(text))
    else:
        raise TypeError(f"unsupported token count type: {type(value).__name__}")
    if result <= 0:
        raise ValueError("token count must be positive")
    return result


def _parse_dataclass(cls, raw: Any, section: str):
    if not isinstance(raw, Mapping):
        raise ValueError(f"{section} must be a mapping")
    allowed = {item.name for item in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown {section} keys: {sorted(unknown)}")
    return cls(**coerce_numeric_strings(cls, dict(raw)))


def _parse_dataloader(raw: Any, section: str, default_drop_last: bool):
    values = _parse_dataclass(DataLoaderRuntimeConfig, raw or {}, section)
    if values.drop_last is None:
        values = replace(values, drop_last=default_drop_last)
    return values


def load_pretraining_config(path: str | Path) -> PretrainConfig:
    """Load and validate a strict pretraining YAML configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("pretraining config must be a YAML mapping")
    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")

    tokenizer_raw = raw.get("tokenizer")
    if not isinstance(tokenizer_raw, Mapping):
        raise ValueError("tokenizer must be a mapping")
    unknown_tokenizer = set(tokenizer_raw) - {"path", "library", "backend"}
    if unknown_tokenizer:
        raise ValueError(
            f"unknown tokenizer keys: {sorted(unknown_tokenizer)}"
        )
    tokenizer_path = tokenizer_raw.get("path")
    if not isinstance(tokenizer_path, str) or not tokenizer_path.strip():
        raise ValueError("tokenizer.path must be a non-empty string")
    tokenizer_backend = tokenizer_raw.get("library", "tokenizers")
    if tokenizer_backend != "tokenizers":
        raise ValueError("only the native 'tokenizers' backend is supported")

    optimizer = _parse_dataclass(
        OptimizerConfig,
        raw.get("optimizer", {}),
        "optimizer",
    )
    scheduler = _parse_dataclass(
        SchedulerConfig,
        raw.get("scheduler", {}),
        "scheduler",
    )
    data = UnifiedDataConfig.from_dict(raw.get("data", {}))

    training_raw = dict(raw.get("training", {}))
    if "target_tokens" in training_raw:
        training_raw["target_tokens"] = parse_token_count(
            training_raw["target_tokens"]
        )
    training_dataloader = _parse_dataloader(
        training_raw.pop("dataloader", {}),
        "training.dataloader",
        True,
    )
    training = _parse_dataclass(
        TrainingConfig,
        training_raw,
        "training",
    )
    training = replace(training, dataloader=training_dataloader)

    evaluating_raw = dict(raw.get("evaluating", {}))
    evaluating_dataloader = _parse_dataloader(
        evaluating_raw.pop("dataloader", {}),
        "evaluating.dataloader",
        False,
    )
    evaluating = _parse_dataclass(
        EvaluatingConfig,
        evaluating_raw,
        "evaluating",
    )
    evaluating = replace(evaluating, dataloader=evaluating_dataloader)

    early_stopping = _parse_dataclass(
        EarlyStoppingConfig,
        raw.get("early_stopping", {}),
        "early_stopping",
    )
    swanlab = _parse_dataclass(
        SwanLabConfig,
        raw.get("swanlab", {}),
        "swanlab",
    )
    return PretrainConfig(
        tokenizer_path=tokenizer_path,
        optimizer=optimizer,
        scheduler=scheduler,
        data=data,
        training=training,
        evaluating=evaluating,
        early_stopping=early_stopping,
        swanlab=swanlab,
    )


def validate_pretraining_config(
    config: PretrainConfig,
    model_config: ModelConfig,
) -> None:
    """Validate relationships that span model, data, and training configs."""
    if config.data.task_type != "pretraining":
        raise ValueError(
            f"pretraining requires data.task_type='pretraining', "
            f"got {config.data.task_type!r}"
        )
    if config.data.packing.max_seq_length > model_config.max_position_embeddings:
        raise ValueError(
            "data.packing.max_seq_length cannot exceed "
            "model.max_position_embeddings"
        )
    if (
        config.evaluating.eval_interval > 0
        and config.data.storage.validation_path is None
    ):
        raise ValueError(
            "validation_path is required when evaluating.eval_interval > 0"
        )


__all__ = [
    "DataLoaderRuntimeConfig",
    "EarlyStoppingConfig",
    "EvaluatingConfig",
    "OptimizerConfig",
    "PretrainConfig",
    "SchedulerConfig",
    "SwanLabConfig",
    "TrainingConfig",
    "load_pretraining_config",
    "parse_token_count",
    "validate_pretraining_config",
]
