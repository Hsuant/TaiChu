"""Optimizer, learning-rate schedule, and step-budget construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from training.config import OptimizerConfig, SchedulerConfig


@dataclass(frozen=True, slots=True)
class StepBudget:
    target_tokens: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    sequence_length: int

    def __post_init__(self) -> None:
        for name in (
            "target_tokens",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "world_size",
            "sequence_length",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def global_batch_size(self) -> int:
        return (
            self.micro_batch_size
            * self.gradient_accumulation_steps
            * self.world_size
        )

    @property
    def tokens_per_step(self) -> int:
        return self.global_batch_size * self.sequence_length

    @property
    def max_steps(self) -> int:
        return math.ceil(self.target_tokens / self.tokens_per_step)


@dataclass(frozen=True, slots=True)
class ScheduleSteps:
    warmup: int
    stable: int
    decay: int
    max_steps: int


def build_optimizer(
    model: torch.nn.Module,
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    """Build AdamW with decay applied only to non-bias, non-normalization weights."""
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "bias" in name or "norm" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    kwargs = {
        "lr": config.learning_rate,
        "betas": (config.beta1, config.beta2),
        "eps": config.epsilon,
    }
    if next(model.parameters()).is_cuda:
        kwargs["fused"] = True
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        **kwargs,
    )


def wsd_lr_lambda(
    steps: ScheduleSteps,
    *,
    min_lr_ratio: float,
    decay_type: str,
) -> Callable[[int], float]:
    """Create a warmup-stable-decay learning-rate multiplier."""
    if decay_type == "cosine":
        def decay(progress: float) -> float:
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    elif decay_type == "linear":
        def decay(progress: float) -> float:
            return 1.0 - progress
    else:
        raise ValueError(f"unsupported decay type {decay_type!r}")

    stable_end = steps.warmup + steps.stable

    def multiplier(current_step: int) -> float:
        step = float(current_step)
        if step < steps.warmup:
            return step / max(1.0, float(steps.warmup))
        if step < stable_end:
            return 1.0
        if step >= steps.max_steps:
            return min_lr_ratio
        progress = min(
            max((step - stable_end) / max(1.0, float(steps.decay)), 0.0),
            1.0,
        )
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay(progress)

    return multiplier


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    max_steps: int,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, ScheduleSteps]:
    """Create the scheduler and the exact phase lengths used to build it."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    warmup = max(1, int(max_steps * config.warmup_ratio))
    stable = int(max_steps * config.stable_ratio)
    if warmup + stable >= max_steps:
        stable = max(0, max_steps - warmup - 1)
    decay = max(0, max_steps - warmup - stable)
    steps = ScheduleSteps(
        warmup=warmup,
        stable=stable,
        decay=decay,
        max_steps=max_steps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        wsd_lr_lambda(
            steps,
            min_lr_ratio=config.min_lr_ratio,
            decay_type=config.decay_type,
        ),
    )
    return scheduler, steps


__all__ = [
    "ScheduleSteps",
    "StepBudget",
    "build_optimizer",
    "build_scheduler",
    "wsd_lr_lambda",
]
