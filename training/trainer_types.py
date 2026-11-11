"""Immutable inputs shared by the task-specific trainer and its assembly layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.amp import GradScaler

from dataset.runtime import DataState
from training.checkpoint import CheckpointManager
from training.diagnostics import TrainingDiagnostics
from training.early_stopping import EarlyStopping
from training.evaluator import CausalLMEvaluator
from training.logging import LogManager, SwanLabLogger
from training.model_artifacts import BestModelStore


@dataclass(slots=True)
class TrainerComponents:
    model: nn.Module
    train_loader: Any
    val_loader: Any | None
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    scaler: GradScaler
    checkpoint_manager: CheckpointManager
    best_model_store: BestModelStore
    final_model_path: Path
    evaluator: CausalLMEvaluator
    diagnostics: TrainingDiagnostics
    log_manager: LogManager
    early_stopping: EarlyStopping | None = None
    swanlab_logger: SwanLabLogger | None = None


@dataclass(frozen=True, slots=True)
class TrainerSettings:
    max_steps: int
    gradient_accumulation_steps: int
    save_interval: int
    eval_interval: int
    train_log_interval: int
    eval_log_interval: int
    max_grad_norm: float
    device: torch.device
    world_size: int
    rank: int
    use_amp: bool
    dtype: torch.dtype
    compiled_model: bool
    generation_prompts: Sequence[str] | None
    num_generate_tokens: int
    compute_empty_prefix: bool
    init_step: int = 0
    best_val_loss: float = float("inf")
    last_val_loss: float | None = None
    last_eval_step: int | None = None
    data_state: DataState | None = None

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm cannot be negative")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid distributed rank configuration")
        if self.init_step < 0:
            raise ValueError("init_step cannot be negative")


__all__ = ["TrainerComponents", "TrainerSettings"]
