"""Shared causal-LM training runtime and task interface."""

from __future__ import annotations

import math
import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler
from torch.amp.autocast_mode import autocast

from dataset.runtime import DataState, skip_batches
from model.config import ModelConfig
from training.batch import move_batch_to_device
from training.builders import build_data_loaders, build_model, load_tokenizer
from training.checkpoint import CheckpointManager, unwrap_model
from training.config import TrainingRunConfig
from training.diagnostics import TrainingDiagnostics
from training.early_stopping import EarlyStopping
from training.environment import (
    barrier,
    configure_torch_runtime,
    initialize_runtime,
)
from training.evaluator import CausalLMEvaluator
from training.logging import LogManager, SwanLabLogger
from training.model_artifacts import BestModelStore, save_model_artifact
from training.model_info import model_size_mb, parameter_count, peak_flops
from training.model_setup import (
    compile_preflight,
    compile_training_model,
    warmup_compiled_training,
    wrap_distributed,
)
from training.optimization import StepBudget, build_optimizer, build_scheduler
from training.paths import (
    ExperimentPaths,
    resolve_experiment_root,
    resolve_model_scale,
)
from training.trainer_types import (
    TrainerComponents,
    TrainerSettings,
    TrainingStepStatistics,
)
from training.validation import (
    compute_bits_per_byte,
    compute_perplexity,
    compute_token_statistics_from_losses,
)


class CausalLMTask(ABC):
    """Task-specific policy used by the shared causal-LM training runtime."""

    stage: str

    def validate_args(self, args: Any) -> None:
        """Validate task-specific CLI arguments before allocating resources."""

    @abstractmethod
    def load_configs(
        self,
        args: Any,
    ) -> tuple[ModelConfig, TrainingRunConfig]:
        """Load and validate the model and training configuration."""

    @abstractmethod
    def build_step_budget(self, config, loaders, runtime) -> StepBudget:
        """Derive the optimizer-step budget for this training objective."""

    def prepare_model(self, args: Any, model: nn.Module) -> str | None:
        """Initialize model weights and return a provenance label."""
        return None

    def prompt_renderer_factory(
        self,
    ) -> Callable[[Any], Callable[[str], str] | None] | None:
        """Return an optional Chat prompt renderer factory."""
        return None


def _resolve_paths(
    args,
    model_config,
    config,
    runtime,
    *,
    stage: str,
) -> ExperimentPaths:
    resume_path = None
    if args.resume is not None:
        resume_path = str(Path(args.resume).expanduser().resolve())
        if not Path(resume_path).is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {resume_path}")
    root = None
    if runtime.is_main:
        root = str(
            resolve_experiment_root(
                experiments_dir=config.training.output_dir,
                stage=stage,
                experiment_name=config.training.experiment_name,
                resume_path=resume_path,
            )
        )
    if runtime.distributed:
        roots = [root]
        torch.distributed.broadcast_object_list(roots, src=0)
        root = roots[0]
    if root is None:
        raise RuntimeError("experiment directory was not resolved")
    return ExperimentPaths.create(
        root,
        model_scale=resolve_model_scale(model_config.model_name),
        stage=stage,
        experiment_name=Path(root).name,
    )


def _create_swanlab_logger(
    *,
    config,
    args,
    runtime,
    checkpoint: dict | None,
    model_config,
    budget: StepBudget,
    schedule,
    use_amp: bool,
    total_params: int,
    trainable_params: int,
) -> SwanLabLogger | None:
    if not config.swanlab.use_swanlab or not runtime.is_main:
        return None
    checkpoint = checkpoint or {}
    init_kwargs = dict(config.swanlab.swanlab_init_kwargs)
    run_id = (
        config.swanlab.swanlab_id
        or checkpoint.get("swanlab_run_id")
        or init_kwargs.get("id")
    )
    workspace = (
        config.swanlab.swanlab_workspace
        or checkpoint.get("swanlab_workspace")
        or init_kwargs.get("workspace")
    )
    resume = config.swanlab.swanlab_resume or init_kwargs.get("resume")
    if args.resume and run_id and not resume:
        resume = "must"
    if config.swanlab.swanlab_mode:
        os.environ["SWANLAB_MODE"] = config.swanlab.swanlab_mode
    return SwanLabLogger(
        project=config.swanlab.swanlab_project,
        experiment_name=config.swanlab.swanlab_experiment_name,
        config={
            "model_name": model_config.model_name,
            "target_tokens": budget.target_tokens,
            "epochs": config.training.epochs,
            "micro_batch_size": budget.micro_batch_size,
            "gradient_accumulation_steps": budget.gradient_accumulation_steps,
            "world_size": budget.world_size,
            "global_batch_size": budget.global_batch_size,
            "tokens_per_step": budget.tokens_per_step,
            "max_steps": budget.max_steps,
            "learning_rate": config.optimizer.learning_rate,
            "weight_decay": config.optimizer.weight_decay,
            "warmup_steps": schedule.warmup,
            "stable_steps": schedule.stable,
            "decay_steps": schedule.decay,
            "min_lr_ratio": config.scheduler.min_lr_ratio,
            "max_seq_length": config.data.packing.max_seq_length,
            "data_pipeline": config.data.to_dict(),
            "use_mixed_precision": use_amp,
            "dtype": config.training.dtype,
            "seed": config.training.seed,
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
        log_dir=config.swanlab.swanlab_log_dir,
        workspace=workspace,
        run_id=run_id,
        resume=resume,
        init_kwargs=init_kwargs,
        global_rank=runtime.rank,
    )


class CausalLMTrainer:
    """Train one causal language model with token-weighted global updates."""

    MAX_CONSECUTIVE_NONFINITE_SKIPS = 10

    def __init__(
        self,
        components: TrainerComponents,
        settings: TrainerSettings,
    ) -> None:
        self.model = components.model
        self.train_loader = components.train_loader
        self.val_loader = components.val_loader
        self.optimizer = components.optimizer
        self.scheduler = components.scheduler
        self.scaler = components.scaler
        self.checkpoint_manager = components.checkpoint_manager
        self.best_model_store = components.best_model_store
        self.final_model_path = Path(components.final_model_path)
        self.evaluator = components.evaluator
        self.diagnostics = components.diagnostics
        self.log_manager = components.log_manager
        self.early_stopping = components.early_stopping
        self.swanlab_logger = components.swanlab_logger

        self.settings = settings
        self.global_step = settings.init_step
        self.best_val_loss = (
            self.best_model_store.best_metric
            if self.best_model_store.best_metric is not None
            else settings.best_val_loss
        )
        self.last_val_loss = settings.last_val_loss
        self.last_eval_step = settings.last_eval_step
        self.data_state = settings.data_state
        self.token_budget = settings.token_budget
        self.tokens_consumed = settings.tokens_consumed
        self.world_size = settings.world_size
        self.global_rank = settings.rank
        self.device = settings.device
        self.dtype = settings.dtype
        self.use_amp = settings.use_amp
        self.max_grad_norm = settings.max_grad_norm
        self.scaler_enabled = bool(
            settings.use_amp and settings.dtype == torch.float16
        )
        self.compiled_model = settings.compiled_model
        self.early_stopped = False
        self.nonfinite_skip_count = 0
        self.consecutive_nonfinite_skips = 0
        self.start_time = 0.0
        self.last_log_time: float | None = None

    def run(self) -> None:
        """Run until max steps, early stopping, or an exception."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.start_time = time.time()
        self.log_manager.info("训练开始")
        try:
            iterator, epoch, batch_index, consumed_samples = (
                self._prepare_iterator()
            )
            supervised_tokens_since_log = 0
            non_padding_tokens_since_log = 0
            padded_tokens_since_log = 0
            steps_since_log = 0
            reader = self.train_loader.dataset.reader
            while (
                self.global_step < self.settings.max_steps
                and not self.early_stopped
                and not self._token_budget_reached()
            ):
                micro_batches, epoch, batch_index, consumed_samples = (
                    self._collect_micro_batches(
                        iterator,
                        epoch,
                        batch_index,
                        consumed_samples,
                    )
                )
                iterator = self._iterator
                data_state = DataState(
                    dataset_id=reader.dataset_id,
                    dataset_version=reader.version,
                    epoch=epoch,
                    batch_index=batch_index,
                    consumed_samples=consumed_samples,
                )
                local_tokens = sum(
                    int((batch["labels"][..., 1:] != -100).sum().item())
                    for batch in micro_batches
                )
                if local_tokens <= 0:
                    raise RuntimeError(
                        "training batch contains no supervised tokens"
                    )
                global_tokens = self._global_sum(local_tokens)
                if global_tokens <= 0:
                    raise RuntimeError(
                        "distributed batch contains no supervised tokens"
                    )

                collect_statistics = self._should_collect_step_statistics()
                (
                    grad_norm,
                    stepped,
                    diagnostics,
                    local_statistics,
                ) = (
                    self._train_micro_batches(
                        micro_batches,
                        global_tokens,
                        collect_statistics=collect_statistics,
                    )
                )
                if not stepped:
                    self._handle_skipped_update(
                        data_state,
                        grad_norm,
                        diagnostics,
                    )
                    continue

                self.consecutive_nonfinite_skips = 0
                self.global_step += 1
                self.data_state = data_state
                statistics = self._reduce_training_statistics(
                    local_statistics
                )
                if statistics.token_count != global_tokens:
                    raise RuntimeError(
                        "token statistics disagree with the global "
                        f"supervised-token count: {statistics.token_count} "
                        f"!= {global_tokens}"
                    )
                loss = statistics.nll_nats / statistics.token_count
                metrics = self.diagnostics.update_step(
                    model=self.model,
                    loss=loss,
                    grad_norm=grad_norm,
                    padded_tokens=statistics.padded_tokens,
                    step=self.global_step,
                )
                self.tokens_consumed += statistics.non_padding_tokens
                supervised_tokens_since_log += global_tokens
                non_padding_tokens_since_log += statistics.non_padding_tokens
                padded_tokens_since_log += statistics.padded_tokens
                steps_since_log += 1
                if (
                    self.settings.train_log_interval > 0
                    and self.global_step % self.settings.train_log_interval == 0
                ):
                    accuracy = (
                        statistics.correct_count / statistics.token_count
                        if collect_statistics
                        else None
                    )
                    bpb = (
                        compute_bits_per_byte(
                            statistics.bpb_nll_nats,
                            statistics.bpb_bytes,
                        )
                        if collect_statistics
                        else None
                    )
                    self._log_training(
                        loss,
                        supervised_tokens_since_log,
                        grad_norm,
                        metrics,
                        accuracy=accuracy,
                        bits_per_byte=bpb,
                        supervised_tokens=statistics.token_count,
                        non_padding_tokens=statistics.non_padding_tokens,
                        padded_tokens=statistics.padded_tokens,
                        non_padding_tokens_window=(
                            non_padding_tokens_since_log
                        ),
                        padded_tokens_window=padded_tokens_since_log,
                        logged_steps=steps_since_log,
                    )
                    supervised_tokens_since_log = 0
                    non_padding_tokens_since_log = 0
                    padded_tokens_since_log = 0
                    steps_since_log = 0

                checkpoint_saved = False
                if (
                    self.global_rank == 0
                    and self.settings.save_interval > 0
                    and self.global_step % self.settings.save_interval == 0
                ):
                    self._save_regular_checkpoint()
                    checkpoint_saved = True

                if (
                    self.settings.eval_interval > 0
                    and self.global_step % self.settings.eval_interval == 0
                ):
                    if self.val_loader is None:
                        raise RuntimeError(
                            "validation loader is required for eval_interval > 0"
                        )
                    if not checkpoint_saved:
                        self._save_regular_checkpoint()
                    self._evaluate_and_record()
            if (
                self.settings.eval_interval > 0
                and self.val_loader is not None
                and self.last_eval_step != self.global_step
            ):
                self._evaluate_and_record()
            if self.early_stopped:
                stop_reason = "early_stopping"
            elif self._token_budget_reached():
                stop_reason = "token_budget"
            else:
                stop_reason = "max_steps"
            self._save_final_checkpoint(stop_reason)
            self.log_manager.info("训练完成!")
        except KeyboardInterrupt:
            self.log_manager.info(
                "训练被用户中断，正在保存检查点后退出"
            )
            self._save_interrupted_checkpoint()
            raise
        except Exception as exc:
            self.log_manager.error(
                f"训练过程中发生异常: {type(exc).__name__}: {exc}",
                exc_info=True,
            )
            self._save_interrupted_checkpoint()
            raise
        finally:
            if self.swanlab_logger is not None:
                self.swanlab_logger.finish()

    def _prepare_iterator(self) -> tuple[Any, int, int, int]:
        reader = getattr(
            getattr(self.train_loader, "dataset", None),
            "reader",
            None,
        )
        if reader is None:
            raise RuntimeError(
                "training loader does not expose a unified storage reader"
            )
        if self.data_state is not None and not self.data_state.compatible_with(
            reader.dataset_id,
            reader.version,
        ):
            raise ValueError(
                "checkpoint DataState does not match the configured dataset"
            )
        epoch = 0 if self.data_state is None else self.data_state.epoch
        batch_index = (
            0 if self.data_state is None else self.data_state.batch_index
        )
        consumed_samples = (
            0 if self.data_state is None else self.data_state.consumed_samples
        )
        if hasattr(self.train_loader, "set_epoch"):
            self.train_loader.set_epoch(epoch)
        if batch_index and hasattr(self.train_loader, "set_resume_batch"):
            self.log_manager.info(
                f"数据恢复位置: epoch={epoch}, batch_index={batch_index}"
            )
            self.train_loader.set_resume_batch(batch_index)
            iterator = iter(self.train_loader)
        else:
            iterator = skip_batches(iter(self.train_loader), batch_index)
        self._iterator = iterator
        return iterator, epoch, batch_index, consumed_samples

    def _collect_micro_batches(
        self,
        iterator,
        epoch: int,
        batch_index: int,
        consumed_samples: int,
    ) -> tuple[list[dict[str, Any]], int, int, int]:
        batches = []
        for _ in range(self.settings.gradient_accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                batch_index = 0
                if hasattr(self.train_loader, "set_epoch"):
                    self.train_loader.set_epoch(epoch)
                iterator = iter(self.train_loader)
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise RuntimeError(
                        "training data pipeline produced no batches"
                    ) from exc
            batch_index += 1
            sample_ids = batch.get("sample_ids")
            consumed_samples += (
                len(sample_ids)
                if sample_ids is not None
                else int(batch["input_ids"].size(0))
            )
            batches.append(batch)
        self._iterator = iterator
        return batches, epoch, batch_index, consumed_samples

    def _train_micro_batches(
        self,
        batches: list[dict[str, Any]],
        global_tokens: int,
        *,
        collect_statistics: bool,
    ) -> tuple[float, bool, list[str], TrainingStepStatistics]:
        loss_sum = 0.0
        token_count = 0
        correct_count = 0
        bpb_nll_nats = 0.0
        bpb_bytes = 0
        non_padding_tokens = 0
        padded_tokens = 0
        finite = True
        for index, raw_batch in enumerate(batches):
            non_padding_tokens += self._batch_non_padding_tokens(raw_batch)
            batch = move_batch_to_device(raw_batch, self.device)
            if self.compiled_model:
                torch.compiler.cudagraph_mark_step_begin()
            no_sync = (
                self.world_size > 1
                and index < len(batches) - 1
                and hasattr(self.model, "no_sync")
            )
            context = self.model.no_sync() if no_sync else nullcontext()
            with context:
                with autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                    dtype=self.dtype,
                ):
                    output = self.model(
                        batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        segment_ids=batch.get("segment_ids"),
                        position_ids=batch.get("position_ids"),
                        labels=batch["labels"],
                        return_logits=False,
                        validate_inputs=False,
                        return_token_statistics=collect_statistics,
                    )
                    if output.loss_sum is None:
                        raise RuntimeError("model did not return a loss sum")
                    backward_loss = output.loss_sum * (
                        self.world_size / global_tokens
                    )
                    if output.aux_loss is not None:
                        backward_loss = backward_loss + output.aux_loss * (
                            self.world_size
                            * batch["input_ids"].numel()
                            / global_tokens
                        )
                    batch_finite = bool(
                        torch.isfinite(backward_loss.detach()).all().item()
                    )
                    finite = finite and batch_finite
                self.scaler.scale(backward_loss).backward()
            if collect_statistics and batch_finite:
                if (
                    output.token_losses is None
                    or output.correct_predictions is None
                ):
                    raise RuntimeError(
                        "model did not return training token statistics"
                    )
                batch_statistics = compute_token_statistics_from_losses(
                    output.token_losses,
                    batch["labels"],
                    correct_count=int(
                        output.correct_predictions.detach().item()
                    ),
                    tokenizer=getattr(self.evaluator, "tokenizer", None),
                )
                loss_sum += batch_statistics.nll_nats
                token_count += batch_statistics.token_count
                correct_count += batch_statistics.correct_count
                bpb_nll_nats += batch_statistics.bpb_nll_nats
                bpb_bytes += batch_statistics.bpb_bytes
            else:
                loss_sum += float(output.loss_sum.detach().float().item())
                token_count += int(
                    (batch["labels"][..., 1:] != -100).sum().item()
                )
            padded_tokens += batch["input_ids"].numel()

        grad_norm, stepped, diagnostics = self._apply_optimizer_step(finite)
        return (
            grad_norm,
            stepped,
            diagnostics,
            TrainingStepStatistics(
                nll_nats=loss_sum,
                token_count=token_count,
                correct_count=correct_count,
                bpb_nll_nats=bpb_nll_nats,
                bpb_bytes=bpb_bytes,
                non_padding_tokens=non_padding_tokens,
                padded_tokens=padded_tokens,
            ),
        )

    def _handle_skipped_update(
        self,
        data_state: DataState,
        grad_norm: float,
        diagnostics: list[str],
    ) -> None:
        self.nonfinite_skip_count += 1
        self.consecutive_nonfinite_skips += 1
        self.data_state = data_state
        reason = (
            "local_nonfinite_gradients=[" + ", ".join(diagnostics) + "]"
            if diagnostics
            else "nonfinite_training_state"
        )
        self.log_manager.warning(
            "检测到非有限训练状态，已原子跳过优化器更新: "
            f"candidate_step={self.global_step + 1}, "
            f"rank={self.global_rank}, grad_norm={grad_norm}, "
            f"consecutive_skips={self.consecutive_nonfinite_skips}, "
            f"reason={reason}"
        )
        if (
            self.consecutive_nonfinite_skips
            >= self.MAX_CONSECUTIVE_NONFINITE_SKIPS
        ):
            raise RuntimeError(
                "training produced non-finite gradients for "
                f"{self.consecutive_nonfinite_skips} consecutive steps"
            )

    def _global_sum(self, value: int) -> int:
        if not torch.distributed.is_initialized():
            return int(value)
        total = torch.tensor(
            float(value),
            dtype=torch.float64,
            device=self.device,
        )
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        return int(total.item())

    def _token_budget_reached(self) -> bool:
        return (
            self.token_budget is not None
            and self.tokens_consumed >= self.token_budget
        )

    @staticmethod
    def _batch_non_padding_tokens(batch: dict[str, Any]) -> int:
        mask = batch.get("attention_mask")
        if mask is not None:
            return int(mask.sum().item())
        segment_ids = batch.get("segment_ids")
        if segment_ids is not None:
            return int((segment_ids >= 0).sum().item())
        return int(batch["input_ids"].numel())

    def _should_collect_step_statistics(self) -> bool:
        interval = self.settings.train_log_interval
        return interval > 0 and (self.global_step + 1) % interval == 0

    def _reduce_training_statistics(
        self,
        statistics: TrainingStepStatistics,
    ) -> TrainingStepStatistics:
        values = [
            statistics.nll_nats,
            float(statistics.token_count),
            float(statistics.correct_count),
            statistics.bpb_nll_nats,
            float(statistics.bpb_bytes),
            float(statistics.non_padding_tokens),
            float(statistics.padded_tokens),
        ]
        totals = torch.tensor(
            values,
            dtype=torch.float64,
            device=self.device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                totals,
                op=torch.distributed.ReduceOp.SUM,
            )
        reduced = totals.tolist()
        return TrainingStepStatistics(
            nll_nats=reduced[0],
            token_count=int(reduced[1]),
            correct_count=int(reduced[2]),
            bpb_nll_nats=reduced[3],
            bpb_bytes=int(reduced[4]),
            non_padding_tokens=int(reduced[5]),
            padded_tokens=int(reduced[6]),
        )

    def _all_ranks_finite(self, finite: bool) -> bool:
        if not torch.distributed.is_initialized():
            return bool(finite)
        flag = torch.tensor(
            int(finite),
            dtype=torch.int32,
            device=self.device,
        )
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        return bool(flag.item())

    def _gradient_total_norm(self) -> torch.Tensor:
        norms = [
            torch.linalg.vector_norm(parameter.grad.detach().float())
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        if not norms:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        return torch.linalg.vector_norm(torch.stack(norms))

    def _nonfinite_gradient_diagnostics(self) -> list[str]:
        diagnostics = []
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            finite = torch.isfinite(parameter.grad)
            if finite.all():
                continue
            diagnostics.append(
                f"{name}: {int((~finite).sum().item())}/{parameter.grad.numel()}"
            )
            if len(diagnostics) == 8:
                diagnostics.append("...")
                break
        return diagnostics

    def _clip_gradients_with_norm(self, norm: torch.Tensor) -> None:
        coefficient = torch.clamp(
            self.max_grad_norm / (norm + 1e-6),
            max=1.0,
        )
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(
                    coefficient.to(dtype=parameter.grad.dtype)
                )

    def _apply_optimizer_step(
        self,
        loss_finite: bool,
    ) -> tuple[float, bool, list[str]]:
        """Apply one optimizer update only if every rank accepts it."""
        if self.scaler_enabled:
            self.scaler.unscale_(self.optimizer)
        norm = self._gradient_total_norm()
        norm_value = float(norm.detach().float().item())
        update_finite = self._all_ranks_finite(
            loss_finite and math.isfinite(norm_value)
        )
        if not update_finite:
            diagnostics = self._nonfinite_gradient_diagnostics()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler_enabled:
                self.scaler.update()
            return norm_value, False, diagnostics

        self._clip_gradients_with_norm(norm)
        if self.scaler_enabled:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return norm_value, True, []

    def _log_training(
        self,
        loss: float,
        tokens: int,
        grad_norm: float,
        diagnostics: dict[str, float],
        *,
        accuracy: float | None = None,
        bits_per_byte: float | None = None,
        supervised_tokens: int | None = None,
        non_padding_tokens: int | None = None,
        padded_tokens: int | None = None,
        non_padding_tokens_window: int | None = None,
        padded_tokens_window: int | None = None,
        logged_steps: int | None = None,
    ) -> None:
        now = time.time()
        elapsed = now - (
            self.start_time
            if self.last_log_time is None
            else self.last_log_time
        )
        self.last_log_time = now
        supervised_tokens_per_second = tokens / max(elapsed, 1e-6)
        non_padding_tokens_per_second = (
            non_padding_tokens_window
            if non_padding_tokens_window is not None
            else tokens
        ) / max(elapsed, 1e-6)
        learning_rate = self.optimizer.param_groups[0]["lr"]
        perplexity = compute_perplexity(loss)
        metrics = {
            "train/loss": loss,
            "train/lr": learning_rate,
            "train/tokens_per_sec": non_padding_tokens_per_second,
            "train/non_padding_tokens_per_sec": (
                non_padding_tokens_per_second
            ),
            "train/supervised_tokens_per_sec": supervised_tokens_per_second,
            "train/supervised_tokens_since_log": tokens,
            "train/grad_norm": grad_norm,
            "train/smoothed_loss": diagnostics.get("smoothed_loss", 0.0),
            "train/loss_trend": diagnostics.get("loss_trend", 0.0),
            "train/loss_variance": diagnostics.get("loss_variance", 0.0),
            "train/grad_norm_sq_variance_ratio": diagnostics.get(
                "grad_norm_sq_variance_ratio",
                0.0,
            ),
            "train/grad_norm_sq_mean": diagnostics.get(
                "grad_norm_sq_mean",
                0.0,
            ),
            "train/grad_norm_sq_std": diagnostics.get(
                "grad_norm_sq_std",
                0.0,
            ),
        }
        if perplexity is not None:
            metrics["train/perplexity"] = perplexity
        if accuracy is not None:
            metrics["train/accuracy"] = accuracy
        if bits_per_byte is not None:
            metrics["train/bpb"] = bits_per_byte
        if padded_tokens_window is not None and elapsed > 0:
            metrics["train/padded_tokens_since_log"] = padded_tokens_window
            metrics["train/padded_tokens_per_sec"] = (
                padded_tokens_window / max(elapsed, 1e-6)
            )
        if non_padding_tokens_window is not None:
            metrics["train/non_padding_tokens_since_log"] = (
                non_padding_tokens_window
            )
        if logged_steps is not None and elapsed > 0:
            metrics["train/steps_per_sec"] = (
                logged_steps / max(elapsed, 1e-6)
            )
        if supervised_tokens is not None and padded_tokens is not None:
            metrics["train/supervised_tokens"] = supervised_tokens
            metrics["train/padded_tokens"] = padded_tokens
            if padded_tokens > 0:
                metrics["train/supervision_ratio"] = (
                    supervised_tokens / padded_tokens
                )
        if non_padding_tokens is not None:
            metrics["train/non_padding_tokens"] = non_padding_tokens
            if padded_tokens is not None and padded_tokens > 0:
                metrics["train/non_padding_ratio"] = (
                    non_padding_tokens / padded_tokens
                )
        settings = getattr(self, "settings", None)
        if settings is not None and settings.max_steps > 0:
            progress = self.global_step / settings.max_steps
            metrics["train/progress"] = progress
            if settings.epochs is not None:
                metrics["train/epoch"] = min(
                    progress * settings.epochs,
                    settings.epochs,
                )
        token_budget = (
            getattr(settings, "token_budget", None)
            if settings is not None
            else None
        )
        if token_budget is not None:
            metrics["train/tokens_consumed"] = self.tokens_consumed
            metrics["train/token_progress"] = (
                self.tokens_consumed / token_budget
            )
        data_state = getattr(self, "data_state", None)
        train_loader = getattr(self, "train_loader", None)
        loader_length = getattr(train_loader, "__len__", None)
        if data_state is not None and loader_length is not None:
            try:
                total_batches = len(self.train_loader)
            except TypeError:
                total_batches = 0
            if total_batches > 0:
                metrics["train/data_epoch"] = (
                    data_state.epoch
                    + data_state.batch_index / total_batches
                )
        for name in (
            "mfu_average",
            "mfu_instant",
            "total_flops_tflops",
            "peak_flops_tflops",
            "elapsed_time",
        ):
            if name in diagnostics:
                metrics[f"train/{name}"] = diagnostics[name]
        ppl_text = "N/A" if perplexity is None else f"{perplexity:.2f}"
        mfu = diagnostics.get("mfu_average")
        accuracy_text = (
            "N/A" if accuracy is None else f"{accuracy:.4f}"
        )
        bpb_text = (
            "N/A" if bits_per_byte is None else f"{bits_per_byte:.4f}"
        )
        self.log_manager.logger.info(
            f"Step {self.global_step} | loss={loss:.4f} | "
            f"smooth_loss={diagnostics.get('smoothed_loss', 0.0):.4f} | "
            f"ppl={ppl_text} | lr={learning_rate:.2e} | "
            f"acc={accuracy_text} | bpb={bpb_text} | "
            f"tok/s={non_padding_tokens_per_second:.0f} | "
            f"sup/s={supervised_tokens_per_second:.0f} | "
            f"tok={supervised_tokens if supervised_tokens is not None else 'N/A'} | "
            f"pad={padded_tokens if padded_tokens is not None else 'N/A'} | "
            f"grad_ratio={diagnostics.get('grad_norm_sq_variance_ratio', 0.0):.2f} | "
            f"mfu={'N/A' if mfu is None else f'{mfu:.3f}'}"
        )
        self._log_metrics(metrics)

    def _evaluate_and_record(self) -> None:
        self.log_manager.info("验证开始")
        result = self.evaluator.evaluate(
            self.val_loader,
            compute_empty_prefix=self.settings.compute_empty_prefix,
            eval_log_interval=self.settings.eval_log_interval,
        )
        self.last_val_loss = result.loss
        self.last_eval_step = self.global_step
        self.best_model_store.save_if_better(
            self.model,
            metrics={
                "val_loss": result.loss,
                "eval_step": self.global_step,
            },
            global_step=self.global_step,
            write=self.global_rank == 0,
        )
        if self.best_model_store.best_metric is not None:
            self.best_val_loss = self.best_model_store.best_metric
        metrics = result.metrics("val/")
        self._log_metrics(metrics)
        if self.settings.generation_prompts:
            samples = self.evaluator.generate(
                self.settings.generation_prompts,
                num_tokens=self.settings.num_generate_tokens,
            )
            for index, sample in enumerate(samples):
                tag = f"generation/prompt_{index}"
                self.log_manager.log_text(
                    tag,
                    sample.generated_text,
                    step=self.global_step,
                )
                if self.swanlab_logger is not None:
                    self.swanlab_logger.log_text(
                        tag,
                        sample.generated_text,
                        step=self.global_step,
                    )
        self.log_manager.info(
            f"当前最佳验证损失: {self.best_val_loss:.4f}"
        )
        if self.early_stopping is not None:
            if self.early_stopping.step(result.loss, self.global_step):
                self.log_manager.info(
                    f"早停触发于步数 {self.global_step}"
                )
                self.early_stopped = True

    def _save_regular_checkpoint(self) -> None:
        if self.global_rank != 0:
            return
        self._save_checkpoint(
            f"checkpoints/checkpoint-{self.global_step}.pt",
            metrics=self._evaluation_metrics(),
            managed=True,
            include_early_stopping=True,
        )
        self.log_manager.info(f"检查点已保存至 step {self.global_step}")

    def _save_final_checkpoint(self, stop_reason: str) -> None:
        if self.global_rank == 0:
            self._save_checkpoint(
                "checkpoints/checkpoint-final.pt",
                metrics=self._evaluation_metrics(),
                managed=False,
                include_early_stopping=True,
            )
            save_model_artifact(
                self.final_model_path,
                self.model,
                variant="final",
                global_step=self.global_step,
                metrics=self._evaluation_metrics(),
                stop_reason=stop_reason,
            )

    def _save_interrupted_checkpoint(self) -> None:
        if self.global_rank != 0:
            return
        try:
            self._save_checkpoint(
                "checkpoints/checkpoint-emergency.pt",
                metrics=self._evaluation_metrics(),
                managed=False,
                include_early_stopping=True,
            )
            self.log_manager.info(
                f"紧急检查点已保存至 step {self.global_step}"
            )
        except Exception as exc:
            self.log_manager.error(f"紧急检查点保存失败: {exc}")

    def _save_checkpoint(
        self,
        filename: str,
        *,
        metrics: dict[str, float | int | None],
        managed: bool,
        include_early_stopping: bool,
    ) -> None:
        kwargs = {
            "data_state": self._data_state_dict(),
            "best_val_loss": (
                self.best_val_loss
                if math.isfinite(self.best_val_loss)
                else None
            ),
            "best_step": self.best_model_store.best_step,
            "last_val_loss": self.last_val_loss,
            "last_eval_step": self.last_eval_step,
            "tokens_consumed": self.tokens_consumed,
            **self._checkpoint_metadata(),
        }
        if include_early_stopping and self.early_stopping is not None:
            kwargs["early_stopping_state"] = self.early_stopping.state_dict()
        self.checkpoint_manager.save(
            filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            global_step=self.global_step,
            metrics=metrics,
            managed=managed,
            **kwargs,
        )

    def _evaluation_metrics(self) -> dict[str, float | None]:
        return {
            "best_val_loss": (
                self.best_val_loss
                if math.isfinite(self.best_val_loss)
                else None
            ),
            "last_val_loss": self.last_val_loss,
            "last_eval_step": self.last_eval_step,
            "tokens_consumed": self.tokens_consumed,
        }

    def _checkpoint_metadata(self) -> dict[str, str]:
        if self.swanlab_logger is None or not self.swanlab_logger.run_id:
            return {}
        metadata = {"swanlab_run_id": self.swanlab_logger.run_id}
        if self.swanlab_logger.project:
            metadata["swanlab_project"] = self.swanlab_logger.project
        if self.swanlab_logger.workspace:
            metadata["swanlab_workspace"] = self.swanlab_logger.workspace
        return metadata

    def _data_state_dict(self) -> dict[str, Any] | None:
        return None if self.data_state is None else self.data_state.to_dict()

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        self.log_manager.log_metrics(metrics, step=self.global_step)
        if self.swanlab_logger is not None:
            self.swanlab_logger.log_metrics(metrics, step=self.global_step)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_experiment_snapshot(
    paths: ExperimentPaths,
    *,
    model_config: ModelConfig,
    config: TrainingRunConfig,
) -> None:
    """Persist the exact resolved configuration used by a new run."""
    model_path = paths.config / "model.yaml"
    training_path = paths.config / "training.yaml"
    metadata_path = paths.metadata
    if model_path.exists() and training_path.exists() and metadata_path.exists():
        return
    training_snapshot = _json_compatible(
        {
            "tokenizer": {
                "path": config.tokenizer_path,
                "backend": "tokenizers",
            },
            "optimizer": asdict(config.optimizer),
            "scheduler": asdict(config.scheduler),
            "data": config.data.to_dict(),
            "training": asdict(config.training),
            "evaluating": asdict(config.evaluating),
            "early_stopping": asdict(config.early_stopping),
            "swanlab": asdict(config.swanlab),
        }
    )
    metadata = _json_compatible(
        {
            "stage": paths.stage,
            "model_scale": paths.model_scale,
            "experiment_name": paths.experiment_name,
            "model_name": model_config.model_name,
        }
    )
    _atomic_write_text(
        model_path,
        yaml.safe_dump(
            _json_compatible(model_config.to_dict()),
            allow_unicode=True,
            sort_keys=False,
        ),
    )
    _atomic_write_text(
        training_path,
        yaml.safe_dump(
            training_snapshot,
            allow_unicode=True,
            sort_keys=False,
        ),
    )
    _atomic_write_text(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )


def _json_compatible(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def run_causal_lm_training(task: CausalLMTask, args) -> None:
    """Assemble and execute a task-configured causal-LM training run."""
    task.validate_args(args)
    configure_torch_runtime()
    model_config, config = task.load_configs(args)
    runtime = initialize_runtime(
        seed=config.training.seed,
        timeout_minutes=config.training.distributed_timeout_minutes,
    )
    paths = _resolve_paths(
        args,
        model_config,
        config,
        runtime,
        stage=task.stage,
    )
    log_manager = LogManager(
        log_dir=str(paths.logs),
        tensorboard=runtime.is_main,
        log_file="training.log" if runtime.is_main else None,
        console_level=20,
    )
    loaders = None
    swanlab_logger = None
    try:
        log_manager.info(f"实验目录: {paths.root}")
        log_manager.info(f"训练任务: {task.stage}")
        log_manager.info(f"模型配置: {model_config.model_name}")
        if runtime.is_main:
            _write_experiment_snapshot(
                paths,
                model_config=model_config,
                config=config,
            )
        tokenizer_bundle = load_tokenizer(
            config.tokenizer_path,
            data_config=config.data,
            model_vocab_size=model_config.vocab_size,
        )
        if (
            tokenizer_bundle.tokenizer.get_vocab_size()
            < model_config.vocab_size
            and runtime.is_main
        ):
            log_manager.warning(
                "Tokenizer 词表小于模型 vocab_size；尾部 Token 不会被训练"
            )
        prompt_renderer_factory = task.prompt_renderer_factory()
        prompt_renderer = (
            prompt_renderer_factory(tokenizer_bundle.tokenizer)
            if prompt_renderer_factory is not None
            else None
        )

        base_model = build_model(model_config, runtime.device)
        if args.resume is None:
            initialization_source = task.prepare_model(args, base_model)
            if runtime.is_main and initialization_source:
                log_manager.info(f"权重初始化: {initialization_source}")
        raw_model = base_model
        compiled = False
        if config.training.compile_model:
            available, reason = compile_preflight()
            if not available and runtime.is_main:
                log_manager.warning(f"跳过 torch.compile: {reason}")
            elif available:
                try:
                    raw_model = compile_training_model(
                        base_model,
                        mode=config.training.compile_mode,
                        dynamic=config.training.compile_dynamic,
                        distributed=runtime.distributed,
                    )
                    compiled = True
                    warmup_compiled_training(
                        raw_model,
                        device=runtime.device,
                        vocab_size=model_config.vocab_size,
                        batch_size=config.training.micro_batch_size,
                        sequence_length=config.data.packing.max_seq_length,
                        use_amp=(
                            config.training.use_mixed_precision
                            and runtime.device.type == "cuda"
                        ),
                        dtype=(
                            torch.bfloat16
                            if config.training.dtype == "bfloat16"
                            else torch.float16
                        ),
                    )
                except Exception as exc:
                    compiled = False
                    raw_model = base_model
                    if runtime.is_main:
                        log_manager.warning(
                            "torch.compile 初始化失败，已回退 eager: "
                            f"{type(exc).__name__}: {exc}"
                        )
                if runtime.device.type == "cuda":
                    torch.cuda.synchronize(runtime.device)
                barrier(runtime)

        model = wrap_distributed(
            raw_model,
            runtime=runtime,
            find_unused_parameters=model_config.ffn_type == "moe",
        )
        total_params = parameter_count(model, trainable_only=False)
        trainable_params = parameter_count(model, trainable_only=True)
        if runtime.is_main:
            log_manager.info(
                f"模型参数量: {total_params:,} "
                f"(可训练: {trainable_params:,})"
            )
            log_manager.info(f"模型大小: {model_size_mb(model):.2f} MB")

        loaders = build_data_loaders(
            config.data,
            config.evaluating,
            micro_batch_size=config.training.micro_batch_size,
            training_dataloader=config.training.dataloader,
            runtime=runtime,
            device=runtime.device,
            pad_token_id=tokenizer_bundle.pad_token_id,
        )
        if runtime.is_main:
            log_manager.info(
                "数据管线配置: "
                + ", ".join(
                    f"{key}={value}"
                    for key, value in config.data.to_dict().items()
                )
            )

        optimizer = build_optimizer(raw_model, config.optimizer)
        budget = task.build_step_budget(config, loaders, runtime)
        scheduler, schedule = build_scheduler(
            optimizer,
            config.scheduler,
            budget.max_steps,
        )
        if runtime.is_main:
            log_manager.info(
                f"目标 Token 数: {budget.target_tokens:,}; "
                f"全局 batch: {budget.global_batch_size}; "
                f"名义每步 Token: {budget.tokens_per_step:,}; "
                f"max_steps: {budget.max_steps}"
            )
            log_manager.info(
                f"WSD 调度: warmup={schedule.warmup}, "
                f"stable={schedule.stable}, decay={schedule.decay}"
            )

        requested_amp = config.training.use_mixed_precision
        use_amp = requested_amp and runtime.device.type == "cuda"
        if requested_amp and not use_amp and runtime.is_main:
            log_manager.warning("CUDA 不可用，已禁用混合精度训练")
        dtype = (
            torch.bfloat16
            if config.training.dtype == "bfloat16"
            else torch.float16
        )
        scaler = GradScaler(
            "cuda",
            enabled=use_amp and dtype == torch.float16,
        )

        checkpoint_manager = CheckpointManager(
            output_dir=str(paths.root),
            keep_last_n=3,
        )
        resume = None
        data_state = None
        resume_path = (
            None
            if args.resume is None
            else str(Path(args.resume).expanduser().resolve())
        )
        if resume_path is not None:
            log_manager.info(f"从检查点恢复: {args.resume}")
            resume = checkpoint_manager.load(
                resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                map_location=str(runtime.device),
            )
            if resume.checkpoint.get("data_state") is not None:
                data_state = DataState.from_dict(
                    resume.checkpoint["data_state"]
                )
            if resume.global_step > budget.max_steps:
                raise ValueError(
                    "checkpoint global_step exceeds the configured step budget: "
                    f"{resume.global_step} > {budget.max_steps}"
                )
            log_manager.info(f"恢复至全局步数: {resume.global_step}")
            checkpoint_manager.protect(resume_path)
        checkpoint_manager.prune()

        resume_checkpoint = {} if resume is None else resume.checkpoint
        best_model_store = BestModelStore(
            paths.best_models / "best_model.pt",
            initial_metric=resume_checkpoint.get("best_val_loss"),
            initial_step=resume_checkpoint.get("best_step"),
            model_config=model_config.to_dict(),
        )
        best_val_loss = (
            best_model_store.best_metric
            if best_model_store.best_metric is not None
            else float("inf")
        )

        early_stopping = None
        if config.early_stopping.enabled:
            early_stopping = EarlyStopping(
                monitor=config.early_stopping.monitor,
                patience=config.early_stopping.patience,
                min_delta=config.early_stopping.min_delta,
                mode=config.early_stopping.mode,
            )
            state = resume_checkpoint.get("early_stopping_state")
            if state is not None:
                early_stopping.load_state_dict(state)

        resolved_peak_flops = (
            args.peak_flops if args.peak_flops is not None else peak_flops()
        )
        diagnostics = TrainingDiagnostics(
            peak_flops=resolved_peak_flops,
            flops_log_file=paths.logs / "mfu_log.jsonl",
        )
        evaluator = CausalLMEvaluator(
            model=model,
            tokenizer=tokenizer_bundle.tokenizer,
            device=runtime.device,
            use_amp=use_amp,
            dtype=dtype,
            bos_token_id=tokenizer_bundle.bos_token_id,
            eos_token_id=tokenizer_bundle.eos_token_id,
            logger=log_manager.logger,
            rank=runtime.rank,
            world_size=runtime.world_size,
            prompt_renderer=prompt_renderer,
        )
        swanlab_logger = _create_swanlab_logger(
            config=config,
            args=args,
            runtime=runtime,
            checkpoint=resume_checkpoint,
            model_config=model_config,
            budget=budget,
            schedule=schedule,
            use_amp=use_amp,
            total_params=total_params,
            trainable_params=trainable_params,
        )
        settings = TrainerSettings(
            max_steps=budget.max_steps,
            gradient_accumulation_steps=(
                config.training.gradient_accumulation_steps
            ),
            save_interval=config.training.save_interval,
            eval_interval=config.evaluating.eval_interval,
            train_log_interval=config.training.log_interval,
            eval_log_interval=config.evaluating.log_interval,
            max_grad_norm=config.training.max_grad_norm,
            device=runtime.device,
            world_size=runtime.world_size,
            rank=runtime.rank,
            use_amp=use_amp,
            dtype=dtype,
            compiled_model=compiled,
            generation_prompts=config.evaluating.prompts,
            num_generate_tokens=config.evaluating.num_generate_tokens,
            compute_empty_prefix=config.evaluating.empty_prefix_eos_ppl,
            init_step=0 if resume is None else resume.global_step,
            epochs=config.training.epochs,
            token_budget=budget.token_budget,
            tokens_consumed=int(
                resume_checkpoint.get("tokens_consumed", 0) or 0
            ),
            best_val_loss=best_val_loss,
            last_val_loss=resume_checkpoint.get("last_val_loss"),
            last_eval_step=resume_checkpoint.get("last_eval_step"),
            data_state=data_state,
        )
        trainer = CausalLMTrainer(
            TrainerComponents(
                model=model,
                train_loader=loaders.train,
                val_loader=loaders.validation,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                checkpoint_manager=checkpoint_manager,
                best_model_store=best_model_store,
                final_model_path=paths.final_models / "final_model.pt",
                evaluator=evaluator,
                diagnostics=diagnostics,
                log_manager=log_manager,
                early_stopping=early_stopping,
                swanlab_logger=swanlab_logger,
            ),
            settings,
        )
        trainer.run()
    finally:
        if loaders is not None:
            loaders.close()
        if swanlab_logger is not None:
            swanlab_logger.finish()
        log_manager.close()


class CausalLMTrainingPipeline:
    """Orchestrate a causal-LM run for one task policy."""

    def __init__(self, task: CausalLMTask) -> None:
        self.task = task

    def run(self, args) -> None:
        run_causal_lm_training(self.task, args)


__all__ = [
    "CausalLMTask",
    "CausalLMTrainingPipeline",
    "CausalLMTrainer",
    "TrainerComponents",
    "TrainerSettings",
    "run_causal_lm_training",
]
