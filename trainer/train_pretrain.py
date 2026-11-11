"""TaiChu pretraining entrypoint, training loop, and optimizer-step policy."""

from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.amp.autocast_mode import autocast

from dataset.runtime import DataState, skip_batches
from model.config import load_model_config
from training.batch import move_batch_to_device
from training.builders import build_data_loaders, build_model, load_tokenizer
from training.checkpoint import CheckpointManager, unwrap_model
from training.cli import apply_cli_overrides, parse_pretraining_args
from training.config import load_pretraining_config, validate_pretraining_config
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
from training.paths import ExperimentPaths, resolve_experiment_root
from training.trainer_types import TrainerComponents, TrainerSettings
from training.validation import compute_perplexity


def _load_configs(args):
    model_config = load_model_config(args.model_config)
    config = load_pretraining_config(args.pretrain_config)
    model_config, config = apply_cli_overrides(model_config, config, args)
    validate_pretraining_config(config, model_config)
    return model_config, config


def _resolve_paths(args, model_config, config, runtime) -> ExperimentPaths:
    resume_path = None
    if args.resume is not None:
        resume_path = str(Path(args.resume).expanduser().resolve())
        if not Path(resume_path).is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {resume_path}")
    root = None
    if runtime.is_main:
        root = str(
            resolve_experiment_root(
                output_dir=config.training.output_dir,
                experiment_name=config.training.experiment_name,
                model_name=model_config.model_name,
                resume_path=resume_path,
            )
        )
    if runtime.distributed:
        roots = [root]
        torch.distributed.broadcast_object_list(roots, src=0)
        root = roots[0]
    if root is None:
        raise RuntimeError("experiment directory was not resolved")
    return ExperimentPaths.create(root)


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
            tokens_since_log = 0
            reader = self.train_loader.dataset.reader
            while self.global_step < self.settings.max_steps and not self.early_stopped:
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

                grad_norm, stepped, diagnostics, loss_sum, padded_tokens = (
                    self._train_micro_batches(
                        micro_batches,
                        global_tokens,
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
                global_loss_sum = self._global_sum_float(loss_sum)
                loss = global_loss_sum / global_tokens
                metrics = self.diagnostics.update_step(
                    model=self.model,
                    loss=loss,
                    grad_norm=grad_norm,
                    padded_tokens=padded_tokens,
                    step=self.global_step,
                )
                tokens_since_log += global_tokens
                if (
                    self.settings.train_log_interval > 0
                    and self.global_step % self.settings.train_log_interval == 0
                ):
                    self._log_training(
                        loss,
                        tokens_since_log,
                        grad_norm,
                        metrics,
                    )
                    tokens_since_log = 0

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
            stop_reason = "early_stopping" if self.early_stopped else "max_steps"
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
    ) -> tuple[float, bool, list[str], float, int]:
        loss_sum = 0.0
        padded_tokens = 0
        finite = True
        for index, raw_batch in enumerate(batches):
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
                    finite = finite and bool(
                        torch.isfinite(backward_loss.detach()).all().item()
                    )
                self.scaler.scale(backward_loss).backward()
            loss_sum += float(output.loss_sum.detach().float().item())
            padded_tokens += batch["input_ids"].numel()

        grad_norm, stepped, diagnostics = self._apply_optimizer_step(finite)
        return grad_norm, stepped, diagnostics, loss_sum, padded_tokens

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

    def _global_sum_float(self, value: float) -> float:
        if not torch.distributed.is_initialized():
            return float(value)
        total = torch.tensor(
            float(value),
            dtype=torch.float64,
            device=self.device,
        )
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        return float(total.item())

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
    ) -> None:
        now = time.time()
        elapsed = now - (
            self.start_time
            if self.last_log_time is None
            else self.last_log_time
        )
        self.last_log_time = now
        tokens_per_second = tokens / max(elapsed, 1e-6)
        learning_rate = self.optimizer.param_groups[0]["lr"]
        perplexity = compute_perplexity(loss)
        metrics = {
            "train/loss": loss,
            "train/lr": learning_rate,
            "train/tokens_per_sec": tokens_per_second,
            "train/grad_norm": grad_norm,
            "train/smoothed_loss": diagnostics.get("smoothed_loss", 0.0),
            "train/grad_norm_sq_variance_ratio": diagnostics.get(
                "grad_norm_sq_variance_ratio",
                0.0,
            ),
        }
        if perplexity is not None:
            metrics["train/perplexity"] = perplexity
        for name in ("mfu_average", "mfu_instant"):
            if name in diagnostics:
                metrics[f"train/{name}"] = diagnostics[name]
        ppl_text = "N/A" if perplexity is None else f"{perplexity:.2f}"
        mfu = diagnostics.get("mfu_average")
        self.log_manager.logger.info(
            f"Step {self.global_step} | loss={loss:.4f} | "
            f"smooth_loss={diagnostics.get('smoothed_loss', 0.0):.4f} | "
            f"ppl={ppl_text} | lr={learning_rate:.2e} | "
            f"tok/s={tokens_per_second:.0f} | "
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
                    sample["generated_text"],
                    step=self.global_step,
                )
                if self.swanlab_logger is not None:
                    self.swanlab_logger.log_text(
                        tag,
                        sample["generated_text"],
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


def run_pretraining(args=None) -> None:
    """Assemble and execute the canonical pretraining run."""
    configure_torch_runtime()
    args = parse_pretraining_args() if args is None else args
    model_config, config = _load_configs(args)
    runtime = initialize_runtime(
        seed=config.training.seed,
        timeout_minutes=config.training.distributed_timeout_minutes,
    )
    paths = _resolve_paths(args, model_config, config, runtime)
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
        log_manager.info(f"模型配置: {model_config.model_name}")
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

        base_model = build_model(model_config, runtime.device)
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
                        batch_size=config.data.batching.micro_batch_size,
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
        budget = StepBudget(
            target_tokens=config.training.target_tokens,
            micro_batch_size=config.data.batching.micro_batch_size,
            gradient_accumulation_steps=(
                config.training.gradient_accumulation_steps
            ),
            world_size=runtime.world_size,
            sequence_length=config.data.packing.max_seq_length,
        )
        scheduler, schedule = build_scheduler(
            optimizer,
            config.scheduler,
            budget.max_steps,
        )
        if runtime.is_main:
            log_manager.info(
                f"目标 Token 数: {budget.target_tokens:,}; "
                f"全局 batch: {budget.global_batch_size}; "
                f"每步 Token: {budget.tokens_per_step:,}; "
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


def main() -> None:
    run_pretraining()


__all__ = [
    "CausalLMTrainer",
    "TrainerComponents",
    "TrainerSettings",
    "main",
    "run_pretraining",
]


if __name__ == "__main__":
    main()
