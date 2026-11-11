#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TaiChu 语言模型预训练入口脚本。

功能：
    - 加载模型结构与训练超参配置（YAML）
    - 初始化 tokenizer、模型、数据集、日志、检查点管理器
    - 支持混合精度训练、梯度累积、warmup + cosine 学习率调度
    - 自动恢复训练、定期验证与保存最佳/最近检查点
    - 所有日志统一通过 LogManager 输出到 TensorBoard 及控制台/文件
    - 集成专业评估指标：
        * 训练阶段：平滑损失、梯度噪声尺度、模型 FLOPs 利用率（MFU）、表示稳定性
        * 验证阶段：Next Token 准确率、分位数交叉熵、空白提示困惑度（可选）

用法：
    # 单卡训练
    python ./trainer/train_pretrain.py --model_config configs/TaiChu_Small_80M.yaml --pretrain_config configs/train_pretrain_config.yaml

    # 多卡训练（通过 torchrun 启动）
    torchrun --nproc_per_node=4 train_pretrain.py --model_config configs/TaiChu_125m.yaml --pretrain_config configs/pretrain_config.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

from model.model import TaiChuModel
from tokenizers import Tokenizer
from typing import Dict, List, Optional

from utils.config_loader import load_model_config, load_pretrain_config, parse_tokens_string
from utils.train_utils import (
    compile_model_for_training,
    get_device,
    get_experiment_dir,
    get_wsd_lr_lambda,
    set_seed,
    training_compile_preflight,
)
from utils.logger import LogManager
from utils.swanlab_logger import SwanLabLogger
from utils.checkpoint import (
    CheckpointManager,
    resolve_experiment_dir_from_checkpoint,
    unwrap_model,
)
from utils.model_utils import ModelInspector
from dataset.runtime import UnifiedDataLoaderFactory as TaiChuDataLoaderFactory
from dataset.runtime import DataState, skip_batches
from utils.metrics import EvaluationMetricsManager, ValidationMetrics


class TaiChuTrainer:
    """TaiChu 预训练器。

    封装训练循环、验证、检查点、日志等功能。支持混合精度、梯度累积、分布式训练，
    并集成专业评估指标。

    Attributes:
        model: 模型实例（可能是 DDP 封装）。
        tokenizer: Tokenizer 实例。
        train_loader: 训练数据加载器。
        val_loader: 验证数据加载器。
        log_manager: 日志管理器。
        checkpoint_manager: 检查点管理器。
        optimizer: 优化器。
        scheduler: 学习率调度器。
        scaler: 混合精度梯度缩放器。
        global_step: 当前全局步数。
        best_val_loss: 最佳验证损失。
        early_stopping: 早停实例（可选）。
        eval_metrics: 训练阶段评估指标管理器。
    """

    MAX_CONSECUTIVE_NONFINITE_SKIPS = 10

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        log_manager: LogManager,
        checkpoint_manager: CheckpointManager,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        scaler: GradScaler,
        gradient_accumulation_steps: int = 1,
        init_step: int = 0,
        max_steps: int = 100000,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        train_log_interval: int = 10,
        eval_log_interval: int = 10,
        early_stop_enabled: bool = False,
        early_stop_monitor: str = "val_loss",
        early_stop_patience: int = 5,
        early_stop_min_delta: float = 1e-4,
        early_stop_mode: str = "min",
        swanlab_logger: Optional[SwanLabLogger] = None,
        device: torch.device = torch.device("cpu"),
        use_amp: bool = False,
        dtype: str = "bfloat16",
        local_rank: int = 0,
        max_seq_len: int = 2048,
        global_rank: int = 0,
        peak_flops: float = 0.0,
        generation_prompts: Optional[List[str]] = None,
        num_generate_tokens: int = 50,
        max_grad_norm: float = 1.0,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        compute_empty_prefix_eos_ppl: bool = False,
        data_state: DataState | None = None,
        compiled_model: bool = False,
    ):
        """初始化训练器。

        Args:
            model: 模型实例（可能是 DDP 封装）。
            tokenizer: Tokenizer 实例。
            train_loader: 训练数据加载器。
            val_loader: 验证数据加载器。
            log_manager: 日志管理器。
            checkpoint_manager: 检查点管理器。
            optimizer: 优化器。
            scheduler: 学习率调度器。
            scaler: 混合精度梯度缩放器。
            gradient_accumulation_steps: 梯度累积步数。
            init_step: 恢复训练时的起始全局步数。
            max_steps: 总训练步数。
            eval_interval: 每隔多少步进行一次验证。
            save_interval: 每隔多少步保存一次常规检查点。
            train_log_interval: 每隔多少步记录一次训练日志。
            eval_log_interval: 每隔多少步记录一次评估日志。
            early_stop_enabled: 是否启用早停机制。
            early_stop_monitor: 监控的验证指标名称（如 'val_loss'）。
            early_stop_patience: 早停耐心值，连续无改善的验证次数上限。
            early_stop_min_delta: 判断是否改善的最小绝对变化量。
            early_stop_mode: 指标优化模式，'min' 表示越小越好，'max' 表示越大越好。
            swanlab_logger: SwanLab 日志记录器实例（仅主进程启用）。
            device: 当前设备。
            use_amp: 是否启用混合精度。
            dtype: 混合精度数据类型（'bfloat16' 或 'float16'）。
            local_rank: 当前进程的本地 Rank（用于设备设置）。
            max_seq_len: 最大序列长度，用于计算 tokens/s。
            global_rank: 全局 Rank（0 为主进程）。
            peak_flops: GPU 理论峰值算力（FLOPs/秒），用于 MFU 计算。
            generation_prompts: 测试 Prompts 列表。
            num_generate_tokens: 测试生成 token 长度。
            max_grad_norm: 梯度裁剪的最大范数。
            bos_token_id: BOS Token ID，用于可选空 Prompt 诊断。
            eos_token_id: EOS Token ID，用于可选空 Prompt 诊断。
            compute_empty_prefix_eos_ppl: 是否记录显式 BOS -> EOS 诊断。
            compiled_model: 模型是否已通过 torch.compile 编译。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_manager = log_manager
        self.checkpoint_manager = checkpoint_manager
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.nonfinite_skip_count = 0
        self.consecutive_nonfinite_skips = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if max_grad_norm < 0:
            raise ValueError("max_grad_norm cannot be negative")
        self.world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        self.max_steps = max_steps
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.train_log_interval = train_log_interval
        self.eval_log_interval = eval_log_interval
        self.generation_prompts = generation_prompts
        self.num_generate_tokens = num_generate_tokens
        self.max_grad_norm = max_grad_norm
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.compute_empty_prefix_eos_ppl = compute_empty_prefix_eos_ppl
        self.compiled_model = compiled_model
        self.data_state = data_state
        self.swanlab_logger = swanlab_logger
        self.device = device
        self.use_amp = use_amp
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.scaler_enabled = bool(use_amp and self.dtype == torch.float16)
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.max_seq_len = max_seq_len

        # 训练状态
        self.global_step = init_step
        self.best_val_loss = float("inf")

        # 早停初始化
        self.early_stopping = None
        if early_stop_enabled:
            from utils.early_stopping import EarlyStopping
            self.early_stopping = EarlyStopping(
                monitor=early_stop_monitor,
                patience=early_stop_patience,
                min_delta=early_stop_min_delta,
                mode=early_stop_mode,
            )
        self.early_stopped = False

        # 日志时间窗口
        self.start_time = 0.0
        self.last_log_time = None

        # ========== 集成训练阶段专业评估指标 ==========
        # 创建评估指标管理器，配置平滑系数和梯度窗口
        self.eval_metrics = EvaluationMetricsManager(config={
            "loss_beta": 0.9,        # 平滑损失系数
            "gns_window": 100        # 梯度噪声尺度滑动窗口
        })
        if peak_flops > 0:
            # 初始化 MFU 计算器（峰值算力可用 --peak_flops 显式指定）
            self.eval_metrics.initialize_mfu(
                peak_flops=peak_flops,
                log_file=os.path.join(log_manager.log_dir, "mfu_log.jsonl")
            )
        # 获取原始模型（处理 DDP 包装），用于表示稳定性钩子
        raw_model = unwrap_model(model)
        self.eval_metrics.initialize_repr_stability(raw_model, device, hook_layer=-1)
        # =================================================

    def train(self) -> None:
        """执行主训练循环。"""
        try:
            self.model.train()
            self.start_time = time.time()
            self.log_manager.info("训练开始")

            # 数据迭代器
            data_epoch = 0 if self.data_state is None else self.data_state.epoch
            batch_index = 0 if self.data_state is None else self.data_state.batch_index
            consumed_samples = (
                0 if self.data_state is None else self.data_state.consumed_samples
            )
            dataset_reader = getattr(
                getattr(self.train_loader, "dataset", None),
                "reader",
                None,
            )
            if dataset_reader is None:
                raise RuntimeError("training loader does not expose unified Storage Reader")
            if self.data_state is not None and not self.data_state.compatible_with(
                dataset_reader.dataset_id,
                dataset_reader.version,
            ):
                raise ValueError(
                    "checkpoint DataState does not match the configured dataset"
                )
            if hasattr(self.train_loader, "set_epoch"):
                self.train_loader.set_epoch(data_epoch)
            fast_resume = (
                batch_index > 0
                and hasattr(self.train_loader, "set_resume_batch")
            )
            if fast_resume:
                self.log_manager.info(
                    "数据恢复位置: "
                    f"epoch={data_epoch}, batch_index={batch_index}"
                )
                self.train_loader.set_resume_batch(batch_index)
            train_iter = iter(self.train_loader)
            if batch_index and not fast_resume:
                train_iter = skip_batches(train_iter, batch_index)
            tokens_since_log = 0

            while self.global_step < self.max_steps and not self.early_stopped:
                micro_batches = []
                local_loss_tokens = 0
                for _ in range(self.gradient_accumulation_steps):
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        data_epoch += 1
                        batch_index = 0
                        if hasattr(self.train_loader, "set_epoch"):
                            self.train_loader.set_epoch(data_epoch)
                        train_iter = iter(self.train_loader)
                        try:
                            batch = next(train_iter)
                        except StopIteration as exc:
                            raise RuntimeError(
                                "training data pipeline produced no batches"
                            ) from exc

                    batch_index += 1
                    batch_sample_ids = batch.get("sample_ids")
                    consumed_samples += (
                        len(batch_sample_ids)
                        if batch_sample_ids is not None
                        else int(batch["input_ids"].size(0))
                    )
                    group_data_state = DataState(
                        dataset_id=dataset_reader.dataset_id,
                        dataset_version=dataset_reader.version,
                        epoch=data_epoch,
                        batch_index=batch_index,
                        consumed_samples=consumed_samples,
                    )
                    micro_batches.append(batch)
                    local_loss_tokens += int(
                        (batch["labels"][..., 1:] != -100).sum().item()
                    )

                if local_loss_tokens <= 0:
                    raise RuntimeError("training batch contains no supervised tokens")
                global_loss_tokens = self._global_sum(local_loss_tokens)
                if global_loss_tokens <= 0:
                    raise RuntimeError(
                        "distributed training batch contains no supervised tokens"
                    )

                accumulated_loss_sum = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=self.device,
                )
                padded_tokens_processed = 0
                loss_finite_flag = torch.ones(
                    (),
                    dtype=torch.bool,
                    device=self.device,
                )
                for micro_index, batch in enumerate(micro_batches):
                    input_ids = batch["input_ids"].to(
                        self.device,
                        non_blocking=self.device.type == "cuda",
                    )
                    labels = batch["labels"].to(
                        self.device,
                        non_blocking=self.device.type == "cuda",
                    )
                    attention_mask = batch.get("attention_mask")
                    if attention_mask is not None:
                        if batch.get("attention_mask_is_right_padded", False):
                            # Right padding is causally invisible to every valid
                            # query, so native causal SDPA is equivalent.
                            attention_mask = None
                        else:
                            attention_mask = attention_mask.to(
                                self.device,
                                non_blocking=self.device.type == "cuda",
                            )
                    segment_ids = batch.get("segment_ids")
                    if segment_ids is not None:
                        segment_ids = segment_ids.to(
                            self.device,
                            non_blocking=self.device.type == "cuda",
                        )
                    position_ids = batch.get("position_ids")
                    if position_ids is not None:
                        position_ids = position_ids.to(
                            self.device,
                            non_blocking=self.device.type == "cuda",
                        )

                    if self.compiled_model:
                        torch.compiler.cudagraph_mark_step_begin()

                    use_no_sync = (
                        self.world_size > 1
                        and micro_index < len(micro_batches) - 1
                        and hasattr(self.model, "no_sync")
                    )
                    sync_context = (
                        self.model.no_sync()
                        if use_no_sync
                        else nullcontext()
                    )
                    with sync_context:
                        with autocast(
                            device_type=self.device.type,
                            enabled=self.use_amp,
                            dtype=self.dtype,
                        ):
                            out = self.model(
                                input_ids,
                                attention_mask=attention_mask,
                                segment_ids=segment_ids,
                                position_ids=position_ids,
                                labels=labels,
                                return_logits=False,
                                validate_inputs=False,
                            )
                            if out.loss_sum is None:
                                raise RuntimeError(
                                    "model did not return a loss sum"
                                )
                            # DDP averages gradients across ranks. Scaling by
                            # world_size yields a global token-weighted mean.
                            backward_loss = (
                                out.loss_sum
                                * (self.world_size / global_loss_tokens)
                            )
                            if out.aux_loss is not None:
                                # Current MoE auxiliary losses are means over
                                # routed tokens in this forward. Keep them
                                # separate from CE and token-weight them for
                                # gradient accumulation and DDP averaging.
                                backward_loss = (
                                    backward_loss
                                    + out.aux_loss
                                    * (
                                        self.world_size
                                        * input_ids.numel()
                                        / global_loss_tokens
                                    )
                                )
                            loss_finite_flag = loss_finite_flag & torch.isfinite(
                                backward_loss.detach()
                            ).all()
                        self.scaler.scale(backward_loss).backward()

                    accumulated_loss_sum += out.loss_sum.detach().float()
                    padded_tokens_processed += input_ids.numel()

                loss_is_finite = bool(loss_finite_flag.item())
                grad_norm_value, stepped, diagnostics = (
                    self._apply_optimizer_step(
                        loss_is_finite,
                    )
                )
                if not stepped:
                    self.nonfinite_skip_count += 1
                    self.consecutive_nonfinite_skips += 1
                    self.data_state = group_data_state
                    if diagnostics:
                        reason = (
                            "local_nonfinite_gradients=["
                            + ", ".join(diagnostics)
                            + "]"
                        )
                    elif not loss_is_finite:
                        reason = "local_loss_is_nonfinite"
                    elif not math.isfinite(grad_norm_value):
                        reason = "local_gradient_norm_is_nonfinite"
                    else:
                        reason = (
                            "local_state_is_finite_"
                            "another_rank_rejected_update"
                        )
                    self.log_manager.warning(
                        "检测到非有限训练状态，已原子跳过优化器更新: "
                        f"candidate_step={self.global_step + 1}, "
                        f"rank={self.global_rank}, "
                        f"loss_finite={loss_is_finite}, "
                        f"grad_norm={grad_norm_value}, "
                        f"consecutive_skips={self.consecutive_nonfinite_skips}, "
                        f"reason={reason}"
                    )
                    if (
                        self.consecutive_nonfinite_skips
                        >= self.MAX_CONSECUTIVE_NONFINITE_SKIPS
                    ):
                        raise RuntimeError(
                            "training produced non-finite gradients for "
                            f"{self.consecutive_nonfinite_skips} consecutive "
                            "optimizer steps; stopped after atomically "
                            "skipping every update"
                        )
                    continue

                self.consecutive_nonfinite_skips = 0
                self.eval_metrics.update_gradient_norm(grad_norm_value)
                self.global_step += 1
                self.data_state = group_data_state

                global_loss_sum = self._global_sum_float(
                    float(accumulated_loss_sum.item())
                )
                avg_loss = global_loss_sum / global_loss_tokens
                self.eval_metrics.update_loss(avg_loss)
                if self.eval_metrics.mfu is not None:
                    step_flops = (
                        self.eval_metrics.mfu.estimate_model_flops_for_tokens(
                            self.model,
                            padded_tokens_processed,
                        )
                    )
                else:
                    step_flops = 0.0
                self.eval_metrics.update_step(step_flops, self.global_step)

                tokens_since_log += global_loss_tokens
                if (
                    self.train_log_interval > 0
                    and self.global_step % self.train_log_interval == 0
                ):
                    self._log_training(
                        avg_loss,
                        tokens_since_log,
                        grad_norm_value,
                    )
                    tokens_since_log = 0

                val_loss = None

                checkpoint_saved_at_step = False
                if (
                    self.global_rank == 0
                    and self.save_interval > 0
                    and self.global_step % self.save_interval == 0
                ):
                    self._save_training_checkpoint(
                        metrics={"val_loss": self.best_val_loss},
                    )
                    checkpoint_saved_at_step = True

                if (
                    self.eval_interval > 0
                    and self.global_step % self.eval_interval == 0
                ):
                    if not checkpoint_saved_at_step:
                        self._save_training_checkpoint(
                            metrics={"val_loss": self.best_val_loss},
                        )
                    self.log_manager.info("验证开始")
                    val_loss = self._evaluate(
                        generate_prompts=self.generation_prompts,
                        num_generate_tokens=self.num_generate_tokens,
                    )
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss

                    if self.global_rank == 0:
                        self.checkpoint_manager.save_best(
                            self.model,
                            metrics={"val_loss": val_loss},
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            scaler=self.scaler,
                            global_step=self.global_step,
                            data_state=self.data_state.to_dict(),
                            **self._checkpoint_swanlab_state(),
                        )
                    self.log_manager.info(
                        f"当前最佳验证损失: {self.best_val_loss:.4f}"
                    )

                    if self.early_stopping is not None:
                        stop = self.early_stopping.step(val_loss, self.global_step)
                        if stop:
                            self.log_manager.info(
                                f"早停触发于步数 {self.global_step}，"
                                f"最佳分数 {self.early_stopping.best_score:.4f} "
                                f"于步数 {self.early_stopping.best_step}"
                            )
                            self.early_stopped = True
                            if self.global_rank == 0:
                                self.checkpoint_manager.save(
                                    f"checkpoints/early_stop_step{self.global_step}.pt",
                                    model=self.model,
                                    optimizer=self.optimizer,
                                    scheduler=self.scheduler,
                                    scaler=self.scaler,
                                    global_step=self.global_step,
                                    metrics={"val_loss": val_loss},
                                    early_stopping_state=self.early_stopping.state_dict(),
                                    data_state=self.data_state.to_dict(),
                                    **self._checkpoint_swanlab_state(),
                                )
                            break

            # 训练结束，保存最终模型
            if self.global_rank == 0:
                self.checkpoint_manager.save(
                    "final_models/final_model.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    global_step=self.global_step,
                    data_state=(
                        None if self.data_state is None else self.data_state.to_dict()
                    ),
                    **self._checkpoint_swanlab_state(),
                )
            self.log_manager.info("训练完成!")
            if self.swanlab_logger:
                self.swanlab_logger.finish()
            self.log_manager.close()
        except KeyboardInterrupt:
            self.log_manager.info("训练被用户中断 (Ctrl+C)，正在保存检查点并退出...")
            self._save_interrupted_checkpoint()
        except Exception as e:
            self.log_manager.error(f"训练过程中发生异常: {type(e).__name__}: {e}")
            self.log_manager.error("正在保存紧急检查点...", exc_info=True)
            self._save_interrupted_checkpoint()
            raise
        finally:
            if self.swanlab_logger:
                self.swanlab_logger.finish()
            self.log_manager.close()
            self.eval_metrics.close()   # 释放钩子资源

    def _save_interrupted_checkpoint(self) -> None:
        """保存中断/异常时的检查点。"""
        if self.global_rank != 0:
            return
        extra = {}
        if self.early_stopping is not None:
            extra["early_stopping_state"] = self.early_stopping.state_dict()
        extra["data_state"] = (
            None if self.data_state is None else self.data_state.to_dict()
        )
        extra.update(self._checkpoint_swanlab_state())
        self.checkpoint_manager.save(
            f"checkpoints/checkpoint-emergency-step{self.global_step}.pt",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            global_step=self.global_step,
            metrics={"val_loss": self.best_val_loss},
            **extra,
        )
        self.log_manager.info(f"紧急检查点已保存至 step {self.global_step}")

    def _save_training_checkpoint(self, metrics: Optional[Dict[str, float]]) -> None:
        """Save resumable state before a validation phase can be interrupted."""
        if self.global_rank != 0:
            return
        extra = {"data_state": self.data_state.to_dict()}
        if self.early_stopping is not None:
            extra["early_stopping_state"] = self.early_stopping.state_dict()
        extra.update(self._checkpoint_swanlab_state())
        self.checkpoint_manager.save(
            f"checkpoints/checkpoint-{self.global_step}.pt",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            global_step=self.global_step,
            metrics=metrics,
            **extra,
        )
        self.log_manager.info(
            f"验证前检查点已保存至 step {self.global_step}"
        )

    def _checkpoint_swanlab_state(self) -> Dict[str, str]:
        """Return SwanLab identity metadata stored with each checkpoint."""
        if self.swanlab_logger is None or not self.swanlab_logger.run_id:
            return {}
        state = {"swanlab_run_id": self.swanlab_logger.run_id}
        if self.swanlab_logger.project:
            state["swanlab_project"] = self.swanlab_logger.project
        if self.swanlab_logger.workspace:
            state["swanlab_workspace"] = self.swanlab_logger.workspace
        return state

    def _global_sum(self, value: int) -> int:
        """Sum a scalar across ranks without changing the compile hot path."""
        if not torch.distributed.is_initialized():
            return int(value)
        total = torch.tensor(float(value), dtype=torch.float64, device=self.device)
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        return int(total.item())

    def _global_sum_float(self, value: float) -> float:
        """Sum a floating-point scalar across ranks."""
        if not torch.distributed.is_initialized():
            return float(value)
        total = torch.tensor(float(value), dtype=torch.float64, device=self.device)
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        return float(total.item())

    def _all_ranks_finite(self, finite: bool) -> bool:
        """Return whether every rank agreed that the training state is finite."""
        if not torch.distributed.is_initialized():
            return bool(finite)
        finite_flag = torch.tensor(
            1 if finite else 0,
            dtype=torch.int32,
            device=self.device,
        )
        torch.distributed.all_reduce(
            finite_flag,
            op=torch.distributed.ReduceOp.MIN,
        )
        return bool(finite_flag.item())

    def _nonfinite_gradient_diagnostics(self) -> List[str]:
        """Describe a bounded sample of parameters whose gradients are invalid."""
        diagnostics = []
        for name, parameter in self.model.named_parameters():
            grad = parameter.grad
            if grad is None:
                continue
            finite_mask = torch.isfinite(grad)
            if finite_mask.all().item():
                continue
            nonfinite_count = int((~finite_mask).sum().item())
            diagnostics.append(
                f"{name}: {nonfinite_count}/{grad.numel()} elements"
            )
            if len(diagnostics) >= 8:
                diagnostics.append("...")
                break
        return diagnostics

    def _gradient_total_norm(self) -> torch.Tensor:
        """Compute the L2 gradient norm in FP32 without mutating gradients."""
        norms = [
            torch.linalg.vector_norm(
                parameter.grad.detach().float(),
                ord=2,
            )
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        if not norms:
            return torch.zeros(
                (),
                dtype=torch.float32,
                device=self.device,
            )
        return torch.linalg.vector_norm(torch.stack(norms), ord=2)

    def _clip_gradients_with_norm(self, total_norm: torch.Tensor) -> None:
        """Apply the standard max-norm coefficient after finite validation."""
        clip_coefficient = torch.clamp(
            self.max_grad_norm / (total_norm + 1e-6),
            max=1.0,
        )
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(
                    clip_coefficient.to(dtype=parameter.grad.dtype)
                )

    def _apply_optimizer_step(
        self,
        loss_finite: bool,
    ) -> tuple[float, bool, List[str]]:
        """Clip and update atomically only when every rank accepts the update.

        A non-finite loss or gradient is never passed to AdamW. All ranks must
        make the same decision before optimizer, scaler, or scheduler state is
        mutated, otherwise a rank-local anomaly can desynchronize DDP training.
        """
        if self.scaler_enabled:
            self.scaler.unscale_(self.optimizer)
        grad_norm = self._gradient_total_norm()
        grad_norm_value = float(grad_norm.detach().float().item())
        update_finite = self._all_ranks_finite(
            loss_finite and math.isfinite(grad_norm_value)
        )

        if not update_finite:
            diagnostics = self._nonfinite_gradient_diagnostics()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler_enabled:
                self.scaler.update()
            return grad_norm_value, False, diagnostics

        self._clip_gradients_with_norm(grad_norm)
        if self.scaler_enabled:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm_value, True, []

    def _log_training(
        self,
        loss: float,
        tokens: int,
        grad_norm: float,
    ) -> None:
        """记录训练指标到 TensorBoard 和日志文件。

        计算 tokens_per_sec 时，使用本次日志与上一次日志之间的时间窗口，
        以反映瞬时训练吞吐量，避免因全程累计时间而导致吞吐量持续下降的假象。

        Args:
            loss: 当前参数更新步的平均损失（已排除梯度累积的缩放）。
            tokens: 上次日志以来全局有效监督 token 数。
            grad_norm: 当前优化步裁剪前的全局梯度范数。
        """
        # 学习率
        lr = self.optimizer.param_groups[0]["lr"]

        # 困惑度
        ppl = ValidationMetrics.compute_perplexity(loss)
        ppl_text = "N/A" if ppl is None else f"{ppl:.2f}"

        # 瞬时 tokens/s
        current_time = time.time()
        if self.last_log_time is None:
            elapsed = current_time - self.start_time
        else:
            elapsed = current_time - self.last_log_time

        tokens_per_sec = tokens / max(elapsed, 1e-6)

        self.last_log_time = current_time

        # 获取训练阶段评估指标
        extra_metrics = self.eval_metrics.get_all_metrics(include_repr=False)

        # 构造完整指标字典（用于 SwanLab 和 TensorBoard）
        metrics = {
            "train/loss": loss,
            "train/lr": lr,
            "train/tokens_per_sec": tokens_per_sec,
            "train/grad_norm": grad_norm,
            "train/smoothed_loss": extra_metrics.get("smoothed_loss", 0.0),
            "train/grad_norm_sq_variance_ratio": extra_metrics.get(
                "grad_norm_sq_variance_ratio",
                0.0,
            ),
        }
        if ppl is not None:
            metrics["train/perplexity"] = ppl
        if "mfu_average" in extra_metrics:
            metrics["train/mfu_average"] = extra_metrics["mfu_average"]
        if "mfu_instant" in extra_metrics:
            metrics["train/mfu_instant"] = extra_metrics["mfu_instant"]

        # 构建控制台日志（包含关键指标）
        mfu_average = extra_metrics.get("mfu_average")
        mfu_text = (
            "N/A" if mfu_average is None else f"{mfu_average:.3f}"
        )
        log_line = (
            f"Step {self.global_step} | "
            f"loss={loss:.4f} | smooth_loss={extra_metrics.get('smoothed_loss', 0):.4f} | "
            f"ppl={ppl_text} | lr={lr:.2e} | tok/s={tokens_per_sec:.0f} | "
            f"grad_ratio={extra_metrics.get('grad_norm_sq_variance_ratio', 0):.2f} | "
            f"mfu={mfu_text}"
        )
        self.log_manager.logger.info(log_line)

        # 记录到 SwanLab 和 TensorBoard（LogManager 内部已支持 TensorBoard）
        if self.swanlab_logger:
            self.swanlab_logger.log_metrics(metrics, step=self.global_step)
        self.log_manager.log_metrics(metrics, step=self.global_step)

    @torch.no_grad()
    def _evaluate(self, generate_prompts: Optional[List[str]] = None, num_generate_tokens: int = 50) -> float:
        """在验证集上计算平均损失，并计算扩展指标（准确率、分位数 CE 等）。

        Args:
            generate_prompts: 用于生成测试的提示词列表，若为 None 则使用默认提示。
            num_generate_tokens: 每个提示生成的 token 数量。

        Returns:
            平均损失值（所有 token 的平均交叉熵）。
        """
        self.model.eval()
        total_nll_nats = 0.0
        total_tokens = 0
        total_correct = 0
        total_bpb_nll_nats = 0.0
        total_bpb_bytes = 0
        max_token_loss = 0.0
        local_token_losses = []

        # 生成测试的提示词列表
        if generate_prompts is None:
            generate_prompts = []

        batch_cnt = 0
        accum_nll = 0.0
        accum_tokens = 0
        try:
            val_batch_count = len(self.val_loader)
        except TypeError:
            val_batch_count = None

        with torch.compiler.set_stance("force_eager"):  # 临时回退到 eager 模式，完全避免 CUDA Graph 问题
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(
                    self.device,
                    non_blocking=self.device.type == "cuda",
                )
                labels = batch["labels"].to(
                    self.device,
                    non_blocking=self.device.type == "cuda",
                )
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    if batch.get("attention_mask_is_right_padded", False):
                        attention_mask = None
                    else:
                        attention_mask = attention_mask.to(
                            self.device,
                            non_blocking=self.device.type == "cuda",
                        )
                segment_ids = batch.get("segment_ids")
                if segment_ids is not None:
                    segment_ids = segment_ids.to(
                        self.device,
                        non_blocking=self.device.type == "cuda",
                    )
                position_ids = batch.get("position_ids")
                if position_ids is not None:
                    position_ids = position_ids.to(
                        self.device,
                        non_blocking=self.device.type == "cuda",
                    )

                with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.dtype):
                    out = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        segment_ids=segment_ids,
                        position_ids=position_ids,
                        labels=labels,
                        return_logits=False,
                        validate_inputs=False,
                        return_token_statistics=True,
                    )

                if out.loss is None or not torch.isfinite(out.loss):
                    raise RuntimeError("model returned a non-finite validation loss")
                if out.token_losses is None or out.correct_predictions is None:
                    raise RuntimeError(
                        "model did not return token statistics for validation"
                    )

                stats = ValidationMetrics.compute_token_statistics_from_losses(
                    out.token_losses,
                    labels,
                    correct_count=int(out.correct_predictions.item()),
                    tokenizer=self.tokenizer,
                )
                total_nll_nats += stats.nll_nats
                total_tokens += stats.token_count
                total_correct += stats.correct_count
                total_bpb_nll_nats += stats.bpb_nll_nats
                total_bpb_bytes += stats.bpb_bytes
                max_token_loss = max(max_token_loss, stats.max_token_loss)
                if stats.token_losses.numel() > 0:
                    local_token_losses.append(stats.token_losses)

                accum_nll += stats.nll_nats
                accum_tokens += stats.token_count
                batch_cnt += 1

                if self.global_rank == 0 and self.eval_log_interval > 0 and batch_cnt % self.eval_log_interval == 0:
                    local_avg_loss = (
                        accum_nll / accum_tokens
                        if accum_tokens > 0
                        else None
                    )
                    local_ppl = ValidationMetrics.compute_perplexity(
                        local_avg_loss
                    )
                    loss_text = (
                        "N/A"
                        if local_avg_loss is None
                        else f"{local_avg_loss:.4f}"
                    )
                    ppl_text = (
                        "N/A"
                        if local_ppl is None
                        else f"{local_ppl:.2f}"
                    )
                    self.log_manager.logger.info(
                        f"[Eval] Step {self.global_step} "
                        f"(batch {batch_cnt}/{val_batch_count or '?'}) | "
                        f"loss={loss_text} | ppl={ppl_text}"
                    )
                    accum_nll = 0.0
                    accum_tokens = 0

        if batch_cnt == 0:
            raise RuntimeError("validation data pipeline produced no batches")

        distributed = torch.distributed.is_initialized()
        if distributed:
            totals = torch.tensor(
                [
                    total_nll_nats,
                    float(total_tokens),
                    float(total_correct),
                    total_bpb_nll_nats,
                    float(total_bpb_bytes),
                    float(batch_cnt),
                ],
                dtype=torch.float64,
                device=self.device,
            )
            torch.distributed.all_reduce(
                totals,
                op=torch.distributed.ReduceOp.SUM,
            )
            (
                total_nll_nats,
                total_tokens_float,
                total_correct_float,
                total_bpb_nll_nats,
                total_bpb_bytes_float,
                total_batches_float,
            ) = totals.tolist()
            total_tokens = int(total_tokens_float)
            total_correct = int(total_correct_float)
            total_bpb_bytes = int(total_bpb_bytes_float)
            total_batches = int(total_batches_float)
            max_loss_device = torch.tensor(
                max_token_loss,
                dtype=torch.float64,
                device=self.device,
            )
            torch.distributed.all_reduce(
                max_loss_device,
                op=torch.distributed.ReduceOp.MAX,
            )
            max_token_loss = float(max_loss_device.item())
        else:
            total_batches = batch_cnt

        if total_tokens <= 0:
            raise RuntimeError(
                "validation produced no supervised next-token positions"
            )
        avg_loss = total_nll_nats / total_tokens
        avg_ppl = ValidationMetrics.compute_perplexity(avg_loss)
        avg_accuracy = total_correct / total_tokens
        val_bpb = ValidationMetrics.compute_bits_per_byte(
            total_bpb_nll_nats,
            total_bpb_bytes,
        )

        local_losses = (
            torch.cat(local_token_losses)
            if local_token_losses
            else torch.empty(0, dtype=torch.float32)
        )
        del local_token_losses
        if distributed:
            gathered_losses = (
                [None] * torch.distributed.get_world_size()
                if self.global_rank == 0
                else None
            )
            torch.distributed.gather_object(
                local_losses,
                object_gather_list=gathered_losses,
                dst=0,
            )
            percentile_error = None
            if self.global_rank == 0:
                try:
                    exact_losses = torch.cat(gathered_losses)
                    percentile_metrics = ValidationMetrics.compute_percentile_ce(
                        exact_losses,
                        percentiles=[0.5, 0.9, 0.95],
                    )
                except Exception as exc:
                    percentile_error = f"{type(exc).__name__}: {exc}"
                    percentile_metrics = {
                        "ce_p50": None,
                        "ce_p90": None,
                        "ce_p95": None,
                    }
            else:
                percentile_metrics = {
                    "ce_p50": None,
                    "ce_p90": None,
                    "ce_p95": None,
                }
            status = [percentile_error]
            torch.distributed.broadcast_object_list(status, src=0)
            torch.distributed.barrier()
            if status[0] is not None:
                raise RuntimeError(
                    f"distributed percentile aggregation failed: {status[0]}"
                )
        else:
            percentile_metrics = ValidationMetrics.compute_percentile_ce(
                local_losses,
                percentiles=[0.5, 0.9, 0.95],
            )

        empty_prefix_metrics = {
            "empty_prefix_eos_ce": None,
            "empty_prefix_eos_ppl": None,
        }
        if self.compute_empty_prefix_eos_ppl:
            raw_model = unwrap_model(self.model)
            empty_prefix_metrics = (
                ValidationMetrics.compute_empty_prefix_eos_metrics(
                    raw_model,
                    bos_token_id=self.bos_token_id,
                    eos_token_id=self.eos_token_id,
                    device=self.device,
                )
            )

        if self.global_rank == 0:
            self.log_manager.set_step(self.global_step)
            ppl_text = "N/A" if avg_ppl is None else f"{avg_ppl:.2f}"
            bpb_text = "N/A" if val_bpb is None else f"{val_bpb:.4f}"
            log_msg = (
                f"Step {self.global_step} - val/loss={avg_loss:.4f} | "
                f"val/ppl={ppl_text} | val/bpb={bpb_text} | "
                f"val/acc={avg_accuracy:.4f} | val/tokens={total_tokens} | "
                f"val/ce_max={max_token_loss:.4f} | "
                f"val/ce_p50={percentile_metrics['ce_p50']:.4f} | "
                f"val/ce_p90={percentile_metrics['ce_p90']:.4f}"
            )
            empty_prefix_ppl = empty_prefix_metrics[
                "empty_prefix_eos_ppl"
            ]
            if empty_prefix_ppl is not None:
                log_msg += f" | val/empty_prefix_eos_ppl={empty_prefix_ppl:.2f}"
            self.log_manager.logger.info(log_msg)

            val_metrics = {
                "val/loss": avg_loss,
                "val/accuracy": avg_accuracy,
                "val/tokens": total_tokens,
                "val/target_bytes": total_bpb_bytes,
                "val/batches": total_batches,
                "val/ce_max_exact": max_token_loss,
                **{f"val/{k}": v for k, v in percentile_metrics.items()},
            }
            if avg_ppl is not None:
                val_metrics["val/perplexity"] = avg_ppl
            if val_bpb is not None:
                val_metrics["val/bpb"] = val_bpb
            for name, value in empty_prefix_metrics.items():
                if value is not None:
                    val_metrics[f"val/{name}"] = value
            if self.swanlab_logger:
                self.swanlab_logger.log_metrics(val_metrics, step=self.global_step)
            self.log_manager.log_metrics(val_metrics, step=self.global_step)

        # 生成测试文本（与原有逻辑一致）
        if self.global_rank == 0 and generate_prompts:
            raw_model = unwrap_model(self.model)
            raw_model.eval()
            with torch.compiler.set_stance("force_eager"):
                for idx, prompt in enumerate(generate_prompts):
                    encoding = self.tokenizer.encode(prompt, add_special_tokens=False)
                    prompt_ids = encoding.ids
                    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
                    generated_ids = raw_model.generate(
                        input_ids=input_tensor,
                        max_new_tokens=num_generate_tokens,
                        temperature=0.8,
                        top_k=50,
                    )
                    generated_text = self.tokenizer.decode(generated_ids[0].tolist())
                    self.log_manager.set_step(self.global_step)
                    self.log_manager.log_text(f"generation/prompt_{idx}", generated_text)
                    self.log_manager.info(
                        f"[生成测试] 提示: {prompt}\n生成结果: {generated_text}\n{'-' * 60}"
                    )
                    if self.swanlab_logger:
                        self.swanlab_logger.log_text(
                            f"generation/prompt_{idx}", generated_text, step=self.global_step
                        )

        if torch.distributed.is_initialized() and generate_prompts:
            torch.distributed.barrier()
        self.model.train()
        return avg_loss


def _warmup_compiled_training(
    model: nn.Module,
    *,
    device: torch.device,
    vocab_size: int,
    batch_size: int,
    sequence_length: int,
    use_amp: bool,
    dtype: torch.dtype,
) -> None:
    """Compile the forward/backward graph before DDP training starts."""
    if batch_size <= 0 or sequence_length < 2:
        raise ValueError("compile warmup requires a non-empty training shape")
    if vocab_size <= 0:
        raise ValueError("compile warmup requires a positive vocabulary size")

    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all()
        if device.type == "cuda"
        else None
    )
    model.train()
    try:
        input_ids = torch.randint(
            0,
            vocab_size,
            (batch_size, sequence_length),
            dtype=torch.long,
            device=device,
        )
        labels = input_ids.clone()
        with torch.enable_grad():
            with autocast(
                device_type=device.type,
                enabled=use_amp,
                dtype=dtype,
            ):
                out = model(
                    input_ids,
                    labels=labels,
                    return_logits=False,
                    validate_inputs=False,
                )
                if out.loss_sum is None:
                    raise RuntimeError(
                        "compile warmup did not produce a loss sum"
                    )
            out.loss_sum.backward()
    finally:
        model.zero_grad(set_to_none=True)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


def main() -> None:
    """主函数：解析参数、构建配置、启动训练。"""
    # 设置 CUDA 内存分配策略，减少显存碎片
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # FP32 矩阵乘使用 TF32（仅 CUDA 生效）；最高精度会关闭该加速。
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # 编译后端不可用（如 CPU 缺少 OpenCL 时首次编译抛 InductorError）时自动回退 eager，
    # 避免训练进程在首次前向时中断
    torch._dynamo.config.suppress_errors = True

    parser = argparse.ArgumentParser(description="TaiChu 预训练")
    parser.add_argument("--model_config", type=str, required=True, help="模型结构 YAML 配置")
    parser.add_argument("--pretrain_config", type=str, required=True, help="预训练超参 YAML 配置")
    parser.add_argument("--resume", type=str, default=None, help="恢复检查点文件路径")

    # ========== 模型结构覆盖参数 ==========
    parser.add_argument("--model_name", type=str, default=None, help="覆盖模型名称")
    parser.add_argument("--hidden_size", type=int, default=None, help="覆盖隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=None, help="覆盖 Transformer 层数")
    parser.add_argument("--num_heads", type=int, default=None, help="覆盖注意力头数")
    parser.add_argument("--vocab_size", type=int, default=None, help="覆盖词表大小")
    parser.add_argument("--max_seq_len", type=int, default=None, help="覆盖最大序列长度")
    parser.add_argument("--dropout", type=float, default=None, help="覆盖 dropout 比率")
    parser.add_argument("--num_experts", type=int, default=None, help="覆盖专家数量")
    parser.add_argument("--top_k", type=int, default=None, help="覆盖路由 top-k")

    # ========== 训练超参数覆盖 ==========
    parser.add_argument("--target_tokens", type=str, default=None, help="覆盖目标总 token 数，支持数字或带单位的字符串，如 2500000000, '2.5B', '100M', '1.2T'")
    parser.add_argument("--train_batch_size", type=int, default=None, help="覆盖训练阶段每卡的批次大小（如 8）")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="覆盖验证阶段的批次大小（如 16）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="覆盖梯度累积步数，用于模拟更大的全局批次（如 4）")
    parser.add_argument("--learning_rate", type=float, default=None, help="覆盖峰值学习率（如 3e-4）")
    parser.add_argument("--warmup_ratio", type=float, default=None, help="覆盖预热步数占总训练步数的比例（如 0.01 表示 1%%）")
    parser.add_argument("--stable_ratio", type=float, default=None, help="覆盖 WSD 调度中恒定阶段占总步数的比例（如 0.8 表示 80%%，推荐 0.7~0.8）")
    parser.add_argument("--weight_decay", type=float, default=None, help="覆盖权重衰减系数（如 0.1）")
    parser.add_argument("--min_lr_ratio", type=float, default=None, help="覆盖最终学习率相对于峰值学习率的比例（WSD 衰减终点，推荐 0.0）")
    parser.add_argument("--decay_type", type=str, default=None, choices=["cosine", "linear"], help="覆盖 WSD 衰减阶段的曲线类型：cosine（余弦，推荐）或 linear（线性）")
    parser.add_argument("--use_mixed_precision", action="store_true", default=None, help="启用混合精度训练（实际生效还依赖 --dtype 参数）")
    parser.add_argument("--dtype", type=str, default=None, choices=["bfloat16", "float16"], help="混合精度数据类型：bfloat16（推荐 Ampere 及以上 GPU）或 float16")
    parser.add_argument("--seed", type=int, default=None, help="覆盖随机种子（不同进程会自动偏移）")
    parser.add_argument("--save_interval", type=int, default=None, help="覆盖常规检查点保存间隔（步数）")
    parser.add_argument("--eval_interval", type=int, default=None, help="覆盖验证间隔（步数）")
    parser.add_argument("--log_interval", type=int, default=None, help="覆盖训练日志打印间隔（步数）")
    parser.add_argument(
        "--compile_model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖是否启用 torch.compile",
    )
    parser.add_argument(
        "--compile_mode",
        choices=[
            "auto",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        default=None,
        help="覆盖 torch.compile 优化模式",
    )
    parser.add_argument(
        "--compile_dynamic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 torch.compile 动态形状模式",
    )
    parser.add_argument("--peak_flops", type=float, default=None,
                        help="GPU 理论峰值算力（FLOPs/秒），用于 MFU 计算；默认按 GPU 型号自动识别")

    # 早停参数覆盖
    # ========== 早停参数覆盖 ==========
    parser.add_argument("--early_stop_enabled", action="store_true", default=None, help="启用早停机制（若未指定，则使用配置文件中的值）")
    parser.add_argument("--early_stop_patience", type=int, default=None, help="早停耐心值，连续无改善的验证次数上限（如 5）")
    parser.add_argument("--early_stop_min_delta", type=float, default=None, help="判定改善的最小绝对变化量（如 1e-4）")

    # SwanLab 参数覆盖
    parser.add_argument(
        "--use_swanlab",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖是否启用 SwanLab",
    )
    parser.add_argument("--swanlab_project", type=str, default=None, help="覆盖 SwanLab 项目名")
    parser.add_argument("--swanlab_workspace", type=str, default=None, help="覆盖 SwanLab 工作区")
    parser.add_argument("--swanlab_id", type=str, default=None, help="指定 SwanLab 实验 ID")
    parser.add_argument("--swanlab_resume", type=str, default=None, help="SwanLab 恢复策略，如 must")
    parser.add_argument("--swanlab_experiment_name", type=str, default=None, help="覆盖 SwanLab 实验名")
    parser.add_argument("--swanlab_log_dir", type=str, default=None, help="覆盖 SwanLab 日志目录")
    parser.add_argument("--swanlab_mode", type=str, default=None, help="覆盖 SwanLab 运行模式")

    args = parser.parse_args()

    # 1. 加载配置
    model_cfg = load_model_config(args.model_config)
    pretrain_cfg = load_pretrain_config(args.pretrain_config)

    # 2. 覆盖配置
    # 覆盖模型配置
    if args.model_name is not None:
        model_cfg.model_name = args.model_name
    if args.hidden_size is not None:
        model_cfg.hidden_size = args.hidden_size
    if args.num_layers is not None:
        model_cfg.num_layers = args.num_layers
    if args.num_heads is not None:
        model_cfg.num_attention_heads = args.num_heads
    if args.vocab_size is not None:
        model_cfg.vocab_size = args.vocab_size
    if args.max_seq_len is not None:
        model_cfg.max_position_embeddings = args.max_seq_len
        pretrain_cfg.data = replace(
            pretrain_cfg.data,
            packing=replace(
                pretrain_cfg.data.packing,
                max_seq_length=args.max_seq_len,
            ),
        )
    if args.dropout is not None:
        model_cfg.attention_dropout = args.dropout
        model_cfg.hidden_dropout = args.dropout
        model_cfg.embedding_dropout = args.dropout
    if args.num_experts is not None:
        model_cfg.moe_num_experts = args.num_experts
    if args.top_k is not None:
        model_cfg.moe_top_k = args.top_k

    # 覆盖训练超参数
    if args.target_tokens is not None:
        pretrain_cfg.training.target_tokens = parse_tokens_string(args.target_tokens)
    if args.train_batch_size is not None:
        pretrain_cfg.training.batch_size = args.train_batch_size
    if args.eval_batch_size is not None:
        pretrain_cfg.evaluating.batch_size = args.eval_batch_size
    if args.gradient_accumulation_steps is not None:
        pretrain_cfg.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.learning_rate is not None:
        pretrain_cfg.optimizer.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        pretrain_cfg.optimizer.weight_decay = args.weight_decay
    if args.warmup_ratio is not None:
        pretrain_cfg.scheduler.warmup_ratio = args.warmup_ratio
    if args.stable_ratio is not None:
        pretrain_cfg.scheduler.stable_ratio = args.stable_ratio
    if args.min_lr_ratio is not None:
        pretrain_cfg.scheduler.min_lr_ratio = args.min_lr_ratio
    if args.decay_type is not None:
        pretrain_cfg.scheduler.decay_type = args.decay_type
    if args.use_mixed_precision is not None:
        pretrain_cfg.training.use_mixed_precision = args.use_mixed_precision
    if args.dtype is not None:
        pretrain_cfg.training.dtype = args.dtype
    if args.seed is not None:
        pretrain_cfg.training.seed = args.seed
    if args.save_interval is not None:
        pretrain_cfg.training.save_interval = args.save_interval
    if args.eval_interval is not None:
        pretrain_cfg.evaluating.eval_interval = args.eval_interval
    if args.log_interval is not None:
        pretrain_cfg.training.log_interval = args.log_interval
    if args.compile_model is not None:
        pretrain_cfg.training.compile_model = args.compile_model
    if args.compile_mode is not None:
        pretrain_cfg.training.compile_mode = args.compile_mode
    if args.compile_dynamic is not None:
        pretrain_cfg.training.compile_dynamic = args.compile_dynamic

    # 覆盖早停配置
    if args.early_stop_enabled is not None:
        pretrain_cfg.early_stopping.enabled = args.early_stop_enabled
    if args.early_stop_patience is not None:
        pretrain_cfg.early_stopping.patience = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        pretrain_cfg.early_stopping.min_delta = args.early_stop_min_delta

    # 覆盖 SwanLab 配置
    if args.use_swanlab is not None:
        pretrain_cfg.swanlab.use_swanlab = args.use_swanlab
    if args.swanlab_project is not None:
        pretrain_cfg.swanlab.swanlab_project = args.swanlab_project
    if args.swanlab_workspace is not None:
        pretrain_cfg.swanlab.swanlab_workspace = args.swanlab_workspace
    if args.swanlab_id is not None:
        pretrain_cfg.swanlab.swanlab_id = args.swanlab_id
    if args.swanlab_resume is not None:
        pretrain_cfg.swanlab.swanlab_resume = args.swanlab_resume
    if args.swanlab_experiment_name is not None:
        pretrain_cfg.swanlab.swanlab_experiment_name = args.swanlab_experiment_name
    if args.swanlab_log_dir is not None:
        pretrain_cfg.swanlab.swanlab_log_dir = args.swanlab_log_dir
    if args.swanlab_mode is not None:
        pretrain_cfg.swanlab.swanlab_mode = args.swanlab_mode

    # 3. 分布式初始化
    local_rank = pretrain_cfg.training.local_rank
    if local_rank == -1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = get_device(local_rank)

    # 只有 torchrun/分布式启动器显式提供 WORLD_SIZE > 1 时才初始化进程组。
    # 单进程脚本即使所在机器有多张 GPU，也不能自行假定其他 rank 存在。
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", local_rank if world_size > 1 else 0))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout_minutes = pretrain_cfg.training.distributed_timeout_minutes
        if timeout_minutes <= 0:
            raise ValueError("distributed_timeout_minutes must be positive")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=backend,
                rank=global_rank,
                world_size=world_size,
                timeout=timedelta(minutes=timeout_minutes),
            )
        world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()

    # 4. 设置随机种子（不同进程不同，避免数据重复）
    base_seed = pretrain_cfg.training.seed
    set_seed(base_seed + global_rank)

    # 5. 生成实验目录；恢复训练时沿用检查点所在实验目录
    base_output_dir = pretrain_cfg.training.output_dir  # 默认 "./experiments"
    experiment_name = pretrain_cfg.training.experiment_name  # 从 YAML 读取，可能为空
    model_name = model_cfg.model_name
    resume_path = None
    if args.resume:
        resume_path = os.path.abspath(args.resume)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"检查点文件不存在: {resume_path}")
        experiment_dir = resolve_experiment_dir_from_checkpoint(resume_path)
    else:
        experiment_dir = get_experiment_dir(base_output_dir, experiment_name, model_name)
    if torch.distributed.is_initialized():
        experiment_dirs = [experiment_dir if global_rank == 0 else None]
        torch.distributed.broadcast_object_list(experiment_dirs, src=0)
        experiment_dir = str(experiment_dirs[0])

    # 创建子目录
    best_dir = os.path.join(experiment_dir, "best_models")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    final_dir = os.path.join(experiment_dir, "final_models")
    log_dir = os.path.join(experiment_dir, "logs")
    for d in [best_dir, checkpoint_dir, final_dir, log_dir]:
        os.makedirs(d, exist_ok=True)

    # 6. 日志管理器（仅主进程写入 TensorBoard 和文件）
    is_main = (global_rank == 0)
    log_manager = LogManager(
        log_dir=log_dir,
        tensorboard=is_main,
        log_file="training.log" if is_main else None,
        console_level=20,  # INFO
    )
    if is_main:
        log_manager.info(f"实验目录: {experiment_dir}")

    # 7. 构建模型
    model = TaiChuModel(model_cfg)
    model.to(device)

    if is_main:
        log_manager.info(f"模型配置: {model_cfg.model_name}")

    compiled_model = False
    if pretrain_cfg.training.compile_model:
        compile_available, compile_reason = training_compile_preflight()
        if not compile_available:
            if is_main:
                log_manager.warning(
                    f"跳过 torch.compile：{compile_reason}"
                )
        else:
            try:
                model = compile_model_for_training(
                    model,
                    mode=pretrain_cfg.training.compile_mode,
                    dynamic=pretrain_cfg.training.compile_dynamic,
                    distributed=world_size > 1,
                )
                compiled_model = True
                if is_main:
                    log_manager.info(
                        "已启用 torch.compile: "
                        f"mode={pretrain_cfg.training.compile_mode}, "
                        f"dynamic={pretrain_cfg.training.compile_dynamic}"
                    )
            except Exception:
                if is_main:
                    log_manager.warning(
                        "torch.compile 初始化失败，已回退到 eager 模式"
                    )

    if compiled_model:
        if is_main:
            log_manager.info("正在预编译训练图（forward + backward）...")
        _warmup_compiled_training(
            model,
            device=device,
            vocab_size=model_cfg.vocab_size,
            batch_size=pretrain_cfg.training.batch_size,
            sequence_length=pretrain_cfg.data.packing.max_seq_length,
            use_amp=(
                pretrain_cfg.training.use_mixed_precision
                and torch.cuda.is_available()
            ),
            dtype=(
                torch.bfloat16
                if pretrain_cfg.training.dtype == "bfloat16"
                else torch.float16
            ),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if world_size > 1:
            if torch.cuda.is_available():
                torch.distributed.barrier(device_ids=[local_rank])
            else:
                torch.distributed.barrier()
        if is_main:
            log_manager.info("训练图预编译完成，开始构建 DDP")

    # 封装 DDP
    if world_size > 1:
        ddp_kwargs = (
            {"device_ids": [local_rank], "gradient_as_bucket_view": True}
            if torch.cuda.is_available()
            else {"gradient_as_bucket_view": True}
        )
        if model_cfg.ffn_type == "moe":
            ddp_kwargs["find_unused_parameters"] = True
        model = nn.parallel.DistributedDataParallel(model, **ddp_kwargs)

    # 打印参数量（仅主进程）
    total_params = None
    trainable_params = None
    if is_main:
        total_params = ModelInspector.get_parameter_count(model)
        trainable_params = ModelInspector.get_parameter_count(model, trainable_only=True)
        log_manager.info(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")
        log_manager.info(f"模型大小: {ModelInspector.get_model_size_mb(model):.2f} MB")

    # 8. 加载 tokenizer
    tokenizer = Tokenizer.from_file(pretrain_cfg.tokenizer_path)
    tokenizer_vocab_size = tokenizer.get_vocab_size()
    if tokenizer_vocab_size > model_cfg.vocab_size:
        raise ValueError(
            f"Tokenizer 词表大小 {tokenizer_vocab_size} 超过模型 vocab_size "
            f"{model_cfg.vocab_size}，会导致 Token Embedding 越界"
        )
    if tokenizer_vocab_size < model_cfg.vocab_size and is_main:
        log_manager.warning(
            f"Tokenizer 词表大小 {tokenizer_vocab_size} 小于模型 vocab_size "
            f"{model_cfg.vocab_size}，尾部 Token 不会被训练"
        )

    # 9. 动态计算 max_steps（基于 target_tokens）
    target_tokens = pretrain_cfg.training.target_tokens
    per_gpu_batch = pretrain_cfg.training.batch_size
    grad_accum = pretrain_cfg.training.gradient_accumulation_steps
    global_batch_size = per_gpu_batch * grad_accum * world_size
    tokens_per_step = global_batch_size * pretrain_cfg.data.packing.max_seq_length
    max_steps = math.ceil(target_tokens / tokens_per_step)

    # 10. 构建统一数据管线
    log_manager.info("构建数据集...")
    pad_token_id = tokenizer.token_to_id(pretrain_cfg.data.pad_token)
    if pad_token_id is None:
        raise ValueError(
            f"configured pad token {pretrain_cfg.data.pad_token!r} "
            "does not exist in the tokenizer"
        )
    data_factory = TaiChuDataLoaderFactory(
        pretrain_cfg.data,
        pad_token_id=pad_token_id,
    )
    train_runtime = pretrain_cfg.training.dataloader
    train_loader = data_factory.create(
        split="train",
        batch_size=pretrain_cfg.training.batch_size,
        rank=global_rank,
        world_size=world_size,
        pin_memory=bool(train_runtime.pin_memory and torch.cuda.is_available()),
        drop_last=train_runtime.drop_last,
        persistent_workers=train_runtime.persistent_workers,
    )
    eval_runtime = pretrain_cfg.evaluating.dataloader
    val_loader = (
        data_factory.create(
            split="validation",
            batch_size=pretrain_cfg.evaluating.batch_size,
            rank=global_rank,
            world_size=world_size,
            pin_memory=bool(eval_runtime.pin_memory and torch.cuda.is_available()),
            drop_last=eval_runtime.drop_last,
            persistent_workers=eval_runtime.persistent_workers,
        )
        if pretrain_cfg.data.storage.validation_path is not None
        else None
    )
    if is_main:
        log_manager.info(
            "数据管线配置: "
            + ", ".join(
                f"{key}={value}"
                for key, value in pretrain_cfg.data.to_dict().items()
            )
        )
        if (
            pretrain_cfg.training.batch_size
            != pretrain_cfg.data.batching.micro_batch_size
        ):
            log_manager.warning(
                "training.batch_size 与 data.batching.micro_batch_size 不一致；"
                "训练 loader 使用 training.batch_size，"
                f"当前为 {pretrain_cfg.training.batch_size}"
            )

    # 11. 优化器
    raw_model_for_optim = unwrap_model(model)
    optimizer = raw_model_for_optim.configure_optimizers(pretrain_cfg.optimizer)

    warmup_ratio = pretrain_cfg.scheduler.warmup_ratio
    warmup_steps = max(1, int(max_steps * warmup_ratio))

    log_manager.info(f"目标总 token 数: {target_tokens:,}")
    log_manager.info(
        f"全局样本 batch size: {global_batch_size} "
        f"(每卡 {per_gpu_batch} × 累积 {grad_accum} × rank 数 {world_size})"
    )
    log_manager.info(f"每步名义 token 数: {tokens_per_step:,}")
    log_manager.info(f"动态计算 max_steps = {max_steps}, warmup_steps = {warmup_steps}")

    # 12. 学习率调度器（warmup + cosine）
    # 12.1. 读取调度配置
    warmup_ratio = pretrain_cfg.scheduler.warmup_ratio
    stable_ratio = getattr(pretrain_cfg.scheduler, 'stable_ratio', 0.0)
    min_lr_ratio = pretrain_cfg.scheduler.min_lr_ratio
    decay_type = getattr(pretrain_cfg.scheduler, 'decay_type', 'cosine')

    # 12.2. 计算各阶段步数（与动态 max_steps 保持一致）
    warmup_steps = max(1, int(max_steps * warmup_ratio))
    stable_steps = int(max_steps * stable_ratio)

    # 若 warmup + stable 超过总步数，则自动裁剪 stable 阶段
    if warmup_steps + stable_steps >= max_steps:
        stable_steps = max(0, max_steps - warmup_steps - 1)  # 至少留 1 步给衰减
        log_manager.warning(
            f"warmup_ratio + stable_ratio 超出 1.0，已自动裁剪 stable_steps 为 {stable_steps}"
        )

    # 12.3. 创建 WSD 学习率 lambda 函数
    lr_lambda = get_wsd_lr_lambda(
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        max_steps=max_steps,
        min_lr_ratio=min_lr_ratio,
        decay_type=decay_type,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 12.4. 记录调度策略详情
    log_manager.info(
        f"WSD 调度: warmup={warmup_steps} 步 ({warmup_ratio:.1%}), "
        f"stable={stable_steps} 步 ({stable_ratio:.1%}), "
        f"decay={max_steps - warmup_steps - stable_steps} 步, "
        f"min_lr_ratio={min_lr_ratio}, decay_type={decay_type}"
    )

    # 13. 混合精度
    requested_amp = pretrain_cfg.training.use_mixed_precision
    use_amp = requested_amp and torch.cuda.is_available()
    if requested_amp and not torch.cuda.is_available():
        log_manager.warning("CUDA 不可用，已自动禁用混合精度训练")
    scaler_enabled = use_amp and pretrain_cfg.training.dtype == "float16"
    scaler = GradScaler("cuda", enabled=scaler_enabled)

    # 14. 检查点管理器
    checkpoint_manager = CheckpointManager(
        output_dir=experiment_dir,
        keep_last_n=3,
        best_metric_name="val_loss",
        best_metric_mode="min",
        best_subdir="best_models",
    )

    # 15. 恢复训练（如果指定）
    resume_step = 0
    best_val_loss_restored = None
    checkpoint = None
    data_state = None
    if resume_path:
        log_manager.info(f"从检查点恢复: {resume_path}")
        resume_step, best_val_loss_restored, checkpoint = checkpoint_manager.load(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=str(device),
        )
        if resume_step > 0 and "scheduler_state_dict" not in checkpoint:
            scheduler.last_epoch = resume_step
        if checkpoint.get("data_state") is not None:
            data_state = DataState.from_dict(checkpoint["data_state"])
        log_manager.info(f"恢复至全局步数: {resume_step}")

    # 16. SwanLab 日志记录器（仅主进程且配置启用）
    swanlab_logger = None
    use_swanlab = pretrain_cfg.swanlab.use_swanlab
    if use_swanlab and is_main:
        swanlab_config = {
            "model_name": model_cfg.model_name,
            "target_tokens": target_tokens,
            "batch_size": pretrain_cfg.training.batch_size,
            "gradient_accumulation_steps": pretrain_cfg.training.gradient_accumulation_steps,
            "learning_rate": pretrain_cfg.optimizer.learning_rate,
            "weight_decay": pretrain_cfg.optimizer.weight_decay,
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "min_lr_ratio": pretrain_cfg.scheduler.min_lr_ratio,
            "max_seq_length": pretrain_cfg.data.packing.max_seq_length,
            "data_pipeline": pretrain_cfg.data.to_dict(),
            "use_mixed_precision": use_amp,
            "dtype": pretrain_cfg.training.dtype,
            "seed": pretrain_cfg.training.seed,
        }
        if 'total_params' in locals():
            swanlab_config["total_params"] = total_params
        if 'trainable_params' in locals():
            swanlab_config["trainable_params"] = trainable_params
        if pretrain_cfg.swanlab.swanlab_mode:
            os.environ["SWANLAB_MODE"] = pretrain_cfg.swanlab.swanlab_mode

        resume_swanlab_id = (
            checkpoint.get("swanlab_run_id")
            if checkpoint is not None
            else None
        )
        swanlab_init_kwargs = pretrain_cfg.swanlab.swanlab_init_kwargs
        swanlab_id = (
            pretrain_cfg.swanlab.swanlab_id
            or resume_swanlab_id
            or swanlab_init_kwargs.get("id")
        )
        swanlab_workspace = (
            pretrain_cfg.swanlab.swanlab_workspace
            or (
                checkpoint.get("swanlab_workspace")
                if checkpoint is not None
                else None
            )
            or swanlab_init_kwargs.get("workspace")
        )
        swanlab_resume = (
            pretrain_cfg.swanlab.swanlab_resume
            or swanlab_init_kwargs.get("resume")
        )
        if resume_path and swanlab_id and not swanlab_resume:
            swanlab_resume = "must"

        swanlab_logger = SwanLabLogger(
            project=pretrain_cfg.swanlab.swanlab_project,
            experiment_name=pretrain_cfg.swanlab.swanlab_experiment_name,
            config=swanlab_config,
            log_dir=pretrain_cfg.swanlab.swanlab_log_dir,
            workspace=swanlab_workspace,
            run_id=swanlab_id,
            resume=swanlab_resume,
            init_kwargs=swanlab_init_kwargs,
            disabled=False,
            global_rank=global_rank,
        )
        log_manager.info(
            "SwanLab 已启用，"
            f"项目: {pretrain_cfg.swanlab.swanlab_project}, "
            f"工作区: {swanlab_workspace or '<default>'}, "
            f"实验: {pretrain_cfg.swanlab.swanlab_experiment_name or 'auto'}, "
            f"ID: {swanlab_logger.run_id or '<new>'}, "
            f"resume: {swanlab_resume or '<disabled>'}"
        )

    # 峰值算力：优先 --peak_flops，否则按 GPU 型号自动识别
    peak_flops = args.peak_flops if args.peak_flops is not None else ModelInspector.get_peak_flops()
    if is_main:
        if peak_flops > 0:
            log_manager.info(f"MFU 峰值算力: {peak_flops / 1e12:.0f} TFLOPS")
        else:
            log_manager.info("MFU 未启用：当前设备峰值算力未定义")

    # 17. 构建 Trainer 并开始训练
    trainer = TaiChuTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        log_manager=log_manager,
        checkpoint_manager=checkpoint_manager,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        gradient_accumulation_steps=pretrain_cfg.training.gradient_accumulation_steps,
        init_step=resume_step,
        max_steps=max_steps,
        save_interval=pretrain_cfg.training.save_interval,
        eval_interval=pretrain_cfg.evaluating.eval_interval,
        train_log_interval=pretrain_cfg.training.log_interval,
        eval_log_interval=pretrain_cfg.evaluating.log_interval,
        early_stop_enabled=pretrain_cfg.early_stopping.enabled,
        early_stop_monitor=pretrain_cfg.early_stopping.monitor,
        early_stop_patience=pretrain_cfg.early_stopping.patience,
        early_stop_min_delta=pretrain_cfg.early_stopping.min_delta,
        early_stop_mode=pretrain_cfg.early_stopping.mode,
        swanlab_logger=swanlab_logger,
        device=device,
        use_amp=use_amp,
        dtype=pretrain_cfg.training.dtype,
        local_rank=local_rank,
        max_seq_len=pretrain_cfg.data.packing.max_seq_length,
        global_rank=global_rank,
        peak_flops=peak_flops,
        generation_prompts=pretrain_cfg.evaluating.prompts,
        num_generate_tokens=pretrain_cfg.evaluating.num_generate_tokens,
        max_grad_norm=pretrain_cfg.training.max_grad_norm,
        bos_token_id=tokenizer.token_to_id(pretrain_cfg.data.bos_token),
        eos_token_id=tokenizer.token_to_id(pretrain_cfg.data.eos_token),
        compute_empty_prefix_eos_ppl=(
            pretrain_cfg.evaluating.empty_prefix_eos_ppl
        ),
        data_state=data_state,
        compiled_model=compiled_model,
    )
    if best_val_loss_restored is not None:
        trainer.best_val_loss = best_val_loss_restored

    # 恢复早停状态
    if (trainer.early_stopping is not None and checkpoint is not None
            and "early_stopping_state" in checkpoint):
        early_state = checkpoint["early_stopping_state"]
        trainer.early_stopping.load_state_dict(early_state)
        log_manager.info(
            f"已恢复早停状态: counter={early_state.get('counter', 0)}, "
            f"best_score={early_state.get('best_score', None)}, "
            f"best_step={early_state.get('best_step', 0)}, "
            f"early_stop={early_state.get('early_stop', False)}"
        )

    trainer.train()


if __name__ == "__main__":
    main()
