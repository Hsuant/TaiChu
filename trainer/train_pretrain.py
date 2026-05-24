#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TaiChu 语言模型预训练入口脚本。

功能：
    - 加载模型结构与训练超参配置（YAML）
    - 初始化 tokenizer、模型、数据集、日志、检查点管理器
    - 支持混合精度训练、梯度累积、warmup + cosine 学习率调度
    - 自动恢复训练、定期验证与保存最佳/最近检查点
    - 所有日志统一通过 LogManager 输出到 TensorBoard 及控制台/文件

用法：
    # 单卡训练
    python -m trainer.train_pretrain --model_config configs/TaiChu_LLM.yaml --pretrain_config configs/train_pretrain_config.yaml

    # 多卡训练（通过 torchrun 启动）
    torchrun --nproc_per_node=4 train_pretrain.py --model_config configs/TaiChu_125m.yaml --pretrain_config configs/pretrain_config.yaml
"""

import argparse
import math
import time
import os
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

from model.model import TaiChuModel
from tokenizers import Tokenizer
from typing import List, Optional

from utils.config_loader import load_model_config, load_pretrain_config
from utils.train_utils import set_seed, get_device, get_cosine_lr_lambda
from utils.logger import LogManager
from utils.swanlab_logger import SwanLabLogger
from utils.checkpoint import CheckpointManager
from utils.model_utils import ModelInspector
from dataset.llm_dataset import build_dataloader


class TaiChuTrainer:
    """TaiChu 预训练器。

    封装训练循环、验证、检查点、日志等功能。支持混合精度、梯度累积、分布式训练。
    """

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
        device: torch.device = torch.device("cpu"),
        use_amp: bool = False,
        dtype: str = "bfloat16",
        local_rank: int = 0,
        max_seq_len: int = 2048,
        global_rank: int = 0,
        generation_prompts: Optional[List[str]] = None,
        num_generate_tokens: int = 50,
        swanlab_logger: Optional[SwanLabLogger] = None,
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
            generation_prompts: 测试Prompts列表
            num_generate_tokens: 测试生成token长度
            swanlab_logger: SwanLab 日志记录器实例（仅主进程启用）。
            device: 当前设备。
            use_amp: 是否启用混合精度。
            dtype: 混合精度数据类型（'bfloat16' 或 'float16'）。
            local_rank: 当前进程的本地 Rank（用于设备设置）。
            max_seq_len: 最大序列长度，用于计算 tokens/s。
            global_rank: 全局 Rank（0 为主进程）。
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
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_steps = max_steps
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.train_log_interval = train_log_interval
        self.eval_log_interval = eval_log_interval
        self.generation_prompts = generation_prompts
        self.num_generate_tokens = num_generate_tokens
        self.swanlab_logger = swanlab_logger
        self.device = device
        self.use_amp = use_amp
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.max_seq_len = max_seq_len

        # 训练状态
        self.global_step = init_step          # 恢复的步数
        self.best_val_loss = float("inf")     # 最佳验证损失

        # 用于计算 tokens_per_sec 的时间窗口
        self.start_time = 0.0                 # 训练开始时间（秒）
        self.last_log_time = None  # 上次记录日志的时间戳
        self.last_log_step = init_step  # 上次记录日志时的全局步数

    def train(self) -> None:
        """执行主训练循环。"""
        self.model.train()
        self.start_time = time.time()
        self.log_manager.info("训练开始")

        # 数据迭代器
        train_iter = iter(self.train_loader)
        accumulated_loss = 0.0          # 累积的原始损失（未除累积步数）
        samples_processed = 0           # 当前累积步内处理的样本数
        micro_step = 0                  # 当前累积步内已处理的微批次计数

        while self.global_step < self.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # 前向传播（混合精度）
            with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.dtype):
                out = self.model(input_ids, labels=labels)
                loss = out.loss
                loss = loss.mean() / self.gradient_accumulation_steps   # 缩放损失

            # 反向传播
            self.scaler.scale(loss).backward()
            accumulated_loss += loss.item() * self.gradient_accumulation_steps   # 恢复原始损失
            samples_processed += input_ids.size(0)
            micro_step += 1

            # 达到梯度累积步数，执行参数更新
            if micro_step == self.gradient_accumulation_steps:
                # 梯度裁剪（防止梯度爆炸）
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # 优化器步进
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                # 更新全局步数
                self.global_step += 1
                avg_loss = accumulated_loss / self.gradient_accumulation_steps

                # 记录训练指标
                if self.global_step % self.train_log_interval == 0:
                    self._log_training(avg_loss, samples_processed)

                # 日志记录完毕再梯度清零
                self.optimizer.zero_grad()

                # 重置累积计数器
                accumulated_loss = 0.0
                samples_processed = 0
                micro_step = 0

                # 验证
                if self.global_step % self.eval_interval == 0:
                    self.log_manager.info("验证开始")
                    val_loss = self._evaluate(
                        generate_prompts=self.generation_prompts,
                        num_generate_tokens=self.num_generate_tokens,
                    )
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                    self.checkpoint_manager.save_best(
                        self.model,
                        metrics={"val_loss": val_loss},
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        global_step=self.global_step,
                    )
                    self.log_manager.info(
                        f"当前最佳验证损失: {self.checkpoint_manager.best_metric_value:.4f}"
                    )

                # 定期保存检查点
                if self.global_step % self.save_interval == 0:
                    self.checkpoint_manager.save(
                        f"checkpoint-{self.global_step}.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        global_step=self.global_step,
                        metrics={"val_loss": val_loss if 'val_loss' in locals() else self.best_val_loss},
                    )

        # 训练结束，保存最终模型
        self.checkpoint_manager.save(
            "final_model.pt",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            global_step=self.global_step,
        )
        self.log_manager.info("训练完成!")
        if self.swanlab_logger:
            self.swanlab_logger.finish()
        self.log_manager.close()

    def _log_training(self, loss: float, samples: int) -> None:
        """记录训练指标到 TensorBoard 和日志文件。

        计算 tokens_per_sec 时，使用本次日志与上一次日志之间的时间窗口，
        以反映瞬时训练吞吐量，避免因全程累计时间而导致吞吐量持续下降的假象。

        Args:
            loss: 当前参数更新步的平均损失（已排除梯度累积的缩放）。
            samples: 当前参数更新步内实际处理的样本数（batch_size × 累积步数）。
        """
        # 获取当前学习率
        lr = self.optimizer.param_groups[0]["lr"]

        # 计算困惑度（perplexity），安全处理 loss 过大导致的溢出
        ppl = math.exp(loss) if loss < 100 else float("inf")

        # ---- 计算 tokens_per_sec（基于滑动窗口的瞬时吞吐量） ----
        current_time = time.time()

        if self.last_log_time is None:
            # 第一次记录日志：没有历史时间基准，使用从训练开始到现在的总时间作为窗口
            elapsed = current_time - self.start_time
            # 总步数差 = 当前全局步数 - 上次记录的步数（初次为 init_step）
            steps_since_log = self.global_step - self.last_log_step
        else:
            # 非首次记录：使用距离上一次记录日志的时间差作为窗口
            elapsed = current_time - self.last_log_time
            steps_since_log = self.global_step - self.last_log_step

        # 窗口内处理的 token 总数 = 步数差 × 每步样本数 × 序列长度
        total_tokens = steps_since_log * samples * self.max_seq_len
        # 避免除零，计算瞬时吞吐量
        tokens_per_sec = total_tokens / max(elapsed, 1e-6)

        # 保存当前时间和步数，供下一次计算使用
        self.last_log_time = current_time
        self.last_log_step = self.global_step

        # 获取梯度统计信息（梯度范数、最大值、最小值等）
        grad_info = ModelInspector.gradient_summary(self.model) or {}
        grad_norm = grad_info.get("grad_norm", 0.0)
        grad_max = grad_info.get("grad_max", 0.0)

        # 记录标量：训练损失、困惑度、学习率、吞吐量

        # 构造指标字典
        metrics = {
            "train/loss": loss,
            "train/perplexity": ppl,
            "train/lr": lr,
            "train/tokens_per_sec": tokens_per_sec,
            "train/grad_norm": grad_norm,
            "train/grad_max": grad_max,
        }

        # 构建单行日志（控制台/文件）
        log_line = (
            f"Step {self.global_step} | "
            f"loss={loss:.4f} | ppl={ppl:.2f} | lr={lr:.2e} | "
            f"tok/s={tokens_per_sec:.0f} | grad_norm={grad_norm:.4f} | grad_max={grad_max:.4f}"
        )
        self.log_manager.logger.info(log_line)

        # 记录到 SwanLab
        if self.swanlab_logger:
            self.swanlab_logger.log_metrics(metrics, step=self.global_step)

        # 更新 LogManager 内部步数（供其他可能调用）
        self.log_manager.set_step(self.global_step)

    @torch.no_grad()
    def _evaluate(self, generate_prompts: list = None, num_generate_tokens: int = 50) -> float:
        """在验证集上计算平均损失，并在主进程中生成文本示例。

        Args:
            generate_prompts: 用于生成测试的提示词列表，若为 None 则使用默认提示。
            num_generate_tokens: 每个提示生成的 token 数量。

        Returns:
            平均损失值（所有 token 的平均交叉熵）。
        """
        # 1. 计算验证损失
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        # 用于中间输出的变量
        batch_cnt = 0
        accum_loss = 0.0
        accum_tokens = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.dtype):
                out = self.model(input_ids, labels=labels)
                loss = out.loss
            batch_tokens = input_ids.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

            # 累积中间输出数据
            accum_loss += loss.item() * batch_tokens
            accum_tokens += batch_tokens
            batch_cnt += 1

            # 每隔 val_log_interval 个 batch 输出一次中间结果（仅主进程）
            if self.global_rank == 0 and self.eval_log_interval > 0 and batch_cnt % self.eval_log_interval == 0:
                local_avg_loss = accum_loss / accum_tokens
                local_ppl = math.exp(local_avg_loss) if local_avg_loss < 100 else float("inf")
                self.log_manager.logger.info(
                    f"[Eval] Step {self.global_step} (batch {batch_cnt}/{len(self.val_loader)}) | "
                    f"loss={local_avg_loss:.4f} | ppl={local_ppl:.2f}"
                )
                # 重置窗口累积器，使下一个窗口独立
                accum_loss = 0.0
                accum_tokens = 0

        # 分布式聚合
        if self.device.type == 'cuda' and torch.distributed.is_initialized():
            loss_tensor = torch.tensor(total_loss, device=self.device)
            tokens_tensor = torch.tensor(total_tokens, device=self.device)
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(tokens_tensor, op=torch.distributed.ReduceOp.SUM)
            total_loss = loss_tensor.item()
            total_tokens = tokens_tensor.item()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        self.model.train()

        # 2. 生成测试（仅主进程，且提供了 prompts）
        if self.global_rank == 0 and generate_prompts:
            # 获取原始模型（处理 DDP 包装）
            raw_model = self.model.module if hasattr(self.model, 'module') else self.model
            raw_model.eval() # type: ignore

            # 对每个提示进行生成
            for idx, prompt in enumerate(generate_prompts):
                # 编码提示并提取 token IDs
                encoding = self.tokenizer.encode(prompt, add_special_tokens=False)
                prompt_ids = encoding.ids
                input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

                # 调用模型的 generate 方法
                generated_ids = raw_model.generate( # type: ignore
                    input_ids=input_tensor,
                    max_new_tokens=num_generate_tokens,
                    temperature=0.8,  # 可调节，建议小于 1.0 以获得更确定的结果
                    top_k=50,  # 可选，常用值 40~60
                )

                # 解码生成的完整文本（包含原始提示）
                generated_text = self.tokenizer.decode(generated_ids[0].tolist())

                # 记录到 TensorBoard 和日志
                self.log_manager.set_step(self.global_step)
                self.log_manager.log_text(f"generation/prompt_{idx}", generated_text)
                self.log_manager.info(
                    f"[生成测试] 提示: {prompt}\n生成结果: {generated_text}\n{'-' * 60}"
                )

                # 记录到 SwanLab（文本）
                if self.swanlab_logger:
                    self.swanlab_logger.log_text(
                        f"generation/prompt_{idx}", generated_text, step=self.global_step
                    )

        # 恢复训练模式
        self.model.train()

        # 3. 记录损失指标(仅主进程)
        if self.global_rank == 0:
            self.log_manager.set_step(self.global_step)
            ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
            self.log_manager.logger.info(
                f"Step {self.global_step} - val/loss={avg_loss:.4f} | val/ppl={ppl:.2f}"
            )

            # 记录到 SwanLab
            if self.swanlab_logger:
                val_metrics = {"val/loss": avg_loss, "val/perplexity": ppl}
                self.swanlab_logger.log_metrics(val_metrics, step=self.global_step)

        return avg_loss


def main() -> None:
    """主函数：解析参数、构建配置、启动训练。"""
    parser = argparse.ArgumentParser(description="TaiChu 预训练")
    parser.add_argument("--model_config", type=str, required=True, help="模型结构 YAML 配置")
    parser.add_argument("--pretrain_config", type=str, required=True, help="预训练超参 YAML 配置")
    parser.add_argument("--resume", type=str, default=None, help="恢复检查点文件路径")
    args = parser.parse_args()

    # 1. 加载配置
    model_cfg = load_model_config(args.model_config)
    pretrain_cfg = load_pretrain_config(args.pretrain_config)

    # 2. 获取分布式信息
    local_rank = pretrain_cfg.training.local_rank
    if local_rank == -1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = get_device(local_rank)

    # 初始化分布式进程组（若多卡）
    world_size = 1
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend="nccl", rank=local_rank, world_size=torch.cuda.device_count())
        world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()
    else:
        global_rank = 0

    # 3. 设置随机种子（每个进程不同，避免数据重复）
    base_seed = pretrain_cfg.training.seed
    set_seed(base_seed + global_rank)

    # 4. 日志管理器（仅主进程写入 TensorBoard 和文件）
    is_main = (global_rank == 0)
    log_manager = LogManager(
        log_dir=pretrain_cfg.training.output_dir,
        tensorboard=is_main,
        log_file="training.log" if is_main else None,
        console_level=20,  # INFO
    )

    # 5. 构建模型
    model = TaiChuModel(model_cfg)
    model.to(device)

    if is_main:
        log_manager.info(f"模型配置: {model_cfg.model_name}")

    # 封装 DDP
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # 打印参数量（仅主进程）
    if is_main:
        total_params = ModelInspector.get_parameter_count(model)
        trainable_params = ModelInspector.get_parameter_count(model, trainable_only=True)
        log_manager.info(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")
        log_manager.info(f"模型大小: {ModelInspector.get_model_size_mb(model):.2f} MB")

    # 6. 加载 tokenizer
    tokenizer = Tokenizer.from_file(pretrain_cfg.data.tokenizer_path)

    # 7. 数据集
    log_manager.info("构建数据集...")
    pin_memory = torch.cuda.is_available()
    train_loader = build_dataloader(
        pretrain_cfg.data, tokenizer,
        split="train",
        batch_size=pretrain_cfg.training.batch_size,
        rank=global_rank,
        world_size=world_size,
        pin_memory=pin_memory,
    )
    val_loader = build_dataloader(
        pretrain_cfg.data, tokenizer,
        split="val",
        batch_size=pretrain_cfg.training.batch_size,
        rank=global_rank,
        world_size=world_size,
        pin_memory=pin_memory,
    )

    # 8. 优化器
    optimizer = model.configure_optimizers(pretrain_cfg.optimizer)

    # 9. 学习率调度器（warmup + cosine）
    lr_lambda = get_cosine_lr_lambda(
        warmup_steps=pretrain_cfg.scheduler.warmup_steps,
        max_steps=pretrain_cfg.scheduler.max_steps,
        min_lr_ratio=pretrain_cfg.scheduler.min_lr_ratio,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 10. 混合精度
    use_amp = pretrain_cfg.training.use_mixed_precision and torch.cuda.is_available()
    if use_amp and not torch.cuda.is_available():
        log_manager.warning("CUDA 不可用，已自动禁用混合精度训练")
    scaler = GradScaler('cuda', enabled=use_amp) if use_amp else GradScaler('cpu', enabled=False)

    # 11. 检查点管理器
    checkpoint_manager = CheckpointManager(
        output_dir=pretrain_cfg.training.output_dir,
        keep_last_n=3,
        best_metric_name="val_loss",
        best_metric_mode="min",
    )

    # 12. SwanLab 日志记录器（仅主进程且配置启用）
    swanlab_logger = None
    use_swanlab = pretrain_cfg.swanlab.use_swanlab
    if use_swanlab and is_main:
        # 收集超参数配置
        swanlab_config = {
            "model_name": model_cfg.model_name,
            "batch_size": pretrain_cfg.training.batch_size,
            "gradient_accumulation_steps": pretrain_cfg.training.gradient_accumulation_steps,
            "learning_rate": pretrain_cfg.optimizer.learning_rate,
            "weight_decay": pretrain_cfg.optimizer.weight_decay,
            "warmup_steps": pretrain_cfg.scheduler.warmup_steps,
            "max_steps": pretrain_cfg.scheduler.max_steps,
            "min_lr_ratio": pretrain_cfg.scheduler.min_lr_ratio,
            "max_seq_length": pretrain_cfg.data.max_seq_length,
            "use_mixed_precision": use_amp,
            "dtype": pretrain_cfg.training.dtype,
            "seed": pretrain_cfg.training.seed,
        }
        # 添加模型参数量（如果在前面已计算）
        if 'total_params' in locals():
            swanlab_config["total_params"] = total_params
        if 'trainable_params' in locals():
            swanlab_config["trainable_params"] = trainable_params

        # 设置环境变量或模式（如果需要）
        if pretrain_cfg.swanlab.swanlab_mode:
            os.environ["SWANLAB_MODE"] = pretrain_cfg.swanlab.swanlab_mode

        swanlab_logger = SwanLabLogger(
            project=pretrain_cfg.swanlab.swanlab_project,
            experiment_name=pretrain_cfg.swanlab.swanlab_experiment_name,
            config=swanlab_config,
            log_dir=pretrain_cfg.swanlab.swanlab_log_dir,
            disabled=False,
            global_rank=global_rank,
        )
        log_manager.info(
            f"SwanLab 已启用，项目: {pretrain_cfg.swanlab.swanlab_project}，实验: {pretrain_cfg.swanlab.swanlab_experiment_name or 'auto'}")

    # 12. 恢复训练（如果指定）
    resume_step = 0
    best_val_loss_restored = None
    if args.resume:
        log_manager.info(f"从检查点恢复: {args.resume}")
        resume_step, best_val_loss_restored = checkpoint_manager.load(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=str(device),
        )
        # 设置调度器的 last_epoch（避免循环调用 step）
        scheduler.last_epoch = resume_step - 1 if resume_step > 0 else -1
        log_manager.info(f"恢复至全局步数: {resume_step}")

    # 13. 构建 Trainer 并开始训练
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
        max_steps=pretrain_cfg.scheduler.max_steps,
        save_interval=pretrain_cfg.training.save_interval,
        eval_interval=pretrain_cfg.evaluating.eval_interval,
        train_log_interval=pretrain_cfg.training.log_interval,
        eval_log_interval=pretrain_cfg.evaluating.log_interval,
        generation_prompts=pretrain_cfg.evaluating.prompts,
        num_generate_tokens=pretrain_cfg.evaluating.num_generate_tokens,
        swanlab_logger=swanlab_logger,
        device=device,
        use_amp=use_amp,
        dtype=pretrain_cfg.training.dtype,
        local_rank=local_rank,
        max_seq_len=pretrain_cfg.data.max_seq_length,
        global_rank=global_rank,
    )
    if best_val_loss_restored is not None:
        trainer.best_val_loss = best_val_loss_restored
    trainer.train()


if __name__ == "__main__":
    main()