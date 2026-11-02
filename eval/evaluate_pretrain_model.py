#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TaiChu 模型独立评估脚本（与预训练目录结构对齐）。

功能：
    - 自动从实验目录的 best_models/best_model.pt 加载最佳模型
    - 在验证集上计算平均损失、准确率、分位数交叉熵、空白提示困惑度
    - 可选生成测试文本，展示模型生成能力
    - 支持单卡/CPU评估，无需启动分布式
    - 评估结果保存至实验目录下的 evaluation/ 子目录

用法：
    # 使用实验目录（自动定位最佳模型）
    python eval.py --model_config configs/TaiChu_125m.yaml --pretrain_config configs/pretrain_config.yaml --exp_dir experiments/TaiChu_LMM_pretrain_v1

    # 手动指定检查点文件
    python eval.py --model_config configs/TaiChu_125m.yaml \
                   --pretrain_config configs/pretrain_config.yaml \
                   --exp_dir experiments/TaiChu_125m_20250301_120000 \
                   --checkpoint experiments/.../checkpoints/checkpoint-10000.pt
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast

# 导入项目内部模块
from model.model import TaiChuModel
from tokenizers import Tokenizer
from utils.config_loader import load_model_config, load_pretrain_config
from utils.train_utils import set_seed
from utils.logger import LogManager
from utils.model_utils import ModelInspector
from utils.checkpoint import strip_checkpoint_prefixes
from dataset.runtime import UnifiedDataLoaderFactory as TaiChuDataLoaderFactory
from utils.metrics import ValidationMetrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="TaiChu 模型独立评估（与训练目录结构对齐）"
    )
    # 必需参数
    parser.add_argument("--model_config", type=str, required=True,
                        help="模型结构 YAML 文件")
    parser.add_argument("--pretrain_config", type=str, required=True,
                        help="预训练超参 YAML 文件（用于数据加载等）")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="训练实验根目录（例如 experiments/TaiChu_125m_xxx）")

    # 可选参数
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="待评估的检查点路径（若不指定，则使用 {exp_dir}/best_models/best_model.pt）")
    parser.add_argument("--device", type=str, default=None,
                        help="设备（cpu / cuda），默认自动选择")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="验证批次大小，覆盖配置文件")
    parser.add_argument("--generate", action="store_true",
                        help="是否执行文本生成测试")
    parser.add_argument("--num_generate_tokens", type=int, default=50,
                        help="生成测试的 token 数量")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（用于确定性）")
    parser.add_argument("--save_model", action="store_true",
                        help="是否导出纯净推理模型（仅权重）到 evaluation/ 目录")
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    tokenizer: Tokenizer,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    generate_prompts: Optional[list] = None,
    num_generate_tokens: int = 50,
    log_manager: LogManager = None,
    compute_empty_prefix_eos_ppl: bool = False,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> tuple[dict, list]:
    """在验证集上评估模型，返回各项指标。"""
    model.eval()

    total_nll_nats = 0.0
    total_tokens = 0
    total_correct = 0
    total_bpb_nll_nats = 0.0
    total_bpb_bytes = 0
    all_token_losses = []

    try:
        val_batch_count = len(val_loader)
    except TypeError:
        val_batch_count = None
    batch_idx = -1
    for batch_idx, batch in enumerate(val_loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            out = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = getattr(out, "logits", None)

        if out.loss is None or not torch.isfinite(out.loss):
            raise RuntimeError("model returned a non-finite validation loss")
        if logits is None:
            raise RuntimeError("model did not return logits for validation")
        stats = ValidationMetrics.compute_token_statistics(
            logits,
            labels,
            tokenizer=tokenizer,
        )
        total_nll_nats += stats.nll_nats
        total_tokens += stats.token_count
        total_correct += stats.correct_count
        total_bpb_nll_nats += stats.bpb_nll_nats
        total_bpb_bytes += stats.bpb_bytes
        if stats.token_losses.numel() > 0:
            all_token_losses.append(stats.token_losses)

        if log_manager and (batch_idx + 1) % 50 == 0:
            local_avg = (
                total_nll_nats / total_tokens
                if total_tokens > 0
                else None
            )
            local_avg_text = (
                "N/A" if local_avg is None else f"{local_avg:.4f}"
            )
            log_manager.logger.info(
                f"评估进度: {batch_idx+1}/{val_batch_count or '?'} 批次, "
                f"当前平均损失: {local_avg_text}"
            )

    if batch_idx < 0:
        raise RuntimeError("validation data pipeline produced no batches")
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

    all_token_losses_tensor = (
        torch.cat(all_token_losses, dim=0)
        if all_token_losses
        else torch.empty(0, dtype=torch.float64)
    )
    percentile_metrics = ValidationMetrics.compute_percentile_ce(
        all_token_losses_tensor,
        percentiles=[0.5, 0.9, 0.95],
    )

    metrics = {
        "val_loss": avg_loss,
        "val_perplexity": avg_ppl,
        "val_bpb": val_bpb,
        "val_accuracy": avg_accuracy,
        "val_tokens": total_tokens,
        "val_target_bytes": total_bpb_bytes,
        "val_batches": batch_idx + 1,
        **{f"val_{k}": v for k, v in percentile_metrics.items()},
    }
    if compute_empty_prefix_eos_ppl:
        metrics.update(
            ValidationMetrics.compute_empty_prefix_eos_metrics(
                model,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                device=device,
            )
        )

    # 生成测试
    generated_samples = []
    if generate_prompts and tokenizer:
        raw_model = model.module if hasattr(model, 'module') else model
        raw_model.eval() # type: ignore
        for idx, prompt in enumerate(generate_prompts):
            encoding = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_ids = encoding.ids
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            generated_ids = raw_model.generate( # type: ignore
                input_ids=input_tensor,
                max_new_tokens=num_generate_tokens,
                temperature=0.8,
                top_k=50,
            )
            generated_text = tokenizer.decode(generated_ids[0].tolist())
            generated_samples.append({
                "prompt": prompt,
                "generated_text": generated_text,
            })
            if log_manager:
                log_manager.logger.info(f"[生成测试 {idx}] 提示: {prompt}\n生成: {generated_text}\n{'-'*60}")

    return metrics, generated_samples


def main():
    args = parse_args()

    # 1. 加载配置
    model_cfg = load_model_config(args.model_config)
    pretrain_cfg = load_pretrain_config(args.pretrain_config)

    if args.eval_batch_size is not None:
        pretrain_cfg.evaluating.batch_size = args.eval_batch_size

    # 2. 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # 3. 确定检查点路径
    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {exp_dir}")
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = exp_dir / "best_models" / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    # 4. 创建评估输出目录（实验目录下的 evaluation/）
    eval_output_dir = exp_dir / "evaluation"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # 5. 日志管理器（输出到 evaluation/eval.log）
    log_manager = LogManager(
        log_dir=str(eval_output_dir),
        tensorboard=False,
        log_file="eval.log",
        console_level=20,
    )
    log_manager.info(f"评估开始，设备: {device}")
    log_manager.info(f"实验目录: {exp_dir}")
    log_manager.info(f"检查点: {checkpoint_path}")
    log_manager.info(f"评估结果输出目录: {eval_output_dir}")

    # 6. 构建模型并加载检查点
    log_manager.info("初始化模型...")
    model = TaiChuModel(model_cfg)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
    if saved_config is not None and saved_config.get("vocab_size") != model_cfg.vocab_size:
        raise ValueError(
            f"检查点 vocab_size={saved_config.get('vocab_size')} 与当前模型配置 "
            f"vocab_size={model_cfg.vocab_size} 不一致"
        )
    # 兼容不同的保存格式
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # 处理 DDP / torch.compile 包装的 key
    new_state_dict = strip_checkpoint_prefixes(state_dict)
    model.load_state_dict(new_state_dict, strict=True)
    log_manager.info(f"成功加载检查点: {checkpoint_path}")

    # ========== 输出模型信息（使用 ModelInspector） ==========
    total_params = ModelInspector.get_parameter_count(model, trainable_only=False)
    trainable_params = ModelInspector.get_parameter_count(model, trainable_only=True)
    model_size_mb = ModelInspector.get_model_size_mb(model, precision=4)
    log_manager.logger.info(f"模型总参数量: {total_params:,}")
    log_manager.logger.info(f"可训练参数量: {trainable_params:,}")
    log_manager.logger.info(f"模型大小（FP32估算）: {model_size_mb:.2f} MB")
    grad_info = ModelInspector.gradient_summary(model)
    log_manager.logger.info(f"梯度范数（评估阶段应接近0）: {grad_info['grad_norm']:.6f}")
    # =====================================================

    # 可选：导出纯净推理模型
    if args.save_model:
        save_path = eval_output_dir / "model_inference.pt"
        torch.save(model.state_dict(), save_path)
        log_manager.info(f"纯净模型权重已保存至: {save_path}")

    # 7. 加载 tokenizer
    tokenizer = Tokenizer.from_file(pretrain_cfg.tokenizer_path)
    if tokenizer.get_vocab_size() > model_cfg.vocab_size:
        raise ValueError(
            f"Tokenizer 词表大小 {tokenizer.get_vocab_size()} 超过模型 vocab_size "
            f"{model_cfg.vocab_size}"
        )

    # 8. 构建验证数据加载器（与训练使用同一配置和数据契约）
    log_manager.info("构建验证数据集...")
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
    runtime = pretrain_cfg.evaluating.dataloader
    val_loader = data_factory.create(
        split="val",
        batch_size=pretrain_cfg.evaluating.batch_size,
        rank=0,
        world_size=1,
        pin_memory=bool(runtime.pin_memory and device.type == "cuda"),
        drop_last=runtime.drop_last,
        persistent_workers=runtime.persistent_workers,
    )
    log_manager.info(
        "验证数据管线: "
        + ", ".join(
            f"{key}={value}"
            for key, value in pretrain_cfg.data.to_dict().items()
        )
    )

    # 9. 混合精度设置
    use_amp = pretrain_cfg.training.use_mixed_precision and device.type == "cuda"
    amp_dtype = torch.bfloat16 if pretrain_cfg.training.dtype == "bfloat16" else torch.float16

    # 10. 准备生成提示词
    generate_prompts = None
    if args.generate:
        generate_prompts = pretrain_cfg.evaluating.prompts
        if not generate_prompts:
            generate_prompts = [
                "人工智能的未来是",
                "在一个宁静的早晨",
                "深度学习模型能够",
            ]
            log_manager.warning("未在配置中找到生成提示词，使用默认提示词")

    # 11. 执行评估
    log_manager.info("开始评估...")
    metrics, generated_samples = evaluate(
        model=model,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        generate_prompts=generate_prompts if args.generate else None,
        num_generate_tokens=args.num_generate_tokens,
        log_manager=log_manager,
        compute_empty_prefix_eos_ppl=(
            pretrain_cfg.evaluating.empty_prefix_eos_ppl
        ),
        bos_token_id=tokenizer.token_to_id(pretrain_cfg.data.bos_token),
        eos_token_id=tokenizer.token_to_id(pretrain_cfg.data.eos_token),
    )

    # 12. 打印主要指标
    log_manager.logger.info("========== 评估结果 ==========")
    log_manager.logger.info(f"验证损失: {metrics['val_loss']:.4f}")
    ppl = metrics.get("val_perplexity")
    bpb = metrics.get("val_bpb")
    log_manager.logger.info(
        f"验证困惑度: {'N/A' if ppl is None else f'{ppl:.2f}'}"
    )
    log_manager.logger.info(
        f"Bits per byte: {'N/A' if bpb is None else f'{bpb:.4f}'}"
    )
    log_manager.logger.info(f"Next Token 准确率: {metrics['val_accuracy']:.4f}")
    for percentile in (50, 90, 95):
        value = metrics[f"val_ce_p{percentile}"]
        log_manager.logger.info(
            f"CE 第{percentile}分位数: "
            f"{'N/A' if value is None else f'{value:.4f}'}"
        )
    if "empty_prefix_eos_ppl" in metrics:
        value = metrics["empty_prefix_eos_ppl"]
        log_manager.logger.info(
            f"空前缀 EOS 困惑度: {'N/A' if value is None else f'{value:.2f}'}"
        )

    # 13. 保存指标为 JSON
    metrics_to_save = {k: v for k, v in metrics.items() if not k.startswith("generated_")}

    result_file = eval_output_dir / "eval_metrics.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(
            metrics_to_save,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    log_manager.logger.info(f"评估指标已保存至: {result_file}")

    # 14. 保存生成样本
    if generated_samples:
        gen_file = eval_output_dir / "generated_samples.json"
        with open(gen_file, "w", encoding="utf-8") as f:
            json.dump(
                generated_samples,
                f,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        log_manager.logger.info(f"生成样本已保存至: {gen_file}")

    log_manager.close()
    print("评估完成。")
    print(f"评估结果保存在: {eval_output_dir}")


if __name__ == "__main__":
    main()
