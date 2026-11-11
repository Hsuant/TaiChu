"""Command-line interface and typed config overrides for pretraining."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Sequence

from model.config import ModelConfig
from training.config import PretrainConfig, parse_token_count


def build_pretraining_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TaiChu pretraining")
    parser.add_argument("--model_config", required=True, help="model YAML")
    parser.add_argument("--pretrain_config", required=True, help="training YAML")
    parser.add_argument("--resume", default=None, help="checkpoint path")

    parser.add_argument("--model_name")
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--max_seq_len", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--num_experts", type=int)
    parser.add_argument("--top_k", type=int)

    parser.add_argument("--target_tokens")
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--stable_ratio", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--min_lr_ratio", type=float)
    parser.add_argument("--decay_type", choices=("cosine", "linear"))
    parser.add_argument(
        "--use_mixed_precision",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save_interval", type=int)
    parser.add_argument("--eval_interval", type=int)
    parser.add_argument("--log_interval", type=int)
    parser.add_argument(
        "--compile_model",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--compile_mode",
        choices=(
            "auto",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
    )
    parser.add_argument(
        "--compile_dynamic",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--peak_flops", type=float)

    parser.add_argument(
        "--early_stop_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--early_stop_patience", type=int)
    parser.add_argument("--early_stop_min_delta", type=float)

    parser.add_argument(
        "--use_swanlab",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--swanlab_project")
    parser.add_argument("--swanlab_workspace")
    parser.add_argument("--swanlab_id")
    parser.add_argument("--swanlab_resume")
    parser.add_argument("--swanlab_experiment_name")
    parser.add_argument("--swanlab_log_dir")
    parser.add_argument("--swanlab_mode")
    return parser


def parse_pretraining_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    return build_pretraining_parser().parse_args(argv)


def apply_cli_overrides(
    model_config: ModelConfig,
    config: PretrainConfig,
    args: argparse.Namespace,
) -> tuple[ModelConfig, PretrainConfig]:
    """Apply CLI values without mutating loaded config objects."""
    model_values = {
        "model_name": args.model_name,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_attention_heads": args.num_heads,
        "vocab_size": args.vocab_size,
    }
    model_values = {
        key: value
        for key, value in model_values.items()
        if value is not None
    }
    if args.max_seq_len is not None:
        model_values["max_position_embeddings"] = args.max_seq_len
    if args.dropout is not None:
        model_values.update(
            attention_dropout=args.dropout,
            hidden_dropout=args.dropout,
            embedding_dropout=args.dropout,
        )
    if args.num_experts is not None:
        model_values["moe_num_experts"] = args.num_experts
    if args.top_k is not None:
        model_values["moe_top_k"] = args.top_k
    if model_values:
        model_config = replace(model_config, **model_values)

    training_values = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
        "use_mixed_precision": args.use_mixed_precision,
        "dtype": args.dtype,
        "seed": args.seed,
        "compile_model": args.compile_model,
        "compile_mode": args.compile_mode,
        "compile_dynamic": args.compile_dynamic,
    }
    training_values = {
        key: value
        for key, value in training_values.items()
        if value is not None
    }
    if args.target_tokens is not None:
        training_values["target_tokens"] = parse_token_count(args.target_tokens)
    training = (
        replace(config.training, **training_values)
        if training_values
        else config.training
    )

    optimizer_values = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    optimizer_values = {
        key: value
        for key, value in optimizer_values.items()
        if value is not None
    }
    optimizer = (
        replace(config.optimizer, **optimizer_values)
        if optimizer_values
        else config.optimizer
    )

    scheduler_values = {
        "warmup_ratio": args.warmup_ratio,
        "stable_ratio": args.stable_ratio,
        "min_lr_ratio": args.min_lr_ratio,
        "decay_type": args.decay_type,
    }
    scheduler_values = {
        key: value
        for key, value in scheduler_values.items()
        if value is not None
    }
    scheduler = (
        replace(config.scheduler, **scheduler_values)
        if scheduler_values
        else config.scheduler
    )

    evaluating_values = {
        "batch_size": args.eval_batch_size,
        "eval_interval": args.eval_interval,
    }
    evaluating_values = {
        key: value
        for key, value in evaluating_values.items()
        if value is not None
    }
    evaluating = (
        replace(config.evaluating, **evaluating_values)
        if evaluating_values
        else config.evaluating
    )

    early_values = {
        "enabled": args.early_stop_enabled,
        "patience": args.early_stop_patience,
        "min_delta": args.early_stop_min_delta,
    }
    early_values = {
        key: value
        for key, value in early_values.items()
        if value is not None
    }
    early_stopping = (
        replace(config.early_stopping, **early_values)
        if early_values
        else config.early_stopping
    )

    swanlab_values = {
        "use_swanlab": args.use_swanlab,
        "swanlab_project": args.swanlab_project,
        "swanlab_workspace": args.swanlab_workspace,
        "swanlab_id": args.swanlab_id,
        "swanlab_resume": args.swanlab_resume,
        "swanlab_experiment_name": args.swanlab_experiment_name,
        "swanlab_log_dir": args.swanlab_log_dir,
        "swanlab_mode": args.swanlab_mode,
    }
    swanlab_values = {
        key: value
        for key, value in swanlab_values.items()
        if value is not None
    }
    swanlab = (
        replace(config.swanlab, **swanlab_values)
        if swanlab_values
        else config.swanlab
    )

    data = config.data
    if args.train_batch_size is not None:
        if args.train_batch_size <= 0:
            raise ValueError("--train_batch_size must be positive")
        data = replace(
            data,
            batching=replace(
                data.batching,
                micro_batch_size=args.train_batch_size,
            ),
        )
    if args.max_seq_len is not None:
        data = replace(
            data,
            packing=replace(
                data.packing,
                max_seq_length=args.max_seq_len,
            ),
        )

    return model_config, replace(
        config,
        optimizer=optimizer,
        scheduler=scheduler,
        data=data,
        training=training,
        evaluating=evaluating,
        early_stopping=early_stopping,
        swanlab=swanlab,
    )


__all__ = [
    "apply_cli_overrides",
    "build_pretraining_parser",
    "parse_pretraining_args",
]
