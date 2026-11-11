"""Standalone evaluation for a trained causal language model."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from model.config import ModelConfig, load_model_config
from training.builders import (
    build_model,
    build_validation_loader,
    load_tokenizer,
)
from training.checkpoint import unwrap_model
from training.config import load_pretraining_config
from training.environment import seed_everything
from training.evaluator import CausalLMEvaluator
from training.logging import LogManager
from training.model_artifacts import load_model_artifact, read_model_artifact
from training.model_info import (
    gradient_summary,
    model_size_mb,
    parameter_count,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="TaiChu model evaluation")
    parser.add_argument("--model_config")
    parser.add_argument("--pretrain_config", required=True)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--artifact",
        choices=("best", "final"),
        default="best",
        help="select best_models/best_model.pt or final_models/final_model.pt",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--num_generate_tokens", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_model", action="store_true")
    return parser.parse_args(argv)


def _resolve_device(requested: str | None) -> torch.device:
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            value,
            stream,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def main(argv=None) -> None:
    args = parse_args(argv)
    config = load_pretraining_config(args.pretrain_config)
    if args.eval_batch_size is not None:
        config = replace(
            config,
            evaluating=replace(
                config.evaluating,
                batch_size=args.eval_batch_size,
            ),
        )
    if config.data.storage.validation_path is None:
        raise ValueError("validation_path is required for standalone evaluation")

    device = _resolve_device(args.device)
    seed_everything(args.seed)
    experiment_dir = Path(args.exp_dir)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"experiment directory not found: {experiment_dir}")
    artifact_dirs = {
        "best": experiment_dir / "best_models" / "best_model.pt",
        "final": experiment_dir / "final_models" / "final_model.pt",
    }
    artifact_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else artifact_dirs[args.artifact]
    )
    if not artifact_path.is_file():
        raise FileNotFoundError(f"model artifact not found: {artifact_path}")
    eval_name = (
        Path(args.checkpoint).stem
        if args.checkpoint
        else args.artifact
    )
    output_dir = experiment_dir / "evaluation" / eval_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_manager = LogManager(
        log_dir=str(output_dir),
        tensorboard=False,
        log_file="eval.log",
        console_level=20,
    )
    loader = None
    try:
        log_manager.info(f"设备: {device}")
        log_manager.info(f"实验目录: {experiment_dir}")
        log_manager.info(f"模型产物: {artifact_path}")
        artifact_metadata = read_model_artifact(artifact_path)
        saved_model_config = ModelConfig.from_dict(
            artifact_metadata["model_config"]
        )
        model_config = saved_model_config
        if args.model_config is not None:
            requested = load_model_config(args.model_config)
            if requested.to_dict() != saved_model_config.to_dict():
                raise ValueError(
                    "--model_config does not match the selected model artifact"
                )
        model = build_model(model_config, torch.device("cpu"))
        load_model_artifact(artifact_path, model)
        model.to(device)
        model.eval()
        log_manager.info(
            f"参数量: {parameter_count(model, trainable_only=False):,}; "
            f"大小: {model_size_mb(model):.2f} MB; "
            f"梯度范数: {gradient_summary(model)['grad_norm']:.6f}"
        )
        if args.save_model:
            path = output_dir / "model_inference.pt"
            torch.save(unwrap_model(model).state_dict(), path)
            log_manager.info(f"推理权重已保存: {path}")

        tokenizer_bundle = load_tokenizer(
            config.tokenizer_path,
            data_config=config.data,
            model_vocab_size=model_config.vocab_size,
        )
        loader = build_validation_loader(
            config.data,
            config.evaluating,
            device=device,
            pad_token_id=tokenizer_bundle.pad_token_id,
        )
        use_amp = (
            config.training.use_mixed_precision and device.type == "cuda"
        )
        dtype = (
            torch.bfloat16
            if config.training.dtype == "bfloat16"
            else torch.float16
        )
        evaluator = CausalLMEvaluator(
            model=model,
            tokenizer=tokenizer_bundle.tokenizer,
            device=device,
            use_amp=use_amp,
            dtype=dtype,
            bos_token_id=tokenizer_bundle.bos_token_id,
            eos_token_id=tokenizer_bundle.eos_token_id,
            logger=log_manager.logger,
        )
        result = evaluator.evaluate(
            loader,
            compute_empty_prefix=config.evaluating.empty_prefix_eos_ppl,
        )
        metrics = result.metrics("val_")
        _write_json(output_dir / "eval_metrics.json", metrics)
        log_manager.info(f"评估指标已保存: {output_dir / 'eval_metrics.json'}")

        prompts = (
            config.evaluating.prompts
            if args.generate
            else None
        )
        if prompts:
            samples = evaluator.generate(
                prompts,
                num_tokens=args.num_generate_tokens,
            )
            _write_json(output_dir / "generated_samples.json", samples)
            log_manager.info(
                f"生成样本已保存: {output_dir / 'generated_samples.json'}"
            )
    finally:
        if loader is not None:
            loader.close()
        log_manager.close()


if __name__ == "__main__":
    main()
