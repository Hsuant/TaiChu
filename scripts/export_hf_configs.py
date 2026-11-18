#!/usr/bin/env python3
"""Generate Hugging Face-compatible model and generation config files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

from chat.config import ChatConfig, SamplingConfig, load_chat_config
from model.config import ModelConfig, load_model_config
from training.model_artifacts import read_model_artifact


_DEFAULT_CHAT_CONFIG = Path("../configs/chat_80m.yaml")
_DEFAULT_MODEL_CONFIG = Path("../configs/TaiChu_80M.yaml")
_DTYPES = ("float32", "float16", "bfloat16")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate config.json and generation_config.json",
    )
    parser.add_argument("--checkpoint", type=Path, help="model artifact path")
    parser.add_argument(
        "--chat-config",
        type=Path,
        default=_DEFAULT_CHAT_CONFIG,
        help=f"chat YAML path (default: {_DEFAULT_CHAT_CONFIG})",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=_DEFAULT_MODEL_CONFIG,
        help=f"model YAML fallback (default: {_DEFAULT_MODEL_CONFIG})",
    )
    parser.add_argument("--tokenizer", type=Path, help="tokenizer directory")
    parser.add_argument("--output-dir", type=Path, help="output directory")
    parser.add_argument(
        "--dtype",
        choices=_DTYPES,
        default="float32",
        help="torch dtype written to config.json (default: float32)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing config files",
    )
    return parser


def _resolve_checkpoint(
    explicit: Path | None,
    chat_config: ChatConfig,
) -> Path | None:
    if explicit is not None:
        checkpoint = explicit.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"model artifact not found: {checkpoint}")
        return checkpoint
    if chat_config.checkpoint.is_file():
        return chat_config.checkpoint
    return None


def _load_model_config(
    checkpoint: Path | None,
    model_config_path: Path,
) -> ModelConfig:
    if checkpoint is not None:
        artifact = read_model_artifact(checkpoint)
        return ModelConfig.from_dict(artifact["model_config"])
    return load_model_config(model_config_path)


def _load_token_ids(tokenizer_dir: Path, vocab_size: int) -> dict[str, int]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.vocab_size > vocab_size:
        raise ValueError(
            f"tokenizer vocabulary size {tokenizer.vocab_size} exceeds "
            f"model vocab_size {vocab_size}"
        )
    values = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(f"tokenizer is missing: {', '.join(missing)}")
    token_ids = {name: int(value) for name, value in values.items()}
    for name, value in token_ids.items():
        if not 0 <= value < vocab_size:
            raise ValueError(f"{name}={value} is outside model vocabulary")
    return token_ids


def build_model_payload(
    config: ModelConfig,
    token_ids: dict[str, int],
    dtype: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "architectures": ["TaiChuModel"],
        "model_type": "taichu",
        "model_name": config.model_name,
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
        "rope_theta": config.rope_theta,
        "rms_norm_eps": config.rms_norm_eps,
        "hidden_act": config.hidden_act,
        "ffn_type": config.ffn_type,
        "tie_word_embeddings": config.tie_word_embeddings,
        "attention_dropout": config.attention_dropout,
        "hidden_dropout": config.hidden_dropout,
        "embedding_dropout": config.embedding_dropout,
        "use_flash_attention": config.use_flash_attention,
        "torch_dtype": dtype,
        "use_cache": True,
        **token_ids,
    }
    if config.ffn_type == "moe":
        payload.update(
            {
                "moe_num_experts": config.moe_num_experts,
                "moe_top_k": config.moe_top_k,
                "moe_expert_intermediate_size": (
                    config.moe_expert_intermediate_size
                ),
                "moe_num_shared_experts": config.moe_num_shared_experts,
                "moe_shared_expert_intermediate_size": (
                    config.moe_shared_expert_intermediate_size
                ),
                "moe_load_balancing_loss_weight": (
                    config.moe_load_balancing_loss_weight
                ),
                "moe_router_z_loss_weight": config.moe_router_z_loss_weight,
            }
        )
    return payload


def build_generation_payload(
    sampling: SamplingConfig,
    token_ids: dict[str, int],
) -> dict[str, Any]:
    do_sample = sampling.temperature > 0
    return {
        "bos_token_id": token_ids["bos_token_id"],
        "eos_token_id": token_ids["eos_token_id"],
        "pad_token_id": token_ids["pad_token_id"],
        "max_new_tokens": sampling.max_new_tokens,
        "do_sample": do_sample,
        "temperature": sampling.temperature if do_sample else 1.0,
        "top_k": sampling.top_k if sampling.top_k is not None else 0,
        "top_p": sampling.top_p if sampling.top_p is not None else 1.0,
        "repetition_penalty": sampling.repetition_penalty,
        "num_beams": 1,
        "use_cache": True,
    }


def _write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; use --force to overwrite")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        chat_config = load_chat_config(args.chat_config)
        checkpoint = _resolve_checkpoint(args.checkpoint, chat_config)
        model_config_path = args.model_config.expanduser().resolve()
        model_config = _load_model_config(checkpoint, model_config_path)
        tokenizer_dir = (
            args.tokenizer.expanduser().resolve()
            if args.tokenizer is not None
            else chat_config.tokenizer
        )
        if not tokenizer_dir.is_dir():
            raise FileNotFoundError(
                f"tokenizer directory not found: {tokenizer_dir}"
            )
        token_ids = _load_token_ids(tokenizer_dir, model_config.vocab_size)

        max_context = (
            chat_config.context_tokens
            if chat_config.context_tokens is not None
            else model_config.max_position_embeddings
        )
        if chat_config.sampling.max_new_tokens >= max_context:
            raise ValueError(
                "sampling.max_new_tokens must be smaller than context "
                f"{max_context}"
            )

        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else (checkpoint.parent if checkpoint is not None else Path.cwd())
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        model_payload = build_model_payload(
            model_config,
            token_ids,
            args.dtype,
        )
        generation_payload = build_generation_payload(
            chat_config.sampling,
            token_ids,
        )
        config_path = output_dir / "config.json"
        generation_path = output_dir / "generation_config.json"
        if not args.force:
            existing = [
                path
                for path in (config_path, generation_path)
                if path.exists()
            ]
            if existing:
                raise FileExistsError(
                    f"{', '.join(str(path) for path in existing)} already "
                    "exist; use --force to overwrite"
                )
        _write_json(
            config_path,
            model_payload,
            force=True,
        )
        _write_json(
            generation_path,
            generation_payload,
            force=True,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    source = checkpoint if checkpoint is not None else model_config_path
    print(f"模型配置来源: {source}")
    print(f"Tokenizer: {tokenizer_dir}")
    print(f"已写入: {config_path}")
    print(f"已写入: {generation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
