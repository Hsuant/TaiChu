"""Model, checkpoint, and tokenizer lifecycle for inference."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from chat.config import ChatConfig
from chat.tokenizer import HFChatTokenizer
from model.config import ModelConfig
from model.model import TaiChuModel
from training.model_artifacts import load_model_artifact, read_model_artifact


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    artifact: Path
    global_step: int | None
    validation_loss: float | None
    parameter_count: int
    max_context_tokens: int
    device: torch.device
    dtype: torch.dtype


@dataclass(slots=True)
class ChatRuntime:
    model: TaiChuModel
    tokenizer: HFChatTokenizer
    info: RuntimeInfo


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "auto":
        if device.type == "cuda":
            return (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        return torch.float32
    resolved = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[requested]
    if device.type == "cpu" and resolved == torch.float16:
        raise ValueError("float16 inference is not supported on CPU")
    return resolved


def _model_config_from_artifact(artifact: dict[str, Any]) -> ModelConfig:
    return ModelConfig.from_dict(artifact["model_config"])


def load_chat_runtime(config: ChatConfig) -> ChatRuntime:
    """Load one isolated inference runtime and release checkpoint-only state."""

    if not config.checkpoint.is_file():
        raise FileNotFoundError(f"model artifact not found: {config.checkpoint}")
    artifact_metadata = read_model_artifact(config.checkpoint)
    tokenizer = HFChatTokenizer.from_path(config.tokenizer)
    model_config = _model_config_from_artifact(artifact_metadata)

    model = TaiChuModel(model_config)
    artifact = load_model_artifact(config.checkpoint, model)

    device = _resolve_device(config.device)
    dtype = _resolve_dtype(config.dtype, device)
    model.to(device=device, dtype=dtype)
    model.eval()

    lexical_size = tokenizer.vocab_size
    if lexical_size > model_config.vocab_size:
        raise ValueError(
            f"tokenizer vocabulary size {lexical_size} exceeds model vocab_size "
            f"{model_config.vocab_size}"
        )

    max_context_tokens = (
        config.context_tokens
        if config.context_tokens is not None
        else model_config.max_position_embeddings
    )
    if max_context_tokens > model_config.max_position_embeddings:
        raise ValueError(
            f"context_tokens {max_context_tokens} exceeds checkpoint maximum "
            f"{model_config.max_position_embeddings}"
        )
    if config.sampling.max_new_tokens >= max_context_tokens:
        raise ValueError(
            "sampling.max_new_tokens must leave room for at least one prompt token"
        )

    metrics = artifact["metrics"]
    validation_loss = next(
        (
            float(metrics[name])
            for name in ("val_loss", "best_val_loss", "last_val_loss")
            if isinstance(metrics.get(name), (int, float))
        ),
        None,
    )

    info = RuntimeInfo(
        artifact=config.checkpoint,
        global_step=int(artifact["global_step"]),
        validation_loss=validation_loss,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        max_context_tokens=max_context_tokens,
        device=device,
        dtype=dtype,
    )

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ChatRuntime(model=model, tokenizer=tokenizer, info=info)


__all__ = ["ChatRuntime", "RuntimeInfo", "load_chat_runtime"]
