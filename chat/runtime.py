"""Model, checkpoint, and tokenizer lifecycle for inference."""

from __future__ import annotations

import gc
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from tqdm import tqdm

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


class _LoadReporter:
    """Render loading stages as a single tqdm bar; sub-steps use postfix."""

    _STAGES = 8

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._started = time.perf_counter()
        self._stage_bar = tqdm(
            total=self._STAGES,
            desc="加载模型",
            unit="阶段",
            dynamic_ncols=True,
            leave=True,
            position=0,
            disable=not enabled,
        )

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._started

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self._stage_bar.set_description(f"加载模型 | {name}")
        try:
            yield
        except BaseException:
            self._stage_bar.set_description(f"加载模型 | {name} 失败")
            raise
        else:
            self._stage_bar.update(1)

    @contextmanager
    def component_stage(
        self, name: str, total: int
    ) -> Iterator[Callable[[str], None]]:
        """A stage whose internal components are shown via postfix text."""
        self._stage_bar.set_description(f"加载模型 | {name}")
        done = 0

        def report(message: str) -> None:
            nonlocal done
            done = min(done + 1, total)
            self._stage_bar.set_postfix_str(
                f"{message} [{done}/{total}]", refresh=False
            )
            self._stage_bar.refresh()

        try:
            yield report
        except BaseException:
            self._stage_bar.set_description(f"加载模型 | {name} 失败")
            raise
        else:
            self._stage_bar.set_postfix_str("", refresh=False)
            self._stage_bar.update(1)

    def note(self, message: str) -> None:
        if self.enabled:
            tqdm.write(message, file=sys.stderr)

    def close(self) -> None:
        self._stage_bar.close()


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


def load_chat_runtime(
    config: ChatConfig,
    *,
    show_progress: bool = False,
) -> ChatRuntime:
    """Load one isolated inference runtime and release checkpoint-only state."""

    reporter = _LoadReporter(show_progress)
    try:
        if not config.checkpoint.is_file():
            raise FileNotFoundError(
                f"model artifact not found: {config.checkpoint}"
            )
        with reporter.stage("读取模型产物清单"):
            artifact_metadata = read_model_artifact(config.checkpoint)
        with reporter.stage("加载 Tokenizer"):
            tokenizer = HFChatTokenizer.from_path(config.tokenizer)
        model_config = _model_config_from_artifact(artifact_metadata)
        reporter.note(
            f"模型结构: {model_config.num_layers} 层 | "
            f"hidden={model_config.hidden_size} | "
            f"vocab={model_config.vocab_size}"
        )

        with reporter.component_stage(
            "构建模型组件", model_config.num_layers + 5
        ) as report_component:
            model = TaiChuModel(
                model_config,
                on_progress=report_component,
            )
        with reporter.stage("载入模型权重"):
            artifact = load_model_artifact(config.checkpoint, model)

        with reporter.stage("选择推理设备"):
            device = _resolve_device(config.device)
            dtype = _resolve_dtype(config.dtype, device)
        reporter.note(f"推理设备: {device} | 精度: {dtype}")
        with reporter.stage("迁移模型参数"):
            model.to(device=device, dtype=dtype)
        model.eval()

        with reporter.stage("校验上下文预算"):
            lexical_size = tokenizer.vocab_size
            if lexical_size > model_config.vocab_size:
                raise ValueError(
                    f"tokenizer vocabulary size {lexical_size} exceeds "
                    f"model vocab_size {model_config.vocab_size}"
                )

            max_context_tokens = (
                config.context_tokens
                if config.context_tokens is not None
                else model_config.max_position_embeddings
            )
            if max_context_tokens > model_config.max_position_embeddings:
                raise ValueError(
                    f"context_tokens {max_context_tokens} exceeds checkpoint "
                    f"maximum {model_config.max_position_embeddings}"
                )
            if config.sampling.max_new_tokens >= max_context_tokens:
                raise ValueError(
                    "sampling.max_new_tokens must leave room for at least "
                    "one prompt token"
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
            parameter_count=sum(
                parameter.numel() for parameter in model.parameters()
            ),
            max_context_tokens=max_context_tokens,
            device=device,
            dtype=dtype,
        )

        with reporter.stage("释放临时内存"):
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        reporter.close()
        reporter.note(
            f"加载完成: 参数 {info.parameter_count:,} | "
            f"上下文 {info.max_context_tokens} | "
            f"总耗时 {reporter.elapsed:.2f}s"
        )
        return ChatRuntime(model=model, tokenizer=tokenizer, info=info)
    finally:
        reporter.close()


__all__ = ["ChatRuntime", "RuntimeInfo", "load_chat_runtime"]