"""Adapter from probe configuration to the shared model runtime."""

from __future__ import annotations

from chat.config import ChatConfig
from chat.runtime import ChatRuntime, load_chat_runtime
from .config import ProbeConfig


def load_probe_runtime(config: ProbeConfig) -> ChatRuntime:
    """Load a self-contained model artifact without chat-specific state."""

    runtime_config = ChatConfig(
        checkpoint=config.checkpoint,
        tokenizer=config.tokenizer,
        system_prompt="",
        device=config.device,
        dtype=config.dtype,
        context_tokens=config.context_tokens,
        sampling=config.sampling,
    )
    return load_chat_runtime(runtime_config)


__all__ = ["load_probe_runtime"]
