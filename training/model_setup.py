"""Model compilation, graph warmup, and DDP wrapping."""

from __future__ import annotations

import shutil
import sys

import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast


def compile_training_model(
    model: nn.Module,
    *,
    mode: str,
    dynamic: bool,
    distributed: bool = False,
) -> nn.Module:
    resolved_mode = mode
    if mode == "auto":
        if distributed:
            resolved_mode = "default"
        elif torch.cuda.is_available():
            resolved_mode = (
                "max-autotune-no-cudagraphs"
                if dynamic
                else "reduce-overhead"
            )
        else:
            resolved_mode = "default"
    return torch.compile(
        model,
        mode=resolved_mode,
        dynamic=dynamic,
        fullgraph=False,
    )


def compile_preflight() -> tuple[bool, str | None]:
    if not hasattr(torch, "compile"):
        return False, "torch.compile is unavailable"
    if torch.cuda.is_available():
        return True, None
    if sys.platform == "win32" and shutil.which("cl") is None:
        return False, "Windows CPU torch.compile requires MSVC cl in PATH"
    return True, None


def warmup_compiled_training(
    model: nn.Module,
    *,
    device: torch.device,
    vocab_size: int,
    batch_size: int,
    sequence_length: int,
    use_amp: bool,
    dtype: torch.dtype,
) -> None:
    """Compile forward/backward with the real shape while preserving RNG."""
    if batch_size <= 0 or sequence_length < 2 or vocab_size <= 0:
        raise ValueError("compile warmup requires a valid non-empty shape")
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    model.train()
    try:
        input_ids = torch.randint(
            0,
            vocab_size,
            (batch_size, sequence_length),
            dtype=torch.long,
            device=device,
        )
        with torch.enable_grad():
            with autocast(
                device_type=device.type,
                enabled=use_amp,
                dtype=dtype,
            ):
                output = model(
                    input_ids,
                    labels=input_ids,
                    return_logits=False,
                    validate_inputs=False,
                )
            if output.loss_sum is None:
                raise RuntimeError("compile warmup did not produce a loss sum")
            output.loss_sum.backward()
    finally:
        model.zero_grad(set_to_none=True)
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def wrap_distributed(
    model: nn.Module,
    *,
    runtime,
    find_unused_parameters: bool,
) -> nn.Module:
    if not runtime.distributed:
        return model
    kwargs = {"gradient_as_bucket_view": True}
    if runtime.device.type == "cuda":
        kwargs["device_ids"] = [runtime.local_rank]
    if find_unused_parameters:
        kwargs["find_unused_parameters"] = True
    return nn.parallel.DistributedDataParallel(model, **kwargs)


__all__ = [
    "compile_preflight",
    "compile_training_model",
    "warmup_compiled_training",
    "wrap_distributed",
]
