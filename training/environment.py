"""Process, device, and distributed runtime initialization."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def configure_torch_runtime() -> None:
    """Apply process-wide CUDA and compiler policies before building tensors."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.suppress_errors = True


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(local_rank: int) -> torch.device:
    """Select the current process device and set the CUDA local device."""
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_runtime(
    *,
    seed: int,
    timeout_minutes: int,
) -> RuntimeContext:
    """Initialize distributed state from the standard torchrun environment."""
    if timeout_minutes <= 0:
        raise ValueError("distributed timeout must be positive")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size <= 0:
        raise ValueError("WORLD_SIZE must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("RANK must be in [0, WORLD_SIZE)")
    if local_rank < 0:
        raise ValueError("LOCAL_RANK cannot be negative")
    if torch.cuda.is_available() and local_rank >= torch.cuda.device_count():
        raise ValueError("LOCAL_RANK is invalid for the available CUDA devices")

    device = get_device(local_rank)
    backend: str | None = None
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(minutes=timeout_minutes),
            )

    seed_everything(seed + rank)
    return RuntimeContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
    )


def barrier(runtime: RuntimeContext) -> None:
    """Synchronize all initialized ranks."""
    if not runtime.distributed:
        return
    if runtime.device.type == "cuda":
        torch.distributed.barrier(device_ids=[runtime.local_rank])
    else:
        torch.distributed.barrier()


__all__ = [
    "RuntimeContext",
    "barrier",
    "configure_torch_runtime",
    "get_device",
    "initialize_runtime",
    "seed_everything",
]
