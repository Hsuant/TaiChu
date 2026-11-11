"""Model size, gradient, and hardware diagnostics."""

from __future__ import annotations

import torch


def parameter_count(
    model: torch.nn.Module,
    *,
    trainable_only: bool = True,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


def model_size_mb(
    model: torch.nn.Module,
    *,
    bytes_per_parameter: int = 4,
) -> float:
    if bytes_per_parameter <= 0:
        raise ValueError("bytes_per_parameter must be positive")
    return (
        parameter_count(model, trainable_only=False)
        * bytes_per_parameter
        / (1024 ** 2)
    )


def gradient_summary(model: torch.nn.Module) -> dict[str, float]:
    gradients = [
        parameter.grad.detach().float()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return {"grad_norm": 0.0, "grad_max": 0.0, "grad_min": 0.0}
    return {
        "grad_norm": float(
            torch.linalg.vector_norm(
                torch.stack(
                    [
                        torch.linalg.vector_norm(gradient)
                        for gradient in gradients
                    ]
                )
            ).item()
        ),
        "grad_max": max(float(gradient.abs().max().item()) for gradient in gradients),
        "grad_min": min(float(gradient.abs().min().item()) for gradient in gradients),
    }


def peak_flops() -> float:
    """Return the configured approximate BF16/FP16 peak for known CUDA GPUs."""
    if not torch.cuda.is_available():
        return 0.0
    name = torch.cuda.get_device_name(0).lower()
    devices = {
        "h100": 989e12,
        "a100": 312e12,
        "a800": 312e12,
        "rtx 4090": 330e12,
        "rtx 4080": 291e12,
        "rtx 3090": 142e12,
        "a6000": 155e12,
        "a5000": 110e12,
        "v100": 125e12,
        "l40": 227e12,
        "l20": 239e12,
        "l4": 121e12,
        "t4": 65e12,
    }
    return next(
        (value for key, value in devices.items() if key in name),
        0.0,
    )


__all__ = [
    "gradient_summary",
    "model_size_mb",
    "parameter_count",
    "peak_flops",
]
