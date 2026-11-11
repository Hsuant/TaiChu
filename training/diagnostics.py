"""Low-overhead training diagnostics."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch


class SmoothedLossTracker:
    """Exponential moving average plus a bounded loss trend window."""

    def __init__(self, beta: float = 0.9) -> None:
        if not 0.0 < beta < 1.0:
            raise ValueError("loss smoothing beta must be in (0, 1)")
        self.beta = beta
        self.smoothed_loss: float | None = None
        self.losses: list[float] = []

    def update(self, loss: float) -> float:
        self.losses.append(float(loss))
        if len(self.losses) > 100:
            self.losses.pop(0)
        if self.smoothed_loss is None:
            self.smoothed_loss = float(loss)
        else:
            self.smoothed_loss = (
                self.beta * self.smoothed_loss
                + (1.0 - self.beta) * float(loss)
            )
        return self.smoothed_loss

    def metrics(self) -> dict[str, float]:
        if self.smoothed_loss is None:
            return {
                "smoothed_loss": 0.0,
                "loss_trend": 0.0,
                "loss_variance": 0.0,
            }
        if len(self.losses) < 2:
            return {
                "smoothed_loss": self.smoothed_loss,
                "loss_trend": 0.0,
                "loss_variance": 0.0,
            }
        mean = sum(self.losses) / len(self.losses)
        variance = sum((value - mean) ** 2 for value in self.losses) / len(
            self.losses
        )
        return {
            "smoothed_loss": self.smoothed_loss,
            "loss_trend": (self.losses[-1] - self.losses[0])
            / max(abs(self.losses[0]), 1e-8),
            "loss_variance": variance,
        }


class GradientNormMonitor:
    """EMA statistics for the already-computed global gradient norm."""

    def __init__(self, beta: float = 0.99) -> None:
        if not 0.0 < beta < 1.0:
            raise ValueError("gradient smoothing beta must be in (0, 1)")
        self.beta = beta
        self.mean: float | None = None
        self.variance: float | None = None
        self.last_ratio: float | None = None

    def update(self, grad_norm: torch.Tensor | float) -> float:
        value = (
            float(grad_norm.detach().float().item())
            if isinstance(grad_norm, torch.Tensor)
            else float(grad_norm)
        )
        if not math.isfinite(value):
            raise ValueError("gradient norm must be finite")
        squared = value * value
        if self.mean is None:
            self.mean = squared
            self.variance = 0.0
        else:
            old_mean = self.mean
            old_variance = self.variance or 0.0
            self.mean = self.beta * old_mean + (1.0 - self.beta) * squared
            second_moment = (
                self.beta * (old_variance + old_mean * old_mean)
                + (1.0 - self.beta) * squared * squared
            )
            self.variance = max(
                second_moment - self.mean * self.mean,
                1e-12,
            )
        self.last_ratio = min(squared / ((self.variance or 0.0) + 1e-8), 1e6)
        return self.last_ratio

    def metrics(self) -> dict[str, float]:
        if self.mean is None:
            return {"grad_norm_sq_variance_ratio": 0.0}
        result = {
            "grad_norm_sq_variance_ratio": float(self.last_ratio or 0.0),
            "grad_norm_sq_mean": math.sqrt(max(self.mean, 0.0)),
        }
        if self.variance is not None:
            result["grad_norm_sq_std"] = math.sqrt(max(self.variance, 0.0))
        return result


class ModelFlopsUtilization:
    """Approximate model FLOPs utilization from actual padded token counts."""

    def __init__(self, peak_flops: float, log_file: str | Path | None = None) -> None:
        if peak_flops <= 0:
            raise ValueError("peak_flops must be positive")
        self.peak_flops = float(peak_flops)
        self.log_file = Path(log_file) if log_file is not None else None
        self.start_time: float | None = None
        self.last_step_time: float | None = None
        self.total_flops = 0.0
        self.last_instant = 0.0
        self.parameter_count: int | None = None
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        self.start_time = time.time()
        self.last_step_time = self.start_time

    def estimate_step_flops(
        self,
        model: torch.nn.Module,
        padded_tokens: int,
    ) -> float:
        if padded_tokens <= 0:
            return 0.0
        if self.parameter_count is None:
            self.parameter_count = sum(
                parameter.numel() for parameter in model.parameters()
            )
        return 6.0 * self.parameter_count * padded_tokens

    def update_step(self, step_flops: float, step: int) -> dict[str, float]:
        now = time.time()
        self.total_flops += step_flops
        metrics: dict[str, float] = {
            "total_flops_tflops": self.total_flops / 1e12,
            "peak_flops_tflops": self.peak_flops / 1e12,
        }
        if self.start_time is not None:
            elapsed = now - self.start_time
            metrics["elapsed_time"] = elapsed
            if elapsed > 0:
                metrics["mfu_average"] = self.total_flops / (
                    self.peak_flops * elapsed
                )
        if self.last_step_time is not None:
            step_elapsed = now - self.last_step_time
            if step_elapsed > 0:
                self.last_instant = step_flops / (
                    self.peak_flops * step_elapsed
                )
                metrics["mfu_instant"] = self.last_instant
        self.last_step_time = now
        if self.log_file is not None and step % 100 == 0:
            with self.log_file.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps({"step": step, **metrics}, sort_keys=True) + "\n"
                )
        return metrics

    def metrics(self) -> dict[str, float]:
        result = {
            "mfu_average": 0.0,
            "mfu_instant": self.last_instant,
            "total_flops_tflops": self.total_flops / 1e12,
            "peak_flops_tflops": self.peak_flops / 1e12,
        }
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                result["mfu_average"] = self.total_flops / (
                    self.peak_flops * elapsed
                )
        return result


class TrainingDiagnostics:
    """Small composition root for the diagnostics used by the trainer."""

    def __init__(
        self,
        *,
        loss_beta: float = 0.9,
        gradient_beta: float = 0.99,
        peak_flops: float = 0.0,
        flops_log_file: str | Path | None = None,
    ) -> None:
        self.loss = SmoothedLossTracker(loss_beta)
        self.gradient = GradientNormMonitor(gradient_beta)
        self.mfu = (
            ModelFlopsUtilization(peak_flops, flops_log_file)
            if peak_flops > 0
            else None
        )
        if self.mfu is not None:
            self.mfu.start()

    def update_gradient(self, grad_norm: torch.Tensor | float) -> float:
        return self.gradient.update(grad_norm)

    def update_step(
        self,
        *,
        model: torch.nn.Module,
        loss: float,
        grad_norm: torch.Tensor | float,
        padded_tokens: int,
        step: int,
    ) -> dict[str, float]:
        self.loss.update(loss)
        self.gradient.update(grad_norm)
        metrics = self.metrics()
        if self.mfu is not None:
            step_flops = self.mfu.estimate_step_flops(model, padded_tokens)
            metrics.update(self.mfu.update_step(step_flops, step))
        return metrics

    def metrics(self) -> dict[str, float]:
        metrics = self.loss.metrics()
        metrics.update(self.gradient.metrics())
        if self.mfu is not None:
            metrics.update(self.mfu.metrics())
        return metrics


__all__ = [
    "GradientNormMonitor",
    "ModelFlopsUtilization",
    "SmoothedLossTracker",
    "TrainingDiagnostics",
]
