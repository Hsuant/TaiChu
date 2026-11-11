"""Minimal, self-contained causal language-model inference artifacts."""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from training.checkpoint import strip_checkpoint_prefixes, unwrap_model


ARTIFACT_TYPE = "taichu-causal-lm"
ARTIFACT_FORMAT_VERSION = 1
ARTIFACT_VARIANTS = {"best", "final"}


def _validate_artifact(artifact: Any) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise ValueError("model artifact must be a dictionary")
    if artifact.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(
            "legacy or unsupported model artifact; run the model artifact "
            "migration before loading it"
        )
    if artifact.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported model artifact format {artifact.get('format_version')!r}"
        )
    if artifact.get("artifact_variant") not in ARTIFACT_VARIANTS:
        raise ValueError("model artifact variant must be 'best' or 'final'")
    required = {"model_state_dict", "model_config", "global_step", "metrics"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"model artifact is missing keys: {sorted(missing)}")
    if not isinstance(artifact["model_state_dict"], Mapping):
        raise ValueError("model_state_dict must be a mapping")
    if not isinstance(artifact["model_config"], Mapping):
        raise ValueError("model_config must be a mapping")
    if not isinstance(artifact["global_step"], int) or artifact["global_step"] < 0:
        raise ValueError("global_step must be a non-negative integer")
    if not isinstance(artifact["metrics"], Mapping):
        raise ValueError("metrics must be a mapping")
    return artifact


def _atomic_torch_save(payload: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def save_model_artifact(
    path: str | Path,
    model: torch.nn.Module,
    *,
    variant: str,
    global_step: int,
    metrics: Mapping[str, float | int | None],
    stop_reason: str | None = None,
) -> Path:
    """Atomically save the minimal data needed for inference and provenance."""
    if variant not in ARTIFACT_VARIANTS:
        raise ValueError("variant must be 'best' or 'final'")
    raw_model = unwrap_model(model)
    config = getattr(raw_model, "config", None)
    if config is None or not hasattr(config, "to_dict"):
        raise ValueError("model must expose a serializable config")
    return save_serialized_model_artifact(
        path,
        variant=variant,
        global_step=global_step,
        metrics=metrics,
        stop_reason=stop_reason,
        model_state_dict=raw_model.state_dict(),
        model_config=config.to_dict(),
    )


def save_serialized_model_artifact(
    path: str | Path,
    *,
    model_state_dict: Mapping[str, Any],
    model_config: Mapping[str, Any],
    variant: str,
    global_step: int,
    metrics: Mapping[str, float | int | None],
    stop_reason: str | None = None,
) -> Path:
    """Save an artifact from already serialized weights and configuration."""
    if variant not in ARTIFACT_VARIANTS:
        raise ValueError("variant must be 'best' or 'final'")
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_variant": variant,
        "model_state_dict": dict(model_state_dict),
        "model_config": dict(model_config),
        "global_step": int(global_step),
        "metrics": dict(metrics),
        "stop_reason": stop_reason,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _validate_artifact(payload)
    return _atomic_torch_save(payload, path)


def read_model_artifact(path: str | Path) -> dict[str, Any]:
    """Read artifact metadata and tensor descriptors with weights-only loading."""
    artifact = torch.load(
        Path(path),
        map_location="meta",
        weights_only=True,
        mmap=True,
    )
    return _validate_artifact(artifact)


def load_model_artifact(
    path: str | Path,
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Load a minimal model artifact into an already constructed CPU model."""
    artifact = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    artifact = _validate_artifact(artifact)
    current_config = getattr(model, "config", None)
    if (
        current_config is not None
        and hasattr(current_config, "to_dict")
        and current_config.to_dict() != artifact["model_config"]
    ):
        raise ValueError("model config does not match the saved artifact")
    unwrap_model(model).load_state_dict(
        strip_checkpoint_prefixes(artifact["model_state_dict"]),
        strict=True,
    )
    return artifact


class BestModelStore:
    """Track the experiment-wide best metric and save a minimal artifact."""

    def __init__(
        self,
        path: str | Path,
        *,
        metric_name: str = "val_loss",
        mode: str = "min",
        initial_metric: float | None = None,
        initial_step: int | None = None,
        model_config: Mapping[str, Any] | None = None,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.path = Path(path)
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric: float | None = None
        self.best_step: int | None = None
        if initial_metric is not None:
            self._observe(float(initial_metric), int(initial_step or 0))
        if self.path.is_file():
            artifact = read_model_artifact(self.path)
            if artifact["artifact_variant"] != "best":
                raise ValueError("best model path contains a non-best artifact")
            if (
                model_config is not None
                and artifact["model_config"] != dict(model_config)
            ):
                raise ValueError(
                    "existing best artifact belongs to a different model config; "
                    "move it to legacy before resuming"
                )
            metric = artifact["metrics"].get(metric_name)
            if metric is None:
                raise ValueError(
                    f"best artifact does not contain metric {metric_name!r}"
                )
            self._observe(float(metric), int(artifact["global_step"]))

    def _is_better(self, value: float) -> bool:
        if self.best_metric is None:
            return True
        if self.mode == "min":
            return value < self.best_metric
        return value > self.best_metric

    def _observe(self, value: float, step: int) -> None:
        if not math.isfinite(value):
            raise ValueError("best-model metric must be finite")
        if self._is_better(value):
            self.best_metric = value
            self.best_step = int(step)

    def save_if_better(
        self,
        model: torch.nn.Module,
        *,
        metrics: Mapping[str, float | int | None],
        global_step: int,
        write: bool,
    ) -> Path | None:
        """Save on one rank, observe on every rank, and advance state after success."""
        if self.metric_name not in metrics:
            raise ValueError(f"metrics must contain {self.metric_name!r}")
        value = float(metrics[self.metric_name])
        if not math.isfinite(value):
            raise ValueError("best-model metric must be finite")
        if not self._is_better(value):
            return None
        saved_path = None
        if write:
            saved_path = save_model_artifact(
                self.path,
                model,
                variant="best",
                global_step=global_step,
                metrics=metrics,
            )
        self._observe(value, global_step)
        return saved_path


__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "ARTIFACT_TYPE",
    "BestModelStore",
    "load_model_artifact",
    "read_model_artifact",
    "save_model_artifact",
    "save_serialized_model_artifact",
]
