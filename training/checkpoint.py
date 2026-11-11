"""原子化训练检查点与恢复状态管理。"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import LRScheduler


CHECKPOINT_VERSION = 2


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """移除 DDP 与 torch.compile 包装，返回实际模型。"""
    while True:
        if hasattr(model, "module"):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model


def strip_checkpoint_prefixes(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """去除 ``module.`` 与 ``_orig_mod.`` 包装前缀，兼容新旧检查点。"""
    prefixes = ("module.", "_orig_mod.")
    normalized = {}
    for key, value in state_dict.items():
        while key.startswith(prefixes):
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
        normalized[key] = value
    return normalized


def resolve_experiment_dir_from_checkpoint(filepath: str) -> str:
    """返回检查点所属实验目录，供断点恢复沿用原输出与日志目录。"""
    checkpoint_path = Path(filepath).expanduser().resolve()
    if checkpoint_path.parent.name in {"checkpoints", "best_models", "final_models"}:
        return str(checkpoint_path.parent.parent)
    return str(checkpoint_path.parent)


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and CUDA RNG state."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    if state is None:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@dataclass(frozen=True, slots=True)
class CheckpointLoadResult:
    global_step: int
    checkpoint: dict[str, Any]


class CheckpointManager:
    """检查点管理器。

    Attributes:
        output_dir: 检查点保存目录。
        keep_last_n: 保留最近 N 个检查点文件（-1 表示保留全部）。
        saved_checkpoints: 已保存的检查点文件列表。
    """

    def __init__(
        self,
        output_dir: str = "./checkpoints",
        keep_last_n: int = 3,
    ):
        """初始化检查点管理器。

        Args:
            output_dir: 检查点保存目录。
            keep_last_n: 保留最近几个检查点，超过则删除最旧的。
        """
        self.output_dir = str(Path(output_dir).resolve())
        os.makedirs(self.output_dir, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.saved_checkpoints: list[str] = []
        self.protected_checkpoints: set[str] = set()
        self._discover_managed_checkpoints()

    def _discover_managed_checkpoints(self) -> None:
        directory = Path(self.output_dir) / "checkpoints"
        if not directory.is_dir():
            return
        paths = sorted(
            directory.glob("checkpoint-[0-9]*.pt"),
            key=lambda item: int(re.search(r"checkpoint-(\d+)", item.name).group(1)),
        )
        self.saved_checkpoints = [str(path) for path in paths]

    def protect(self, filepath: str | Path) -> None:
        """Keep a checkpoint during retention even when it is not the newest."""
        self.protected_checkpoints.add(str(Path(filepath)))

    def prune(self) -> list[Path]:
        """Apply the rolling retention while preserving explicitly protected files."""
        if self.keep_last_n < 0 or len(self.saved_checkpoints) <= self.keep_last_n:
            return []

        def step(path: str) -> int:
            match = re.search(r"checkpoint-(\d+)", Path(path).name)
            return int(match.group(1)) if match else -1

        ordered = sorted(self.saved_checkpoints, key=step)
        protected = set(self.protected_checkpoints)
        keep = {path for path in ordered if path in protected}
        kept_unprotected = 0
        for path in reversed(ordered):
            if path in protected:
                continue
            if kept_unprotected >= self.keep_last_n:
                break
            keep.add(path)
            kept_unprotected += 1

        removed = []
        for path in ordered:
            if path in keep:
                continue
            target = Path(path)
            target.unlink(missing_ok=True)
            removed.append(target)
        self.saved_checkpoints = [path for path in ordered if path in keep]
        return removed

    def _build_state(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        scaler: Optional[Any] = None,
        global_step: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """构建检查点字典。

        Args:
            model: 模型实例。
            optimizer: 优化器。
            scheduler: 学习率调度器。
            scaler: 混合精度缩放器。
            global_step: 全局步数。
            metrics: 评估指标字典。
            kwargs: 其他自定义状态。

        Returns:
            检查点状态字典。
        """
        raw_model = unwrap_model(model)
        state = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "global_step": global_step,
            "model_state_dict": raw_model.state_dict(),
            "rng_state": capture_rng_state(),
        }
        model_config = getattr(raw_model, "config", None)
        if model_config is not None and hasattr(model_config, "to_dict"):
            state["model_config"] = model_config.to_dict()
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()
        if metrics is not None:
            state["metrics"] = metrics
        state.update(kwargs)
        return state

    def save(
        self,
        filename: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        scaler: Optional[Any] = None,
        global_step: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        managed: bool = True,
        **kwargs,
    ) -> str:
        """保存检查点。

        Args:
            filename: 文件名（如 "checkpoint-1000.pt"）。
            model: 模型。
            optimizer: 优化器。
            scheduler: 调度器。
            scaler: 混合精度缩放器。
            global_step: 全局步数。
            metrics: 评估指标。
            managed: 是否将此检查点纳入滚动删除管理（默认 True）。
                 设为 False 时，文件不会被自动删除，也不计入最近检查点列表。
            kwargs: 其他自定义状态。

        Returns:
            实际保存的文件路径。
        """
        filepath = str(Path(self.output_dir) / filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = self._build_state(model, optimizer, scheduler, scaler, global_step, metrics, **kwargs)
        temporary_path = f"{filepath}.tmp-{os.getpid()}"
        try:
            with open(temporary_path, "wb") as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, filepath)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

        # 维护 saved_checkpoints 列表
        if managed:
            if filepath in self.saved_checkpoints:
                self.saved_checkpoints.remove(filepath)
            self.saved_checkpoints.append(filepath)
            self.prune()
        return filepath

    def load(
        self,
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        scaler: Optional[Any] = None,
        map_location: str = "cpu",
    ) -> CheckpointLoadResult:
        """从检查点文件恢复训练状态。

        Args:
            filepath: 检查点路径。
            model: 模型实例。
            optimizer: 优化器。
            scheduler: 调度器。
            scaler: 混合精度缩放器。
            map_location: 设备映射。

        Returns:
            Immutable load result containing the global step and the raw
            checkpoint needed for task-specific state restoration.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"检查点文件不存在: {filepath}")
        checkpoint = torch.load(
            filepath,
            map_location=map_location,
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint root must be a dictionary")
        if "artifact_type" in checkpoint:
            raise ValueError(
                "inference artifacts cannot be used to resume training; "
                "select a checkpoint from checkpoints/"
            )
        model_state_dict = strip_checkpoint_prefixes(checkpoint["model_state_dict"])
        unwrap_model(model).load_state_dict(model_state_dict)
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        restore_rng_state(checkpoint.get("rng_state"))
        global_step = checkpoint.get("global_step", 0)
        if not isinstance(global_step, int) or global_step < 0:
            raise ValueError("checkpoint global_step must be a non-negative integer")
        return CheckpointLoadResult(
            global_step=global_step,
            checkpoint=checkpoint,
        )

    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新的检查点文件路径。

        Returns:
            最新检查点路径，若不存在则返回 None。
        """
        if self.saved_checkpoints:
            return self.saved_checkpoints[-1]
        directory = Path(self.output_dir) / "checkpoints"
        if not directory.is_dir():
            return None
        files = [path for path in directory.glob("*.pt")]
        if not files:
            return None
        files.sort(key=lambda path: path.stat().st_mtime)
        return str(files[-1])


__all__ = [
    "CHECKPOINT_VERSION",
    "CheckpointLoadResult",
    "CheckpointManager",
    "capture_rng_state",
    "resolve_experiment_dir_from_checkpoint",
    "restore_rng_state",
    "strip_checkpoint_prefixes",
    "unwrap_model",
]
