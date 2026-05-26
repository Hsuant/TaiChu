"""检查点管理器。

提供模型、优化器、调度器等的保存与恢复功能，支持：
- 按指定步数/评估指标保存最佳模型
- 自动保存最近 N 个检查点（滚动删除）
- 从检查点完整恢复训练状态
"""

import os
import torch
from typing import Optional, Dict, Any, Tuple

from torch.optim.lr_scheduler import LRScheduler


class CheckpointManager:
    """检查点管理器。

    Attributes:
        output_dir: 检查点保存目录。
        keep_last_n: 保留最近 N 个检查点文件（-1 表示保留全部）。
        best_metric_name: 用于选择最佳模型的指标名称（如 "val_loss"）。
        best_metric_mode: "min" 或 "max"，越小越好还是越大越好。
        best_metric_value: 当前最佳指标值。
        saved_checkpoints: 已保存的检查点文件列表。
    """

    def __init__(
        self,
        output_dir: str = "./checkpoints",
        keep_last_n: int = 3,
        best_metric_name: str = "val_loss",
        best_metric_mode: str = "min",
    ):
        """初始化检查点管理器。

        Args:
            output_dir: 检查点保存目录。
            keep_last_n: 保留最近几个检查点，超过则删除最旧的。
            best_metric_name: 最佳模型评估指标名称。
            best_metric_mode: 指标优化方向，'min' 或 'max'。
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.best_metric_name = best_metric_name
        self.best_metric_mode = best_metric_mode
        self.best_metric_value = float("inf") if best_metric_mode == "min" else float("-inf")
        self.saved_checkpoints = []

    def _build_state(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        scaler: Optional[Any] = None,
        global_step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
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
        state = {
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
        }
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
        metrics: Optional[Dict[str, float]] = None,
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
            kwargs: 其他自定义状态。

        Returns:
            实际保存的文件路径。
        """
        filepath = os.path.join(self.output_dir, filename)
        state = self._build_state(model, optimizer, scheduler, scaler, global_step, metrics, **kwargs)
        # 保存当前最佳指标值以便恢复
        state["best_metric_value"] = self.best_metric_value
        torch.save(state, filepath)

        # 维护 saved_checkpoints 列表
        self.saved_checkpoints.append(filepath)
        # 如果超出保留数量，删除旧检查点
        if self.keep_last_n > 0 and len(self.saved_checkpoints) > self.keep_last_n:
            old_file = self.saved_checkpoints.pop(0)
            if os.path.exists(old_file):
                os.remove(old_file)
        return filepath

    def save_best(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, float],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        scaler: Optional[Any] = None,
        global_step: int = 0,
        **kwargs
    ) -> Optional[str]:
        """根据指标保存最佳模型。

        若当前指标优于历史最佳，则保存为 'best_model.pt'。

        Args:
            model: 模型。
            metrics: 包含评估指标的字典，需包含 self.best_metric_name。
            optimizer: 优化器。
            scheduler: 调度器。
            scaler: 混合精度缩放器。
            global_step: 全局步数。
            **kwargs: 额外状态（如 early_stopping_state）

        Returns:
            若保存成功则返回文件路径，否则返回 None。
        """
        if self.best_metric_name not in metrics:
            return None
        current_val = metrics[self.best_metric_name]
        is_better = (
            (self.best_metric_mode == "min" and current_val < self.best_metric_value) or
            (self.best_metric_mode == "max" and current_val > self.best_metric_value)
        )
        if is_better:
            self.best_metric_value = current_val
            return self.save(
                "best_model.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
                metrics=metrics,
                **kwargs,
            )
        return None

    def load(
        self,
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        scaler: Optional[Any] = None,
        map_location: str = "cpu",
    ) -> Tuple[int, Optional[float], Dict[str, Any]]:
        """从检查点文件恢复训练状态。

        Args:
            filepath: 检查点路径。
            model: 模型实例。
            optimizer: 优化器。
            scheduler: 调度器。
            scaler: 混合精度缩放器。
            map_location: 设备映射。

        Returns:
            (global_step, best_metric_value, checkpoint):
            - 恢复后的全局步数
            - 最佳指标值
            - 原始检查点字典（用于恢复额外状态，如早停计数器）
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"检查点文件不存在: {filepath}")
        checkpoint = torch.load(filepath, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "best_metric_value" in checkpoint:
            self.best_metric_value = checkpoint["best_metric_value"]
        global_step = checkpoint.get("global_step", 0)
        best_metric_value = checkpoint.get("best_metric_value", None)
        return global_step, best_metric_value, checkpoint

    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新的检查点文件路径。

        Returns:
            最新检查点路径，若不存在则返回 None。
        """
        if self.saved_checkpoints:
            return self.saved_checkpoints[-1]
        # 也可扫描目录
        files = [f for f in os.listdir(self.output_dir) if f.endswith(".pt") and f != "best_model.pt"]
        if not files:
            return None
        files.sort(key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)))
        return os.path.join(self.output_dir, files[-1])