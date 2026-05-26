# -*- coding: utf-8 -*-
"""早停机制模块。

提供 EarlyStopping 类，用于监控验证指标并在模型不再改善时提前终止训练。
支持配置监控指标、耐心值、最小改善阈值和优化模式（min/max）。
"""

from typing import Dict, Any, Optional


class EarlyStopping:
    """早停类。

    监控指定指标，如果连续 patience 次评估都没有达到最小改善阈值则触发停止。

    Attributes:
        monitor: 监控的指标名称（如 'val_loss'）。
        patience: 允许连续无改善的评估次数。
        min_delta: 最小改善阈值，小于该变化视为无改善。
        mode: 'min' 表示指标越小越好，'max' 表示指标越大越好。
        counter: 当前连续无改善的次数。
        best_score: 当前最佳指标值。
        best_step: 取得最佳指标时的步数。
        early_stop: 是否触发早停。
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = 'min',
    ):
        """初始化早停实例。

        Args:
            monitor: 监控的指标名称。
            patience: 早停耐心值，连续无改善的评估次数阈值。
            min_delta: 视为改善的最小变化量。
            mode: 优化方向，'min' 或 'max'。
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        # 状态变量
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_step = 0
        self.early_stop = False

    def _is_better(self, current: float) -> bool:
        """判断当前指标是否优于最佳指标。

        Args:
            current: 当前指标值。

        Returns:
            True 表示优于最佳，False 表示不优于。
        """
        if self.best_score is None:
            return True
        if self.mode == 'min':
            return current < self.best_score - self.min_delta
        else:  # mode == 'max'
            return current > self.best_score + self.min_delta

    def step(self, current: float, step: int) -> bool:
        """每轮验证后更新早停状态。

        调用该方法后，会判断是否触发早停。

        Args:
            current: 当前验证指标值。
            step: 当前全局步数。

        Returns:
            bool: 是否触发早停（True 表示训练应停止）。
        """
        if self.best_score is None:
            # 首次记录，设定最佳分数
            self.best_score = current
            self.best_step = step
            self.counter = 0
        elif self._is_better(current):
            # 有改善，重置计数器，更新最佳分数
            self.best_score = current
            self.best_step = step
            self.counter = 0
        else:
            # 无改善，增加计数器
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def reset(self) -> None:
        """重置早停状态（可用于新训练）。"""
        self.counter = 0
        self.best_score = None
        self.best_step = 0
        self.early_stop = False

    def state_dict(self) -> Dict[str, Any]:
        """返回早停状态字典，用于检查点保存。

        Returns:
            包含 counter, best_score, best_step, early_stop 的字典。
        """
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'best_step': self.best_step,
            'early_stop': self.early_stop,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """从状态字典加载早停状态，用于恢复训练。

        Args:
            state_dict: 包含早停状态信息的字典。
        """
        self.counter = state_dict.get('counter', 0)
        self.best_score = state_dict.get('best_score', None)
        self.best_step = state_dict.get('best_step', 0)
        self.early_stop = state_dict.get('early_stop', False)