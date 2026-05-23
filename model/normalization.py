"""归一化模块。

提供 RMSNorm 实现，作为 Transformer 中的高效归一化方法。
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMS 归一化（Root Mean Square Layer Normalization）。

    与 LayerNorm 相比，去除了减均值的操作，仅保留缩放，计算更高效。
    公式: RMS(x) = x * weight / sqrt(mean(x^2) + eps)

    Attributes:
        weight: 可学习的缩放参数，形状 (hidden_size,)
        eps: 数值稳定用的小常数
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """初始化 RMSNorm 层。

        Args:
            hidden_size: 特征维度
            eps: 防止除零的小常数
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状 (batch_size, seq_len, hidden_size) 或任意最后维度为 hidden_size 的张量

        Returns:
            归一化后的张量，形状与输入相同
        """
        # 计算均方根
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x.float() / rms
        # 应用可学习缩放，并转换回原始数据类型
        return (self.weight * x_normed).type_as(x)