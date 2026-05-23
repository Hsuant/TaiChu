"""输出投影头模块。

将最后一层隐藏状态映射回词表维度以生成 logits。
"""

import torch
import torch.nn as nn


class OutputHead(nn.Module):
    """语言模型输出头。

    通常是一个线性层，将 hidden_size 映射到 vocab_size。

    Attributes:
        linear: 线性变换层，无偏置
    """

    def __init__(self, hidden_size: int, vocab_size: int, tie_weights: bool = False):
        """初始化输出头。

        Args:
            hidden_size: 隐藏层维度
            vocab_size: 词表大小
            tie_weights: 是否与词嵌入层共享权重，初始时不赋值，由外部绑定
        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            hidden_states: 最后一层的隐藏状态，形状 (batch_size, seq_len, hidden_size)

        Returns:
            logits，形状 (batch_size, seq_len, vocab_size)
        """
        return self.linear(hidden_states)