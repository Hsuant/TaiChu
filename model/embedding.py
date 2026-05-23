"""词嵌入模块。

将输入的 token ID 序列映射为稠密向量表示。
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """词嵌入层。

    将形状为 (batch_size, seq_len) 的整数张量映射到
    形状为 (batch_size, seq_len, hidden_size) 的浮点张量。

    Attributes:
        embedding: 核心嵌入矩阵，形状 (vocab_size, hidden_size)
        dropout: 可选的嵌入层 dropout
    """

    def __init__(self, vocab_size: int, hidden_size: int, dropout: float = 0.0):
        """初始化词嵌入层。

        Args:
            vocab_size: 词表大小
            hidden_size: 隐藏层维度（d_model）
            dropout: 嵌入层 dropout 概率
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            input_ids: token ID 序列，形状 (batch_size, seq_len)

        Returns:
            嵌入向量序列，形状 (batch_size, seq_len, hidden_size)
        """
        hidden_states = self.embedding(input_ids)
        return self.dropout(hidden_states)