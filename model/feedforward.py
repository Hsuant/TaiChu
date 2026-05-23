"""前馈网络模块。

提供 SwiGLU 激活的前馈网络、标准 FFN 以及 MoE 前馈网络，
均可通过配置切换。所有构造参数均显式传入。
"""

import torch
import torch.nn as nn
from model.moe import MoEConfig, SparseMoE
from typing import Optional


class SwiGLUFeedForward(nn.Module):
    """使用 SwiGLU 激活的前馈网络。

    SwiGLU 形式: output = (x @ W_gate * SILU(x @ W_up)) @ W_down

    Attributes:
        gate_proj: 门控投影矩阵
        up_proj: 上投影矩阵
        down_proj: 下投影矩阵
        dropout: 残差 dropout
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        """初始化 SwiGLU 前馈网络。

        Args:
            hidden_size: 输入/输出维度
            intermediate_size: 中间层维度
            dropout: 残差 dropout 概率
        """
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 初始化权重，参考常见实践
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状 (batch_size, seq_len, hidden_size)

        Returns:
            输出张量，形状与输入相同
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = gate * torch.nn.functional.silu(up)
        output = self.down_proj(hidden)
        return self.dropout(output)


class StandardFFN(nn.Module):
    """标准的前馈网络（GELU激活），用于消融实验。"""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        """初始化标准 FFN。

        Args:
            hidden_size: 输入/输出维度
            intermediate_size: 中间层维度
            dropout: dropout 概率
        """
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class MoEFeedForward(nn.Module):
    """MoE 前馈网络适配器。

    将 SparseMoE 封装为与普通 FFN 相同的接口，
    并存储辅助损失供外部收集。

    Attributes:
        moe: SparseMoE 实例
        aux_loss: 最近一次前向的辅助损失（标量张量）
    """

    def __init__(self, hidden_size: int, moe_config: MoEConfig):
        """初始化 MoE 前馈网络。

        Args:
            hidden_size: 隐藏层维度
            moe_config: MoE 专用配置对象
        """
        super().__init__()
        self.moe = SparseMoE(hidden_size, moe_config)
        self.aux_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状 (batch_size, seq_len, hidden_size)

        Returns:
            输出张量，形状与输入相同。
        """
        output, aux_loss = self.moe(x)
        self.aux_loss = aux_loss
        return output

    def get_aux_loss(self):
        """获取AUX损失
        Returns：
            aux_loss:返回最近一次前向的辅助损失，若未计算则返回 0 张量
        """
        return self.aux_loss if self.aux_loss is not None else torch.tensor(0.0)


def create_ffn(
    ffn_type: str,
    hidden_size: int,
    intermediate_size: int,
    dropout: float = 0.0,
    moe_config: Optional[MoEConfig] = None,
) -> nn.Module:
    """根据类型创建前馈网络。

    Args:
        ffn_type: "swiglu"、"standard" 或 "moe"
        hidden_size: 隐藏层维度
        intermediate_size: 普通 FFN 的中间层维度（对 MoE 无效）
        dropout: FFN 后的 dropout 概率
        moe_config: 当 ffn_type 为 "moe" 时必须提供

    Returns:
        前馈网络模块实例
    """
    if ffn_type == "swiglu":
        return SwiGLUFeedForward(hidden_size, intermediate_size, dropout)
    elif ffn_type == "standard":
        return StandardFFN(hidden_size, intermediate_size, dropout)
    elif ffn_type == "moe":
        if moe_config is None:
            raise ValueError("ffn_type 为 'moe' 时，必须提供 moe_config 参数")
        return MoEFeedForward(hidden_size, moe_config)
    else:
        raise ValueError(f"不支持的 ffn_type: {ffn_type}")