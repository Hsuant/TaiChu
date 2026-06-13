"""旋转位置编码（RoPE）模块。

实现 Rotary Position Embedding，将相对位置信息注入查询和键向量。
"""

import torch
import torch.nn as nn


class RoPEPositionEncoding(nn.Module):
    """旋转位置编码（RoPE）。

    该模块不包含可学习参数，仅负责预计算频率并返回
    用于旋转的 cos 和 sin 值。

    Attributes:
        cos_cached: 预先计算的余弦值，形状 (1, max_seq_len, 1, head_dim)
        sin_cached: 预先计算的正弦值，形状 (1, max_seq_len, 1, head_dim)
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        """初始化 RoPE 模块。

        Args:
            head_dim: 每个注意力头的维度
            max_seq_len: 支持的最大序列长度
            theta: 旋转频率的基频
        """
        super().__init__()
        assert head_dim % 2 == 0, "head_dim 必须为偶数"

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 预计算频率张量
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        # 创建位置索引，形状 (max_seq_len, 1)
        positions = torch.arange(max_seq_len).unsqueeze(1).float()
        # 计算角度矩阵，形状 (max_seq_len, head_dim/2)
        freqs = torch.matmul(positions, inv_freq.unsqueeze(0))
        # 拼接成完整的 cos 和 sin 缓存，形状 (max_seq_len, head_dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        # 扩展维度以便广播: (1, max_seq_len, 1, head_dim)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(2))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(2))

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """获取指定序列长度的 cos 和 sin 值。

        Args:
            seq_len: 当前输入序列的实际长度

        Returns:
            (cos, sin) 元组，每个张量形状为 (1, seq_len, 1, head_dim)
        """
        if seq_len > self.max_seq_len:
            raise IndexError(
                f"Requested sequence length {seq_len} exceeds maximum supported {self.max_seq_len}"
            )
        return (
            self.cos_cached[:, :seq_len, :, :],
            self.sin_cached[:, :seq_len, :, :],
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将输入张量的最后一维分为两半并执行旋转操作。

    具体操作：将 ``x`` 的前半部分与后半部分交换，并对原先的后半部分取反。
    相当于将向量对视为复数时乘以 i 的旋转效果。

    Args:
        x: 输入张量，最后一维长度必须为偶数。

    Returns:
        旋转后的张量，形状与输入相同。
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """对张量应用旋转位置编码（使用 ``rotate_half`` 方法）。

    按照公式：
        x_rot = x * cos + rotate_half(x) * sin
    该实现完全基于实数运算，避免复数转换开销，是 LLaMA、GPT-NeoX 等
    主流模型的标准做法。

    Args:
        x: 输入张量，形状 ``(batch_size, seq_len, num_heads, head_dim)``。
        cos: 余弦值，形状 ``(1, seq_len, 1, head_dim)``（可广播）。
        sin: 正弦值，形状 ``(1, seq_len, 1, head_dim)``（可广播）。

    Returns:
        应用旋转后的张量，形状与输入相同，数据类型与输入一致。
    """
    # 确保 cos/sin 与输入 x 的 dtype 一致
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)

    # 直接使用输入的数据类型进行运算，避免不必要的 float() 转换
    return (x * cos) + (_rotate_half(x) * sin)