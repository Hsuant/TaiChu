"""Transformer 解码器块。

组合因果自注意力和前馈网络，形成单个 Transformer 层。
支持 KV 缓存以加速自回归生成。
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from model.attention import CausalSelfAttention
from model.feedforward import create_ffn
from model.normalization import RMSNorm
from model.moe import MoEConfig


class TransformerBlock(nn.Module):
    """Pre-Norm 形式的 Transformer 解码器块。

    数据流:
        x = x + CausalSelfAttention(RMSNorm(x))
        x = x + FeedForward(RMSNorm(x))

    Attributes:
        norm1: 注意力子层前的归一化
        attention: 因果自注意力模块
        norm2: 前馈网络前的归一化
        feed_forward: 前馈网络模块
        is_moe: 是否为 MoE 层（用于收集辅助损失）
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        ffn_type: str,
        attention_dropout: float,
        hidden_dropout: float,
        rms_norm_eps: float,
        moe_config: Optional[MoEConfig] = None,
    ):
        """初始化 Transformer 块。

        Args:
            hidden_size: 隐藏层维度
            num_attention_heads: 注意力头数
            num_key_value_heads: 键/值头数
            intermediate_size: FFN 中间维度
            ffn_type: FFN 类型，"swiglu"、"standard" 或 "moe"
            attention_dropout: 注意力 dropout
            hidden_dropout: FFN 后的 dropout
            rms_norm_eps: RMSNorm 的 epsilon
            moe_config: MoE 专家网络参数
        """
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attention = CausalSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_dropout=attention_dropout,
        )
        self.norm2 = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.feed_forward = create_ffn(
            ffn_type=ffn_type,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=hidden_dropout,
            moe_config=moe_config,
        )
        self.is_moe = (ffn_type == "moe")

    def get_moe_aux_loss(self) -> torch.Tensor:
        """获取 MoE 层的辅助损失（负载均衡 + z-loss）。

        Returns:
            辅助损失标量张量，若当前层不是 MoE 则返回 0。
        """
        if self.is_moe and hasattr(self.feed_forward, 'get_aux_loss'):
            return self.feed_forward.get_aux_loss()
        # 返回 0 张量，设备与模型参数一致
        device = next(self.parameters()).device
        return torch.tensor(0.0, device=device)


    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """前向传播（支持 KV 缓存，用于自回归生成）。

        Args:
            hidden_states: 输入张量，形状 (batch, seq_len, hidden_size)（通常 seq_len=1）
            cos: RoPE 余弦，形状 (1, seq_len, 1, head_dim)（已偏移至正确位置）
            sin: RoPE 正弦，形状 (1, seq_len, 1, head_dim)
            attention_mask: 可选的注意力掩码
            past_key_value: 上一时刻的 KV 缓存
            use_cache: 是否返回新的 KV 缓存

        Returns:
            (output, present_key_value) 元组
        """
        # ========== 1.注意力子层（Pre-Norm + 残差） ==========
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, present_key_value = self.attention(
            hidden_states, cos, sin, attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # ========== 2.前馈网络子层（Pre-Norm + 残差） ==========
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        # ========== 3.收集 MoE 辅助损失 ==========
        aux_loss = self.get_moe_aux_loss()

        return hidden_states, present_key_value, aux_loss