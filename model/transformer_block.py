"""Transformer 解码器块。

组合因果自注意力和前馈网络，形成单个 Transformer 层。
支持 KV 缓存以加速自回归生成。
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from model.attention import TaiChuCausalSelfAttention
from model.feedforward import create_ffn
from model.normalization import RMSNorm


class TaiChuBlock(nn.Module):
    """Pre-Norm 形式的 Transformer 解码器块。

    组成结构：
        输入
         ├── RMSNorm
         ├── 因果自注意力（含 RoPE、GQA、KV 缓存）
         ├── 残差连接（+ Dropout）
         ├── RMSNorm
         ├── 前馈网络（SwiGLU / Standard / MoE）
         └── 残差连接（+ Dropout）
         输出

    数据流:
        x = x + CausalSelfAttention(RMSNorm(x))
        x = x + FeedForward(RMSNorm(x))

    Attributes:
        input_norm: 注意力子层前的归一化
        self_attention: 因果自注意力模块
        post_attention_norm: 前馈网络前的归一化
        feed_forward: 前馈网络模块
        is_moe: 是否为 MoE 层（用于收集辅助损失）
    """

    def __init__(self, config, layer_idx: int = 0):
        """初始化 Transformer 块。

        Args:
            config: 包含所有超参数。
            layer_idx: 该层在模型中的索引（备用，当前未使用）。
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # ========== 1. 因果自注意力 ==========
        self.self_attention = TaiChuCausalSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            attention_dropout=config.attention_dropout,
            use_flash_attention=config.use_flash_attention,
        )

        # ========== 2. 前馈网络（根据 ffn_type 动态选择） ==========
        self.feed_forward = create_ffn(
            ffn_type=config.ffn_type,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.hidden_dropout,
            moe_config=config.moe_config,
        )
        self.is_moe = (config.ffn_type == "moe")

        # ========== 3. 归一化层 ==========
        # 使用 RMSNorm（更轻量且效果与 LayerNorm 相当）
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ========== 4. 残差路径上的 Dropout ==========
        if config.hidden_dropout > 0:
            self.hidden_dropout = nn.Dropout(config.hidden_dropout)
        else:
            self.hidden_dropout = nn.Identity()

    def get_aux_loss(self) -> torch.Tensor:
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
            hidden_states: 输入隐藏状态，形状 ``(batch, seq_len, hidden_size)``。
            cos: RoPE 余弦值，形状 ``(1, seq_len, 1, head_dim)``。
            sin: RoPE 正弦值，形状 ``(1, seq_len, 1, head_dim)``。
            attention_mask: 可选的加性掩码，形状可广播至
                ``(batch, 1, seq_len_q, total_len)``，padding 处为 ``-inf``。
            past_key_value: 之前的 KV 缓存，布局为
                ``(batch, num_kv_heads, total_len, head_dim)`` 的元组 ``(k, v)``。
            use_cache: 是否返回本次计算的 KV 缓存。

        Returns:
            Tuple:
                - output: 该层的输出，形状 ``(batch, seq_len, hidden_size)``。
                - present_key_value: 若 ``use_cache=True`` 则返回新的 KV 缓存，
                  ``(k, v)`` 元组，否则为 ``None``。
                - aux_loss: MoE 辅助损失（标量），非 MoE 层为 0。
        """
        # ========== 1.注意力子层（Pre-Norm + 残差） ==========
        residual = hidden_states
        # 注意力前的归一化
        normed_hidden = self.input_norm(hidden_states)
        attn_output, present_key_value = self.self_attention(
            hidden_states=normed_hidden,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        # 残差连接 + Dropout
        hidden_states = residual + self.hidden_dropout(attn_output)

        # ========== 2.前馈网络子层（Pre-Norm + 残差） ==========
        residual = hidden_states
        # FFN 前的归一化
        normed_hidden = self.post_attention_norm(hidden_states)
        ffn_output = self.feed_forward(normed_hidden)

        # 残差连接 + Dropout
        hidden_states = residual + self.hidden_dropout(ffn_output)

        # 获取本层的 MoE 辅助损失（若不是 MoE 层则返回 0）
        aux_loss = self.get_aux_loss()

        return hidden_states, present_key_value, aux_loss