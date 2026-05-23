"""因果自注意力模块。

支持多头注意力（MHA）与分组查询注意力（GQA），
集成 RoPE、KV Cache，并通过 torch.nn.functional.scaled_dot_product_attention
实现高效计算（内部自动选用 FlashAttention、Memory-Efficient Attention 等最优后端）。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.positional_encoding import apply_rotary_pos_emb


class CausalSelfAttention(nn.Module):
    """因果自注意力层，支持 GQA、RoPE 和 KV 缓存。

    Attributes:
        num_attention_heads: 查询头数。
        num_key_value_heads: 键/值头数。当它小于查询头数时实现 GQA。
        head_dim: 每个注意力头的维度。
        q_proj, k_proj, v_proj, o_proj: 投影矩阵。
        attn_dropout: 注意力权重的 Dropout 层。
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_dropout: float = 0.0,
    ):
        """初始化自注意力层。

        Args:
            hidden_size: 隐藏层维度。
            num_attention_heads: 查询注意力头数。
            num_key_value_heads: 键/值头数。
            attention_dropout: 注意力权重 Dropout 概率。
        """
        super().__init__()

        # ========== 参数校验 ==========
        # 保证隐藏层维度能被查询头数整除，从而每个头的维度为整数
        assert hidden_size % num_attention_heads == 0, (
            f"hidden_size ({hidden_size}) 必须能被 num_attention_heads ({num_attention_heads}) 整除"
        )
        # 键/值头数不能超过查询头数（GQA 的特性）
        assert num_key_value_heads <= num_attention_heads, "键/值头数不能大于查询头数"
        # 查询头数必须是键/值头数的整数倍，便于分组复制
        assert num_attention_heads % num_key_value_heads == 0, "查询头数必须是键/值头数的整数倍"

        # ========== 基本属性保存 ==========
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        # 每个注意力头的维度
        self.head_dim = hidden_size // num_attention_heads
        # 每个键/值头对应多少个查询头（GQA 分组数）
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        # ========== 线性投影层 ==========
        # Q 投影：hidden_size -> num_attention_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        # K 投影：hidden_size -> num_key_value_heads * head_dim（键头数可能更少）
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        # V 投影：同 K
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        # 输出投影：将多头结果合并回 hidden_size
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

        # 注意力权重的 dropout 层
        self.attn_dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """前向传播（支持 KV 缓存）。

        Args:
            hidden_states: 输入，形状 (batch_size, seq_len, hidden_size)。
            cos: RoPE 余弦部分，形状 (1, seq_len, 1, head_dim)。
            sin: RoPE 正弦部分，形状 (1, seq_len, 1, head_dim)。
            attention_mask: 可选的外部注意力掩码（加性，如 padding mask），
                形状应可广播至 (batch, 1, seq_len, total_len)。
            past_key_value: 过去的 KV 缓存。
            use_cache: 是否返回新的 KV 缓存。

        Returns:
            output: 形状 (batch, seq_len, hidden_size)。
            present_key_value: 若 use_cache=True 则返回 (k, v)，否则 None。
        """
        # 获取批次大小和当前序列长度
        batch_size, seq_len, _ = hidden_states.shape

        # ========== 1. 投影到 Q, K, V ==========
        q = self.q_proj(hidden_states)  # (batch, seq_len, num_attention_heads * head_dim)
        k = self.k_proj(hidden_states)  # (batch, seq_len, num_key_value_heads * head_dim)
        v = self.v_proj(hidden_states)  # (batch, seq_len, num_key_value_heads * head_dim)

        # ========== 2. 重塑为多头形式 ==========
        # 将最后一个维度拆分为 (头数, 头维度)
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # ========== 3. 应用 RoPE ==========
        # 对 Q 和 K 施加旋转位置编码，注入相对位置信息
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # ========== 4. KV 缓存拼接 ==========
        if past_key_value is not None:
            past_k, past_v = past_key_value  # 历史键和值
            # 将当前 K,V 与历史 K,V 在序列长度维度拼接
            k = torch.cat([past_k, k], dim=1)  # (batch, total_len, num_kv_heads, head_dim)
            v = torch.cat([past_v, v], dim=1)  # (batch, total_len, num_kv_heads, head_dim)

        # 若需要缓存，则保存当前拼接后的 K,V 用于后续步骤
        present_key_value = (k, v) if use_cache else None

        # ========== 5. GQA 扩展：将 K,V 扩展至与 Q 相同头数 ==========
        if self.num_key_value_groups > 1:
            # 通过 unsqueeze 增加一个维度，然后 expand 复制，最后 reshape 合并到头维度
            # 步骤：(batch, total_len, num_kv_heads, head_dim)
            #   -> unsqueeze(3) 得到 (batch, total_len, num_kv_heads, 1, head_dim)
            #   -> expand 得到 (batch, total_len, num_kv_heads, num_key_value_groups, head_dim)
            #   -> reshape 得到 (batch, total_len, num_attention_heads, head_dim)
            k = k.unsqueeze(3).expand(-1, -1, -1, self.num_key_value_groups, -1)
            k = k.reshape(batch_size, -1, self.num_attention_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.num_key_value_groups, -1)
            v = v.reshape(batch_size, -1, self.num_attention_heads, self.head_dim)

        # ========== 6. 转换为 (batch, num_heads, seq_len, head_dim) 以适配 SDPA ==========
        # SDPA 要求输入形状为 (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)  # (batch, num_attention_heads, seq_len_q, head_dim)
        k = k.transpose(1, 2)  # (batch, num_attention_heads, total_len, head_dim)
        v = v.transpose(1, 2)  # (batch, num_attention_heads, total_len, head_dim)

        # ========== 7. 注意力计算 ==========
        # 训练时使用设定的 dropout 概率，推理时 dropout 为 0
        dropout_p = self.attn_dropout.p if self.training else 0.0

        if attention_mask is None:
            # 无外部掩码：直接使用因果掩码
            # is_causal=True 告知后端自动构建下三角因果掩码，无需手动传入
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout_p,
                is_causal=True,
            )
        else:
            # 有外部掩码：需要将因果掩码与外部掩码合并
            total_len = k.size(2)          # 键序列总长度（past_len + seq_len）
            seq_len_q = q.size(2)         # 查询序列长度（当前步长）
            past_len = total_len - seq_len_q  # 历史长度

            # 生成因果掩码（布尔）：只屏蔽“键位置 > 查询位置”的部分
            # 查询位置索引：从 past_len 开始，到 past_len + seq_len_q - 1
            q_pos = torch.arange(past_len, past_len + seq_len_q,
                                 device=q.device).unsqueeze(1)   # (seq_len_q, 1)
            # 键位置索引：0 到 total_len - 1
            k_pos = torch.arange(total_len,
                                 device=k.device).unsqueeze(0)   # (1, total_len)
            causal_mask_bool = k_pos > q_pos                     # (seq_len_q, total_len)

            # 转换为加性掩码（0 表示允许，-inf 表示屏蔽）
            causal_mask = torch.zeros(seq_len_q, total_len,
                                      device=q.device, dtype=q.dtype)
            causal_mask.masked_fill_(causal_mask_bool, float("-inf"))

            # 将因果掩码与外部掩码（如 padding mask）相加
            # 先将 causal_mask 扩展为 (1, 1, seq_len_q, total_len) 以便广播
            mask = causal_mask.unsqueeze(0).unsqueeze(0) + attention_mask

            # 使用合并后的掩码调用 SDPA
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=dropout_p,
            )

        # ========== 8. 重塑输出 ==========
        # 将 (batch, num_heads, seq_len_q, head_dim) 转回 (batch, seq_len_q, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(
            batch_size, seq_len, -1
        )

        # ========== 9. 输出投影 ==========
        output = self.o_proj(attn_output)  # (batch, seq_len, hidden_size)
        return output, present_key_value