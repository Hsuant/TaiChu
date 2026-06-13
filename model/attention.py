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


class TaiChuCausalSelfAttention(nn.Module):
    """因果自注意力层，支持 GQA、RoPE 和 KV 缓存。

    Attributes:
        num_attention_heads: 查询头数。
        num_key_value_heads: 键/值头数。当它小于查询头数时实现 GQA。
        head_dim: 每个注意力头的维度。
        q_proj, k_proj, v_proj, o_proj: 投影矩阵。
        attn_dropout: 注意力权重的 Dropout 层。
        use_flash_attention: 是否优先使用 FlashAttention 后端。
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_dropout: float = 0.0,
        use_flash_attention: bool = True,
    ):
        """初始化自注意力层。

        Args:
            hidden_size: 隐藏层维度。
            num_attention_heads: 查询注意力头数。
            num_key_value_heads: 键/值头数。
            attention_dropout: 注意力权重 Dropout 概率。
            use_flash_attention: 是否启用 FlashAttention 后端。
                若硬件支持，推荐开启以提升训练/推理速度。
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
        # FlashAttention
        self.use_flash_attention = use_flash_attention

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
            attention_mask: 可选的外部加性掩码，必须为形状 ``(batch, 1, seq_len_q, total_len)``。
                padding 位置为 ``-inf``，可见位置为 ``0.0``。内部会自动叠加因果掩码。
                若为 ``None``，则仅使用因果掩码。
            past_key_value: 过去的 KV 缓存。
                ``(batch, num_kv_heads, total_len, head_dim)`` 的元组。
            use_cache: 是否返回新的 KV 缓存。

        Returns:
            - output: 注意力输出，形状 ``(batch, seq_len, hidden_size)``。
            - present_key_value: 若 use_cache=True 则返回新的缓存
              ``(k, v)``，形状均为 ``(batch, num_kv_heads, total_len, head_dim)``；
              否则为 ``None``。
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

        # ========== 4. 转换为缓存友好布局： (batch, heads, seq, head_dim) ==========
        # 该布局可避免在拼接缓存时进行昂贵的 transpose 操作。
        q = q.permute(0, 2, 1, 3)  # (batch, num_q_heads, seq_len, head_dim)
        k = k.permute(0, 2, 1, 3)  # (batch, num_kv_heads, seq_len, head_dim)
        v = v.permute(0, 2, 1, 3)  # (batch, num_kv_heads, seq_len, head_dim)

        # ========== 5. KV 缓存拼接 ==========
        if past_key_value is not None:
            past_k, past_v = past_key_value  # 历史键和值，均为 (batch, num_kv_heads, past_len, head_dim)
            # 将当前 K,V 与历史 K,V 在序列长度维度拼接
            k = torch.cat([past_k, k], dim=2)  # (batch, total_len, num_kv_heads, head_dim)
            v = torch.cat([past_v, v], dim=2)  # (batch, total_len, num_kv_heads, head_dim)

        # 若需要缓存，则保存当前拼接后的 K,V 用于后续步骤
        present_key_value = (k, v) if use_cache else None

        # ========== 6. GQA 扩展：将 K,V 扩展至与 Q 相同头数 ==========
        if self.num_key_value_groups > 1:
            # ================================================================
            #  GQA (分组查询注意力) 扩展：将键（K）和值（V）从较少的头数
            #  复制到与查询（Q）相同的头数，以便进行多头注意力计算。
            #
            #  输入形状：
            #    k, v: (batch_size, total_len, num_key_value_heads, head_dim)
            #  输出形状：
            #    k, v: (batch_size, total_len, num_attention_heads, head_dim)
            #
            #  实现方式：
            #    repeat_interleave(dim=1, repeats=num_key_value_groups)
            #    - dim=1 表示在“头维度”上复制每个 kv 头。
            #    - 例如 num_kv_heads=2, num_q_heads=8 → groups=4，
            #      则第 0 个 kv 头复制 4 次，第 1 个 kv 头复制 4 次，
            #      最终得到 8 个头的 k/v。
            # ================================================================
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)  # 显式复制，无隐式拷贝
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # ========== 7. 注意力计算 ==========
        # 训练时使用设定的 dropout 概率，推理时 dropout 为 0
        dropout_p = self.attn_dropout.p if self.training else 0.0

        # 根据配置选择 FlashAttention 后端
        if self.use_flash_attention:
            # 上下文管理器确保在当前计算中优先使用 FlashAttention
            backend_context = torch.nn.attention.sdpa_kernel(
                [torch.nn.attention.SDPBackend.FLASH_ATTENTION]
            )
        else:
            # 回退到默认的自动后端选择
            backend_context = torch.nn.attention.sdpa_kernel(
                [
                    torch.nn.attention.SDPBackend.MATH,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
                ]
            )

        with backend_context:
            if attention_mask is None:
                # 无外部掩码：仅使用因果掩码
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=dropout_p,
                    is_causal=True,
                )
            else:
                # 手动叠加因果掩码与外部 padding 掩码
                # 外部掩码规定为 (batch, 1, q_len, total_len) 的加性掩码，直接使用
                total_len = k.size(2)  # K 经过 GQA 扩展后的总序列长度
                q_len = q.size(2)      # 当前查询序列长度
                # 构造因果加性掩码：上三角区域填充 -inf
                causal_mask = torch.triu(
                    torch.full((q_len, total_len), float("-inf"), device=q.device, dtype=q.dtype),
                    diagonal=1,
                )
                # 直接相加，-inf + 任意值 = -inf，正确屏蔽未来+padding位置
                combined_mask = attention_mask + causal_mask

                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=combined_mask,
                    dropout_p=dropout_p,
                    is_causal=False,  # 已手动叠加，不再使用自动因果
                )

        # ========== 8. 重塑输出 ==========
        # attn_output 形状为 (batch, num_attention_heads, seq_len_q, head_dim)
        # 转换为 (batch, seq_len_q, hidden_size)
        attn_output = (
            attn_output.transpose(1, 2)  # (batch, seq_len_q, num_heads, head_dim)
            .contiguous()
            .reshape(batch_size, seq_len, -1)
        )

        # ========== 9. 输出投影 ==========
        output = self.o_proj(attn_output)  # (batch, seq_len, hidden_size)
        return output, present_key_value