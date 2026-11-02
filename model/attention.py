"""因果自注意力模块。

支持多头注意力（MHA）与分组查询注意力（GQA），
集成 RoPE、KV Cache，并通过 torch.nn.functional.scaled_dot_product_attention
实现高效计算（内部自动选用 FlashAttention、Memory-Efficient Attention 等最优后端）。
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.positional_encoding import apply_rotary_pos_emb


@dataclass(frozen=True)
class AttentionMask:
    """SDPA 使用的统一布尔掩码计划。

    该计划绑定到一个具体的 batch、Query 长度、Key 长度和设备，禁止跨形状复用。
    """

    attn_mask: Optional[torch.Tensor]
    is_causal: bool
    batch_size: int
    q_len: int
    total_len: int
    device: torch.device


def _causal_visibility(
    q_len: int,
    total_len: int,
    device: torch.device,
) -> torch.Tensor:
    """返回 (q_len, total_len) 的布尔因果可见性矩阵。"""
    if total_len <= 0:
        raise ValueError("total_len 必须为正整数")
    if q_len <= 0:
        raise ValueError("q_len 必须为正整数")
    if total_len < q_len:
        raise ValueError(f"total_len ({total_len}) 不能小于 q_len ({q_len})")
    query_positions = torch.arange(
        total_len - q_len,
        total_len,
        device=device,
    )
    key_positions = torch.arange(total_len, device=device)
    return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)


def normalize_padding_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    total_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """校验并规范化外部二维 Padding Mask。

    契约：
        - 形状必须严格为 (batch_size, total_len)；
        - 仅接受 bool 或整数张量；
        - 非零表示有效 Token，零表示 Padding；
        - 如位于其他设备，显式迁移到目标设备。
    """
    if attention_mask is None:
        return None
    if attention_mask.dim() != 2:
        raise ValueError(
            "attention_mask 仅支持二维 Padding Mask，"
            f"形状应为 (batch, total_len)，当前为 {tuple(attention_mask.shape)}"
        )
    if attention_mask.shape != (batch_size, total_len):
        raise ValueError(
            "attention_mask 形状必须为 "
            f"({batch_size}, {total_len})，当前为 {tuple(attention_mask.shape)}"
        )
    if attention_mask.is_floating_point() or attention_mask.is_complex():
        raise ValueError(
            "attention_mask 仅接受 bool 或整数张量，"
            f"当前 dtype 为 {attention_mask.dtype}"
        )
    return attention_mask.to(device=device).bool()


def _ensure_no_empty_rows(visibility: torch.Tensor) -> torch.Tensor:
    """仅对没有任何可见 Key 的行开放其全局位置对应的自身 Key。"""
    batch_size, q_len, total_len = visibility.shape
    self_key_indices = (
        torch.arange(total_len - q_len, total_len, device=visibility.device)
        .view(1, q_len, 1)
        .expand(batch_size, -1, -1)
    )
    row_has_key = visibility.any(dim=-1, keepdim=True)
    self_key_visible = visibility.gather(-1, self_key_indices)
    fallback_value = self_key_visible | ~row_has_key
    return visibility.scatter(-1, self_key_indices, fallback_value)


def validate_attention_mask(
    plan: AttentionMask,
    batch_size: int,
    q_len: int,
    total_len: int,
    device: torch.device,
) -> None:
    """验证预构造 Mask Plan 与当前注意力计算完全兼容。"""
    if plan.batch_size != batch_size:
        raise ValueError(
            f"Mask Plan batch_size={plan.batch_size} 与当前 batch={batch_size} 不匹配"
        )
    if plan.q_len != q_len or plan.total_len != total_len:
        raise ValueError(
            f"Mask Plan (q_len={plan.q_len}, total_len={plan.total_len}) "
            f"与当前 (q_len={q_len}, total_len={total_len}) 不匹配"
        )
    if plan.device != device:
        raise ValueError(f"Mask Plan device={plan.device} 与当前 device={device} 不匹配")
    if plan.is_causal and plan.attn_mask is not None:
        raise ValueError("is_causal=True 时不得同时提供显式 attn_mask")
    if plan.is_causal and q_len != total_len:
        raise ValueError("非方形注意力不能使用 is_causal=True 快路径")
    if plan.attn_mask is None:
        if not plan.is_causal and q_len != 1:
            raise ValueError("无显式 Mask 且 is_causal=False 仅适用于单 Token 增量解码")
        return

    mask = plan.attn_mask
    if mask.dtype != torch.bool:
        raise ValueError(f"内部 attn_mask 必须为 bool，当前 dtype 为 {mask.dtype}")
    if mask.device != device:
        raise ValueError(f"内部 attn_mask device={mask.device} 与当前 device={device} 不匹配")
    if mask.dim() != 4:
        raise ValueError(f"内部 attn_mask 必须为四维，当前为 {mask.dim()} 维")
    if mask.size(0) not in (1, batch_size):
        raise ValueError(
            f"内部 attn_mask batch 维必须为 1 或 {batch_size}，当前为 {mask.size(0)}"
        )
    if mask.size(1) != 1 or tuple(mask.shape[-2:]) != (q_len, total_len):
        raise ValueError(
            "内部 attn_mask 形状必须为 "
            f"(1-or-{batch_size}, 1, {q_len}, {total_len})，"
            f"当前为 {tuple(mask.shape)}"
        )


def build_attention_mask(
    q_len: int,
    total_len: int,
    batch_size: int,
    attention_mask: Optional[torch.Tensor],
    device: torch.device,
) -> AttentionMask:
    """统一构造全序列、增量解码和 padding 的 SDPA 掩码计划。

    快路径：
        - 无外部 mask 且 q_len == total_len：使用 SDPA is_causal=True；
        - 无外部 mask 且 q_len == 1：无需显式掩码。

    通用路径按全局位置构造布尔可见性 ``key_position <= query_position``，
    因此同一套逻辑适用于普通训练、chunked prefill 和多 Token 增量解码。
    Query 必须对应 Key 序列末尾的连续位置。
    """
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
    if total_len <= 0:
        raise ValueError("total_len 必须为正整数")
    if q_len <= 0 or q_len > total_len:
        raise ValueError(f"必须满足 0 < q_len <= total_len，当前 q_len={q_len}, total_len={total_len}")

    key_is_valid = normalize_padding_mask(
        attention_mask,
        batch_size=batch_size,
        total_len=total_len,
        device=device,
    )

    if key_is_valid is None:
        if q_len == total_len:
            return AttentionMask(
                attn_mask=None,
                is_causal=True,
                batch_size=batch_size,
                q_len=q_len,
                total_len=total_len,
                device=device,
            )
        if q_len == 1:
            return AttentionMask(
                attn_mask=None,
                is_causal=False,
                batch_size=batch_size,
                q_len=q_len,
                total_len=total_len,
                device=device,
            )

        visibility = _causal_visibility(q_len, total_len, device)
        return AttentionMask(
            attn_mask=visibility[None, None, :, :],
            is_causal=False,
            batch_size=batch_size,
            q_len=q_len,
            total_len=total_len,
            device=device,
        )

    visibility = _causal_visibility(q_len, total_len, device).unsqueeze(0)
    visibility = visibility & key_is_valid[:, None, :]
    visibility = _ensure_no_empty_rows(visibility)
    return AttentionMask(
        attn_mask=visibility[:, None, :, :],
        is_causal=False,
        batch_size=batch_size,
        q_len=q_len,
        total_len=total_len,
        device=device,
    )


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
        mask: Optional[AttentionMask] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """前向传播（支持 KV 缓存）。

        Args:
            hidden_states: 输入，形状 (batch_size, seq_len, hidden_size)。
            cos: RoPE 余弦部分，形状 (1, seq_len, 1, head_dim)。
            sin: RoPE 正弦部分，形状 (1, seq_len, 1, head_dim)。
            attention_mask: 可选的二维 Padding Mask，形状
                ``(batch, total_len)``。非零/True 表示有效 Token，
                零/False 表示 Padding。因果约束始终在内部生成。
            mask: 可选的预构造掩码计划。提供后所有层可复用同一掩码，
                此时 ``attention_mask`` 必须为 ``None``。
            past_key_value: 过去的 KV 缓存。
                ``(batch, num_kv_heads, total_len, head_dim)`` 的元组。
            use_cache: 是否返回新的 KV 缓存。

        Returns:
            - output: 注意力输出，形状 ``(batch, seq_len, hidden_size)``。
            - present_key_value: 若 use_cache=True 则返回新的缓存
              ``(k, v)``，形状均为 ``(batch, num_kv_heads, total_len, head_dim)``；
              否则为 ``None``。
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states 必须为三维张量，当前形状为 {tuple(hidden_states.shape)}"
            )
        batch_size, seq_len, _ = hidden_states.shape
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("hidden_states 的 batch 和序列长度必须为正整数")
        if cos.shape[-1] != self.head_dim or sin.shape[-1] != self.head_dim:
            raise ValueError("cos/sin 的 head_dim 与注意力层不一致")
        if cos.size(1) != seq_len or sin.size(1) != seq_len:
            raise ValueError("cos/sin 的序列长度必须与当前 Query 长度一致")
        if cos.shape != sin.shape:
            raise ValueError("cos 与 sin 的形状必须一致")

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
            if past_k.shape[0] != batch_size or past_v.shape[0] != batch_size:
                raise ValueError("KV cache 的 batch 维度与当前输入不一致")
            if past_k.shape[1] != self.num_key_value_heads or past_v.shape[1] != self.num_key_value_heads:
                raise ValueError("KV cache 的 head 维度与模型配置不一致")
            if past_k.shape[-1] != self.head_dim or past_v.shape[-1] != self.head_dim:
                raise ValueError("KV cache 的 head_dim 与模型配置不一致")
            if past_k.shape[2] != past_v.shape[2]:
                raise ValueError("KV cache 的 K/V 序列长度不一致")
            if past_k.device != q.device or past_v.device != q.device:
                raise ValueError("KV cache 的 device 与当前计算不一致")
            if past_k.dtype != q.dtype or past_v.dtype != q.dtype:
                raise ValueError("KV cache 的 dtype 与当前计算不一致")
            # 将当前 K,V 与历史 K,V 在序列长度维度拼接
            k = torch.cat([past_k, k], dim=2)  # (batch, num_kv_heads, total_len, head_dim)
            v = torch.cat([past_v, v], dim=2)  # (batch, num_kv_heads, total_len, head_dim)

        # 若需要缓存，则保存当前拼接后的 K,V 用于后续步骤
        present_key_value = (k, v) if use_cache else None

        # ========== 6. GQA 扩展：将 K,V 扩展至与 Q 相同头数 ==========
        if self.num_key_value_groups > 1:
            # ================================================================
            #  GQA (分组查询注意力) 扩展：将键（K）和值（V）从较少的头数
            #  复制到与查询（Q）相同的头数，以便进行多头注意力计算。
            #
            #  输入形状：
            #    k, v: (batch_size, num_key_value_heads, total_len, head_dim)
            #  输出形状：
            #    k, v: (batch_size, num_attention_heads, total_len, head_dim)
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
        total_len = k.size(2)
        q_len = q.size(2)
        if mask is None:
            mask = build_attention_mask(
                q_len=q_len,
                total_len=total_len,
                batch_size=batch_size,
                attention_mask=attention_mask,
                device=q.device,
            )
        elif attention_mask is not None:
            raise ValueError("attention_mask 与 mask 不能同时提供")
        validate_attention_mask(
            plan=mask,
            batch_size=batch_size,
            q_len=q_len,
            total_len=total_len,
            device=q.device,
        )

        # 根据配置选择 SDPA 后端集合
        if self.use_flash_attention:
            # 优先 FlashAttention，输入不支持时按可用性自动降级到其余后端
            backend_context = torch.nn.attention.sdpa_kernel(
                [
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                    torch.nn.attention.SDPBackend.MATH,
                ]
            )
        else:
            # 显式排除 FlashAttention，仅允许常规后端
            backend_context = torch.nn.attention.sdpa_kernel(
                [
                    torch.nn.attention.SDPBackend.MATH,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
                ]
            )

        with backend_context:
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask.attn_mask,
                dropout_p=dropout_p,
                is_causal=mask.is_causal,
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
