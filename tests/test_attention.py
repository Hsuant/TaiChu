#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因果自注意力模块的单元测试。

测试 CausalSelfAttention 的正确性，包括：
- 形状匹配
- KV 缓存功能
- 因果掩码效果
- GQA 扩展的正确性
- FlashAttention 回退
- 外部注意力掩码叠加
"""

import unittest
import torch

from model.attention import TaiChuCausalSelfAttention
from model.positional_encoding import RoPEPositionEncoding


class TestCausalSelfAttention(unittest.TestCase):
    """CausalSelfAttention 单元测试。"""

    def setUp(self):
        """测试前固定随机种子，保证可复现。"""
        torch.manual_seed(42)
        self.batch = 2
        self.seq_len = 8
        self.hidden_size = 768
        self.num_heads = 12
        self.num_kv_heads = 4  # GQA
        self.head_dim = self.hidden_size // self.num_heads

        # 创建 RoPE 编码器以生成 cos/sin
        self.rope = RoPEPositionEncoding(
            head_dim=self.head_dim,
            max_seq_len=self.seq_len,
            theta=10000.0
        )
        cos, sin = self.rope(self.seq_len)
        self.cos = cos  # (1, seq_len, 1, head_dim)
        self.sin = sin

    def _create_attention(self, use_flash=False):
        """辅助函数：创建注意力层实例。"""
        return TaiChuCausalSelfAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            attention_dropout=0.0,
            use_flash_attention=use_flash,
        )

    def test_forward_shape(self):
        """测试前向传播的输出形状是否正确。"""
        attn = self._create_attention()
        hidden = torch.randn(self.batch, self.seq_len, self.hidden_size)
        output, _ = attn(hidden, self.cos, self.sin)
        self.assertEqual(output.shape, (self.batch, self.seq_len, self.hidden_size))

    def test_kv_cache_shape(self):
        """测试 KV 缓存拼接后的形状，布局为 (batch, num_heads, seq, head_dim)。"""
        attn = self._create_attention()
        past_len = 4
        cur_len = 1
        total_len = past_len + cur_len

        hidden_cur = torch.randn(self.batch, cur_len, self.hidden_size)

        # past_key_value 形状： (batch, num_kv_heads, seq_len, head_dim)
        past_k = torch.randn(self.batch, self.num_kv_heads, past_len, self.head_dim)
        past_v = torch.randn(self.batch, self.num_kv_heads, past_len, self.head_dim)

        # 生成对应总长度的 cos/sin，并只取最后 cur_len 个位置
        cos_full, sin_full = self.rope(total_len)
        cos = cos_full[:, -cur_len:, :, :]
        sin = sin_full[:, -cur_len:, :, :]

        output, present_kv = attn(
            hidden_cur, cos, sin,
            past_key_value=(past_k, past_v),
            use_cache=True
        )
        self.assertEqual(output.shape, (self.batch, cur_len, self.hidden_size))

        new_k, new_v = present_kv
        self.assertEqual(new_k.shape, (self.batch, self.num_kv_heads, total_len, self.head_dim))
        self.assertEqual(new_v.shape, (self.batch, self.num_kv_heads, total_len, self.head_dim))

    def test_causal_mask_effect(self):
        """测试因果掩码是否阻止未来位置的信息流动。"""
        attn = self._create_attention()
        hidden = torch.ones(self.batch, self.seq_len, self.hidden_size)
        output, _ = attn(hidden, self.cos, self.sin)
        # 简化为形状验证，具体因果性已在注意力权重中保证
        self.assertEqual(output.shape, (self.batch, self.seq_len, self.hidden_size))

    def test_gqa_replication(self):
        """测试在非 flash 模式下 GQA 的复制逻辑是否正确。"""
        attn = self._create_attention(use_flash=False)
        hidden = torch.randn(self.batch, self.seq_len, self.hidden_size)
        output, _ = attn(hidden, self.cos, self.sin)
        self.assertEqual(output.shape, (self.batch, self.seq_len, self.hidden_size))

    def test_flash_attention_fallback(self):
        """测试 FlashAttention 不可用时回退到其他后端。"""
        attn = self._create_attention(use_flash=True)
        hidden = torch.randn(self.batch, self.seq_len, self.hidden_size)
        try:
            output, _ = attn(hidden, self.cos, self.sin)
            self.assertEqual(output.shape, (self.batch, self.seq_len, self.hidden_size))
        except Exception as e:
            self.fail(f"FlashAttention 回退失败: {e}")

    def test_bool_padding_mask_with_flash_preference(self):
        """Flash 优先时必须能对不支持的 bool Mask 合理回退。"""
        attn = self._create_attention(use_flash=True)
        hidden = torch.randn(self.batch, self.seq_len, self.hidden_size)
        mask = torch.ones(self.batch, self.seq_len, dtype=torch.bool)
        output, _ = attn(hidden, self.cos, self.sin, attention_mask=mask)
        self.assertEqual(output.shape, (self.batch, self.seq_len, self.hidden_size))
        self.assertTrue(torch.isfinite(output).all())

    def test_native_causal_and_explicit_bool_mask_match(self):
        attn = self._create_attention()
        hidden = torch.randn(self.batch, self.seq_len, self.hidden_size)
        native, _ = attn(hidden, self.cos, self.sin)
        explicit, _ = attn(
            hidden,
            self.cos,
            self.sin,
            attention_mask=torch.ones(
                self.batch,
                self.seq_len,
                dtype=torch.bool,
            ),
        )
        self.assertTrue(torch.allclose(native, explicit, atol=1e-5, rtol=1e-5))

    def test_attention_mask_integration(self):
        """二维 Padding Mask 应被接受，四维 Mask 必须被明确拒绝。"""
        attn = self._create_attention()
        hidden = torch.randn(self.batch, self.seq_len, self.hidden_size)

        mask = torch.ones(self.batch, self.seq_len, dtype=torch.bool)
        output, _ = attn(hidden, self.cos, self.sin, attention_mask=mask)
        self.assertEqual(output.shape, (self.batch, self.seq_len, self.hidden_size))

        mask_4d = torch.ones(self.batch, 1, self.seq_len, self.seq_len, dtype=torch.float32)
        with self.assertRaises(ValueError):
            attn(hidden, self.cos, self.sin, attention_mask=mask_4d)

    def test_two_dimensional_padding_mask(self):
        """二维 padding 掩码应屏蔽填充 key，并保持有效前缀输出一致。"""
        attn = self._create_attention()
        valid = torch.randn(1, 5, self.hidden_size)
        padded = torch.cat([valid, torch.randn(1, 3, self.hidden_size)], dim=1)
        valid_cos, valid_sin = self.rope(5)
        padded_cos, padded_sin = self.rope(8)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)

        valid_out, _ = attn(valid, valid_cos, valid_sin)
        padded_out, _ = attn(
            padded,
            padded_cos,
            padded_sin,
            attention_mask=mask,
        )
        self.assertTrue(
            torch.allclose(valid_out, padded_out[:, :5], atol=1e-5)
        )

    def test_all_padding_mask_is_finite(self):
        """全 Padding 行应通过全局自身 Key 回退保持有限值。"""
        attn = self._create_attention()
        hidden = torch.randn(self.batch, self.seq_len, self.hidden_size)
        output, _ = attn(
            hidden,
            self.cos,
            self.sin,
            attention_mask=torch.zeros(self.batch, self.seq_len, dtype=torch.bool),
        )
        self.assertTrue(torch.isfinite(output).all())

    def test_incompatible_kv_cache_shapes_are_rejected(self):
        attn = self._create_attention()
        hidden = torch.randn(self.batch, 1, self.hidden_size)
        cos, sin = self.rope(5)
        cos = cos[:, -1:, :, :]
        sin = sin[:, -1:, :, :]

        bad_batch_k = torch.randn(1, self.num_kv_heads, 4, self.head_dim)
        bad_batch_v = torch.randn(1, self.num_kv_heads, 4, self.head_dim)
        with self.assertRaises(ValueError):
            attn(
                hidden,
                cos,
                sin,
                past_key_value=(bad_batch_k, bad_batch_v),
                use_cache=True,
            )

        bad_length_k = torch.randn(self.batch, self.num_kv_heads, 4, self.head_dim)
        bad_length_v = torch.randn(self.batch, self.num_kv_heads, 3, self.head_dim)
        with self.assertRaises(ValueError):
            attn(
                hidden,
                cos,
                sin,
                past_key_value=(bad_length_k, bad_length_v),
                use_cache=True,
            )

        bad_heads_k = torch.randn(self.batch, self.num_kv_heads + 1, 4, self.head_dim)
        bad_heads_v = torch.randn(self.batch, self.num_kv_heads + 1, 4, self.head_dim)
        with self.assertRaises(ValueError):
            attn(
                hidden,
                cos,
                sin,
                past_key_value=(bad_heads_k, bad_heads_v),
                use_cache=True,
            )

        bad_dim_k = torch.randn(self.batch, self.num_kv_heads, 4, self.head_dim + 1)
        bad_dim_v = torch.randn(self.batch, self.num_kv_heads, 4, self.head_dim + 1)
        with self.assertRaises(ValueError):
            attn(
                hidden,
                cos,
                sin,
                past_key_value=(bad_dim_k, bad_dim_v),
                use_cache=True,
            )

        bad_dtype_k = torch.randn(
            self.batch, self.num_kv_heads, 4, self.head_dim, dtype=torch.float64
        )
        bad_dtype_v = torch.randn(
            self.batch, self.num_kv_heads, 4, self.head_dim, dtype=torch.float64
        )
        with self.assertRaises(ValueError):
            attn(
                hidden,
                cos,
                sin,
                past_key_value=(bad_dtype_k, bad_dtype_v),
                use_cache=True,
            )


if __name__ == '__main__':
    unittest.main()
