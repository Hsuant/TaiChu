#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""旋转位置编码模块的单元测试。

测试 RoPEPositionEncoding 的缓存和形状，以及 apply_rotary_pos_emb 函数的正确性。
"""

import unittest
import torch

from model.positional_encoding import RoPEPositionEncoding, apply_rotary_pos_emb


class TestRoPEPositionEncoding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.head_dim = 64
        self.max_seq_len = 2048
        self.theta = 10000.0

    def test_cache_shape(self):
        rope = RoPEPositionEncoding(self.head_dim, self.max_seq_len, self.theta)
        seq_len = 128
        cos, sin = rope(seq_len)
        self.assertEqual(cos.shape, (1, seq_len, 1, self.head_dim))
        self.assertEqual(sin.shape, (1, seq_len, 1, self.head_dim))

    def test_cache_consistency(self):
        rope = RoPEPositionEncoding(self.head_dim, self.max_seq_len, self.theta)
        cos1, sin1 = rope(100)
        cos2, sin2 = rope(100)
        # 同一长度两次调用应返回相同缓存切片
        self.assertTrue(torch.equal(cos1, cos2))
        self.assertTrue(torch.equal(sin1, sin2))

    def test_longer_than_cache(self):
        rope = RoPEPositionEncoding(self.head_dim, self.max_seq_len, self.theta)
        # 请求超过 max_seq_len 的长度会触发错误，因为缓存只有 max_seq_len
        with self.assertRaises(IndexError):
            rope(self.max_seq_len + 1)


class TestApplyRotaryPosEmb(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch = 2
        self.seq_len = 8
        self.num_heads = 12
        self.head_dim = 64
        self.rope = RoPEPositionEncoding(self.head_dim, self.seq_len)

    def test_apply_shape(self):
        x = torch.randn(self.batch, self.seq_len, self.num_heads, self.head_dim)
        cos, sin = self.rope(self.seq_len)
        out = apply_rotary_pos_emb(x, cos, sin)
        self.assertEqual(out.shape, x.shape)

    def test_apply_rotation_property(self):
        """测试旋转后的向量模长不变（旋转是正交变换）。"""
        x = torch.randn(self.batch, self.seq_len, self.num_heads, self.head_dim)
        cos, sin = self.rope(self.seq_len)
        out = apply_rotary_pos_emb(x, cos, sin)
        # 计算每个向量的模长
        norm_x = torch.norm(x, dim=-1)
        norm_out = torch.norm(out, dim=-1)
        self.assertTrue(torch.allclose(norm_x, norm_out, atol=1e-5))

    def test_apply_dtype_preservation(self):
        x = torch.randn(self.batch, self.seq_len, self.num_heads, self.head_dim, dtype=torch.float16)
        cos, sin = self.rope(self.seq_len)
        out = apply_rotary_pos_emb(x, cos, sin)
        self.assertEqual(out.dtype, torch.float16)


if __name__ == '__main__':
    unittest.main()