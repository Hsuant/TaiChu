#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RMSNorm 模块的单元测试。

验证 RMSNorm 的前向传播、数值稳定性及形状。
"""

import unittest
import torch
import torch.nn as nn

from model.normalization import RMSNorm


class TestRMSNorm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.hidden_size = 768
        self.batch = 2
        self.seq_len = 8
        self.eps = 1e-6

    def test_forward_shape(self):
        norm = RMSNorm(self.hidden_size, eps=self.eps)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    def test_output_mean_rms(self):
        """检查 RMS 归一化后的结果：均方根应接近 1（由于 eps 影响略有偏差）。"""
        norm = RMSNorm(self.hidden_size, eps=self.eps)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        out = norm(x)
        # 计算每个样本每个序列位置的 RMS（最后一维）
        rms = torch.sqrt(torch.mean(out.float() ** 2, dim=-1) + self.eps)
        # 由于可学习 weight 初始为 1，因此输出 RMS 应接近 1
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-5))

    def test_weight_scale(self):
        """测试可学习权重能够正确缩放输出。"""
        norm = RMSNorm(self.hidden_size, eps=self.eps)
        # 手动设置权重全为 2
        nn.init.constant_(norm.weight, 2.0)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        out = norm(x)
        # 计算 RMS of out，应为 2
        rms = torch.sqrt(torch.mean(out.float() ** 2, dim=-1) + self.eps)
        self.assertTrue(torch.allclose(rms, 2.0 * torch.ones_like(rms), atol=1e-5))

    def test_dtype_preservation(self):
        """输出数据类型应与输入一致。"""
        norm = RMSNorm(self.hidden_size, eps=self.eps)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size, dtype=torch.float16)
        out = norm(x)
        self.assertEqual(out.dtype, torch.float16)


if __name__ == '__main__':
    unittest.main()