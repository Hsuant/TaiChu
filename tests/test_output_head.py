#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""输出头模块的单元测试。

测试 OutputHead 的前向传播形状及权重共享（外部绑定）。
"""

import unittest
import torch


from model.output_head import OutputHead


class TestOutputHead(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.hidden_size = 768
        self.vocab_size = 50304
        self.batch = 2
        self.seq_len = 8

    def test_forward_shape(self):
        head = OutputHead(self.hidden_size, self.vocab_size)
        hidden = torch.randn(self.batch, self.seq_len, self.hidden_size)
        logits = head(hidden)
        self.assertEqual(logits.shape, (self.batch, self.seq_len, self.vocab_size))

    def test_no_bias(self):
        head = OutputHead(self.hidden_size, self.vocab_size)
        self.assertIsNone(head.linear.bias)

    def test_weight_sharing_external(self):
        """验证外部权重绑定功能（tie_weights 参数仅用于文档，实际绑定由外部完成）。"""
        head = OutputHead(self.hidden_size, self.vocab_size, tie_weights=True)
        # 创建一个可训练参数作为共享权重
        shared_weight = torch.nn.Parameter(torch.randn(self.vocab_size, self.hidden_size))
        head.linear.weight = shared_weight
        # 验证共享后参数相同
        self.assertIs(head.linear.weight, shared_weight)


if __name__ == '__main__':
    unittest.main()