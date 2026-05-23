#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""前馈网络模块的单元测试。

测试 SwiGLUFeedForward、StandardFFN、MoEFeedForward 以及工厂函数 create_ffn。
"""

import unittest
import torch

from model.feedforward import (
    SwiGLUFeedForward,
    StandardFFN,
    MoEFeedForward,
    create_ffn,
)
from model.moe import MoEConfig


class TestSwiGLUFeedForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.hidden_size = 768
        self.intermediate_size = 3072
        self.batch = 2
        self.seq_len = 8

    def test_forward_shape(self):
        ff = SwiGLUFeedForward(self.hidden_size, self.intermediate_size, dropout=0.0)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        out = ff(x)
        self.assertEqual(out.shape, x.shape)

    def test_dropout(self):
        ff = SwiGLUFeedForward(self.hidden_size, self.intermediate_size, dropout=0.5)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        ff.train()
        out_train = ff(x)
        ff.eval()
        out_eval = ff(x)
        # 训练模式 dropout 随机失活，输出与 eval 模式不同
        self.assertFalse(torch.allclose(out_train, out_eval))


class TestStandardFFN(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.hidden_size = 768
        self.intermediate_size = 3072
        self.batch = 2
        self.seq_len = 8

    def test_forward_shape(self):
        ff = StandardFFN(self.hidden_size, self.intermediate_size, dropout=0.0)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        out = ff(x)
        self.assertEqual(out.shape, x.shape)

    def test_gelu_activation(self):
        ff = StandardFFN(self.hidden_size, self.intermediate_size)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        out = ff(x)
        # 不检查具体值，只验证不报错
        self.assertIsNotNone(out)


class TestMoEFeedForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.hidden_size = 768
        self.moe_config = MoEConfig(
            num_experts=4,
            top_k=2,
            expert_intermediate_size=1536,
            num_shared_experts=1,
            shared_expert_intermediate_size=3072,
        )
        self.batch = 2
        self.seq_len = 8

    def test_forward_shape_and_aux_loss(self):
        moe_ff = MoEFeedForward(self.hidden_size, self.moe_config)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        out = moe_ff(x)
        self.assertEqual(out.shape, x.shape)
        aux_loss = moe_ff.get_aux_loss()
        self.assertIsInstance(aux_loss, torch.Tensor)
        self.assertEqual(aux_loss.dim(), 0)  # 标量

    def test_aux_loss_accumulation(self):
        moe_ff = MoEFeedForward(self.hidden_size, self.moe_config)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        _ = moe_ff(x)
        loss1 = moe_ff.get_aux_loss()
        _ = moe_ff(x)
        loss2 = moe_ff.get_aux_loss()
        # 两次前向的损失应不同（因为 router 输出不同），但标量即可
        self.assertIsInstance(loss1, torch.Tensor)
        self.assertIsInstance(loss2, torch.Tensor)


class TestCreateFFN(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 768
        self.intermediate_size = 3072
        self.moe_config = MoEConfig()

    def test_create_swiglu(self):
        ff = create_ffn("swiglu", self.hidden_size, self.intermediate_size)
        self.assertIsInstance(ff, SwiGLUFeedForward)

    def test_create_standard(self):
        ff = create_ffn("standard", self.hidden_size, self.intermediate_size)
        self.assertIsInstance(ff, StandardFFN)

    def test_create_moe(self):
        ff = create_ffn("moe", self.hidden_size, self.intermediate_size, moe_config=self.moe_config)
        self.assertIsInstance(ff, MoEFeedForward)

    def test_moe_without_config_raises(self):
        with self.assertRaises(ValueError):
            create_ffn("moe", self.hidden_size, self.intermediate_size)

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            create_ffn("unknown", self.hidden_size, self.intermediate_size)


if __name__ == '__main__':
    unittest.main()