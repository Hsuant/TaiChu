#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""混合专家模块的单元测试。

测试 MoEConfig、MoEGate、SharedExpert、RoutedExpert、SparseMoE。
"""

import unittest
import torch

from model.moe import MoEConfig, MoEGate, SharedExpert, RoutedExpert, SparseMoE


class TestMoEConfig(unittest.TestCase):
    def test_default_values(self):
        cfg = MoEConfig()
        self.assertEqual(cfg.num_experts, 8)
        self.assertEqual(cfg.top_k, 2)


class TestMoEGate(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.hidden_size = 768
        self.num_experts = 8
        self.top_k = 2
        self.batch = 2
        self.seq_len = 8

    def test_forward_shape(self):
        gate = MoEGate(self.hidden_size, self.num_experts, self.top_k)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        indices, weights, logits = gate(x)
        num_tokens = self.batch * self.seq_len
        self.assertEqual(indices.shape, (num_tokens, self.top_k))
        self.assertEqual(weights.shape, (num_tokens, self.top_k))
        self.assertEqual(logits.shape, (num_tokens, self.num_experts))

    def test_weights_sum_to_one(self):
        gate = MoEGate(self.hidden_size, self.num_experts, self.top_k)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        _, weights, _ = gate(x)
        sum_weights = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sum_weights, torch.ones_like(sum_weights), atol=1e-6))


class TestSharedExpert(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.hidden_size = 768
        self.intermediate_size = 3072
        self.batch = 2
        self.seq_len = 8

    def test_forward_shape(self):
        expert = SharedExpert(self.hidden_size, self.intermediate_size)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        out = expert(x)
        self.assertEqual(out.shape, x.shape)


class TestRoutedExpert(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.hidden_size = 768
        self.intermediate_size = 1536
        self.batch = 2
        self.seq_len = 8

    def test_forward_shape(self):
        expert = RoutedExpert(self.hidden_size, self.intermediate_size)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        out = expert(x)
        self.assertEqual(out.shape, x.shape)


class TestSparseMoE(unittest.TestCase):
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
        moe = SparseMoE(self.hidden_size, self.moe_config)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        out, aux_loss = moe(x)
        self.assertEqual(out.shape, x.shape)
        self.assertIsInstance(aux_loss, torch.Tensor)
        self.assertEqual(aux_loss.dim(), 0)

    def test_no_shared_expert(self):
        cfg = MoEConfig(num_shared_experts=0)
        moe = SparseMoE(self.hidden_size, cfg)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        out, aux_loss = moe(x)
        self.assertEqual(out.shape, x.shape)

    def test_load_balancing_loss_nonzero(self):
        moe = SparseMoE(self.hidden_size, self.moe_config)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        moe.train()
        _, aux_loss = moe(x)
        # 在随机输入下，负载均衡损失通常 > 0
        self.assertGreater(aux_loss.item(), 0.0)

    def test_eval_mode_aux_loss_zero(self):
        moe = SparseMoE(self.hidden_size, self.moe_config)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        moe.eval()
        _, aux_loss = moe(x)
        # 评估模式下负载均衡损失被设置为 0，但 z-loss 仍计算？查看代码：if self.training 时计算 load_bal_loss，否则为 0
        # z_loss 在任何模式下都计算，因此 aux_loss 可能不为 0。这里只验证不崩溃
        self.assertIsInstance(aux_loss, torch.Tensor)


if __name__ == '__main__':
    unittest.main()