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
        out, components = moe(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(
            set(components),
            {"moe_load_balancing_loss", "moe_router_z_loss"},
        )
        self.assertTrue(
            all(component.dim() == 0 for component in components.values())
        )

    def test_no_shared_expert(self):
        cfg = MoEConfig(num_shared_experts=0)
        moe = SparseMoE(self.hidden_size, cfg)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        out, components = moe(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(len(components), 2)

    def test_load_balancing_loss_nonzero(self):
        moe = SparseMoE(self.hidden_size, self.moe_config)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        moe.train()
        _, components = moe(x)
        # 在随机输入下，负载均衡损失通常 > 0
        self.assertGreater(components["moe_load_balancing_loss"].item(), 0.0)

    def test_eval_mode_aux_loss_zero(self):
        moe = SparseMoE(self.hidden_size, self.moe_config)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        moe.eval()
        _, components = moe(x)
        self.assertEqual(float(components["moe_load_balancing_loss"]), 0.0)
        self.assertEqual(float(components["moe_router_z_loss"]), 0.0)


if __name__ == '__main__':
    unittest.main()
