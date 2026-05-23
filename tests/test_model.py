#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TaiChu 语言模型主模块的集成测试。

测试完整的 TaiChuModel，包括前向传播、损失计算、KV 缓存生成以及参数统计。
"""

import unittest
import torch
from types import SimpleNamespace

from model.model import TaiChuModel
from model.moe import MoEConfig


def get_test_config():
    """生成一个用于测试的小型配置（不加载 yaml，直接构造 SimpleNamespace）。"""
    config = SimpleNamespace()
    config.model_name = "TestModel"
    config.vocab_size = 1000
    config.hidden_size = 128
    config.num_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.intermediate_size = 512
    config.max_position_embeddings = 512
    config.rope_theta = 10000.0
    config.rms_norm_eps = 1e-6
    config.tie_word_embeddings = True
    config.ffn_type = "swiglu"
    config.attention_dropout = 0.0
    config.hidden_dropout = 0.0
    config.embedding_dropout = 0.0
    config.use_flash_attention = False
    config.weight_decay = 0.01
    config.learning_rate = 1e-4
    config.beta1 = 0.9
    config.beta2 = 0.95
    config.epsilon = 1e-8
    # MoE 配置（默认为 None）
    config.moe_config = None
    return config


class TestTaiChuModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = get_test_config()
        self.model = TaiChuModel(self.config)
        self.batch = 2
        self.seq_len = 16

    def test_forward_shape_no_labels(self):
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch, self.seq_len))
        output = self.model(input_ids)
        logits, loss = output.logits, output.loss
        self.assertEqual(logits.shape, (self.batch, self.seq_len, self.config.vocab_size))
        self.assertIsNone(loss)

    def test_forward_with_labels(self):
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch, self.seq_len))
        labels = torch.randint(0, self.config.vocab_size, (self.batch, self.seq_len))
        output = self.model(input_ids, labels=labels)
        self.assertIsNotNone(output.loss)
        self.assertGreater(output.loss.item(), 0.0)

    def test_forward_with_attention_mask(self):
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch, self.seq_len))
        # 构造一个简单的因果掩码（下三角）
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len)).unsqueeze(0).unsqueeze(0)
        mask = mask.to(torch.float32)
        mask = mask.masked_fill(mask == 0, float('-inf'))
        output = self.model(input_ids, attention_mask=mask)
        self.assertEqual(output.logits.shape, (self.batch, self.seq_len, self.config.vocab_size))

    def test_forward_with_cache(self):
        # 测试单步生成
        cur_len = 1
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch, cur_len))
        output = self.model.forward(input_ids, use_cache=True)
        logits, past_kv = output.logits, output.past_key_values
        self.assertEqual(logits.shape, (self.batch, cur_len, self.config.vocab_size))
        self.assertEqual(len(past_kv), self.config.num_layers)
        # 检查第一层缓存的形状
        first_k, first_v = past_kv[0]
        self.assertEqual(first_k.shape, (self.batch, cur_len, self.config.num_key_value_heads,
                                         self.config.hidden_size // self.config.num_attention_heads))
        self.assertEqual(first_v.shape, first_k.shape)

    def test_generate(self):
        # 测试生成功能（只生成少量 token 以避免过长时间）
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch, 5))
        generated = self.model.generate(input_ids, max_new_tokens=3, temperature=0.8, top_k=50)
        self.assertEqual(generated.shape, (self.batch, 5 + 3))

    def test_configure_optimizers(self):
        # 仅测试不报错
        optimizer = self.model.configure_optimizers(self.config)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        # 检查参数分组
        self.assertEqual(len(optimizer.param_groups), 2)
        # decay 组 weight_decay 应为 config.weight_decay，no_decay 组为 0
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], self.config.weight_decay)
        self.assertEqual(optimizer.param_groups[1]['weight_decay'], 0.0)

    def test_get_num_params(self):
        num_params = self.model.get_num_params()
        self.assertGreater(num_params, 0)
        # 可训练参数总和应与模型总参数相同
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(num_params, total_params)

    def test_weight_tieing(self):
        # 验证 weight tying 生效：output_head 的权重应与 token_embedding 的权重共享
        self.assertIs(self.model.output_head.linear.weight, self.model.token_embedding.embedding.weight)


class TestTaiChuModelWithMoE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = get_test_config()
        self.config.ffn_type = "moe"
        self.config.moe_config = MoEConfig(
            num_experts=4,
            top_k=2,
            expert_intermediate_size=256,
            num_shared_experts=1,
            shared_expert_intermediate_size=512,
        )
        self.model = TaiChuModel(self.config)
        self.batch = 2
        self.seq_len = 8

    def test_forward_with_moe(self):
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch, self.seq_len))
        output = self.model(input_ids, labels=input_ids)
        self.assertIsNotNone(output.loss)
        self.assertGreater(output.loss.item(), 0.0)


if __name__ == '__main__':
    unittest.main()