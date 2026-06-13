#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transformer 解码器块的单元测试。

测试 TransformerBlock 的前向传播、KV 缓存以及 MoE 辅助损失收集。
"""

import unittest
import torch
from dataclasses import dataclass
from typing import Optional

from model.transformer_block import TaiChuBlock
from model.moe import MoEConfig
from model.positional_encoding import RoPEPositionEncoding


# 简单配置类，模拟真实 config 对象
@dataclass
class BlockConfig:
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    intermediate_size: int = 3072
    ffn_type: str = "swiglu"
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    use_flash_attention: bool = False
    moe_config: Optional[MoEConfig] = None


class TestTransformerBlock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.hidden_size = 768
        self.num_heads = 12
        self.num_kv_heads = 4
        self.intermediate_size = 3072
        self.batch = 2
        self.seq_len = 8
        self.head_dim = self.hidden_size // self.num_heads
        self.rope = RoPEPositionEncoding(self.head_dim, max_seq_len=self.seq_len)
        self.cos, self.sin = self.rope(self.seq_len)

    def _create_block(self, ffn_type="swiglu", moe_config=None):
        config = BlockConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            intermediate_size=self.intermediate_size,
            ffn_type=ffn_type,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            rms_norm_eps=1e-6,
            use_flash_attention=False,
            moe_config=moe_config,
        )
        return TaiChuBlock(config, layer_idx=0)

    def test_forward_shape(self):
        block = self._create_block()
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        out, _ = block(x, self.cos, self.sin)
        self.assertEqual(out.shape, x.shape)

    def test_kv_cache(self):
        block = self._create_block()
        past_len = 4
        cur_len = 1
        total_len = past_len + cur_len
        x_cur = torch.randn(self.batch, cur_len, self.hidden_size)

        # 内部布局：(batch, num_kv_heads, past_len, head_dim)
        past_k = torch.randn(self.batch, self.num_kv_heads, past_len, self.head_dim)
        past_v = torch.randn(self.batch, self.num_kv_heads, past_len, self.head_dim)

        cos_full, sin_full = self.rope(total_len)
        cos = cos_full[:, -cur_len:, :, :]
        sin = sin_full[:, -cur_len:, :, :]

        out, present_kv = block(
            x_cur, cos, sin,
            past_key_value=(past_k, past_v),
            use_cache=True
        )
        self.assertEqual(out.shape, (self.batch, cur_len, self.hidden_size))
        self.assertIsNotNone(present_kv)
        new_k, new_v = present_kv
        # 期望形状与输入一致：(batch, num_kv_heads, total_len, head_dim)
        self.assertEqual(new_k.shape, (self.batch, self.num_kv_heads, total_len, self.head_dim))
        self.assertEqual(new_v.shape, (self.batch, self.num_kv_heads, total_len, self.head_dim))

    def test_moe_aux_loss_collection(self):
        moe_config = MoEConfig(num_experts=4, top_k=2)
        block = self._create_block(ffn_type="moe", moe_config=moe_config)
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        out, _ = block(x, self.cos, self.sin)
        aux_loss = block.get_moe_aux_loss()
        self.assertIsInstance(aux_loss, torch.Tensor)
        self.assertEqual(aux_loss.dim(), 0)

    def test_non_moe_aux_loss_zero(self):
        block = self._create_block(ffn_type="swiglu")
        x = torch.randn(self.batch, self.seq_len, self.hidden_size)
        out, _ = block(x, self.cos, self.sin)
        aux_loss = block.get_moe_aux_loss()
        self.assertEqual(aux_loss.item(), 0.0)


if __name__ == '__main__':
    unittest.main()