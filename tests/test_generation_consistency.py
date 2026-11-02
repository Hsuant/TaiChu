#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KV 缓存增量解码与全序列前向的一致性测试。

验证 generate() 依赖的两条增量解码路径：
  1. past_len 定位（model.py 从 KV 缓存取真实序列长度，而非 kv_heads 数）；
  2. 增量注意力掩码（q_len < total_len 时查询应可见全部历史 keys）。

覆盖 GQA / MHA / MoE 三种架构与 FlashAttention 开关，
以及 generate() 采样路径与逐 token 全序列重算的等价性。
"""

import unittest

import torch

from model.config import ModelConfig
from model.model import TaiChuModel


def _make_model(ffn_type: str = "swiglu", use_flash: bool = False) -> TaiChuModel:
    """构造小规模测试模型（GQA: 4 查询头 / 2 KV 头）。"""
    return TaiChuModel(ModelConfig(
        vocab_size=2048, hidden_size=64, num_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=192, max_position_embeddings=256,
        ffn_type=ffn_type, tie_word_embeddings=True,
        use_flash_attention=use_flash,
    )).eval()


def _check_consistency(model: TaiChuModel, seq: torch.Tensor) -> None:
    """全序列前向 vs 逐 token 增量解码，logits 应一致（容差 1e-4）。"""
    with torch.no_grad():
        full = model(seq).logits                      # (B, S, V)
        dec_logits = []
        past = None
        for t in range(seq.size(1)):
            out = model(seq[:, t:t + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            dec_logits.append(out.logits)             # (B, 1, V)
        dec_logits = torch.cat(dec_logits, dim=1)
    diff = (full - dec_logits).abs().max().item()
    assert diff < 1e-4, f"增量解码与全序列 logits 最大差 {diff:.3e} 超出容差"


class TestKVCacheConsistency(unittest.TestCase):
    """增量解码与全序列前向的 logits 一致性。"""

    def test_gqa_swiglu(self):
        torch.manual_seed(0)
        _check_consistency(_make_model("swiglu", use_flash=False),
                          torch.randint(0, 1000, (2, 24)))

    def test_gqa_flash(self):
        torch.manual_seed(0)
        _check_consistency(_make_model("swiglu", use_flash=True),
                          torch.randint(0, 1000, (2, 24)))

    def test_mha(self):
        torch.manual_seed(0)
        cfg = ModelConfig(
            vocab_size=2048, hidden_size=64, num_layers=2,
            num_attention_heads=4, num_key_value_heads=4,  # MHA
            intermediate_size=192, max_position_embeddings=256,
            ffn_type="swiglu", use_flash_attention=False,
        )
        _check_consistency(TaiChuModel(cfg).eval(), torch.randint(0, 1000, (2, 24)))

    def test_moe(self):
        torch.manual_seed(0)
        cfg = ModelConfig(
            vocab_size=2048, hidden_size=64, num_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=192, max_position_embeddings=256,
            ffn_type="moe", moe_num_experts=4, moe_top_k=2,
            moe_expert_intermediate_size=128,
            moe_num_shared_experts=1, moe_shared_expert_intermediate_size=192,
            use_flash_attention=False,
        )
        _check_consistency(TaiChuModel(cfg).eval(), torch.randint(0, 1000, (2, 24)))

    def test_multitoken_incremental(self):
        """带缓存的批量增量（q_len>1）：按全局位置对齐因果，不得泄漏未来 token。"""
        torch.manual_seed(0)
        model = _make_model("swiglu", use_flash=False)
        seq = torch.randint(0, 1000, (2, 8))
        past_len = 5
        new_len = seq.size(1) - past_len  # 3 个新 token 一次前向

        with torch.no_grad():
            # 逐 token 增量（正确参考）
            past = model(seq[:, :past_len], use_cache=True).past_key_values
            step_logits = []
            p = past
            for t in range(past_len, seq.size(1)):
                out = model(seq[:, t:t + 1], past_key_values=p, use_cache=True)
                p = out.past_key_values
                step_logits.append(out.logits)
            step_logits = torch.cat(step_logits, dim=1)

            # 批量增量：一次前向全部新 token
            out = model(seq[:, past_len:], past_key_values=past, use_cache=True)
            multi_logits = out.logits

        diff = (multi_logits - step_logits).abs().max().item()
        self.assertLess(diff, 1e-4,
                        f"多 token 增量泄漏未来，与逐 token 参考最大差 {diff:.3e}")

    def test_generate_matches_reference(self):
        """generate()（KV 缓存）在贪心模式下应与逐 token 全序列重算一致。"""
        torch.manual_seed(0)
        model = _make_model("swiglu", use_flash=False)
        prompt = torch.tensor([[3, 7, 9, 12, 5]])

        gen = model.generate(prompt.clone(), max_new_tokens=6, top_k=1)  # 确定性贪心

        ref = prompt.clone()
        for _ in range(6):
            with torch.no_grad():
                logits = model(ref).logits
            ref = torch.cat([ref, logits[:, -1, :].argmax(-1, keepdim=True)], dim=-1)

        self.assertTrue(torch.equal(gen, ref),
                        f"generate(KV缓存) {gen.tolist()} ≠ 逐token重算 {ref.tolist()}")

    def test_generate_temperature_zero(self):
        """temperature=0 不应产生 NaN，应等价于贪心。"""
        torch.manual_seed(0)
        model = _make_model("swiglu", use_flash=False)
        prompt = torch.tensor([[3, 7, 9, 12, 5]])
        gen_zero = model.generate(prompt.clone(), max_new_tokens=6, temperature=0.0)
        gen_greedy = model.generate(prompt.clone(), max_new_tokens=6, top_k=1)
        self.assertFalse(torch.isnan(gen_zero).any(), "temperature=0 产生 NaN")
        self.assertTrue(torch.equal(gen_zero, gen_greedy),
                        f"temperature=0 {gen_zero.tolist()} ≠ 贪心 {gen_greedy.tolist()}")


if __name__ == "__main__":
    unittest.main()
