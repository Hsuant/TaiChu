#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decoder-only 架构、数学、训练、KV Cache 与生成的核心验收测试。"""

import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.attention import TaiChuCausalSelfAttention
from model.config import ModelConfig
from model.model import CausalLMOutput, TaiChuModel
from model.positional_encoding import RoPEPositionEncoding, apply_rotary_pos_emb
from utils.checkpoint import (
    CheckpointManager,
    strip_checkpoint_prefixes,
)
from utils.config_loader import load_model_config, load_pretrain_config


ROOT = Path(__file__).resolve().parents[1]
GPT2_TOKENIZER_PATH = ROOT / "taichu_tokenizer" / "gpt2" / "tokenizer.json"
GPT2_MODEL_CONFIG_PATH = ROOT / "configs" / "audit_gpt2_model.yaml"
GPT2_PRETRAIN_CONFIG_PATH = ROOT / "configs" / "audit_gpt2_pretrain.yaml"
GPT2_AUDIT_ASSETS_AVAILABLE = all(
    path.exists()
    for path in (
        GPT2_TOKENIZER_PATH,
        GPT2_MODEL_CONFIG_PATH,
        GPT2_PRETRAIN_CONFIG_PATH,
    )
)


def make_model(
    vocab_size: int = 128,
    use_flash_attention: bool = False,
    ffn_type: str = "swiglu",
) -> TaiChuModel:
    config = ModelConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        ffn_type=ffn_type,
        use_flash_attention=use_flash_attention,
    )
    return TaiChuModel(config)


class TestTokenizerAndConfig(unittest.TestCase):
    @unittest.skipUnless(
        GPT2_AUDIT_ASSETS_AVAILABLE,
        "optional GPT-2 audit assets are not present",
    )
    def test_mature_tokenizer_matches_audit_config(self):
        tokenizer = Tokenizer.from_file(str(GPT2_TOKENIZER_PATH))
        config = load_model_config(str(ROOT / "configs" / "audit_gpt2_model.yaml"))
        pretrain = load_pretrain_config(
            str(ROOT / "configs" / "audit_gpt2_pretrain.yaml")
        )

        self.assertEqual(tokenizer.get_vocab_size(), 50257)
        self.assertEqual(tokenizer.token_to_id("<|endoftext|>"), 50256)
        self.assertEqual(config.vocab_size, tokenizer.get_vocab_size())
        self.assertEqual(
            Path(pretrain.tokenizer_path).resolve(),
            GPT2_TOKENIZER_PATH.resolve(),
        )
        for text in ["The capital of France is", "Hello 世界"]:
            token_ids = tokenizer.encode(text, add_special_tokens=False).ids
            self.assertTrue(all(0 <= token_id < config.vocab_size for token_id in token_ids))

    def test_invalid_architecture_constraints_are_rejected(self):
        with self.assertRaises(ValueError):
            ModelConfig(hidden_size=40, num_attention_heads=8, num_layers=1)
        with self.assertRaises(ValueError):
            ModelConfig(hidden_size=65, num_attention_heads=1, num_layers=1)
        with self.assertRaises(ValueError):
            ModelConfig(hidden_size=64, num_attention_heads=4, num_layers=1, rope_theta=0)
        with self.assertRaises(ValueError):
            ModelConfig(
                hidden_size=64,
                num_attention_heads=4,
                num_layers=1,
                ffn_type="moe",
                moe_num_shared_experts=2,
            )

    def test_bundled_sample_configs_cover_bundled_tokenizer(self):
        tokenizer = Tokenizer.from_file(str(ROOT / "taichu_tokenizer" / "tokenizer.json"))
        for name in ("TaiChu_LLM.yaml", "TaiChu_77M.yaml", "TaiChu_256M.yaml"):
            config = load_model_config(str(ROOT / "configs" / name))
            self.assertEqual(config.vocab_size, tokenizer.get_vocab_size(), name)

    def test_small_model_presets_match_parameter_and_token_budgets(self):
        cases = [
            ("TaiChu_Small_80M.yaml", "train_small_80m.yaml", 77_611_520, 4_000_000_000),
            ("TaiChu_Small_150M.yaml", "train_small_150m.yaml", 149_057_280, 5_200_000_000),
            ("TaiChu_Small_190M.yaml", "train_small_190m.yaml", 191_394_816, 7_000_000_000),
        ]
        for model_name, train_name, expected_params, expected_tokens in cases:
            with self.subTest(model=model_name):
                model_config = load_model_config(
                    str(ROOT / "configs" / model_name)
                )
                pretrain = load_pretrain_config(
                    str(ROOT / "configs" / train_name)
                )
                with torch.device("meta"):
                    model = TaiChuModel(model_config)
                actual_params = sum(
                    parameter.numel() for parameter in model.parameters()
                )
                self.assertEqual(actual_params, expected_params)
                self.assertEqual(
                    pretrain.training.target_tokens,
                    expected_tokens,
                )
                self.assertEqual(
                    pretrain.data.storage.format,
                    "binary_mmap",
                )
                self.assertTrue(pretrain.data.packing.enabled)
                self.assertEqual(pretrain.data.packing.strategy, "stream")
                self.assertTrue(pretrain.data.storage.train_path)
                self.assertIsNotNone(pretrain.data.storage.validation_path)
                self.assertLessEqual(actual_params, 200_000_000)

class TestModelMath(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = make_model(vocab_size=128).eval()

    def test_forward_shape_softmax_and_parameter_count(self):
        input_ids = torch.randint(0, 128, (2, 9))
        with torch.no_grad():
            logits = self.model(input_ids).logits

        self.assertEqual(logits.shape, (2, 9, 128))
        self.assertTrue(torch.isfinite(logits).all())
        probs = F.softmax(logits[:, -1, :], dim=-1)
        self.assertTrue(torch.isfinite(probs).all())
        self.assertTrue((probs >= 0).all())
        self.assertTrue(
            torch.allclose(
                probs.sum(dim=-1),
                torch.ones(2),
                atol=1e-6,
            )
        )
        predicted = logits[:, -1, :].argmax(dim=-1)
        self.assertTrue(((predicted >= 0) & (predicted < 128)).all())
        tokenizer = Tokenizer.from_file(
            str(ROOT / "taichu_tokenizer" / "tokenizer.json")
        )
        self.assertIsInstance(tokenizer.decode([predicted[0].item()]), str)
        self.assertIs(
            self.model.output_head.linear.weight,
            self.model.token_embedding.embedding.weight,
        )

        config = self.model.config
        head_dim = config.hidden_size // config.num_attention_heads
        attention = (
            config.hidden_size * config.hidden_size
            + 2 * config.hidden_size * config.num_key_value_heads * head_dim
            + config.hidden_size * config.hidden_size
        )
        ffn = 3 * config.hidden_size * config.intermediate_size
        per_layer = attention + ffn + 2 * config.hidden_size
        expected = (
            config.vocab_size * config.hidden_size
            + config.num_layers * per_layer
            + config.hidden_size
        )
        self.assertEqual(self.model.get_num_params(), expected)

    def test_future_tokens_do_not_change_prefix_logits(self):
        input_ids = torch.randint(0, 128, (1, 12))
        changed = input_ids.clone()
        changed[:, 6:] = torch.randint(0, 128, (1, 6))

        with torch.no_grad():
            original_logits = self.model(input_ids).logits
            changed_logits = self.model(changed).logits

        max_diff = (original_logits[:, :6] - changed_logits[:, :6]).abs().max().item()
        self.assertLess(max_diff, 1e-6)

    def test_padding_mask_does_not_change_last_prompt_logits(self):
        prompt = torch.tensor([[3, 7, 9, 12, 5]])
        padded = torch.tensor([[3, 7, 9, 12, 5, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)

        with torch.no_grad():
            unpadded_logits = self.model(prompt).logits[:, -1, :]
            padded_logits = self.model(padded, attention_mask=mask).logits[:, 4, :]

        self.assertTrue(torch.allclose(unpadded_logits, padded_logits, atol=1e-5))

    def test_attention_matches_dense_reference(self):
        torch.manual_seed(17)
        attention = TaiChuCausalSelfAttention(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_dropout=0.0,
            use_flash_attention=False,
        ).eval()
        rope = RoPEPositionEncoding(head_dim=8, max_seq_len=8)
        hidden = torch.randn(2, 6, 32)
        cos, sin = rope(6)

        with torch.no_grad():
            output, _ = attention(hidden, cos, sin)
            q = attention.q_proj(hidden).view(2, 6, 4, 8)
            k = attention.k_proj(hidden).view(2, 6, 2, 8)
            v = attention.v_proj(hidden).view(2, 6, 2, 8)
            q = apply_rotary_pos_emb(q, cos, sin).permute(0, 2, 1, 3)
            k = apply_rotary_pos_emb(k, cos, sin).permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            k = k.repeat_interleave(2, dim=1)
            v = v.repeat_interleave(2, dim=1)
            scores = (q @ k.transpose(-2, -1)) / (8 ** 0.5)
            causal_mask = torch.triu(
                torch.full((6, 6), float("-inf")),
                diagonal=1,
            )
            weights = torch.softmax(scores + causal_mask, dim=-1)
            reference_hidden = (weights @ v).transpose(1, 2).reshape(2, 6, 32)
            reference = attention.o_proj(reference_hidden)

        max_diff = (output - reference).abs().max().item()
        self.assertLess(max_diff, 1e-5, f"attention max diff={max_diff:.3e}")

    def test_loss_matches_manual_cross_entropy_and_backward(self):
        model = make_model(vocab_size=128).train()
        input_ids = torch.randint(0, 128, (2, 10))
        labels = input_ids.clone()
        labels[0, -2:] = -100
        attention_mask = torch.ones_like(input_ids)

        before = model.token_embedding.embedding.weight.detach().clone()
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        manual_loss = F.cross_entropy(
            output.logits[..., :-1, :].reshape(-1, 128),
            labels[..., 1:].reshape(-1),
            ignore_index=-100,
        )
        self.assertTrue(torch.allclose(output.loss, manual_loss, atol=1e-6))
        self.assertTrue(torch.isfinite(output.loss))

        output.loss.backward()
        self.assertIsNotNone(model.token_embedding.embedding.weight.grad)
        self.assertTrue(torch.isfinite(model.token_embedding.embedding.weight.grad).all())

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.token_embedding.embedding.weight))

    def test_cpu_autocast_with_padding_mask_is_finite(self):
        model = make_model(vocab_size=128).eval()
        input_ids = torch.randint(0, 128, (3, 10))
        attention_mask = torch.ones_like(input_ids)
        attention_mask[1, -3:] = 0
        attention_mask[2, :] = 0

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model(
                input_ids,
                attention_mask=attention_mask,
            ).logits

        self.assertEqual(logits.shape, (3, 10, 128))
        self.assertTrue(torch.isfinite(logits).all())


class TestIncrementalGeneration(unittest.TestCase):
    def test_padded_batch_generation_matches_unpadded_prompt(self):
        torch.manual_seed(3)
        model = make_model(vocab_size=128).eval()
        prompt = torch.tensor([[3, 7, 9, 12, 5]])
        left_padded = torch.tensor([[0, 0, 3, 7, 9, 12, 5]])
        mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1]], dtype=torch.bool)

        output = model.generate(prompt, max_new_tokens=4, temperature=0.0)
        padded_output = model.generate(
            left_padded,
            max_new_tokens=4,
            temperature=0.0,
            attention_mask=mask,
        )

        self.assertTrue(torch.equal(output, padded_output[:, -output.size(1):]))

    def test_generate_restores_training_mode_and_honors_zero_tokens(self):
        model = make_model(vocab_size=128)
        model.train()
        prompt = torch.tensor([[1, 2, 3]])
        output = model.generate(prompt, max_new_tokens=0)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(output, prompt))

        single_token = model.generate(
            torch.tensor([[5]]),
            max_new_tokens=2,
            temperature=0.0,
        )
        self.assertEqual(single_token.shape, (1, 3))
        self.assertTrue(((single_token >= 0) & (single_token < 128)).all())

    def test_context_limit_is_explicit(self):
        model = make_model(vocab_size=128).eval()
        near_limit = torch.randint(0, 128, (1, 63))
        output = model.generate(near_limit, max_new_tokens=1, temperature=0.0)
        self.assertEqual(output.shape, (1, 64))

        with self.assertRaises(ValueError):
            model.generate(torch.randint(0, 128, (1, 64)), max_new_tokens=1)
        with self.assertRaises(ValueError):
            model.generate(torch.randint(0, 128, (1, 65)), max_new_tokens=0)

    def test_generate_rejects_unsupported_mask_dimensions_and_dtypes(self):
        model = make_model(vocab_size=128).eval()
        prompt = torch.randint(0, 128, (1, 5))
        with self.assertRaises(ValueError):
            model.generate(
                prompt,
                max_new_tokens=1,
                attention_mask=torch.ones(1, 1, 5, 5, dtype=torch.bool),
            )
        with self.assertRaises(ValueError):
            model.generate(
                prompt,
                max_new_tokens=1,
                attention_mask=torch.ones(1, 5, dtype=torch.float32),
            )

    def test_sampling_is_seeded_and_top_filters_have_expected_support(self):
        torch.manual_seed(11)
        model = make_model(vocab_size=128).eval()
        prompt = torch.tensor([[3, 7, 9]])
        torch.manual_seed(123)
        first = model.generate(prompt, max_new_tokens=4, temperature=1.0, top_k=5)
        torch.manual_seed(123)
        second = model.generate(prompt, max_new_tokens=4, temperature=1.0, top_k=5)
        self.assertTrue(torch.equal(first, second))

        logits = torch.tensor([[0.9, 0.6, 0.2, -1.0]])
        top_k = TaiChuModel._filter_logits_for_sampling(logits.clone(), top_k=2)
        self.assertEqual(torch.isfinite(top_k).sum().item(), 2)

        top_p = TaiChuModel._filter_logits_for_sampling(logits.clone(), top_p=0.7)
        self.assertEqual(torch.isfinite(top_p).sum().item(), 2)
        self.assertEqual(torch.isfinite(top_p)[0].tolist(), [True, True, False, False])

        top_p_small = TaiChuModel._filter_logits_for_sampling(logits.clone(), top_p=0.4)
        self.assertEqual(
            torch.isfinite(top_p_small)[0].tolist(),
            [True, False, False, False],
        )

    def test_batch_eos_is_tracked_per_sample(self):
        model = make_model(vocab_size=8)

        def forced_forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None,
            past_key_values=None,
            use_cache=False,
        ):
            batch_size, cur_len = input_ids.shape
            past_len = past_key_values[0][0].size(2) if past_key_values else 0
            total_len = past_len + cur_len
            logits = torch.full((batch_size, cur_len, 8), -10.0)
            logits[0, -1, 1] = 10.0
            logits[1, -1, 2] = 10.0
            cache = [
                (
                    torch.zeros(batch_size, 1, total_len, 1),
                    torch.zeros(batch_size, 1, total_len, 1),
                )
                for _ in self.layers
            ]
            return CausalLMOutput(logits=logits, past_key_values=cache)

        model.forward = types.MethodType(forced_forward, model)
        output = model.generate(
            torch.tensor([[0, 0], [0, 0]]),
            max_new_tokens=4,
            temperature=0.0,
            eos_token_id=2,
        )
        self.assertEqual(output[0].tolist(), [0, 0, 1, 1, 1, 1])
        self.assertEqual(output[1].tolist(), [0, 0, 2, 2, 2, 2])


class TestCheckpointCompatibility(unittest.TestCase):
    def test_compiled_checkpoint_loads_into_plain_model(self):
        torch.manual_seed(5)
        plain = torch.nn.Linear(4, 3)
        compiled = torch.compile(plain)

        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(output_dir=tmp_dir)
            path = manager.save("compiled.pt", model=compiled)

            target = torch.nn.Linear(4, 3)
            manager.load(path, model=target)

        for name, value in plain.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[name]))

    def test_prefix_normalization_handles_nested_wrappers(self):
        normalized = strip_checkpoint_prefixes(
            {
                "module._orig_mod.weight": torch.tensor(1),
                "_orig_mod.bias": torch.tensor(2),
                "plain": torch.tensor(3),
            }
        )
        self.assertEqual(set(normalized), {"weight", "bias", "plain"})


if __name__ == "__main__":
    unittest.main()
