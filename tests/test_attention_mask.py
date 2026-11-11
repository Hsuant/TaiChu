#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一 AttentionMask 的边界、因果、Padding 与复用测试。"""

import unittest
from unittest.mock import patch

import torch

import model.model as model_module
from model.attention import (
    AttentionMask,
    build_attention_mask,
    validate_attention_mask,
)
from model.config import ModelConfig
from model.model import TaiChuModel


CPU = torch.device("cpu")


def plan_visibility(plan: AttentionMask) -> torch.Tensor:
    """返回 (batch, q_len, total_len) 的布尔可见性。"""
    if plan.attn_mask is None:
        raise AssertionError("当前 Plan 没有显式 Mask")
    return plan.attn_mask[:, 0, :, :]


class TestAttentionMask(unittest.TestCase):
    def test_full_sequence_fast_path_uses_native_causal(self):
        plan = build_attention_mask(
            q_len=6,
            total_len=6,
            batch_size=2,
            attention_mask=None,
            device=CPU,
        )
        self.assertIsNone(plan.attn_mask)
        self.assertTrue(plan.is_causal)
        validate_attention_mask(plan, 2, 6, 6, CPU)

    def test_single_token_fast_path_uses_all_cached_keys(self):
        plan = build_attention_mask(
            q_len=1,
            total_len=7,
            batch_size=2,
            attention_mask=None,
            device=CPU,
        )
        self.assertIsNone(plan.attn_mask)
        self.assertFalse(plan.is_causal)
        validate_attention_mask(plan, 2, 1, 7, CPU)

    def test_chunked_decode_uses_bottom_right_causal_alignment(self):
        plan = build_attention_mask(
            q_len=3,
            total_len=5,
            batch_size=1,
            attention_mask=None,
            device=CPU,
        )
        self.assertFalse(plan.is_causal)
        expected = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, False],
                [True, True, True, True, True],
            ]
        )
        self.assertTrue(torch.equal(plan_visibility(plan)[0], expected))

    def test_integer_padding_mask_is_accepted_with_nonzero_semantics(self):
        padding_mask = torch.tensor(
            [
                [0, 0, 0, 2, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.int64,
        )
        plan = build_attention_mask(
            q_len=2,
            total_len=6,
            batch_size=2,
            attention_mask=padding_mask,
            device=CPU,
        )
        self.assertFalse(plan.is_causal)
        self.assertEqual(plan.attn_mask.dtype, torch.bool)
        expected = torch.tensor(
            [
                [
                    [False, False, False, True, True, False],
                    [False, False, False, True, True, True],
                ],
                [
                    [True, True, True, True, True, False],
                    [True, True, True, True, True, True],
                ],
            ]
        )
        self.assertTrue(torch.equal(plan_visibility(plan), expected))

    def test_left_padding_only_falls_back_for_empty_rows(self):
        padding_mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.bool)
        plan = build_attention_mask(
            q_len=4,
            total_len=4,
            batch_size=1,
            attention_mask=padding_mask,
            device=CPU,
        )
        visibility = plan_visibility(plan)[0]
        self.assertEqual(visibility[0].tolist(), [True, False, False, False])
        self.assertEqual(visibility[1].tolist(), [False, True, False, False])
        self.assertEqual(visibility[2].tolist(), [False, False, True, False])
        self.assertEqual(visibility[3].tolist(), [False, False, True, True])

    def test_right_padding_never_opens_padding_keys_for_nonempty_rows(self):
        padding_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        plan = build_attention_mask(
            q_len=4,
            total_len=4,
            batch_size=1,
            attention_mask=padding_mask,
            device=CPU,
        )
        visible = plan_visibility(plan)[0]
        expected = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, False, False],
                [True, True, False, False],
            ]
        )
        self.assertTrue(torch.equal(visible, expected))

    def test_segment_ids_create_block_diagonal_causal_attention(self):
        plan = build_attention_mask(
            q_len=4,
            total_len=4,
            batch_size=1,
            attention_mask=None,
            segment_ids=torch.tensor([[0, 0, 1, 1]]),
            device=CPU,
        )
        expected = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [False, False, True, False],
                [False, False, True, True],
            ]
        )
        self.assertTrue(torch.equal(plan_visibility(plan)[0], expected))
        self.assertFalse(plan.is_causal)

    def test_packed_positions_match_independent_sequences(self):
        torch.manual_seed(31)
        model = TaiChuModel(
            ModelConfig(
                vocab_size=64,
                hidden_size=32,
                num_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=64,
                max_position_embeddings=32,
                use_flash_attention=False,
            )
        ).eval()
        packed_ids = torch.tensor([[5, 6, 7, 8]])
        separate_ids = torch.tensor([[5, 6], [7, 8]])
        with torch.no_grad():
            packed = model(
                packed_ids,
                segment_ids=torch.tensor([[0, 0, 1, 1]]),
                position_ids=torch.tensor([[0, 1, 0, 1]]),
            ).logits
            separate = model(separate_ids).logits
        self.assertTrue(
            torch.allclose(packed[:, :2], separate[:1], atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(packed[:, 2:], separate[1:], atol=1e-6)
        )

    def test_all_padding_rows_fallback_to_global_self_key(self):
        plan = build_attention_mask(
            q_len=4,
            total_len=4,
            batch_size=2,
            attention_mask=torch.zeros(2, 4, dtype=torch.long),
            device=CPU,
        )
        expected = torch.eye(4, dtype=torch.bool)
        self.assertTrue(torch.equal(plan_visibility(plan)[0], expected))
        self.assertTrue(torch.equal(plan_visibility(plan)[1], expected))

    def test_single_token_with_padding_does_not_use_unconditional_fast_path(self):
        plan = build_attention_mask(
            q_len=1,
            total_len=5,
            batch_size=1,
            attention_mask=torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool),
            device=CPU,
        )
        self.assertFalse(plan.is_causal)
        self.assertIsNotNone(plan.attn_mask)
        self.assertEqual(
            plan_visibility(plan)[0, 0].tolist(),
            [True, True, True, False, False],
        )

    def test_invalid_lengths_shapes_and_dtypes_are_rejected(self):
        cases = [
            dict(batch_size=0, q_len=1, total_len=5, attention_mask=None),
            dict(batch_size=1, q_len=0, total_len=5, attention_mask=None),
            dict(batch_size=1, q_len=6, total_len=5, attention_mask=None),
            dict(batch_size=2, q_len=2, total_len=5, attention_mask=torch.ones(1, 5)),
            dict(batch_size=2, q_len=2, total_len=5, attention_mask=torch.ones(2, 4)),
            dict(batch_size=2, q_len=2, total_len=5, attention_mask=torch.ones(2, 1, 2, 5)),
            dict(batch_size=2, q_len=2, total_len=5, attention_mask=torch.ones(2, 5, 1)),
            dict(batch_size=2, q_len=2, total_len=5, attention_mask=torch.ones(2, 5)),
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    build_attention_mask(device=CPU, **case)

    def test_incompatible_mask_plan_is_rejected(self):
        valid_plan = build_attention_mask(
            q_len=2,
            total_len=6,
            batch_size=2,
            attention_mask=torch.ones(2, 6, dtype=torch.bool),
            device=CPU,
        )
        with self.assertRaises(ValueError):
            validate_attention_mask(valid_plan, 1, 2, 6, CPU)
        with self.assertRaises(ValueError):
            validate_attention_mask(valid_plan, 2, 3, 6, CPU)
        with self.assertRaises(ValueError):
            validate_attention_mask(valid_plan, 2, 2, 6, torch.device("meta"))

        float_mask_plan = AttentionMask(
            attn_mask=torch.ones(2, 1, 2, 6, dtype=torch.float32),
            is_causal=False,
            batch_size=2,
            q_len=2,
            total_len=6,
            device=CPU,
        )
        with self.assertRaises(ValueError):
            validate_attention_mask(float_mask_plan, 2, 2, 6, CPU)

        conflicting_plan = AttentionMask(
            attn_mask=torch.ones(2, 1, 2, 6, dtype=torch.bool),
            is_causal=True,
            batch_size=2,
            q_len=2,
            total_len=6,
            device=CPU,
        )
        with self.assertRaises(ValueError):
            validate_attention_mask(conflicting_plan, 2, 2, 6, CPU)

    def test_model_builds_mask_plan_once_for_all_layers(self):
        torch.manual_seed(23)
        model = TaiChuModel(
            ModelConfig(
                vocab_size=64,
                hidden_size=32,
                num_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=64,
                max_position_embeddings=32,
                use_flash_attention=False,
            )
        ).eval()
        input_ids = torch.randint(0, 64, (2, 8))
        attention_mask = torch.ones_like(input_ids)

        with patch.object(
            model_module,
            "build_attention_mask",
            wraps=model_module.build_attention_mask,
        ) as builder:
            with torch.no_grad():
                model(input_ids, attention_mask=attention_mask)

        builder.assert_called_once()

    def test_model_rejects_four_dimensional_mask(self):
        model = TaiChuModel(
            ModelConfig(
                vocab_size=64,
                hidden_size=32,
                num_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=64,
                max_position_embeddings=32,
                use_flash_attention=False,
            )
        ).eval()
        with self.assertRaises(ValueError):
            model(
                torch.randint(0, 64, (1, 5)),
                attention_mask=torch.ones(1, 1, 5, 5, dtype=torch.bool),
            )

    def test_model_rejects_inconsistent_layer_cache_lengths(self):
        model = TaiChuModel(
            ModelConfig(
                vocab_size=64,
                hidden_size=32,
                num_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=64,
                max_position_embeddings=32,
                use_flash_attention=False,
            )
        ).eval()
        cache_a = (
            torch.zeros(1, 2, 4, 8),
            torch.zeros(1, 2, 4, 8),
        )
        cache_b = (
            torch.zeros(1, 2, 3, 8),
            torch.zeros(1, 2, 3, 8),
        )
        with self.assertRaises(ValueError):
            model(
                torch.randint(0, 64, (1, 1)),
                past_key_values=[cache_a, cache_b],
            )
        with self.assertRaises(ValueError):
            model(
                torch.randint(0, 64, (1, 1)),
                past_key_values=[None, cache_b],
            )


if __name__ == "__main__":
    unittest.main()
