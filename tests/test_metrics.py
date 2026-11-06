#!/usr/bin/env python3
"""Formula-level tests for validation and model-loss metrics."""

import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.config import ModelConfig
from model.model import TaiChuModel
from model.moe import MoEConfig, SparseMoE
from utils.metrics import (
    EfficientGradientNoiseScale,
    EvaluationMetricsManager,
    ValidationMetrics,
)
from utils.model_utils import ModelInspector


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "taichu_tokenizer" / "tokenizer.json"


def _tiny_model(vocab_size: int = 64) -> TaiChuModel:
    return TaiChuModel(
        ModelConfig(
            vocab_size=vocab_size,
            hidden_size=32,
            num_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            max_position_embeddings=16,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            embedding_dropout=0.0,
            use_flash_attention=False,
        )
    )


class TestValidationMetricFormulas(unittest.TestCase):
    def test_token_statistics_match_manual_loss_accuracy_and_bpb(self):
        torch.manual_seed(71)
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        bos_id = tokenizer.token_to_id("<|taichu_start|>")
        eos_id = tokenizer.token_to_id("<|taichu_end|>")
        text_ids = tokenizer.encode(
            "metric formula 中英",
            add_special_tokens=False,
        ).ids
        input_ids = torch.tensor([[bos_id, *text_ids, eos_id]])
        labels = input_ids.clone()
        logits = torch.randn(
            1,
            input_ids.size(1),
            tokenizer.get_vocab_size(),
        )

        stats = ValidationMetrics.compute_token_statistics(
            logits,
            labels,
            tokenizer=tokenizer,
        )
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        manual_losses = F.cross_entropy(
            shift_logits.reshape(-1, logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).view(shift_labels.shape)
        self.assertEqual(stats.token_count, len(text_ids) + 1)
        self.assertAlmostEqual(
            stats.nll_nats,
            manual_losses.sum().item(),
            places=5,
        )
        expected_correct = (
            shift_logits.argmax(dim=-1) == shift_labels
        ).sum().item()
        self.assertEqual(stats.correct_count, expected_correct)
        self.assertAlmostEqual(
            stats.nll_nats / stats.token_count,
            manual_losses.mean().item(),
            places=5,
        )

        special_ids = set(tokenizer.get_added_tokens_decoder())
        content_positions = [
            index
            for index, token_id in enumerate(shift_labels[0].tolist())
            if token_id not in special_ids
        ]
        expected_text = tokenizer.decode(
            [shift_labels[0, index].item() for index in content_positions],
            skip_special_tokens=True,
        )
        expected_bytes = len(expected_text.encode("utf-8"))
        expected_nll = manual_losses[0, content_positions].sum().item()
        self.assertEqual(stats.bpb_bytes, expected_bytes)
        self.assertAlmostEqual(stats.bpb_nll_nats, expected_nll, places=5)

        bpb = ValidationMetrics.compute_bits_per_byte(
            stats.bpb_nll_nats,
            stats.bpb_bytes,
        )
        self.assertAlmostEqual(
            bpb,
            expected_nll / (math.log(2.0) * expected_bytes),
            places=6,
        )

    def test_exact_percentiles_use_linear_quantiles(self):
        losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
        with patch(
            "utils.metrics.torch.quantile",
            side_effect=RuntimeError("quantile() input tensor is too large"),
        ):
            result = ValidationMetrics.compute_percentile_ce(
                losses,
                [0.5, 0.9, 0.95],
            )
        self.assertAlmostEqual(result["ce_p50"], 2.5)
        self.assertAlmostEqual(result["ce_p90"], 3.7)
        self.assertAlmostEqual(result["ce_p95"], 3.85)
        empty = ValidationMetrics.compute_percentile_ce(
            torch.empty(0),
            [0.5],
        )
        self.assertIsNone(empty["ce_p50"])

    def test_perplexity_is_none_instead_of_inf(self):
        self.assertEqual(ValidationMetrics.compute_perplexity(0.0), 1.0)
        self.assertIsNotNone(ValidationMetrics.compute_perplexity(10.0))
        self.assertIsNone(ValidationMetrics.compute_perplexity(float("inf")))
        self.assertIsNone(ValidationMetrics.compute_perplexity(float("nan")))
        self.assertIsNone(ValidationMetrics.compute_perplexity(-1.0))
        self.assertIsNone(ValidationMetrics.compute_perplexity(1000.0))

    def test_empty_prefix_metric_is_explicit_and_restores_mode(self):
        torch.manual_seed(73)
        model = _tiny_model()
        model.train()
        bos_id, eos_id = 1, 2
        result = ValidationMetrics.compute_empty_prefix_eos_metrics(
            model,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            logits = model(torch.tensor([[bos_id, eos_id]])).logits
            expected_ce = F.cross_entropy(
                logits[:, 0, :],
                torch.tensor([eos_id]),
            ).item()
        self.assertTrue(model.training)
        self.assertAlmostEqual(
            result["empty_prefix_eos_ce"],
            expected_ce,
            places=6,
        )
        self.assertAlmostEqual(
            result["empty_prefix_eos_ppl"],
            math.exp(expected_ce),
            places=5,
        )

        invalid = ValidationMetrics.compute_empty_prefix_eos_metrics(
            model,
            bos_token_id=2,
            eos_token_id=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(
            invalid,
            {
                "empty_prefix_eos_ce": None,
                "empty_prefix_eos_ppl": None,
            },
        )

    def test_legacy_blank_ppl_returns_none_and_restores_mode(self):
        model = _tiny_model()
        model.train()
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        self.assertIsNone(
            ValidationMetrics.compute_blank_ppl(
                model,
                tokenizer,
                device=torch.device("cpu"),
            )
        )
        self.assertTrue(model.training)

    def test_metrics_json_is_strict(self):
        metrics = {
            "val_loss": 1.5,
            "val_perplexity": ValidationMetrics.compute_perplexity(1.5),
            "empty_prefix_eos_ppl": None,
        }
        encoded = json.dumps(metrics, allow_nan=False)
        self.assertIn("null", encoded)

    def test_gradient_metric_is_labeled_as_variance_ratio(self):
        model = _tiny_model()
        loss = model(
            torch.tensor([[1, 2, 3]]),
            labels=torch.tensor([[1, 2, 3]]),
        ).loss
        loss.backward()
        manager = EvaluationMetricsManager()
        manager.update_gradient(model)
        metrics = manager.get_all_metrics()
        self.assertIn("grad_norm_sq_variance_ratio", metrics)
        self.assertNotIn("gradient_noise_scale", metrics)

    @unittest.skipIf(torch.cuda.is_available(), "CPU-only MFU assertion")
    def test_cpu_peak_flops_disables_mfu(self):
        self.assertEqual(ModelInspector.get_peak_flops(), 0.0)
        manager = EvaluationMetricsManager()
        self.assertNotIn("mfu_average", manager.get_all_metrics())

    def test_efficient_gradient_diagnostic_collects_rmsnorm(self):
        model = _tiny_model()
        tracker = EfficientGradientNoiseScale(model)
        self.assertGreater(len(tracker.norm_layers), 0)
        loss = model(
            torch.tensor([[1, 2, 3]]),
            labels=torch.tensor([[1, 2, 3]]),
        ).loss
        loss.backward()
        value = tracker.update(model)
        self.assertTrue(math.isfinite(value))
        self.assertIn(
            "grad_norm_sq_variance_ratio",
            tracker.get_metrics(),
        )


class TestModelLossValidation(unittest.TestCase):
    def test_padding_labels_must_use_minus_100(self):
        model = _tiny_model()
        input_ids = torch.tensor([[1, 2, 3, 0, 0]])
        labels = input_ids.clone()
        labels[0, 3:] = -100
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
        loss = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss
        self.assertTrue(torch.isfinite(loss))

        labels[0, 3] = 0
        with self.assertRaises(ValueError):
            model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

    def test_no_supervised_next_token_is_rejected(self):
        model = _tiny_model()
        input_ids = torch.tensor([[1, 2, 3]])
        labels = torch.tensor([[1, -100, -100]])
        with self.assertRaises(ValueError):
            model(input_ids, labels=labels)

    def test_invalid_label_dtype_and_range_are_rejected(self):
        model = _tiny_model()
        input_ids = torch.tensor([[1, 2, 3]])
        with self.assertRaises(ValueError):
            model(input_ids, labels=input_ids.float())
        invalid = input_ids.clone()
        invalid[0, 1] = model.config.vocab_size
        with self.assertRaises(ValueError):
            model(input_ids, labels=invalid)

    def test_gradient_summary_without_gradients_is_finite(self):
        summary = ModelInspector.gradient_summary(_tiny_model())
        self.assertEqual(summary["grad_norm"], 0.0)
        self.assertEqual(summary["grad_max"], 0.0)
        self.assertEqual(summary["grad_min"], 0.0)


class TestMoEAuxiliaryLossFormula(unittest.TestCase):
    def test_load_balance_and_z_loss_match_reference(self):
        moe = SparseMoE(
            hidden_size=8,
            moe_config=MoEConfig(
                num_experts=3,
                top_k=2,
                expert_intermediate_size=16,
                num_shared_experts=0,
                load_balancing_loss_weight=0.1,
                router_z_loss_weight=0.01,
            ),
        ).train()
        logits = torch.tensor(
            [
                [2.0, 0.5, -1.0],
                [0.2, 1.5, -0.5],
                [-1.0, 0.1, 2.0],
                [0.4, 0.3, -0.2],
            ]
        )
        topk_indices = torch.topk(logits, k=2, dim=-1).indices
        components = moe._compute_auxiliary_loss_components(
            logits,
            topk_indices,
        )
        actual = sum(components.values())

        probs = torch.softmax(logits, dim=-1)
        density = torch.zeros(3)
        for index in range(2):
            density.scatter_add_(
                0,
                topk_indices[:, index],
                torch.ones(logits.size(0)),
            )
        density = density / (logits.size(0) * 2)
        load_balance = 3 * (probs.mean(dim=0) * density).sum()
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        expected = 0.1 * load_balance + 0.01 * z_loss
        self.assertAlmostEqual(actual.item(), expected.item(), places=6)


if __name__ == "__main__":
    unittest.main()
