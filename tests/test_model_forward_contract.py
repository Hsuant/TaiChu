"""Top-level model forward contract and loss semantics."""

import copy
import math
import unittest

import torch
import torch.nn.functional as F

from model.config import ModelConfig
from model.model import CausalLMOutput, TaiChuModel


def _tiny_model(
    *,
    vocab_size: int = 37,
    num_layers: int = 1,
    ffn_type: str = "swiglu",
) -> TaiChuModel:
    return TaiChuModel(
        ModelConfig(
            vocab_size=vocab_size,
            hidden_size=16,
            num_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=24,
            max_position_embeddings=32,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            embedding_dropout=0.0,
            use_flash_attention=False,
            ffn_type=ffn_type,
            moe_num_experts=4,
            moe_top_k=2,
            moe_expert_intermediate_size=12,
            moe_num_shared_experts=1,
            moe_shared_expert_intermediate_size=16,
        )
    )


def _assert_gradients_close(
    test: unittest.TestCase,
    first: TaiChuModel,
    second: TaiChuModel,
) -> None:
    for (name_a, param_a), (name_b, param_b) in zip(
        first.named_parameters(),
        second.named_parameters(),
    ):
        test.assertEqual(name_a, name_b)
        if param_a.grad is None or param_b.grad is None:
            test.assertIs(param_a.grad, param_b.grad, msg=name_a)
            continue
        test.assertTrue(
            torch.allclose(param_a.grad, param_b.grad, atol=2e-6, rtol=1e-5),
            msg=name_a,
        )


class TestOutputProtocol(unittest.TestCase):
    def test_output_switches_use_one_dataclass(self):
        torch.manual_seed(1)
        model = _tiny_model().eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        labels = input_ids.clone()

        inference = model(input_ids)
        self.assertIsInstance(inference, CausalLMOutput)
        self.assertEqual(inference.logits.shape, (1, 4, 37))
        self.assertIsNone(inference.loss)
        self.assertIsNone(inference.loss_sum)
        self.assertIsNone(inference.aux_loss)
        self.assertIsNone(inference.loss_components)
        self.assertIsNone(inference.past_key_values)

        hidden_only = model(input_ids, return_logits=False)
        self.assertIsNone(hidden_only.logits)
        self.assertIsNone(hidden_only.loss)

        training = model(
            input_ids,
            labels=labels,
            return_logits=False,
        )
        self.assertIsNone(training.logits)
        self.assertIsNotNone(training.loss)
        self.assertIsNotNone(training.loss_sum)
        self.assertIsNone(training.token_losses)
        self.assertIsNone(training.correct_predictions)
        self.assertEqual(set(training.loss_components), {"ce_loss"})

    def test_inference_without_logits_skips_output_head(self):
        model = _tiny_model().eval()
        calls = []
        handle = model.output_head.register_forward_hook(
            lambda module, inputs, output: calls.append(module)
        )
        try:
            output = model(
                torch.tensor([[1, 2, 3]]),
                return_logits=False,
            )
        finally:
            handle.remove()
        self.assertEqual(calls, [])
        self.assertIsNone(output.logits)

    def test_training_without_returned_logits_still_uses_output_head(self):
        model = _tiny_model()
        calls = []
        handle = model.output_head.register_forward_hook(
            lambda module, inputs, output: calls.append(module)
        )
        input_ids = torch.tensor([[1, 2, 3]])
        try:
            output = model(
                input_ids,
                labels=input_ids,
                return_logits=False,
            )
        finally:
            handle.remove()

        self.assertEqual(len(calls), 1)
        self.assertIsNone(output.logits)
        self.assertIsNotNone(output.loss)


class TestCrossEntropyContract(unittest.TestCase):
    def test_returned_and_hidden_logits_paths_match_values_and_gradients(self):
        torch.manual_seed(2)
        full_model = _tiny_model()
        hidden_logits_model = copy.deepcopy(full_model)
        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        )
        labels = input_ids.clone()
        labels[0, 3:] = -100
        labels[1, 5:] = -100
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
        )

        full = full_model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_token_statistics=True,
        )
        hidden_logits = hidden_logits_model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_logits=False,
            return_token_statistics=True,
        )

        self.assertIsNone(hidden_logits.logits)
        self.assertEqual(full.num_loss_tokens.dtype, torch.int64)
        self.assertEqual(hidden_logits.num_loss_tokens.dtype, torch.int64)
        self.assertEqual(full.loss_sum.dtype, torch.float32)
        self.assertEqual(hidden_logits.loss_sum.dtype, torch.float32)
        self.assertEqual(hidden_logits.token_losses.dtype, torch.float32)
        self.assertEqual(int(hidden_logits.num_loss_tokens), 6)
        self.assertTrue(
            torch.allclose(full.loss_sum, hidden_logits.loss_sum, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(full.loss, hidden_logits.loss, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                full.token_losses,
                hidden_logits.token_losses,
                atol=1e-6,
            )
        )
        self.assertEqual(
            int(full.correct_predictions),
            int(hidden_logits.correct_predictions),
        )

        full.loss_sum.backward()
        hidden_logits.loss_sum.backward()
        _assert_gradients_close(self, full_model, hidden_logits_model)

    def test_statistics_switch_does_not_change_loss_or_gradients(self):
        torch.manual_seed(8)
        without_statistics = _tiny_model()
        with_statistics = copy.deepcopy(without_statistics)
        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        )
        labels = input_ids.clone()
        labels[0, 4:] = -100
        labels[1, 3:] = -100

        plain = without_statistics(input_ids, labels=labels)
        detailed = with_statistics(
            input_ids,
            labels=labels,
            return_token_statistics=True,
        )

        self.assertIsNone(plain.token_losses)
        self.assertIsNone(plain.correct_predictions)
        self.assertIsNotNone(detailed.token_losses)
        self.assertIsNotNone(detailed.correct_predictions)
        self.assertTrue(
            torch.allclose(plain.loss_sum, detailed.loss_sum, atol=1e-6)
        )

        plain.loss_sum.backward()
        detailed.loss_sum.backward()
        _assert_gradients_close(
            self,
            without_statistics,
            with_statistics,
        )

    def test_empty_supervision_is_finite(self):
        model = _tiny_model()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        labels = torch.full_like(input_ids, -100)

        output = model(
            input_ids,
            labels=labels,
            return_logits=False,
            validate_inputs=False,
            return_token_statistics=True,
        )

        self.assertEqual(int(output.num_loss_tokens), 0)
        self.assertEqual(output.token_losses.numel(), 0)
        self.assertEqual(int(output.correct_predictions), 0)
        self.assertEqual(float(output.loss_sum.detach()), 0.0)
        self.assertTrue(math.isfinite(float(output.loss.detach())))
        self.assertEqual(float(output.loss.detach()), 0.0)

    def test_token_weighted_microbatch_accumulation_matches_combined_batch(self):
        torch.manual_seed(3)
        combined_model = _tiny_model()
        accumulated_model = copy.deepcopy(combined_model)
        first_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        second_ids = torch.tensor([[9, 10, 11, 12], [13, 14, 15, 16]])
        first_labels = first_ids.clone()
        second_labels = second_ids.clone()
        first_labels[:, 2:] = -100
        second_labels[:, 3:] = -100
        input_ids = torch.cat([first_ids, second_ids], dim=0)
        labels = torch.cat([first_labels, second_labels], dim=0)
        token_count = int((labels[:, 1:] != -100).sum())

        combined = combined_model(
            input_ids,
            labels=labels,
            return_logits=False,
        )
        (combined.loss_sum / token_count).backward()

        for micro_ids, micro_labels in (
            (first_ids, first_labels),
            (second_ids, second_labels),
        ):
            output = accumulated_model(
                micro_ids,
                labels=micro_labels,
                return_logits=False,
            )
            (output.loss_sum / token_count).backward()

        _assert_gradients_close(self, combined_model, accumulated_model)

    def test_padding_and_ignore_index_are_not_counted(self):
        model = _tiny_model().eval()
        input_ids = torch.tensor([[1, 2, 3, 0, 0]])
        labels = torch.tensor([[1, 2, 3, -100, -100]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]])

        output = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_token_statistics=True,
        )

        self.assertEqual(int(output.num_loss_tokens), 2)
        self.assertEqual(output.token_losses.numel(), 2)
        expected = F.cross_entropy(
            output.logits[:, :-1].float().reshape(-1, 37),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        self.assertTrue(torch.allclose(output.loss_sum, expected, atol=1e-6))

    def test_padding_label_validation_is_retained(self):
        model = _tiny_model().eval()
        with self.assertRaises(ValueError):
            model(
                torch.tensor([[1, 2, 3, 0]]),
                attention_mask=torch.tensor([[1, 1, 1, 0]]),
                labels=torch.tensor([[1, 2, 3, 0]]),
            )


class TestMoELossContract(unittest.TestCase):
    def test_auxiliary_loss_is_separate_and_not_token_scaled(self):
        torch.manual_seed(4)
        model = _tiny_model(num_layers=2, ffn_type="moe")
        input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        labels = input_ids.clone()

        output = model(input_ids, labels=labels)
        manual_ce = F.cross_entropy(
            output.logits[:, :-1].float().reshape(-1, 37),
            labels[:, 1:].reshape(-1),
            reduction="sum",
        )

        self.assertTrue(
            torch.allclose(output.loss_sum, manual_ce, atol=1e-6)
        )
        ce_loss = output.loss_sum / output.num_loss_tokens
        self.assertTrue(
            torch.allclose(
                output.loss,
                ce_loss + output.aux_loss,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                output.loss_components["moe_aux_loss"],
                output.aux_loss,
            )
        )
        self.assertTrue(
            torch.allclose(
                (
                    output.loss_components["moe_load_balancing_loss"]
                    + output.loss_components["moe_router_z_loss"]
                ),
                output.aux_loss,
                atol=1e-6,
            )
        )
        self.assertFalse(
            torch.allclose(
                output.loss_sum,
                manual_ce + output.aux_loss * output.num_loss_tokens,
            )
        )

        output.aux_loss.backward()
        gate_gradients = [
            layer.feed_forward.moe.gate.weight.grad
            for layer in model.layers
        ]
        self.assertTrue(all(grad is not None for grad in gate_gradients))
        self.assertTrue(
            all(torch.isfinite(grad).all() for grad in gate_gradients)
        )

    def test_zero_auxiliary_loss_is_reported(self):
        model = _tiny_model(ffn_type="moe")
        for layer in model.layers:
            layer.feed_forward.moe.load_balancing_loss_weight = 0.0
            layer.feed_forward.moe.router_z_loss_weight = 0.0

        output = model(
            torch.tensor([[1, 2, 3, 4]]),
            labels=torch.tensor([[1, 2, 3, 4]]),
            return_logits=False,
        )

        self.assertIsNotNone(output.aux_loss)
        self.assertEqual(float(output.aux_loss.detach()), 0.0)
        self.assertEqual(
            float(
                output.loss_components["moe_load_balancing_loss"].detach()
            ),
            0.0,
        )
        self.assertEqual(
            float(output.loss_components["moe_router_z_loss"].detach()),
            0.0,
        )

    def test_auxiliary_only_objective_handles_zero_ce_tokens(self):
        model = _tiny_model(ffn_type="moe")
        input_ids = torch.tensor([[1, 2, 3, 4]])
        output = model(
            input_ids,
            labels=torch.full_like(input_ids, -100),
            return_logits=False,
            validate_inputs=False,
        )

        self.assertEqual(int(output.num_loss_tokens), 0)
        self.assertEqual(float(output.loss_sum.detach()), 0.0)
        self.assertTrue(
            torch.allclose(output.loss, output.aux_loss)
        )

        inference = model(input_ids)
        self.assertIsNone(inference.aux_loss)
        self.assertIsNone(inference.loss_components)


class TestKVCacheContract(unittest.TestCase):
    def test_prefill_decode_and_cache_switches(self):
        torch.manual_seed(5)
        model = _tiny_model().eval()
        sequence = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with torch.no_grad():
            prefill = model(sequence[:, :4], use_cache=True)
            self.assertEqual(len(prefill.past_key_values), 1)
            self.assertIsNone(prefill.loss)

            first = model(
                sequence[:, 4:5],
                past_key_values=prefill.past_key_values,
                use_cache=True,
            )
            second = model(
                sequence[:, 5:6],
                past_key_values=first.past_key_values,
                use_cache=True,
            )
            full = model(sequence)
            no_cache = model(sequence[:, 5:6], use_cache=False)

        decoded = torch.cat([prefill.logits, first.logits, second.logits], dim=1)
        self.assertTrue(torch.allclose(decoded, full.logits, atol=1e-5))
        self.assertIsNone(no_cache.past_key_values)

    def test_cache_mask_and_labels_validation(self):
        model = _tiny_model().eval()
        sequence = torch.tensor([[1, 2, 3]])
        cache = model(sequence, use_cache=True).past_key_values

        with self.assertRaises(ValueError):
            model(
                torch.tensor([[4]]),
                attention_mask=torch.ones(1, 3, dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
            )
        with self.assertRaises(ValueError):
            model(
                torch.tensor([[4]]),
                labels=torch.tensor([[4]]),
                past_key_values=cache,
                use_cache=True,
            )


class TestMixedPrecisionContract(unittest.TestCase):
    def test_float32_and_bfloat16_loss_are_stable(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(6)
                model = _tiny_model(vocab_size=32).to(dtype=dtype)
                input_ids = torch.tensor([[1, 2, 3, 4, 5]])
                output = model(
                    input_ids,
                    labels=input_ids,
                    return_logits=False,
                )
                self.assertEqual(output.loss_sum.dtype, torch.float32)
                self.assertTrue(torch.isfinite(output.loss))
                output.loss.backward()
                self.assertTrue(
                    all(
                        param.grad is None
                        or torch.isfinite(param.grad).all()
                        for param in model.parameters()
                    )
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA FP16 unavailable")
    def test_float16_loss_is_stable_on_cuda(self):
        model = _tiny_model(vocab_size=32).to(device="cuda", dtype=torch.float16)
        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5]],
            device="cuda",
        )
        output = model(
            input_ids,
            labels=input_ids,
            return_logits=False,
        )
        self.assertEqual(output.loss_sum.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertTrue(
            all(
                param.grad is None
                or torch.isfinite(param.grad).all()
                for param in model.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
