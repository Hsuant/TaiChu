#!/usr/bin/env python3
"""Cross-component correctness checks for tokenizer, data, model, and training."""

import io
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unicodedata
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from dataset.batching import PretrainingBatchBuilder
from dataset.config import StorageConfig, UnifiedDataConfig
from dataset.packing import PretrainingPacker
from dataset.schema import PretrainingSample
from model.config import ModelConfig
from model.model import TaiChuModel
from tokenizer.special_tokens import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
)
from utils.checkpoint import CheckpointManager
from utils.config_loader import load_model_config, load_pretrain_config
from utils.logger import LogManager
from utils.train_utils import (
    compile_model_for_training,
    get_wsd_lr_lambda,
    training_compile_preflight,
)


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
            max_position_embeddings=32,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            embedding_dropout=0.0,
            use_flash_attention=False,
        )
    )


class TestBundledTokenizerIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    def test_special_token_protocol_and_batch_equivalence(self):
        expected_ids = {
            token: index for index, token in enumerate(SPECIAL_TOKENS)
        }
        actual_ids = {
            token: self.tokenizer.token_to_id(token)
            for token in SPECIAL_TOKENS
        }
        self.assertEqual(actual_ids, expected_ids)
        self.assertIsNotNone(actual_ids[BOS_TOKEN])
        self.assertIsNotNone(actual_ids[EOS_TOKEN])
        self.assertIsNotNone(actual_ids[PAD_TOKEN])

        texts = [
            "alpha beta",
            "中文 mixed 123",
            "line1\nline2",
            "Emoji: 😀🚀",
        ]
        single = [
            self.tokenizer.encode(text, add_special_tokens=False).ids
            for text in texts
        ]
        batched = [
            encoding.ids
            for encoding in self.tokenizer.encode_batch(
                texts,
                add_special_tokens=False,
            )
        ]
        self.assertEqual(single, batched)

    def test_random_unicode_roundtrip_matches_normalizer(self):
        rng = random.Random(20261026)
        fixed = [
            "",
            " ",
            "\n\t",
            "人工智能改变了世界。",
            "Byte-level BPE keeps spaces and\nnewlines stable.",
            "\x00\x01\x7f",
            "Emoji: 😀🚀 中文标点：，。！？",
        ]
        generated = []
        while len(generated) < 100:
            chars = []
            for _ in range(32):
                codepoint = rng.randrange(1, 0x110000)
                if 0xD800 <= codepoint <= 0xDFFF:
                    continue
                chars.append(chr(codepoint))
            generated.append("".join(chars))

        for text in fixed + generated:
            with self.subTest(text=repr(text)):
                ids = self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                ).ids
                decoded = self.tokenizer.decode(
                    ids,
                    skip_special_tokens=False,
                )
                self.assertEqual(decoded, unicodedata.normalize("NFC", text))


class TestConfigurationIntegrity(unittest.TestCase):
    def test_training_configs_match_bundled_tokenizer(self):
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        cases = [
            (
                ROOT / "configs" / "train_pretrain_config.yaml",
                ROOT / "configs" / "TaiChu_80M.yaml",
            ),
            (
                ROOT / "tests" / "test_train" / "pretrain_sim.yaml",
                ROOT / "tests" / "test_train" / "model_sim.yaml",
            ),
        ]
        for pretrain_path, model_path in cases:
            with self.subTest(config=pretrain_path.name):
                pretrain = load_pretrain_config(str(pretrain_path))
                model_config = load_model_config(str(model_path))
                self.assertEqual(
                    tokenizer.token_to_id(pretrain.data.bos_token),
                    tokenizer.token_to_id(BOS_TOKEN),
                )
                self.assertEqual(
                    tokenizer.token_to_id(pretrain.data.eos_token),
                    tokenizer.token_to_id(EOS_TOKEN),
                )
                self.assertEqual(
                    tokenizer.token_to_id(pretrain.data.pad_token),
                    tokenizer.token_to_id(PAD_TOKEN),
                )
                self.assertGreaterEqual(
                    model_config.vocab_size,
                    tokenizer.get_vocab_size(),
                )
                self.assertLessEqual(
                    pretrain.data.packing.max_seq_length,
                    model_config.max_position_embeddings,
                )
                self.assertTrue(Path(pretrain.tokenizer_path).resolve().exists())
                self.assertTrue(pretrain.data.storage.train_path)
                self.assertTrue(pretrain.data.storage.validation_path)
                self.assertIn(
                    pretrain.training.compile_mode,
                    {
                        "auto",
                        "default",
                        "reduce-overhead",
                        "max-autotune",
                        "max-autotune-no-cudagraphs",
                    },
                )
                self.assertIsInstance(
                    pretrain.training.compile_dynamic,
                    bool,
                )
                self.assertGreater(
                    pretrain.training.distributed_timeout_minutes,
                    0,
                )

    def test_dataset_config_has_one_unified_schema(self):
        config = UnifiedDataConfig(
            task_type="pretraining",
            dataset_id="test",
            storage=StorageConfig(train_path="./train"),
        )
        self.assertEqual(config.pad_token, PAD_TOKEN)
        self.assertEqual(config.storage.format, "binary_mmap")
        self.assertEqual(config.task_type, "pretraining")
        self.assertFalse(hasattr(config, "train_paths"))


class TestDataAndLossIntegrity(unittest.TestCase):
    def test_collator_masks_padding_but_keeps_real_eos(self):
        eos_token_id = 2
        collator = PretrainingBatchBuilder(pad_token_id=eos_token_id)
        batch = collator(
            [
                PretrainingSample(sample_id="a", input_ids=[1, 10, 11, 2]),
                PretrainingSample(sample_id="b", input_ids=[1, 12, 2]),
            ]
        )
        self.assertEqual(batch["labels"][0, :4].tolist(), [1, 10, 11, 2])
        self.assertEqual(batch["labels"][1, :3].tolist(), [1, 12, 2])
        self.assertTrue((batch["labels"][3:] == -100).all())
        self.assertTrue(batch["attention_mask_is_right_padded"])

    def test_stream_packing_masks_cross_document_target(self):
        packer = PretrainingPacker(
            max_length=8,
            pad_id=0,
            pad_to_max=False,
            drop_last_remainder=False,
        )
        packed = list(
            packer.pack(
                [
                    PretrainingSample(sample_id="a", input_ids=[1, 2]),
                    PretrainingSample(sample_id="b", input_ids=[3, 4]),
                ]
            )
        )[0]
        self.assertEqual(packed.input_ids, [1, 2, 3, 4])
        self.assertEqual(packed.labels, [1, 2, -100, 4])

    def test_model_loss_ignores_padding(self):
        torch.manual_seed(17)
        model = _tiny_model()
        input_ids = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 0]])
        labels = input_ids.clone()
        labels[0, 4:] = -100
        labels[1, 5:] = -100
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]]
        )

        loss = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss
        logits = model(
            input_ids,
            attention_mask=attention_mask,
        ).logits
        manual = F.cross_entropy(
            logits[..., :-1, :].reshape(-1, logits.size(-1)),
            labels[..., 1:].reshape(-1),
            ignore_index=-100,
        )
        self.assertTrue(torch.allclose(loss, manual, atol=1e-6))

    def test_hidden_logits_loss_matches_full_logits_and_reports_statistics(self):
        torch.manual_seed(19)
        model = _tiny_model()
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        labels = input_ids.clone()

        full = model(input_ids, labels=labels)
        hidden_logits = model(
            input_ids,
            labels=labels,
            return_logits=False,
            validate_inputs=False,
            return_token_statistics=True,
        )

        self.assertIsNone(hidden_logits.logits)
        self.assertEqual(int(hidden_logits.num_loss_tokens.item()), 4)
        self.assertEqual(hidden_logits.token_losses.numel(), 4)
        self.assertEqual(int(hidden_logits.correct_predictions.item()), 0)
        self.assertTrue(
            torch.allclose(full.loss_sum, hidden_logits.loss_sum, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(full.loss, hidden_logits.loss, atol=1e-6)
        )


class TestOptimizationIntegrity(unittest.TestCase):
    @staticmethod
    def _optimizer_step_trainer():
        from trainer.train_pretrain import TaiChuTrainer

        torch.manual_seed(41)
        model = _tiny_model(vocab_size=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda _: 1.0,
        )
        trainer = TaiChuTrainer.__new__(TaiChuTrainer)
        trainer.model = model
        trainer.optimizer = optimizer
        trainer.scheduler = scheduler
        trainer.scaler = Mock()
        trainer.scaler_enabled = False
        trainer.max_grad_norm = 1.0
        trainer.device = torch.device("cpu")
        trainer.world_size = 1
        trainer.global_rank = 0
        return trainer, model, optimizer, scheduler

    def test_finite_optimizer_step_updates_model_and_scheduler(self):
        trainer, model, optimizer, scheduler = self._optimizer_step_trainer()
        before = [
            parameter.detach().clone()
            for parameter in model.parameters()
        ]
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)

        grad_norm, stepped, diagnostics = trainer._apply_optimizer_step(True)

        self.assertTrue(stepped)
        self.assertEqual(diagnostics, [])
        self.assertGreater(grad_norm, 0.0)
        self.assertTrue(math.isfinite(grad_norm))
        self.assertGreater(len(optimizer.state), 0)
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertTrue(
            any(
                not torch.equal(expected, parameter)
                for expected, parameter in zip(
                    before,
                    model.parameters(),
                    strict=True,
                )
            )
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.parameters())
        )

    def test_nan_gradient_skips_update_and_clears_gradients(self):
        trainer, model, optimizer, scheduler = self._optimizer_step_trainer()
        before = [
            parameter.detach().clone()
            for parameter in model.parameters()
        ]
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
        first_parameter = next(model.parameters())
        first_parameter.grad.view(-1)[0] = float("nan")

        grad_norm, stepped, diagnostics = trainer._apply_optimizer_step(True)

        self.assertFalse(stepped)
        self.assertTrue(math.isnan(grad_norm))
        self.assertTrue(diagnostics)
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertTrue(
            all(parameter.grad is None for parameter in model.parameters())
        )
        for expected, parameter in zip(
            before,
            model.parameters(),
            strict=True,
        ):
            self.assertTrue(torch.equal(expected, parameter))

    def test_nonfinite_loss_forces_atomic_skip(self):
        trainer, model, optimizer, scheduler = self._optimizer_step_trainer()
        before = [
            parameter.detach().clone()
            for parameter in model.parameters()
        ]
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)

        _, stepped, diagnostics = trainer._apply_optimizer_step(False)

        self.assertFalse(stepped)
        self.assertEqual(diagnostics, [])
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertTrue(
            all(parameter.grad is None for parameter in model.parameters())
        )
        for expected, parameter in zip(
            before,
            model.parameters(),
            strict=True,
        ):
            self.assertTrue(torch.equal(expected, parameter))

    def test_distributed_finite_decision_requires_every_rank(self):
        trainer, _, _, _ = self._optimizer_step_trainer()

        def reject_on_one_rank(tensor, op):
            tensor.fill_(0)

        with (
            patch(
                "trainer.train_pretrain.torch.distributed.is_initialized",
                return_value=True,
            ),
            patch(
                "trainer.train_pretrain.torch.distributed.all_reduce",
                side_effect=reject_on_one_rank,
            ) as all_reduce_mock,
        ):
            accepted = trainer._all_ranks_finite(True)

        self.assertFalse(accepted)
        all_reduce_mock.assert_called_once()

    def test_compile_preflight_skips_missing_windows_compiler(self):
        with (
            patch(
                "utils.train_utils.torch.cuda.is_available",
                return_value=False,
            ),
            patch("utils.train_utils.sys.platform", "win32"),
            patch("utils.train_utils.shutil.which", return_value=None),
        ):
            available, reason = training_compile_preflight()
        self.assertFalse(available)
        self.assertIn("MSVC cl", reason)

    def test_compile_helper_uses_requested_mode(self):
        model = _tiny_model()
        with patch(
            "utils.train_utils.torch.compile",
            return_value=model,
        ) as compile_mock:
            result = compile_model_for_training(
                model,
                mode="default",
                dynamic=False,
            )
        self.assertIs(result, model)
        compile_mock.assert_called_once_with(
            model,
            mode="default",
            dynamic=False,
            fullgraph=False,
        )

    def test_compile_helper_avoids_autotune_for_distributed_auto_mode(self):
        model = _tiny_model()
        with (
            patch(
                "utils.train_utils.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "utils.train_utils.torch.compile",
                return_value=model,
            ) as compile_mock,
        ):
            compile_model_for_training(
                model,
                mode="auto",
                dynamic=True,
                distributed=True,
            )
        compile_mock.assert_called_once_with(
            model,
            mode="default",
            dynamic=True,
            fullgraph=False,
        )

    def test_compile_warmup_preserves_parameters_and_rng(self):
        from trainer.train_pretrain import _warmup_compiled_training

        torch.manual_seed(29)
        model = _tiny_model(vocab_size=32)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        rng_before = torch.random.get_rng_state().clone()

        _warmup_compiled_training(
            model,
            device=torch.device("cpu"),
            vocab_size=32,
            batch_size=1,
            sequence_length=8,
            use_amp=False,
            dtype=torch.bfloat16,
        )

        self.assertTrue(
            torch.equal(rng_before, torch.random.get_rng_state())
        )
        for expected, parameter in zip(before, model.parameters(), strict=True):
            self.assertTrue(torch.equal(expected, parameter))
            self.assertIsNone(parameter.grad)

    def test_tiny_model_overfits_fixed_batch(self):
        torch.manual_seed(23)
        model = _tiny_model(vocab_size=32)
        input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 9],
            ]
        )
        labels = input_ids.clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

        losses = []
        for _ in range(24):
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids, labels=labels).loss
            loss.backward()
            self.assertTrue(
                all(
                    parameter.grad is None
                    or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                )
            )
            optimizer.step()
            losses.append(loss.item())

        self.assertTrue(math.isfinite(losses[-1]))
        self.assertLess(losses[-1], losses[0] * 0.7)

    def test_wsd_schedule_has_expected_phases(self):
        schedule = get_wsd_lr_lambda(
            warmup_steps=4,
            stable_steps=6,
            max_steps=20,
            min_lr_ratio=0.1,
            decay_type="cosine",
        )
        self.assertEqual(schedule(0), 0.0)
        self.assertAlmostEqual(schedule(2), 0.5)
        self.assertEqual(schedule(4), 1.0)
        self.assertEqual(schedule(9), 1.0)
        self.assertLess(schedule(15), 1.0)
        self.assertGreaterEqual(schedule(15), 0.1)
        self.assertEqual(schedule(20), 0.1)

    def test_checkpoint_restores_logits_exactly(self):
        torch.manual_seed(31)
        model = _tiny_model()
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        expected = model(input_ids).logits

        with tempfile.TemporaryDirectory() as directory:
            manager = CheckpointManager(output_dir=directory)
            path = manager.save("model.pt", model=model, global_step=7)
            self.assertFalse(list(Path(directory).glob("*.tmp-*")))
            restored = _tiny_model()
            step, _, _ = manager.load(path, model=restored)
            actual = restored(input_ids).logits

        self.assertEqual(step, 7)
        self.assertTrue(torch.equal(expected, actual))

    def test_checkpoint_restores_scheduler_epoch_without_override(self):
        model = _tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: 0.5,
        )
        optimizer.step()
        scheduler.step()

        with tempfile.TemporaryDirectory() as directory:
            manager = CheckpointManager(output_dir=directory)
            path = manager.save(
                "model.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=1,
            )
            restored_optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
            restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
                restored_optimizer,
                lambda step: 0.5,
            )
            manager.load(
                path,
                model=model,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
            )

        self.assertEqual(restored_scheduler.last_epoch, 1)
        self.assertAlmostEqual(
            restored_optimizer.param_groups[0]["lr"],
            0.05,
        )


class TestLoggingIntegrity(unittest.TestCase):
    def test_training_step_log_has_no_duplicate_repr_line(self):
        from trainer.train_pretrain import TaiChuTrainer

        trainer = TaiChuTrainer.__new__(TaiChuTrainer)
        trainer.global_step = 100
        trainer.optimizer = type(
            "Optimizer",
            (),
            {"param_groups": [{"lr": 3.0e-6}]},
        )()
        trainer.eval_metrics = Mock()
        trainer.eval_metrics.get_all_metrics.return_value = {
            "smoothed_loss": 10.31,
            "grad_norm_sq_variance_ratio": 0.09,
        }
        trainer.last_log_time = None
        trainer.start_time = 0.0
        trainer.log_manager = Mock()
        trainer.swanlab_logger = None

        trainer._log_training(10.26, 100, 0.5)

        trainer.log_manager.logger.info.assert_called_once()
        log_line = trainer.log_manager.logger.info.call_args.args[0]
        self.assertEqual(log_line.count("Step 100"), 1)
        self.assertNotIn("repr_metrics", log_line)

    def test_console_logging_survives_non_gbk_text(self):
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="gbk", errors="strict")
        original_stdout = sys.stdout
        try:
            sys.stdout = stream
            with tempfile.TemporaryDirectory() as directory:
                manager = LogManager(
                    log_dir=directory,
                    tensorboard=False,
                    log_file=None,
                )
                manager.info("invalid byte marker: \ufffd")
                manager.close()
            stream.flush()
            self.assertIn(b"\\ufffd", buffer.getvalue())
        finally:
            sys.stdout = original_stdout
            stream.detach()


class TestEntryPoints(unittest.TestCase):
    def test_train_pretrain_help_is_available(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "trainer.train_pretrain",
                "--help",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model_config", result.stdout)
        self.assertIn("--pretrain_config", result.stdout)

    def test_offline_preparation_help_is_available(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dataset.offline.prepare",
                "--help",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--config", result.stdout)


if __name__ == "__main__":
    unittest.main()
