"""Integration tests for adapters, sampling, packing and unified runtime."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from dataset.batching import (
    PreferenceBatchBuilder,
    RewardBatchBuilder,
    TrajectoryBatchBuilder,
)
from dataset.config import (
    BatchingConfig,
    SamplingConfig,
    StorageConfig,
    UnifiedDataConfig,
    UnifiedPackingConfig,
)
from dataset.exceptions import TaiChuConfigurationError
from dataset.metrics import DataMetrics, PackingMetrics, ThroughputMetrics
from dataset.packing import PretrainingPacker, SampleIsolatedPacker
from dataset.offline.adapters import (
    PreferencePreprocessor,
    PretrainingPreprocessor,
    RewardPreprocessor,
    RoleTagRenderer,
    SFTPreprocessor,
    TextTokenizer,
    TrajectoryPreprocessor,
)
from dataset.offline.unified import UnifiedPreparationPipeline
from dataset.runtime import (
    DataState,
    ExperienceBuffer,
    ResumableBatchIterator,
    UnifiedDataLoaderFactory,
)
from dataset.sampling import DatasetMixer, DistributedSampler, MixSource
from dataset.schema import (
    PreferenceSample,
    PretrainingSample,
    RewardSample,
    SFTSequenceSample,
    TrajectorySample,
)
from dataset.storage import BinaryDatasetReader, BinarySampleWriter


class TinyTokenizer:
    vocab_size = 512
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    unk_token = "<unk>"

    def __call__(
        self,
        texts,
        *,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    ):
        return {
            "input_ids": [
                [4 + ord(character) % (self.vocab_size - 4) for character in text]
                for text in texts
            ]
        }


def make_text_tokenizer() -> TextTokenizer:
    return TextTokenizer(
        TinyTokenizer(),
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )


class TestPreprocessingAdapters(unittest.TestCase):
    def test_sft_multi_turn_assistant_only_mask(self):
        preprocessor = SFTPreprocessor(
            make_text_tokenizer(),
            RoleTagRenderer(),
            assistant_only_loss=True,
            train_on_eos=True,
        )
        sample = preprocessor.prepare(
            {
                "sample_id": "sft-one",
                "messages": [
                    {"role": "system", "content": "rules"},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "answer"},
                    {"role": "user", "content": "second"},
                    {"role": "assistant", "content": "follow-up"},
                ],
            }
        )
        self.assertIsInstance(sample, SFTSequenceSample)
        self.assertEqual(len(sample.messages), 5)
        self.assertTrue(any(sample.loss_mask))
        self.assertFalse(all(sample.loss_mask))
        for mask, label in zip(sample.loss_mask, sample.labels, strict=True):
            if not mask:
                self.assertEqual(label, -100)

    def test_preference_reward_and_trajectory_adapters(self):
        tokenizer = make_text_tokenizer()
        renderer = RoleTagRenderer()
        preference = PreferencePreprocessor(
            tokenizer,
            renderer,
            add_eos=False,
        ).prepare(
            {
                "sample_id": "pref",
                "pair_id": "pair",
                "prompt_messages": [{"role": "user", "content": "question"}],
                "chosen": "good",
                "rejected": "bad",
            }
        )
        self.assertEqual(preference.pair_id, "pair")
        self.assertNotEqual(
            preference.chosen_input_ids,
            preference.rejected_input_ids,
        )

        reward = RewardPreprocessor(
            tokenizer,
            renderer,
            add_eos=False,
        ).prepare(
            {
                "sample_id": "reward",
                "prompt_messages": [{"role": "user", "content": "question"}],
                "response": "answer",
                "reward_label": 0.5,
            }
        )
        self.assertEqual(reward.reward_label, 0.5)

        trajectory = TrajectoryPreprocessor(
            tokenizer,
            renderer,
            add_eos=False,
        ).prepare(
            {
                "sample_id": "trajectory",
                "prompt_messages": [{"role": "user", "content": "question"}],
                "response": "answer",
                "rewards": [1.0],
                "group_id": "group",
            }
        )
        self.assertEqual(trajectory.group_id, "group")

    def test_unified_preparation_writes_binary_dataset(self):
        adapter = PretrainingPreprocessor(
            make_text_tokenizer(),
            add_eos=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepared"
            writer = BinarySampleWriter(
                output,
                dataset_id="prepared",
                sample_type="pretraining",
            )
            stats = UnifiedPreparationPipeline(adapter, writer).run(
                [
                    {"sample_id": "one", "text": "alpha"},
                    {"sample_id": "two", "text": "beta"},
                ]
            )
            writer.commit()
            self.assertEqual(stats.written_samples, 2)
            self.assertTrue((output / "manifest.json").is_file())


class TestSamplingPackingAndBatching(unittest.TestCase):
    def test_distributed_sampler_is_epoch_stable_and_disjoint(self):
        rank_values = []
        for rank in range(2):
            sampler = DistributedSampler(
                10,
                rank=rank,
                world_size=2,
                shuffle=True,
                seed=7,
            )
            first = list(sampler)
            self.assertEqual(len(sampler), 5)
            sampler.set_epoch(0)
            self.assertEqual(first, list(sampler))
            rank_values.append(first)
        self.assertEqual(sorted(rank_values[0] + rank_values[1]), list(range(10)))
        self.assertTrue(set(rank_values[0]).isdisjoint(rank_values[1]))

    def test_weighted_dataset_mixer_samples_registered_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for name, sample_type, samples in (
                (
                    "a",
                    "pretraining",
                    [
                        PretrainingSample(
                            sample_id="a-0",
                            dataset_id="a",
                            input_ids=[1, 2],
                        )
                    ],
                ),
                (
                    "b",
                    "pretraining",
                    [
                        PretrainingSample(
                            sample_id="b-0",
                            dataset_id="b",
                            input_ids=[3, 4],
                        ),
                        PretrainingSample(
                            sample_id="b-1",
                            dataset_id="b",
                            input_ids=[5, 6],
                        ),
                    ],
                ),
            ):
                path = root / name
                writer = BinarySampleWriter(
                    path,
                    dataset_id=name,
                    sample_type=sample_type,
                )
                for sample in samples:
                    writer.write(sample)
                writer.commit()
                sources.append(
                    MixSource(
                        name=name,
                        reader=BinaryDatasetReader(path),
                        weight=1.0,
                    )
                )
            mixer = DatasetMixer(sources, seed=9, sample_budget=32)
            mixed = list(mixer)
            self.assertEqual(len(mixed), 32)
            self.assertEqual(
                {sample.dataset_id for sample in mixed},
                {"a", "b"},
            )
            for source in sources:
                source.reader.close()

    def test_pretraining_and_conversation_aware_packing(self):
        packer = PretrainingPacker(
            8,
            pad_id=0,
            pad_to_max=True,
            drop_last_remainder=False,
        )
        packed = list(
            packer.pack(
                [
                    PretrainingSample(
                        sample_id="a",
                        input_ids=[3, 4, 5],
                        document_ids=[0, 0, 0],
                    ),
                    PretrainingSample(
                        sample_id="b",
                        input_ids=[6, 7, 8],
                        document_ids=[0, 0, 0],
                    ),
                ]
            )
        )
        self.assertEqual(len(packed), 1)
        self.assertEqual(packed[0].input_ids[:6], [3, 4, 5, 6, 7, 8])
        self.assertEqual(packed[0].attention_mask[-2:], [0, 0])
        self.assertNotEqual(packed[0].document_ids[2], packed[0].document_ids[3])

        sft = [
            SFTSequenceSample(
                sample_id="s1",
                input_ids=[3, 4],
                labels=[3, 4],
                loss_mask=[1, 1],
            ),
            SFTSequenceSample(
                sample_id="s2",
                input_ids=[5, 6],
                labels=[5, 6],
                loss_mask=[1, 1],
            ),
        ]
        isolated = list(SampleIsolatedPacker(6, pad_id=0).pack(sft))[0]
        self.assertEqual(isolated.segment_ids[:4], [0, 0, 1, 1])
        self.assertEqual(isolated.loss_mask[-2:], [0, 0])

    def test_preference_reward_and_trajectory_batch_contracts(self):
        tokenizer = make_text_tokenizer()
        renderer = RoleTagRenderer()
        preference = PreferencePreprocessor(
            tokenizer,
            renderer,
            add_eos=False,
        ).prepare(
            {
                "sample_id": "pref",
                "pair_id": "pair",
                "prompt_messages": [{"role": "user", "content": "q"}],
                "chosen": "long",
                "rejected": "x",
            }
        )
        preference.chosen_mask[-1] = 0
        preference_batch = PreferenceBatchBuilder(0)([preference])
        self.assertEqual(preference_batch["pair_ids"], ["pair"])
        self.assertEqual(
            preference_batch["chosen_input_ids"].size(0),
            preference_batch["rejected_input_ids"].size(0),
        )
        prompt_length = len(preference.prompt_input_ids)
        self.assertEqual(
            int(preference_batch["chosen_labels"][0, prompt_length + 3]),
            -100,
        )
        self.assertEqual(
            int(preference_batch["chosen_attention_mask"][0, prompt_length + 3]),
            0,
        )

        reward = RewardPreprocessor(
            tokenizer,
            renderer,
            add_eos=False,
        ).prepare(
            {
                "sample_id": "reward",
                "prompt_messages": [{"role": "user", "content": "q"}],
                "response": "ab",
                "reward_label": 1.0,
            }
        )
        reward.reward_mask[-1] = 0
        reward_batch = RewardBatchBuilder(0)([reward])
        self.assertEqual(reward_batch["reward_label"].tolist(), [1.0])
        self.assertTrue(torch.is_tensor(reward_batch["input_ids"]))
        self.assertEqual(
            int(reward_batch["attention_mask"][0, len(reward.prompt_input_ids) + 1]),
            0,
        )
        self.assertEqual(
            int(reward_batch["labels"][0, len(reward.prompt_input_ids) + 1]),
            -100,
        )

        trajectory = TrajectorySample(
            sample_id="t",
            prompt_input_ids=[3],
            response_input_ids=[4, 5],
            advantages=[0.5, -0.5],
            old_log_probs=[-1.0, -2.0],
            reference_log_probs=[-1.5, -2.5],
            values=[0.1, 0.2],
        )
        trajectory_batch = TrajectoryBatchBuilder(0)([trajectory])
        self.assertEqual(trajectory_batch["advantages"].shape, (1, 2))
        self.assertEqual(trajectory_batch["response_input_ids"].shape, (1, 2))


class TestUnifiedConfigurationAndRuntime(unittest.TestCase):
    def _config(self, dataset_path: Path) -> UnifiedDataConfig:
        return UnifiedDataConfig(
            task_type="pretraining",
            dataset_id="runtime-test",
            storage=StorageConfig(
                train_path=dataset_path,
                num_workers=0,
            ),
            sampling=SamplingConfig(
                strategy="random",
                shuffle=False,
                drop_last=False,
            ),
            packing=UnifiedPackingConfig(
                enabled=True,
                strategy="stream",
                max_seq_length=6,
            ),
            batching=BatchingConfig(
                strategy="fixed_length",
                micro_batch_size=2,
            ),
        )

    def test_configuration_is_strict_and_runtime_produces_model_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            writer = BinarySampleWriter(
                output,
                dataset_id="runtime-test",
                sample_type="pretraining",
            )
            for index in range(4):
                writer.write(
                    PretrainingSample(
                        sample_id=f"s-{index}",
                        input_ids=[3, 10 + index, 4],
                        document_ids=[0, 0, 0],
                    )
                )
            writer.commit()

            config = self._config(output)
            factory = UnifiedDataLoaderFactory(config, pad_token_id=0)
            loader = factory.create(
                batch_size=2,
                pin_memory=False,
                persistent_workers=False,
            )
            try:
                loader.set_epoch(0)
                batch = next(iter(loader))
                self.assertEqual(batch["input_ids"].shape, (1, 6))
                self.assertTrue(torch.all(batch["attention_mask"] == 1))
                model_inputs = factory.input_adapter()(batch)
                self.assertEqual(
                    set(model_inputs),
                    {"input_ids", "attention_mask", "labels"},
                )
            finally:
                loader.close()

    def test_invalid_task_and_packing_combination_is_rejected(self):
        config = UnifiedDataConfig(
            task_type="preference",
            dataset_id="preference",
            storage=StorageConfig(train_path=Path("unused")),
            packing=UnifiedPackingConfig(
                enabled=True,
                strategy="stream",
            ),
        )
        with self.assertRaises(TaiChuConfigurationError):
            UnifiedDataLoaderFactory(config, pad_token_id=0)

        raw = config.to_dict()
        raw["unexpected"] = True
        with self.assertRaises(TaiChuConfigurationError):
            UnifiedDataConfig.from_dict(raw)

    def test_resume_state_and_experience_buffer_contracts(self):
        state = DataState(
            dataset_id="d",
            dataset_version="v1",
            epoch=2,
            batch_index=10,
            consumed_samples=80,
        )
        restored = DataState.from_dict(state.to_dict())
        self.assertTrue(restored.compatible_with("d", "v1"))
        self.assertFalse(restored.compatible_with("other", "v1"))

        trajectory = TrajectorySample(
            sample_id="online",
            prompt_input_ids=[1],
            response_input_ids=[2],
            rewards=[1.0],
        )
        buffer = ExperienceBuffer(max_samples=2)
        buffer.append(trajectory)
        self.assertEqual(len(buffer), 1)
        self.assertEqual(buffer.snapshot()[0].sample_id, "online")


class TestMultiTaskRuntimeIntegration(unittest.TestCase):
    @staticmethod
    def _write(path: Path, sample_type: str, samples) -> None:
        writer = BinarySampleWriter(
            path,
            dataset_id=f"{sample_type}-dataset",
            sample_type=sample_type,
        )
        for sample in samples:
            writer.write(sample)
        writer.commit()

    def test_sft_runtime_is_rank_sharded_and_keeps_loss_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sft"
            samples = [
                SFTSequenceSample(
                    sample_id=f"sft-{index}",
                    input_ids=[3, 10 + index, 11 + index, 4],
                    labels=[-100, 10 + index, 11 + index, 4],
                    loss_mask=[0, 1, 1, 1],
                    attention_mask=[1, 1, 1, 1],
                    segment_ids=[0, 0, 0, 0],
                )
                for index in range(2)
            ]
            self._write(path, "sft", samples)
            config = UnifiedDataConfig(
                task_type="sft",
                dataset_id="sft-dataset",
                storage=StorageConfig(train_path=path, num_workers=0),
                sampling=SamplingConfig(
                    strategy="distributed",
                    shuffle=False,
                ),
                packing=UnifiedPackingConfig(
                    enabled=True,
                    strategy="sample_isolated",
                    max_seq_length=8,
                ),
                batching=BatchingConfig(
                    strategy="fixed_length",
                    micro_batch_size=1,
                    pad_to_multiple_of=1,
                ),
            )
            seen = []
            for rank in range(2):
                loader = UnifiedDataLoaderFactory(
                    config,
                    pad_token_id=0,
                ).create(
                    batch_size=1,
                    rank=rank,
                    world_size=2,
                    pin_memory=False,
                )
                try:
                    batch = next(iter(loader))
                    seen.extend(batch["sample_ids"])
                    self.assertEqual(batch["input_ids"].shape, (1, 8))
                    self.assertTrue(torch.all(batch["attention_mask"][:, :4] == 1))
                    self.assertTrue(torch.all(batch["attention_mask"][:, 4:] == 0))
                finally:
                    loader.close()
            self.assertEqual(sorted(seen), ["sft-0", "sft-1"])

    def test_preference_runtime_preserves_pairwise_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preference"
            samples = [
                PreferenceSample(
                    sample_id=f"sample-{index}",
                    pair_id=f"pair-{index}",
                    prompt_input_ids=[3],
                    chosen_input_ids=[7, 4],
                    rejected_input_ids=[8],
                    chosen_mask=[1, 1],
                    rejected_mask=[1],
                )
                for index in range(2)
            ]
            self._write(path, "preference", samples)
            config = UnifiedDataConfig(
                task_type="preference",
                dataset_id="preference-dataset",
                storage=StorageConfig(train_path=path, num_workers=0),
                sampling=SamplingConfig(
                    strategy="pair_aware",
                    shuffle=False,
                ),
                packing=UnifiedPackingConfig(
                    enabled=False,
                    strategy="none",
                ),
                batching=BatchingConfig(
                    strategy="preference_pair",
                    micro_batch_size=2,
                ),
            )
            loader = UnifiedDataLoaderFactory(
                config,
                pad_token_id=0,
            ).create(pin_memory=False)
            try:
                batch = next(iter(loader))
                self.assertEqual(batch["pair_ids"], ["pair-0", "pair-1"])
                self.assertEqual(batch["chosen_input_ids"].size(0), 2)
                self.assertEqual(batch["rejected_input_ids"].size(0), 2)
            finally:
                loader.close()

    def test_sft_padding_free_runtime_emits_flat_sequence_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft-flat"
            samples = [
                SFTSequenceSample(
                    sample_id=f"sft-{index}",
                    input_ids=[3, 10 + index, 11 + index, 4],
                    labels=[-100, 10 + index, 11 + index, 4],
                    loss_mask=[0, 1, 1, 1],
                    attention_mask=[1, 1, 1, 1],
                    segment_ids=[0, 0, 0, 0],
                )
                for index in range(2)
            ]
            self._write(path, "sft", samples)
            config = UnifiedDataConfig(
                task_type="sft",
                dataset_id="sft-flat",
                storage=StorageConfig(train_path=path, num_workers=0),
                sampling=SamplingConfig(
                    strategy="random",
                    shuffle=False,
                ),
                packing=UnifiedPackingConfig(
                    enabled=True,
                    strategy="sample_isolated",
                    max_seq_length=8,
                ),
                batching=BatchingConfig(
                    strategy="padding_free",
                    micro_batch_size=2,
                    pad_to_multiple_of=1,
                ),
            )
            loader = UnifiedDataLoaderFactory(
                config,
                pad_token_id=0,
            ).create(pin_memory=False)
            try:
                batch = next(iter(loader))
                self.assertEqual(batch["input_ids"].shape, (8,))
                self.assertEqual(batch["position_ids"].tolist(), [0, 1, 2, 3, 0, 1, 2, 3])
                self.assertEqual(batch["cu_seqlens"].tolist(), [0, 4, 8])
                self.assertEqual(batch["sample_ids"], ["sft-0", "sft-1"])
            finally:
                loader.close()

    def test_reward_and_trajectory_runtime_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reward_path = root / "reward"
            self._write(
                reward_path,
                "reward",
                [
                    RewardSample(
                        sample_id="reward-1",
                        prompt_input_ids=[3],
                        response_input_ids=[7, 4],
                        reward_label=0.5,
                        reward_mask=[1, 1],
                    )
                ],
            )
            reward_config = UnifiedDataConfig(
                task_type="reward",
                dataset_id="reward-dataset",
                storage=StorageConfig(train_path=reward_path, num_workers=0),
                sampling=SamplingConfig(strategy="random", shuffle=False),
                packing=UnifiedPackingConfig(enabled=False, strategy="none"),
                batching=BatchingConfig(
                    strategy="reward_pair",
                    micro_batch_size=1,
                ),
            )
            reward_loader = UnifiedDataLoaderFactory(
                reward_config,
                pad_token_id=0,
            ).create(pin_memory=False)
            try:
                reward_batch = next(iter(reward_loader))
                self.assertEqual(reward_batch["reward_label"].tolist(), [0.5])
            finally:
                reward_loader.close()

            trajectory_path = root / "trajectory"
            self._write(
                trajectory_path,
                "trajectory",
                [
                    TrajectorySample(
                        sample_id="trajectory-1",
                        prompt_input_ids=[3],
                        response_input_ids=[7, 4],
                        group_id="group-1",
                        candidate_index=0,
                        advantages=[0.5, -0.5],
                        old_log_probs=[-0.1, -0.2],
                        reference_log_probs=[-0.2, -0.3],
                        values=[0.0, 1.0],
                    )
                ],
            )
            trajectory_config = UnifiedDataConfig(
                task_type="ppo",
                dataset_id="trajectory-dataset",
                storage=StorageConfig(
                    train_path=trajectory_path,
                    num_workers=0,
                ),
                sampling=SamplingConfig(strategy="random", shuffle=False),
                packing=UnifiedPackingConfig(enabled=False, strategy="none"),
                batching=BatchingConfig(
                    strategy="trajectory",
                    micro_batch_size=1,
                ),
            )
            trajectory_loader = UnifiedDataLoaderFactory(
                trajectory_config,
                pad_token_id=0,
            ).create(pin_memory=False)
            try:
                trajectory_batch = next(iter(trajectory_loader))
                self.assertEqual(trajectory_batch["group_ids"], ["group-1"])
                self.assertEqual(trajectory_batch["advantages"].shape, (1, 2))
            finally:
                trajectory_loader.close()

    def test_resumable_iterator_skips_completed_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume"
            self._write(
                path,
                "pretraining",
                [
                    PretrainingSample(
                        sample_id=f"sample-{index}",
                        input_ids=[3, 10 + index, 4],
                        document_ids=[0, 0, 0],
                    )
                    for index in range(4)
                ],
            )
            config = UnifiedDataConfig(
                task_type="pretraining",
                dataset_id="resume-dataset",
                storage=StorageConfig(train_path=path, num_workers=0),
                sampling=SamplingConfig(
                    strategy="random",
                    shuffle=False,
                ),
                packing=UnifiedPackingConfig(
                    enabled=False,
                    strategy="none",
                ),
                batching=BatchingConfig(
                    strategy="fixed_length",
                    micro_batch_size=1,
                ),
            )
            first_loader = UnifiedDataLoaderFactory(
                config,
                pad_token_id=0,
            ).create(pin_memory=False)
            try:
                first = ResumableBatchIterator(first_loader)
                first_iter = iter(first)
                next(first_iter)
                next(first_iter)
                state = first.state
            finally:
                first_loader.close()
            self.assertEqual(state.batch_index, 2)

            second_loader = UnifiedDataLoaderFactory(
                config,
                pad_token_id=0,
            ).create(pin_memory=False)
            try:
                resumed = next(iter(ResumableBatchIterator(second_loader, state)))
                self.assertEqual(resumed["sample_ids"], ["sample-2"])
            finally:
                second_loader.close()

    def test_data_packing_and_throughput_metrics(self):
        data = DataMetrics()
        data.record_batch(samples=4, tokens=256, padding_tokens=16)
        data.record_quality(accepted=10, rejected=2, duplicates=1)
        data.record_skipped(1)
        snapshot = data.snapshot()
        self.assertEqual(snapshot["samples"], 4)
        self.assertEqual(snapshot["accepted_documents"], 10)

        packing = PackingMetrics()
        packing.record(
            source_sequences=4,
            packed_sequences=1,
            source_tokens=24,
            packed_tokens=24,
            padding_tokens=8,
            segments=4,
        )
        packing_snapshot = packing.snapshot()
        self.assertEqual(packing_snapshot["utilization"], 0.75)

        clock_values = iter([0.0, 2.0])
        throughput = ThroughputMetrics(clock=lambda: next(clock_values))
        throughput.record(batches=1, samples=4, tokens=256, bytes_read=128)
        throughput_snapshot = throughput.snapshot()
        self.assertEqual(throughput_snapshot["samples_per_second"], 2.0)
        self.assertEqual(throughput_snapshot["bytes_per_second"], 64.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
