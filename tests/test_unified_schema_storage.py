"""Layer-one tests for unified schemas and mmap token storage."""
from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from dataset.schema import (
    ConversationMessage,
    MultimodalSample,
    PreferenceSample,
    PretrainingSample,
    RewardSample,
    SFTSequenceSample,
    TrajectorySample,
)
from dataset.storage import (
    BinaryDatasetReader,
    BinarySampleWriter,
    DatasetManifest,
    ShardManager,
)


class TestUnifiedSchemas(unittest.TestCase):
    def test_task_samples_validate_and_store_optional_arrays(self):
        samples = [
            PretrainingSample(
                sample_id="pre-1",
                input_ids=[3, 4, 5],
                document_ids=[0, 0, 1],
                metadata={"language": "zh"},
            ),
            SFTSequenceSample(
                sample_id="sft-1",
                input_ids=[3, 7, 8, 4],
                labels=[-100, 7, 8, 4],
                loss_mask=[0, 1, 1, 1],
                messages=[
                    ConversationMessage(role="user", content="hi"),
                    ConversationMessage(role="assistant", content="hello"),
                ],
            ),
            PreferenceSample(
                sample_id="pref-1",
                pair_id="pair-1",
                prompt_input_ids=[3],
                chosen_input_ids=[7, 4],
                rejected_input_ids=[8, 4],
                chosen_mask=[1, 1],
                rejected_mask=[1, 1],
            ),
            RewardSample(
                sample_id="reward-1",
                prompt_input_ids=[3],
                response_input_ids=[7, 4],
                reward_label=0.75,
                reward_mask=[0, 1],
            ),
            TrajectorySample(
                sample_id="traj-1",
                prompt_input_ids=[3],
                response_input_ids=[7, 8],
                rewards=[1.0],
                advantages=[0.5, -0.5],
                old_log_probs=[-0.1, -0.2],
                reference_log_probs=[-0.2, -0.3],
                group_id="group-1",
            ),
            MultimodalSample(
                sample_id="mm-1",
                input_ids=[3, 4],
                image_refs=["object://image-1"],
                image_pixel_values=[0.1, 0.2, 0.3],
                image_shape=[1, 3, 1],
            ),
        ]
        for sample in samples:
            with self.subTest(sample_type=sample.sample_type):
                sample.validate()
                self.assertTrue(sample.sample_id)
                self.assertEqual(type(sample).__name__, sample.__class__.__name__)

    def test_sft_requires_ignore_index_outside_loss_mask(self):
        with self.assertRaises(ValueError):
            SFTSequenceSample(
                sample_id="bad",
                input_ids=[1, 2],
                labels=[1, 2],
                loss_mask=[1, 0],
            )


class TestBinaryStorage(unittest.TestCase):
    def test_binary_roundtrip_manifest_checksums_and_shards(self):
        samples = [
            PretrainingSample(
                sample_id=f"sample-{index}",
                input_ids=[3, 10 + index, 4],
                document_ids=[0, 0, 1],
                metadata={"index": index},
            )
            for index in range(5)
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            writer = BinarySampleWriter(
                output,
                dataset_id="pretraining-corpus",
                sample_type="pretraining",
                version="2026.11",
                shard_size=2,
                metadata={"tokenizer": "test"},
            )
            for sample in samples:
                writer.write(sample)
            manifest = writer.commit()

            self.assertEqual(manifest.sample_count, 5)
            self.assertEqual(len(manifest.shards), 3)
            self.assertTrue((output / "manifest.json").is_file())
            loaded = DatasetManifest.load(output)
            self.assertEqual(loaded.version, "2026.11")
            self.assertEqual(loaded.content_hash, manifest.content_hash)

            with BinaryDatasetReader(output, checksums=True) as reader:
                self.assertEqual(len(reader), 5)
                self.assertEqual(reader.sample_type, "pretraining")
                self.assertEqual(reader[0].sample_id, "sample-0")
                self.assertEqual(reader[-1].input_ids, [3, 14, 4])
                self.assertEqual(
                    [sample.metadata["index"] for sample in reader],
                    list(range(5)),
                )
                self.assertEqual(reader[4].labels, None)
                restored = pickle.loads(pickle.dumps(reader))
                try:
                    self.assertEqual(restored[4].sample_id, "sample-4")
                finally:
                    restored.close()

    def test_optional_arrays_and_floats_roundtrip(self):
        sample = TrajectorySample(
            sample_id="trajectory-1",
            prompt_input_ids=[1],
            response_input_ids=[2, 3],
            rewards=[0.25],
            advantages=[1.5, -2.5],
            old_log_probs=[-0.125, -0.25],
            reference_log_probs=[-0.5, -0.75],
            values=[0.0, 1.0],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trajectory"
            writer = BinarySampleWriter(
                output,
                dataset_id="rollouts",
                sample_type="trajectory",
            )
            writer.write(sample)
            writer.commit()
            with BinaryDatasetReader(output) as reader:
                restored = reader[0]
            self.assertIsInstance(restored, TrajectorySample)
            self.assertEqual(restored.advantages, [1.5, -2.5])
            self.assertEqual(restored.response_mask, None)

    def test_multimodal_tensor_references_roundtrip(self):
        sample = MultimodalSample(
            sample_id="image-1",
            input_ids=[3, 4, 5],
            image_refs=["object://bucket/image.png"],
            image_pixel_values=[0.25, 0.5, 0.75],
            image_shape=[1, 3, 1],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "multimodal"
            writer = BinarySampleWriter(
                output,
                dataset_id="multimodal",
                sample_type="multimodal",
            )
            writer.write(sample)
            writer.commit()
            with BinaryDatasetReader(output, checksums=True) as reader:
                restored = reader[0]
            self.assertIsInstance(restored, MultimodalSample)
            self.assertEqual(restored.image_refs, sample.image_refs)
            self.assertEqual(restored.image_shape, [1, 3, 1])

    def test_writer_does_not_overwrite_and_abort_cleans_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            writer = BinarySampleWriter(
                output,
                dataset_id="d",
                sample_type="pretraining",
            )
            writer.abort()
            self.assertFalse(output.exists())

            completed = BinarySampleWriter(
                output,
                dataset_id="d",
                sample_type="pretraining",
            )
            completed.write(
                PretrainingSample(sample_id="one", input_ids=[1, 2])
            )
            completed.commit()
            with self.assertRaises(FileExistsError):
                BinarySampleWriter(
                    output,
                    dataset_id="d",
                    sample_type="pretraining",
                )


class TestShardManager(unittest.TestCase):
    def test_ranges_are_disjoint_and_cover_expected_indices(self):
        manager = ShardManager(10)
        ranges = manager.ranges(3)
        values = [index for item in ranges for index in item]
        self.assertEqual(values, list(range(10)))
        self.assertEqual([len(item) for item in ranges], [3, 3, 4])
        dropped = manager.ranges(3, drop_last=True)
        values = [index for item in dropped for index in item]
        self.assertEqual(values, list(range(9)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
