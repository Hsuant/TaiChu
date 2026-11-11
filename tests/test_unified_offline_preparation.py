"""End-to-end tests for canonical offline preparation."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dataset.config import UnifiedDataConfig
from dataset.offline.config import load_preparation_config
from dataset.offline.prepare import UnifiedDataPreparation
from dataset.offline.pretraining import load_pretraining_config
from dataset.offline.readers import RecordSourceReader
from dataset.storage import BinaryDatasetReader


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "taichu_tokenizer" / "tokenizer.json"


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def sft_record(text: str, index: int = 0) -> dict:
    return {
        "sample_id": f"sft-{index}",
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"answer {index}"},
        ],
    }


def write_config(
    path: Path,
    output: Path,
    train: Path,
    validation: Path,
    *,
    batch_size: int = 256,
    workers: int = 1,
) -> None:
    path.write_text(
        "\n".join(
            [
                "task_type: sft",
                "dataset_id: offline-test",
                'version: "1"',
                f'tokenizer_path: "{TOKENIZER_PATH.as_posix()}"',
                f'output_dir: "{output.as_posix()}"',
                "max_seq_length: 128",
                f'pad_token: "<|pad|>"',
                f'bos_token: "<|taichu_start|>"',
                f'eos_token: "<|taichu_end|>"',
                "add_eos: true",
                "assistant_only_loss: true",
                "train_on_eos: true",
                "invalid_policy: error",
                "shard_size: 2",
                f"preprocess_batch_size: {batch_size}",
                f"preprocess_workers: {workers}",
                "splits:",
                "  train:",
                f'    - "{train.as_posix()}"',
                "  validation:",
                f'    - "{validation.as_posix()}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


class TestUnifiedOfflinePreparation(unittest.TestCase):
    def test_reader_reports_processed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            write_jsonl(source, [{"text": "first"}, {"text": "second"}])
            observed = []
            records = list(
                RecordSourceReader().read(
                    source,
                    progress_callback=observed.append,
                )
            )
            self.assertEqual(len(records), 2)
            self.assertEqual(sum(observed), source.stat().st_size)

    def test_raw_jsonl_to_binary_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            write_jsonl(
                train,
                [
                    sft_record("first training document", 0),
                    sft_record("second training document", 1),
                    sft_record("third training document", 2),
                ],
            )
            write_jsonl(validation, [sft_record("validation document", 3)])
            output = root / "prepared"
            config_path = root / "prepare.yaml"
            write_config(config_path, output, train, validation)

            config = load_preparation_config(config_path)
            manifest = UnifiedDataPreparation(config).run()
            self.assertEqual(manifest["sample_type"], "sft")
            self.assertTrue((output / "dataset_manifest.json").is_file())

            with BinaryDatasetReader(output / "train") as reader:
                self.assertEqual(reader.sample_type, "sft")
                self.assertEqual(len(reader), 3)
                self.assertTrue(reader[0].input_ids)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.jsonl"
            write_jsonl(source, [sft_record("document one")])
            output = root / "prepared"
            config_path = root / "prepare.yaml"
            write_config(config_path, output, source, source)
            config = load_preparation_config(config_path)
            UnifiedDataPreparation(config).run()
            with self.assertRaises(FileExistsError):
                UnifiedDataPreparation(config).run()

    def test_raw_parquet_is_an_offline_source_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.parquet"
            pq.write_table(
                pa.table(
                    {
                        "messages": [
                            sft_record("first document", 0)["messages"],
                            sft_record("second document", 1)["messages"],
                        ]
                    }
                ),
                source,
                row_group_size=1,
            )
            output = root / "prepared"
            config_path = root / "prepare.yaml"
            write_config(config_path, output, source, source)
            config = load_preparation_config(config_path)
            UnifiedDataPreparation(config).run()
            with BinaryDatasetReader(output / "train") as reader:
                self.assertEqual(len(reader), 2)

    def test_parallel_preparation_writes_binary_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            write_jsonl(
                train,
                [
                    sft_record(f"training document {index}", index)
                    for index in range(8)
                ],
            )
            write_jsonl(validation, [sft_record("validation document", 99)])
            output = root / "prepared"
            config_path = root / "prepare.yaml"
            write_config(
                config_path,
                output,
                train,
                validation,
                batch_size=2,
                workers=2,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dataset.offline.prepare",
                    "--config",
                    str(config_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("RuntimeWarning", result.stderr)
            with BinaryDatasetReader(output / "train") as reader:
                self.assertEqual(len(reader), 8)

    def test_cli_help_is_available(self):
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
        self.assertIn("--restart", result.stdout)
        self.assertNotIn("RuntimeWarning", result.stderr)


class TestUnifiedConfigurationFiles(unittest.TestCase):
    def test_unified_training_examples_parse(self):
        names = (
            "data_pretraining_unified.yaml",
            "data_sft_unified.yaml",
            "data_preference_unified.yaml",
            "data_reward_unified.yaml",
            "data_trajectory_unified.yaml",
        )
        for name in names:
            with self.subTest(name=name):
                config = UnifiedDataConfig.from_yaml(ROOT / "configs" / name)
                self.assertTrue(config.dataset_id)
                self.assertEqual(config.storage.format, "binary_mmap")

    def test_unified_preparation_examples_parse(self):
        pretraining = load_pretraining_config(
            ROOT / "configs" / "prepare_pretrain_data.yaml"
        )
        self.assertTrue(pretraining.dataset_id)
        self.assertTrue(pretraining.sources)
        names = (
            "prepare_sft_data.yaml",
            "prepare_preference_data.yaml",
            "prepare_reward_data.yaml",
            "prepare_trajectory_data.yaml",
        )
        for name in names:
            with self.subTest(name=name):
                config = load_preparation_config(ROOT / "configs" / name)
                self.assertTrue(config.dataset_id)
                self.assertTrue(config.splits)


if __name__ == "__main__":
    unittest.main(verbosity=2)
