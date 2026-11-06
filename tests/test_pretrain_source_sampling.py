"""Tests for deterministic GB-based pretraining source sampling."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dataset.offline.sample_sources as sampler_module
from dataset.offline.sample_sources import (
    load_sampling_config,
    run_sampling,
)


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for index in range(count):
            text = f"sample-{index:04d} " + ("x" * 64)
            stream.write(json.dumps({"text": text}) + "\n")


def write_config(
    root: Path,
    *,
    source_a: Path,
    source_b: Path,
    output: Path,
    target_total_gb: float = 0.00002,
    min_chars: int = 16,
    max_chars: int = 200_000,
) -> Path:
    path = root / "sample.yaml"
    path.write_text(
        "\n".join(
            [
                f"target_total_gb: {target_total_gb}",
                f'output_dir: "{output.as_posix()}"',
                "seed: 7",
                "validation_fraction: 0.25",
                f"min_chars: {min_chars}",
                f"max_chars: {max_chars}",
                "sources:",
                "  - name: source_a",
                '    language: "zh"',
                "    patterns:",
                f'      - "{source_a.as_posix()}"',
                "    weight_gb: 1.0",
                "  - name: source_b",
                '    language: "en"',
                "    patterns:",
                f'      - "{source_b.as_posix()}"',
                "    weight_gb: 2.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


class TestPretrainSourceSampling(unittest.TestCase):
    def test_samples_with_stable_ratio_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "a.jsonl"
            source_b = root / "b.jsonl"
            write_jsonl(source_a, 100)
            write_jsonl(source_b, 200)
            output = root / "sampled"
            config = load_sampling_config(
                write_config(
                    root,
                    source_a=source_a,
                    source_b=source_b,
                    output=output,
                )
            )

            first = run_sampling(config, show_progress=False)
            self.assertTrue((output / "sampling_manifest.json").is_file())
            self.assertEqual(
                first["train_records"] + first["validation_records"],
                sum(item["selected_records"] for item in first["sources"]),
            )
            self.assertTrue((output / "train" / "source_a.jsonl").is_file())
            self.assertTrue((output / "validation" / "source_b.jsonl").is_file())
            self.assertGreater(first["actual_total_gb"], 0)

            with self.assertRaises(FileExistsError):
                run_sampling(config, show_progress=False)

    def test_dry_run_resolves_globs_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "a-0001.jsonl"
            source_b = root / "b.jsonl"
            write_jsonl(source_a, 20)
            write_jsonl(source_b, 20)
            config_path = write_config(
                root,
                source_a=root / "a-*.jsonl",
                source_b=source_b,
                output=root / "sampled",
            )
            config = load_sampling_config(config_path)
            summary = run_sampling(
                config,
                target_total_gb=0.000002,
                dry_run=True,
                show_progress=False,
            )
            self.assertEqual(len(summary["sources"]), 2)
            self.assertEqual(len(summary["sources"][0]["files"]), 1)
            self.assertFalse((root / "sampled").exists())

    def test_filters_records_before_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "a.jsonl"
            source_b = root / "b.jsonl"
            lengths = (5, 20, 101)
            for path in (source_a, source_b):
                path.write_text(
                    "".join(
                        json.dumps({"text": "x" * length}) + "\n"
                        for length in lengths
                    ),
                    encoding="utf-8",
                )
            output = root / "sampled"
            config = load_sampling_config(
                write_config(
                    root,
                    source_a=source_a,
                    source_b=source_b,
                    output=output,
                    target_total_gb=0.000000003,
                    min_chars=10,
                    max_chars=100,
                )
            )

            summary = run_sampling(config, show_progress=False)
            self.assertEqual(summary["short_records"], 2)
            self.assertEqual(summary["long_records"], 2)
            self.assertEqual(summary["eligible_records"], 2)
            self.assertEqual(summary["min_chars"], 10)
            self.assertEqual(summary["max_chars"], 100)

            for line in (
                output / "train" / "source_a.jsonl"
            ).read_text(encoding="utf-8").splitlines():
                self.assertEqual(len(json.loads(line)["text"]), 20)

    def test_directory_commit_falls_back_to_file_moves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "a.jsonl"
            source_b = root / "b.jsonl"
            write_jsonl(source_a, 20)
            write_jsonl(source_b, 20)
            output = root / "sampled"
            config = load_sampling_config(
                write_config(
                    root,
                    source_a=source_a,
                    source_b=source_b,
                    output=output,
                    target_total_gb=0.000002,
                )
            )
            staging = output.with_name(f".{output.name}.staging")
            real_replace = os.replace

            def fail_directory_replace(source, target):
                if Path(source) == staging:
                    raise PermissionError("directory rename denied")
                return real_replace(source, target)

            with (
                mock.patch.object(
                    sampler_module.os,
                    "replace",
                    side_effect=fail_directory_replace,
                ),
                mock.patch.object(sampler_module.time, "sleep"),
            ):
                run_sampling(config, show_progress=False)

            self.assertTrue((output / "sampling_manifest.json").is_file())
            self.assertFalse(staging.exists())

    def test_commit_failure_preserves_completed_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "a.jsonl"
            source_b = root / "b.jsonl"
            write_jsonl(source_a, 20)
            write_jsonl(source_b, 20)
            output = root / "sampled"
            config = load_sampling_config(
                write_config(
                    root,
                    source_a=source_a,
                    source_b=source_b,
                    output=output,
                    target_total_gb=0.000002,
                )
            )
            staging = output.with_name(f".{output.name}.staging")

            with (
                mock.patch.object(
                    sampler_module.os,
                    "replace",
                    side_effect=PermissionError("file move denied"),
                ),
                mock.patch.object(sampler_module.time, "sleep"),
                self.assertRaises(PermissionError),
            ):
                run_sampling(config, show_progress=False)

            self.assertTrue(staging.is_dir())
            self.assertTrue((staging / "sampling_manifest.json").is_file())

            run_sampling(config, show_progress=False)
            self.assertTrue((output / "sampling_manifest.json").is_file())
            self.assertFalse(staging.exists())

    def test_cli_help_is_available(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dataset.offline.sample_sources",
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
        self.assertIn("--min-chars", result.stdout)
        self.assertIn("--max-chars", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
