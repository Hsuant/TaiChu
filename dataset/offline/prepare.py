#!/usr/bin/env python3
"""Prepare task-specific raw records into unified binary datasets."""
from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from tqdm import tqdm

from ..storage import BinarySampleWriter
from ..storage.fs import atomic_replace_directory
from ..tasks import get_task
from .adapters import (
    BasePreprocessingAdapter,
    PreferencePreprocessor,
    RewardPreprocessor,
    RoleTagRenderer,
    SFTPreprocessor,
    TextTokenizer,
    TrajectoryPreprocessor,
)
from .config import PreparationConfig, load_preparation_config
from .readers import RecordSourceReader
from .unified import UnifiedPreparationPipeline


@dataclass(slots=True)
class SplitPreparationStats:
    source_records: int = 0
    cleaned_records: int = 0
    duplicate_records: int = 0
    rejected_records: int = 0
    written_records: int = 0


AdapterFactory = Callable[[PreparationConfig, TextTokenizer], BasePreprocessingAdapter]


def _sft_adapter(
    config: PreparationConfig,
    tokenizer: TextTokenizer,
) -> BasePreprocessingAdapter:
    return SFTPreprocessor(
        tokenizer,
        RoleTagRenderer(),
        max_length=config.max_seq_length,
        truncation=config.truncation,
        assistant_only_loss=config.assistant_only_loss,
        add_eos=config.add_eos,
        train_on_eos=config.train_on_eos,
    )


def _preference_adapter(
    config: PreparationConfig,
    tokenizer: TextTokenizer,
) -> BasePreprocessingAdapter:
    return PreferencePreprocessor(
        tokenizer,
        RoleTagRenderer(),
        add_eos=config.add_eos,
        max_response_length=config.max_seq_length,
    )


def _reward_adapter(
    config: PreparationConfig,
    tokenizer: TextTokenizer,
) -> BasePreprocessingAdapter:
    return RewardPreprocessor(
        tokenizer,
        RoleTagRenderer(),
        add_eos=config.add_eos,
    )


def _trajectory_adapter(
    config: PreparationConfig,
    tokenizer: TextTokenizer,
) -> BasePreprocessingAdapter:
    return TrajectoryPreprocessor(
        tokenizer,
        RoleTagRenderer(),
        add_eos=config.add_eos,
    )


ADAPTER_REGISTRY: dict[str, AdapterFactory] = {
    "sft": _sft_adapter,
    "instruction_tuning": _sft_adapter,
    "preference": _preference_adapter,
    "dpo": _preference_adapter,
    "ipo": _preference_adapter,
    "kto": _preference_adapter,
    "reward": _reward_adapter,
    "rlhf": _trajectory_adapter,
    "ppo": _trajectory_adapter,
    "grpo": _trajectory_adapter,
}


class UnifiedDataPreparation:
    def __init__(self, config: PreparationConfig) -> None:
        self.config = config
        if not config.tokenizer_path.is_file():
            raise FileNotFoundError(f"tokenizer does not exist: {config.tokenizer_path}")
        try:
            adapter_factory = ADAPTER_REGISTRY[config.task_type]
        except KeyError as exc:
            raise ValueError(
                f"unsupported preparation task_type {config.task_type!r}; "
                f"available: {sorted(ADAPTER_REGISTRY)}"
            ) from exc
        self.task = get_task(config.task_type)
        self.tokenizer = TextTokenizer(
            Tokenizer.from_file(str(config.tokenizer_path)),
            pad_token=config.pad_token,
            bos_token=config.bos_token,
            eos_token=config.eos_token,
        )
        self.adapter = adapter_factory(config, self.tokenizer)
        self.reader = RecordSourceReader()

    def run(self, *, restart: bool = False) -> dict[str, Any]:
        if self.config.output_dir.exists():
            raise FileExistsError(
                f"preparation output already exists: {self.config.output_dir}"
            )
        staging = self.config.output_dir.with_name(
            f".{self.config.output_dir.name}.staging"
        )
        if staging.exists():
            if not restart:
                raise FileExistsError(
                    f"preparation staging already exists: {staging}; "
                    "use --restart to discard it"
                )
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        split_manifests: dict[str, Any] = {}
        try:
            for split, paths in self.config.splits.items():
                split_manifests[split] = self._prepare_split(
                    split,
                    paths,
                    staging,
                )
            root_index = {
                "schema_version": 1,
                "dataset_id": self.config.dataset_id,
                "version": self.config.version,
                "task_type": self.config.task_type,
                "sample_type": self.task.sample_type,
                "splits": split_manifests,
            }
            with (staging / "dataset_manifest.json").open(
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump(root_index, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            atomic_replace_directory(staging, self.config.output_dir)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        return root_index

    def _prepare_split(
        self,
        split: str,
        paths: tuple[Path, ...],
        staging: Path,
    ) -> dict[str, Any]:
        output = staging / split
        writer = BinarySampleWriter(
            output,
            dataset_id=f"{self.config.dataset_id}:{split}",
            sample_type=self.task.sample_type,
            version=self.config.version,
            shard_size=self.config.shard_size,
            metadata={
                "task_type": self.config.task_type,
                "split": split,
                "tokenizer_path": str(self.config.tokenizer_path),
            },
        )
        stats = SplitPreparationStats()
        pipeline = UnifiedPreparationPipeline(
            self.adapter,
            writer,
            invalid_policy=self.config.invalid_policy,
            batch_size=self.config.preprocess_batch_size,
        )
        pipeline_stats = pipeline.run(self._records(paths, split, stats))
        manifest = writer.commit()
        stats.written_records = pipeline_stats.written_samples
        stats.rejected_records += pipeline_stats.skipped_samples
        return {
            "manifest": (Path(split) / "manifest.json").as_posix(),
            "sample_count": manifest.sample_count,
            "content_hash": manifest.content_hash,
            "stats": asdict(stats),
        }

    def _records(
        self,
        paths: tuple[Path, ...],
        split: str,
        stats: SplitPreparationStats,
    ) -> Iterator[dict[str, Any]]:
        index = 0
        files = self.reader.resolve(paths)
        total_bytes = sum(path.stat().st_size for path in files)
        with tqdm(
            total=total_bytes,
            desc=f"prepare {split}",
            unit="B",
            unit_scale=True,
            unit_divisor=1000,
            dynamic_ncols=True,
        ) as progress:
            for path in files:
                plain_jsonl = path.suffix.lower() == ".jsonl"
                callback = progress.update if plain_jsonl else None
                for raw in self.reader.read(
                    path,
                    progress_callback=callback,
                ):
                    stats.source_records += 1
                    record = dict(raw)
                    record.setdefault(
                        "sample_id",
                        f"{self.config.dataset_id}-{split}-{index:010d}",
                    )
                    record.setdefault("dataset_id", self.config.dataset_id)
                    record.setdefault("source_id", str(path))
                    index += 1
                    stats.cleaned_records += 1
                    if stats.source_records % 1000 == 0:
                        progress.set_postfix(
                            kept=stats.cleaned_records,
                            duplicate=stats.duplicate_records,
                            rejected=stats.rejected_records,
                        )
                    yield record
                if not plain_jsonl:
                    progress.update(path.stat().st_size)
            progress.set_postfix(
                kept=stats.cleaned_records,
                duplicate=stats.duplicate_records,
                rejected=stats.rejected_records,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare document-aware pretraining data or generic task records "
            "into unified binary storage"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard an existing incomplete staging directory before preparation",
    )
    args = parser.parse_args()
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load preparation configs") from exc
    with Path(args.config).open("r", encoding="utf-8") as stream:
        raw_config = yaml.safe_load(stream) or {}
    if (
        isinstance(raw_config, dict)
        and raw_config.get("task_type") in {"pretraining", "cpt"}
    ):
        from .pretraining import PretrainingDataPreparation, load_pretraining_config

        config = load_pretraining_config(args.config)
        manifest = PretrainingDataPreparation(config).run(restart=args.restart)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    config = load_preparation_config(args.config)
    manifest = UnifiedDataPreparation(config).run(restart=args.restart)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "ADAPTER_REGISTRY",
    "SplitPreparationStats",
    "UnifiedDataPreparation",
    "main",
]
