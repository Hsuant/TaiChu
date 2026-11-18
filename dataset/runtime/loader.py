"""Unified PyTorch DataLoader integration for indexed offline datasets."""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from ..config import UnifiedDataConfig
from ..exceptions import TaiChuConfigurationError
from ..sampling import DistributedSampler, RandomSampler
from ..storage import ParquetDatasetReader
from ..tasks import get_task
from .dataset import IndexedTokenDataset


class TaskCollator:
    def __init__(self, packer, builder) -> None:
        self.packer = packer
        self.builder = builder

    def __call__(self, samples):
        packed = samples
        if self.packer is not None:
            packed = list(self.packer.pack(samples))
            if not packed:
                raise RuntimeError(
                    "packing emitted no sequences for this batch; "
                    "increase training.micro_batch_size or disable "
                    "drop_last_remainder"
                )
        return self.builder(packed)


class UnifiedDataLoader(DataLoader):
    def set_epoch(self, epoch: int) -> None:
        sampler = getattr(self, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def set_resume_batch(self, batch_index: int) -> None:
        """Skip completed samples without loading and discarding batches."""
        if batch_index < 0:
            raise ValueError("resume batch index cannot be negative")
        if self.batch_size is None:
            raise ValueError("resume batch index requires a batch size")
        sampler = getattr(self, "sampler", None)
        if sampler is None or not hasattr(sampler, "set_skip_samples"):
            raise ValueError("sampler does not support resume offsets")
        sampler.set_skip_samples(int(batch_index) * int(self.batch_size))

    def close(self) -> None:
        reader = getattr(getattr(self, "dataset", None), "reader", None)
        if reader is not None and hasattr(reader, "close"):
            reader.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class UnifiedDataLoaderFactory:
    def __init__(
        self,
        config: UnifiedDataConfig,
        *,
        pad_token_id: int,
    ) -> None:
        self.config = config
        self.pad_token_id = int(pad_token_id)
        self.task = get_task(config.task_type)
        if config.batching.strategy not in self.task.batching_strategies:
            raise TaiChuConfigurationError(
                f"batching strategy {config.batching.strategy!r} is invalid for "
                f"task {config.task_type!r}; expected one of "
                f"{sorted(self.task.batching_strategies)}"
            )
        if config.packing.enabled and not self.task.supports_packing:
            raise TaiChuConfigurationError(
                f"task {config.task_type!r} does not support packing"
            )
        if (
            config.packing.enabled
            and config.packing.strategy not in self.task.packing_strategies
        ):
            raise TaiChuConfigurationError(
                f"packing strategy {config.packing.strategy!r} is invalid for "
                f"task {config.task_type!r}; expected one of "
                f"{sorted(self.task.packing_strategies)}"
            )

    @staticmethod
    def _distributed_context(
        rank: int | None,
        world_size: int | None,
    ) -> tuple[int, int]:
        if rank is None and world_size is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return torch.distributed.get_rank(), torch.distributed.get_world_size()
            return 0, 1
        if rank is None or world_size is None:
            raise TaiChuConfigurationError(
                "rank and world_size must be provided together"
            )
        if world_size <= 0 or not 0 <= rank < world_size:
            raise TaiChuConfigurationError("rank must be in [0, world_size)")
        return int(rank), int(world_size)

    def _validate_dataset_identity(self, reader, split: str) -> None:
        canonical_split = (
            "train"
            if split in {"train", "training"}
            else "validation"
            if split in {"validation", "valid", "eval", "test"}
            else split
        )
        expected_ids = {
            self.config.dataset_id,
            f"{self.config.dataset_id}:{canonical_split}",
        }
        if reader.dataset_id not in expected_ids:
            actual_id = reader.dataset_id
            reader.close()
            raise TaiChuConfigurationError(
                f"dataset identity mismatch for split {split!r}: "
                f"configured {self.config.dataset_id!r}, manifest contains "
                f"{actual_id!r}"
            )

    def create(
        self,
        *,
        split: str = "train",
        batch_size: int,
        rank: int | None = None,
        world_size: int | None = None,
        num_workers: int | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        drop_last: bool | None = None,
    ) -> UnifiedDataLoader:
        if int(batch_size) <= 0:
            raise TaiChuConfigurationError("batch_size must be positive")
        rank, world_size = self._distributed_context(rank, world_size)
        reader = ParquetDatasetReader(
            self.config.storage.path_for(split),
            checksums=self.config.storage.verify_checksums,
        )
        if reader.sample_type != self.task.sample_type:
            reader.close()
            raise TaiChuConfigurationError(
                f"task {self.config.task_type!r} expects {self.task.sample_type!r} "
                f"but dataset contains {reader.sample_type!r}"
            )
        self._validate_dataset_identity(reader, split)
        dataset = IndexedTokenDataset(reader)
        strategy = self.config.sampling.strategy
        if strategy == "pair_aware":
            if self.task.sample_type != "preference":
                reader.close()
                raise TaiChuConfigurationError(
                    "pair_aware sampling is valid only for preference datasets"
                )
            sampler = DistributedSampler(
                len(dataset),
                rank=rank,
                world_size=world_size,
                seed=self.config.sampling.seed,
                shuffle=self.config.sampling.shuffle,
                drop_last=self.config.sampling.drop_last,
                pad_to_world_size=self.config.sampling.pad_to_world_size,
                pair_aware=True,
            )
        elif strategy == "distributed" or world_size > 1:
            sampler = DistributedSampler(
                len(dataset),
                rank=rank,
                world_size=world_size,
                seed=self.config.sampling.seed,
                shuffle=self.config.sampling.shuffle,
                drop_last=self.config.sampling.drop_last,
                pad_to_world_size=self.config.sampling.pad_to_world_size,
            )
        elif strategy == "random":
            sampler = RandomSampler(
                len(dataset),
                seed=self.config.sampling.seed,
                shuffle=self.config.sampling.shuffle,
                drop_last=self.config.sampling.drop_last,
            )
        else:
            reader.close()
            raise TaiChuConfigurationError(
                f"unknown sampling strategy {strategy!r}"
            )

        workers = (
            self.config.storage.num_workers
            if num_workers is None
            else int(num_workers)
        )
        if workers < 0:
            reader.close()
            raise TaiChuConfigurationError("num_workers cannot be negative")
        builder = self.task.batch_builder_factory(
            self.config,
            self.pad_token_id,
        )
        packer = (
            self.task.packer_factory(self.config, self.pad_token_id)
            if self.config.packing.enabled
            else None
        )
        collator = TaskCollator(packer, builder)
        kwargs: dict[str, Any] = {}
        if workers:
            kwargs["prefetch_factor"] = self.config.storage.prefetch_factor
        return UnifiedDataLoader(
            dataset=dataset,
            batch_size=int(batch_size),
            sampler=sampler,
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=bool(persistent_workers and workers > 0),
            drop_last=(
                self.config.sampling.drop_last
                if drop_last is None
                else bool(drop_last)
            ),
            collate_fn=collator,
            **kwargs,
        )

    def input_adapter(self):
        return self.task.input_adapter_factory()


def build_sft_data_loader(
    config: UnifiedDataConfig,
    *,
    pad_token_id: int,
    batch_size: int,
    split: str = "train",
    rank: int | None = None,
    world_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    drop_last: bool | None = None,
) -> UnifiedDataLoader:
    """Build an SFT loader through the unified runtime pipeline."""
    if config.task_type not in {"sft", "instruction_tuning"}:
        raise TaiChuConfigurationError(
            "SFT loader requires task_type='sft' or 'instruction_tuning'"
        )
    return UnifiedDataLoaderFactory(
        config,
        pad_token_id=pad_token_id,
    ).create(
        split=split,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )


def build_pretrain_data_loader(
    config: UnifiedDataConfig,
    *,
    pad_token_id: int,
    batch_size: int,
    split: str = "train",
    rank: int | None = None,
    world_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    drop_last: bool | None = None,
) -> UnifiedDataLoader:
    """Build a pretraining loader over the Parquet dataset backend."""
    if config.task_type not in {"pretraining", "cpt"}:
        raise TaiChuConfigurationError(
            "pretraining loader requires task_type='pretraining' or 'cpt'"
        )
    return UnifiedDataLoaderFactory(
        config,
        pad_token_id=pad_token_id,
    ).create(
        split=split,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )


__all__ = [
    "UnifiedDataLoader",
    "UnifiedDataLoaderFactory",
    "build_pretrain_data_loader",
    "build_sft_data_loader",
]
