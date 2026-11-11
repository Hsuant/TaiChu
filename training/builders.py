"""Construction of model, tokenizer, and unified data loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

from dataset.config import UnifiedDataConfig
from dataset.runtime import UnifiedDataLoader, UnifiedDataLoaderFactory
from model.config import ModelConfig
from model.model import TaiChuModel
from training.config import DataLoaderRuntimeConfig, EvaluatingConfig
from training.environment import RuntimeContext


@dataclass(slots=True)
class TokenizerBundle:
    tokenizer: Tokenizer
    pad_token_id: int
    bos_token_id: int | None
    eos_token_id: int | None


@dataclass(slots=True)
class DataLoaderBundle:
    train: UnifiedDataLoader
    validation: UnifiedDataLoader | None

    def close(self) -> None:
        self.train.close()
        if self.validation is not None:
            self.validation.close()


def build_model(config: ModelConfig, device: torch.device) -> TaiChuModel:
    model = TaiChuModel(config)
    return model.to(device)


def load_tokenizer(
    path: str | Path,
    *,
    data_config: UnifiedDataConfig,
    model_vocab_size: int,
) -> TokenizerBundle:
    tokenizer = Tokenizer.from_file(str(path))
    vocabulary_size = tokenizer.get_vocab_size()
    if vocabulary_size > model_vocab_size:
        raise ValueError(
            f"tokenizer vocabulary size {vocabulary_size} exceeds "
            f"model vocab_size {model_vocab_size}"
        )
    pad_id = tokenizer.token_to_id(data_config.pad_token)
    if pad_id is None:
        raise ValueError(
            f"configured pad token {data_config.pad_token!r} is missing"
        )
    bos_id = tokenizer.token_to_id(data_config.bos_token)
    eos_id = tokenizer.token_to_id(data_config.eos_token)
    if bos_id is None:
        raise ValueError(
            f"configured BOS token {data_config.bos_token!r} is missing"
        )
    if eos_id is None:
        raise ValueError(
            f"configured EOS token {data_config.eos_token!r} is missing"
        )
    return TokenizerBundle(
        tokenizer=tokenizer,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
    )


def _create_loader(
    factory: UnifiedDataLoaderFactory,
    *,
    split: str,
    batch_size: int,
    runtime: RuntimeContext,
    device: torch.device,
    options: DataLoaderRuntimeConfig,
) -> UnifiedDataLoader:
    return factory.create(
        split=split,
        batch_size=batch_size,
        rank=runtime.rank,
        world_size=runtime.world_size,
        pin_memory=bool(options.pin_memory and device.type == "cuda"),
        drop_last=options.drop_last,
        persistent_workers=options.persistent_workers,
    )


def build_data_loaders(
    data_config: UnifiedDataConfig,
    evaluating: EvaluatingConfig,
    *,
    training_dataloader: DataLoaderRuntimeConfig,
    runtime: RuntimeContext,
    device: torch.device,
    pad_token_id: int,
) -> DataLoaderBundle:
    factory = UnifiedDataLoaderFactory(
        data_config,
        pad_token_id=pad_token_id,
    )
    train = _create_loader(
        factory,
        split="train",
        batch_size=data_config.batching.micro_batch_size,
        runtime=runtime,
        device=device,
        options=training_dataloader,
    )
    validation = None
    if data_config.storage.validation_path is not None:
        try:
            validation = _create_loader(
                factory,
                split="validation",
                batch_size=evaluating.batch_size,
                runtime=runtime,
                device=device,
                options=evaluating.dataloader,
            )
        except Exception:
            train.close()
            raise
    return DataLoaderBundle(train=train, validation=validation)


def build_validation_loader(
    data_config: UnifiedDataConfig,
    evaluating: EvaluatingConfig,
    *,
    device: torch.device,
    pad_token_id: int,
) -> UnifiedDataLoader:
    runtime = RuntimeContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=device,
        backend=None,
    )
    factory = UnifiedDataLoaderFactory(
        data_config,
        pad_token_id=pad_token_id,
    )
    return _create_loader(
        factory,
        split="validation",
        batch_size=evaluating.batch_size,
        runtime=runtime,
        device=device,
        options=evaluating.dataloader,
    )


__all__ = [
    "DataLoaderBundle",
    "TokenizerBundle",
    "build_data_loaders",
    "build_model",
    "build_validation_loader",
    "load_tokenizer",
]
