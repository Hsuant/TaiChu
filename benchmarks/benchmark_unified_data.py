#!/usr/bin/env python3
"""Microbenchmark the unified binary storage and DataLoader path."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer

from dataset.config import UnifiedDataConfig
from dataset.metrics import ThroughputMetrics
from dataset.runtime import UnifiedDataLoaderFactory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()
    if args.batches <= 0:
        raise ValueError("--batches must be positive")

    config = UnifiedDataConfig.from_yaml(args.config)
    tokenizer_path = Path(
        config.metadata.get(
            "tokenizer_path",
            "./taichu_tokenizer/tokenizer.json",
        )
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_token_id = tokenizer.token_to_id(config.pad_token)
    if pad_token_id is None:
        raise ValueError(f"unknown pad token: {config.pad_token!r}")

    loader = UnifiedDataLoaderFactory(
        config,
        pad_token_id=pad_token_id,
    ).create(
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    metrics = ThroughputMetrics()
    try:
        for index, batch in enumerate(loader):
            input_ids = batch["input_ids"]
            metrics.record(
                batches=1,
                samples=int(input_ids.size(0)),
                tokens=int(input_ids.numel()),
            )
            if index + 1 >= args.batches:
                break
    finally:
        loader.close()
    print(json.dumps(metrics.snapshot(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
