"""Validate an existing Tokenizer directory against the training protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/TaiChu_Tokenizer_config.yaml",
    )
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--report")
    args = parser.parse_args()

    from tokenizer.config import load_config
    from tokenizer.resource_monitor import ResourceMonitor
    from tokenizer.validator import validate_tokenizer

    config_path = Path(args.config).resolve()
    config = load_config(config_path, require_data_files=False)
    monitor = ResourceMonitor(interval_seconds=0.02)
    monitor.start()
    started = time.perf_counter()
    try:
        report = validate_tokenizer(
            Path(args.tokenizer_dir).resolve(),
            config.trainer.special_tokens.ordered_tokens,
            args.vocab_size or config.trainer.vocab_size,
        )
    finally:
        elapsed = time.perf_counter() - started
        monitor.stop()
    output = {
        "validation": report,
        "elapsed_seconds": elapsed,
        "resources": monitor.report(),
    }
    if args.report:
        report_path = Path(args.report).resolve()
        report_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(output, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
