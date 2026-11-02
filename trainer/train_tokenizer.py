"""CLI entry point for TaiChu Byte-Level BPE Tokenizer training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TaiChu Byte-Level BPE Tokenizer training"
    )
    parser.add_argument(
        "--config",
        default="configs/TaiChu_Tokenizer_config.yaml",
        help="YAML 配置文件路径",
    )
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--min_frequency", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--data_files", nargs="+")
    parser.add_argument("--output_dir")
    parser.add_argument(
        "--workers",
        type=int,
        help="兼容旧参数；仅推荐 1，训练并行由 Tokenizers 原生实现",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="限制 Tokenizers 原生线程数为 1，降低峰值内存",
    )
    parser.add_argument("--report", help="将阶段内存报告写入 JSON 文件")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_raw_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("配置文件顶层必须是映射")
    return config


def apply_overrides(
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if args.vocab_size is not None:
        config.setdefault("trainer", {})["vocab_size"] = args.vocab_size
    if args.min_frequency is not None:
        config.setdefault("trainer", {})["min_frequency"] = args.min_frequency
    if args.epoch is not None:
        config.setdefault("data", {})["epoch"] = args.epoch
    if args.data_files:
        config.setdefault("data", {})["files"] = [
            str(Path(path).expanduser().resolve()) for path in args.data_files
        ]
    if args.workers is not None:
        config.setdefault("data", {})["num_workers"] = args.workers
    if args.output_dir:
        config.setdefault("save", {})["directory"] = str(
            Path(args.output_dir).expanduser().resolve()
        )
    return config


def configure_runtime(low_memory: bool) -> None:
    if not low_memory:
        return
    # These variables must be set before tokenizers' native library is imported.
    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config_path = Path(args.config).expanduser().resolve()
    try:
        raw_config = load_raw_config(config_path)
        raw_config = apply_overrides(raw_config, args)
        configure_runtime(args.low_memory)

        # Imported only after low-memory environment variables are applied.
        from tokenizer.config import parse_config
        from tokenizer.tokenizer import TaiChuTokenizerTrainer

        config = parse_config(raw_config, config_path.parent)
        logging.getLogger(__name__).info(
            "启动 Tokenizer 训练: files=%d vocab_size=%d num_workers=%d low_memory=%s",
            len(config.data.files),
            config.trainer.vocab_size,
            config.data.num_workers,
            args.low_memory,
        )
        trainer = TaiChuTokenizerTrainer(config)
        report = trainer.run()
        if args.report:
            report_path = Path(args.report).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0
    except Exception as exc:
        logging.getLogger(__name__).error("Tokenizer 训练失败: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
