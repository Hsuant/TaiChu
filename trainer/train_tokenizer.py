"""
TaiChu Tokenizer 训练入口脚本。

读取 YAML 配置文件，支持通过命令行参数动态覆盖部分关键配置，
然后调用训练主控器完成 Byte-Level BPE 分词器的训练与保存。

使用方法：
    python -m trainer.train_tokenizer --config ./configs/tokenizer_config.yaml
    python -m trainer.train_tokenizer --vocab_size 64000 --data_files data/*.jsonl

Author: TaiChu Team
Version: 2.0.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
from tokenizer.tokenizer import TaiChuTokenizerTrainer


def main() -> None:
    """解析命令行参数、加载配置文件、启动训练流程。

    命令行参数允许在不修改配置文件的情况下，覆盖以下关键设置：
        - 词表大小 (--vocab_size)
        - 最小词频 (--min_frequency)
        - 训练数据文件列表 (--data_files)
        - 训练轮数 (--epoch)
        - 中文单字拆分开关 (--split_chinese)
        - 数据预取进程数 (--num_workers)
    """
    # ========================================================================
    # 1. 定义命令行参数
    # ========================================================================
    parser = argparse.ArgumentParser(
        description="TaiChu BPE Tokenizer 训练脚本，支持动态覆盖配置参数。"
    )

    # ---------- 配置文件路径 ----------
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/tokenizer_config.yaml",
        help="YAML 配置文件的路径（默认: ./configs/tokenizer_config.yaml）"
    )

    # ---------- 训练核心参数 ----------
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="词表大小，例如 64000"
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        help="最小词频阈值，低于此频次的 token 将不进入合并候选，例如 2 或 3"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        help="数据集遍历次数，大数据量时通常设置为 1"
    )

    # ---------- 数据相关参数 ----------
    parser.add_argument(
        "--data_files",
        type=str,
        nargs="+",
        help="训练数据文件路径列表，支持多个文件，用空格分隔"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="数据预取使用的进程数，0 表示自动选择 CPU 核心数"
    )

    # 解析已知参数，忽略其余未知参数（避免与其他脚本混用时出错）
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"警告：忽略未知命令行参数: {unknown}", file=sys.stderr)

    # ========================================================================
    # 2. 加载 YAML 配置文件
    # ========================================================================
    config_path = args.config
    print("--- TaiChu Tokenizer 训练启动 ---")
    print(f"加载配置文件: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误：配置文件不存在 - {config_path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"错误：解析 YAML 文件失败 - {e}", file=sys.stderr)
        sys.exit(1)

    # ========================================================================
    # 3. 使用命令行参数覆盖配置文件中的对应字段
    # ========================================================================
    if args.vocab_size is not None:
        config.setdefault("trainer", {})["vocab_size"] = args.vocab_size
        print(f"[覆盖] vocab_size = {args.vocab_size}")

    if args.min_frequency is not None:
        config.setdefault("trainer", {})["min_frequency"] = args.min_frequency
        print(f"[覆盖] min_frequency = {args.min_frequency}")

    if args.data_files is not None:
        config.setdefault("data", {})["files"] = args.data_files
        print(f"[覆盖] data_files = {args.data_files}")

    if args.epoch is not None:
        config.setdefault("data", {})["epoch"] = args.epoch
        print(f"[覆盖] epoch = {args.epoch}")

    if args.num_workers is not None:
        config.setdefault("data", {})["num_workers"] = args.num_workers
        print(f"[覆盖] num_workers = {args.num_workers}")

    # ========================================================================
    # 4. 初始化训练主控器并执行训练
    # ========================================================================
    try:
        trainer = TaiChuTokenizerTrainer(config)
        trainer.run()
    except Exception as e:
        print(f"训练过程中发生错误: {e}", file=sys.stderr)
        sys.exit(1)

    print("--- TaiChu Tokenizer 训练完成 ---")


if __name__ == "__main__":
    main()