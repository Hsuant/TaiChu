"""
程序入口，用于读取配置文件并启动BPE Tokenizer的训练。
核心训练器模块。用于配置并创建Hugging Face Tokenizers库的BPE Trainer。
"""

import sys
import yaml
import argparse
from tokenizer.tokenizer import TaiChuTokenizerTrainer


def main() -> None:
    """主函数，解析参数、加载配置并启动训练。"""
    # ========== 1. 解析命令行参数 ==========
    parser = argparse.ArgumentParser(
        description="TaiChu BPE Tokenizer 训练脚本，支持动态覆盖配置参数"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/tokenizer_config.yaml",
        help="YAML 配置文件的路径（默认: ./configs/tokenizer_config.yaml）"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="词表大小（例如 64000）"
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        help="最小词频阈值（例如 2 或 3）"
    )
    parser.add_argument(
        "--limit_alphabet",
        type=int,
        help="初始字母表大小（中文场景建议 5000）"
    )
    parser.add_argument(
        "--data_files",
        type=str,
        nargs="+",
        help="训练数据文件列表，支持多个文件，用空格分隔"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        help="训练迭代次数（通常设为 1）"
    )

    # 解析命令行，若遇到未知参数则忽略（避免与其它脚本混用时出错）
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"警告：忽略未知命令行参数: {unknown}", file=sys.stderr)

    config_path = args.config

    # ========== 2. 加载 YAML 配置 ==========
    print(f"--- TaiChu Tokenizer 训练启动 ---")
    print(f"加载配置文件: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误：配置文件不存在 - {config_path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"错误：解析 YAML 文件失败 - {e}", file=sys.stderr)
        sys.exit(1)

    # ========== 3. 用命令行参数覆盖配置 ==========
    if args.vocab_size is not None:
        config['trainer']['vocab_size'] = args.vocab_size

    if args.min_frequency is not None:
        config['trainer']['min_frequency'] = args.min_frequency

    if args.limit_alphabet is not None:
        config['trainer']['limit_alphabet'] = args.limit_alphabet

    if args.data_files is not None:
        config['data']['files'] = args.data_files

    if args.epoch is not None:
        config['data']['epoch'] = args.epoch

    # ========== 4. 初始化训练器并执行 ==========
    try:
        trainer = TaiChuTokenizerTrainer(config)
        trainer.run()
    except Exception as e:
        print(f"训练过程中发生错误: {e}", file=sys.stderr)
        sys.exit(1)

    print("--- TaiChu Tokenizer 训练完成 ---")


if __name__ == "__main__":
    main()