"""
程序入口，用于读取配置文件并启动BPE Tokenizer的训练。
核心训练器模块。用于配置并创建Hugging Face Tokenizers库的BPE Trainer。
"""

import yaml
import argparse
from tokenizer.tokenizer import TaiChuTokenizerTrainer


def main(config_path):
    """
    主函数，加载配置并执行训练。

    Args:
        config_path (str): YAML配置文件的路径。
    """
    # 1. 加载配置
    print(f"--- TaiChu Tokenizer 训练启动 ---")
    print(f"加载配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. 初始化训练器
    trainer = TaiChuTokenizerTrainer(config)

    # 3. 执行训练流程
    trainer.run()
    print("--- TaiChu Tokenizer 训练完成 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TaiChu BPE Tokenizer 训练脚本")
    parser.add_argument("--config", type=str, default="./configs/tokenizer_config.yaml",
                        help="YAML配置文件的路径")
    args = parser.parse_args()
    main(args.config)