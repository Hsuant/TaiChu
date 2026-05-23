"""
预训练启动脚本。

支持单卡训练和分布式训练（通过 torchrun 启动）。
从 YAML 配置文件读取所有参数。
"""

import argparse
from model.model import TaiChuModel
from utils.config_loader import load_model_config, load_pretrain_config
from trainer.train_pretrain import Trainer
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="XuanYan 模型预训练")
    parser.add_argument("--model_config", type=str, required=True,
                        help="模型结构配置文件路径 (YAML)")
    parser.add_argument("--pretrain_config", type=str, required=True,
                        help="预训练参数配置文件路径 (YAML)")
    args = parser.parse_args()

    # 加载模型结构配置
    model_cfg = load_model_config(args.model_config)
    # 加载预训练流程配置
    pretrain_cfg = load_pretrain_config(args.pretrain_config)

    # 初始化分词器
    tokenizer = Tokenizer.from_file(pretrain_cfg.data.tokenizer_path)

    # 构建模型
    model = TaiChuModel(model_cfg)

    # 创建训练器并启动训练
    trainer = Trainer(pretrain_cfg, model, tokenizer)
    trainer.train()


if __name__ == "__main__":
    main()