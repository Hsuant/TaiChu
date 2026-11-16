"""SFT 通用离线构建命令行入口。

该入口位于 dataset 包内部，不依赖 scripts 目录。配置文件同时定义 source
过滤/配额和 unified Binary Dataset 的 tokenization 参数。
"""
from __future__ import annotations

import argparse
import json

from .config import load_sft_build_config
from .pipeline import SFTDataPreparation


def main() -> None:
    """加载 YAML 配置、执行构建并输出可审计的 JSON 报告。"""
    parser = argparse.ArgumentParser(description="Build a unified SFT dataset")
    # --restart 只清理不完整的 staging，不会覆盖已完成的正式输出。
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--profile",
        default=None,
        help="named model-size data profile from the config",
    )
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    report = SFTDataPreparation(
        load_sft_build_config(args.config, profile=args.profile)
    ).run(restart=args.restart)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
