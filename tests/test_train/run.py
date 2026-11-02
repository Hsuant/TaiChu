#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键运行模拟预训练（CPU 可跑通，约 1 分钟）。

用法：
    cd D:\\Objects\\TaiChu
    python tests/test_train/run.py

或直接从任意目录运行，脚本会自动切换到项目根目录。
"""
import os
import sys
from pathlib import Path

# 切换到项目根目录（taichu_tokenizer、dataset、model 等均相对项目根）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

from dataset.offline.config import load_preparation_config
from dataset.offline.prepare import UnifiedDataPreparation

data_dir = Path(_PROJECT_ROOT) / "tests" / "test_train" / "unified"
if not data_dir.exists():
    config = load_preparation_config(
        Path(_PROJECT_ROOT) / "tests" / "test_train" / "prepare_sim.yaml"
    )
    UnifiedDataPreparation(config).run()

sys.argv = [
    "train_pretrain",
    "--model_config", "tests/test_train/model_sim.yaml",
    "--pretrain_config", "tests/test_train/pretrain_sim.yaml",
]

import runpy
runpy.run_module("trainer.train_pretrain", run_name="__main__")
