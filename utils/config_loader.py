"""统一配置加载模块。

包含训练、数据、优化器相关的配置数据类，
以及从 YAML 文件加载配置的工厂函数。

模型结构配置统一由 config.ModelConfig 负责，
本模块的 load_model_config 直接加载并返回该对象。
"""

import yaml
from dataclasses import dataclass, field
from typing import List, Optional

# 导入统一的模型配置类
from model.config import ModelConfig


# ==================== 训练流程配置 ====================
@dataclass
class OptimizerConfig:
    """优化器参数。"""
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8


@dataclass
class SchedulerConfig:
    """学习率调度参数。"""
    warmup_steps: int = 1000
    max_steps: int = 100000
    min_lr_ratio: float = 0.1


@dataclass
class DataConfig:
    """训练/验证数据配置。"""
    train_files: List[str] = field(default_factory=list)
    val_files: List[str] = field(default_factory=list)
    tokenizer_path: str = "./taichu_tokenizer/tokenizer.json"
    text_field: str = "text"
    max_seq_length: int = 2048
    from_pre_tokenized: bool = False
    pre_tokenized_dir: str = "./data/tokenized"
    num_workers: int = 2


@dataclass
class TrainingConfig:
    """训练循环配置。"""
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 100000
    save_interval: int = 5000
    log_interval: int = 10
    output_dir: str = "./checkpoints"
    use_mixed_precision: bool = True
    dtype: str = "bfloat16"
    seed: int = 42
    local_rank: int = -1  # 由环境变量自动填充

@dataclass
class EvaluatingConfig:
    """验证与生成测试配置。"""
    batch_size: int = 10
    eval_interval: int = 1000
    log_interval: int = 10
    prompts: Optional[List[str]] = None
    num_generate_tokens: int = 50

@dataclass
class EarlyStoppingConfig:
    """早停配置。"""
    enabled: bool = False
    monitor: str = "val_loss"
    patience: int = 5
    min_delta: float = 1e-4
    mode: str = "min"               # 优化方向，'min' 或 'max'

@dataclass
class SwanLabLoggingConfig:
    """SwanLab 实验跟踪配置"""
    use_swanlab: bool = True
    swanlab_project: str = "TaiChu-Project"
    swanlab_experiment_name: str = "TaiChu_Experiment_Name"
    swanlab_log_dir: str = "./swanlogs"
    swanlab_mode: str = "cloud"  # 可选 "cloud" 或 "local"

@dataclass
class PretrainConfig:
    """完整的预训练配置，聚合训练、数据、优化器和调度器。"""
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluating: EvaluatingConfig = field(default_factory=EvaluatingConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    swanlab: SwanLabLoggingConfig = field(default_factory=SwanLabLoggingConfig)


# ==================== 加载函数 ====================
def load_model_config(yaml_path: str) -> ModelConfig:
    """从 YAML 文件加载模型结构配置（扁平结构 + 可选 moe 子块）。

    Args:
        yaml_path: 模型结构 YAML 文件路径，格式必须符合 ModelConfig 的字段要求。

    Returns:
        ModelConfig 实例。
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    return ModelConfig.from_dict(raw)


def load_pretrain_config(yaml_path: str) -> PretrainConfig:
    """从 YAML 文件加载预训练流程配置。

    Args:
        yaml_path: 预训练配置 YAML 文件路径。

    Returns:
        PretrainConfig 实例。
    """
    import os
    with open(yaml_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    optim = OptimizerConfig(**raw.get('optimizer', {}))
    sched = SchedulerConfig(**raw.get('scheduler', {}))
    data = DataConfig(**raw.get('data', {}))
    train = TrainingConfig(**raw.get('training', {}))
    train.local_rank = int(os.environ.get('LOCAL_RANK', -1))
    eval_cfg = EvaluatingConfig(**raw.get('evaluating', {}))
    early_stop_cfg = EarlyStoppingConfig(**raw.get('early_stopping', {}))
    swanlab_cfg = SwanLabLoggingConfig(**raw.get('swanlab', {}))
    return PretrainConfig(
        optimizer=optim,
        scheduler=sched,
        data=data,
        training=train,
        evaluating=eval_cfg,
        early_stopping=early_stop_cfg,
        swanlab=swanlab_cfg,
    )
