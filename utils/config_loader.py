"""统一配置加载模块。

包含训练、数据、优化器相关的配置数据类，
以及从 YAML 文件加载配置的工厂函数。

模型结构配置统一由 config.ModelConfig 负责，
本模块的 load_model_config 直接加载并返回该对象。
"""

import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 导入统一的模型配置类与数值字符串还原工具
from model.config import ModelConfig, _coerce_numeric_strings
from dataset.config import StorageConfig, UnifiedDataConfig


# ==================== 训练流程配置 ====================
@dataclass
class OptimizerConfig:
    """优化器参数。"""
    name: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8


@dataclass
class SchedulerConfig:
    """学习率调度参数。"""
    name: str = "warmup_stable_decay"
    warmup_ratio: float = 0.01
    stable_ratio: float = 0.8
    max_steps: int = 100000
    min_lr_ratio: float = 0.0
    decay_type: str = "cosine"


@dataclass
class DataLoaderRuntimeConfig:
    """Runtime options forwarded to TaiChuDataLoaderFactory.create()."""

    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool | None = None


@dataclass
class TrainingConfig:
    """训练循环配置。"""
    target_tokens: int = 2000000000
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    save_interval: int = 5000
    log_interval: int = 10
    output_dir: str = "./experiments"
    experiment_name: str = ""
    use_mixed_precision: bool = True
    dtype: str = "bfloat16"
    seed: int = 42
    local_rank: int = -1  # 由环境变量自动填充
    compile_model: bool = True
    compile_mode: str = "auto"
    compile_dynamic: bool = True
    distributed_timeout_minutes: int = 30
    max_grad_norm: float = 1.0
    dataloader: DataLoaderRuntimeConfig = field(
        default_factory=lambda: DataLoaderRuntimeConfig(drop_last=True)
    )

@dataclass
class EvaluatingConfig:
    """验证与生成测试配置。"""
    batch_size: int = 10
    eval_interval: int = 1000
    log_interval: int = 10
    prompts: Optional[List[str]] = None
    num_generate_tokens: int = 50
    empty_prefix_eos_ppl: bool = False
    dataloader: DataLoaderRuntimeConfig = field(
        default_factory=lambda: DataLoaderRuntimeConfig(drop_last=False)
    )

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
    swanlab_workspace: Optional[str] = None
    swanlab_id: Optional[str] = None
    swanlab_resume: Optional[str] = None
    swanlab_init_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PretrainConfig:
    """完整的预训练配置，聚合训练、数据、优化器和调度器。"""
    tokenizer_path: str = "./taichu_tokenizer/tokenizer.json"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: UnifiedDataConfig = field(
        default_factory=lambda: UnifiedDataConfig(
            task_type="pretraining",
            dataset_id="default",
            storage=StorageConfig(
                train_path="./data/unified/pretraining/train"
            ),
        )
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluating: EvaluatingConfig = field(default_factory=EvaluatingConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    swanlab: SwanLabLoggingConfig = field(default_factory=SwanLabLoggingConfig)

# ==================== 解析函数 ====================
def parse_tokens_string(value) -> int:
    """将 "5.2B", "100M", "1.2T", "50K" 等字符串转换为整数 token 数。
    支持 K (10^3), M (10^6), B (10^9), T (10^12)，忽略大小写。
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip().upper()
        if value.endswith('K'):
            return int(float(value[:-1]) * 1e3)
        elif value.endswith('B'):
            return int(float(value[:-1]) * 1e9)
        elif value.endswith('M'):
            return int(float(value[:-1]) * 1e6)
        elif value.endswith('T'):
            return int(float(value[:-1]) * 1e12)
        else:
            # 纯数字字符串
            return int(float(value))
    raise ValueError(f"无法解析 target_tokens: {value}")

# ==================== 加载函数 ====================
def load_model_config(yaml_path: str) -> ModelConfig:
    """从 YAML 文件加载模型结构配置（结构化或扁平结构，可选 moe 子块）。

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

    # PyYAML 会把不带小数点的科学计数法（如 3e-4、1e-8）解析为字符串，
    # 构造各配置前按字段声明类型统一还原为数值
    optim = OptimizerConfig(**_coerce_numeric_strings(OptimizerConfig, raw.get('optimizer', {})))
    sched = SchedulerConfig(**_coerce_numeric_strings(SchedulerConfig, raw.get('scheduler', {})))
    tokenizer = raw.get('tokenizer', {})
    if not isinstance(tokenizer, dict):
        raise ValueError("tokenizer config must be a mapping")
    tokenizer_path = tokenizer.get('path')
    if not isinstance(tokenizer_path, str) or not tokenizer_path:
        raise ValueError("tokenizer.path must be a non-empty string")
    data = UnifiedDataConfig.from_dict(raw.get('data', {}))

    train_raw = raw.get('training', {}).copy()
    if 'target_tokens' in train_raw:
        train_raw['target_tokens'] = parse_tokens_string(train_raw['target_tokens'])
    train_dataloader = _build_dataloader_config(
        train_raw.pop('dataloader', {}),
        default_drop_last=True,
    )

    # local_rank 保留 YAML 配置值（默认 -1 表示自动），
    # 自动模式在训练入口中由环境变量 LOCAL_RANK 动态决定
    train = TrainingConfig(
        **_coerce_numeric_strings(TrainingConfig, train_raw),
        dataloader=train_dataloader,
    )

    eval_raw = raw.get('evaluating', {}).copy()
    eval_dataloader = _build_dataloader_config(
        eval_raw.pop('dataloader', {}),
        default_drop_last=False,
    )
    eval_cfg = EvaluatingConfig(
        **_coerce_numeric_strings(EvaluatingConfig, eval_raw),
        dataloader=eval_dataloader,
    )
    early_stop_cfg = EarlyStoppingConfig(**_coerce_numeric_strings(EarlyStoppingConfig, raw.get('early_stopping', {})))
    swanlab_cfg = SwanLabLoggingConfig(**_coerce_numeric_strings(SwanLabLoggingConfig, raw.get('swanlab', {})))
    if not isinstance(swanlab_cfg.swanlab_init_kwargs, dict):
        raise ValueError("swanlab.swanlab_init_kwargs must be a mapping")

    return PretrainConfig(
        tokenizer_path=tokenizer_path,
        optimizer=optim,
        scheduler=sched,
        data=data,
        training=train,
        evaluating=eval_cfg,
        early_stopping=early_stop_cfg,
        swanlab=swanlab_cfg,
    )


def _build_dataloader_config(raw, *, default_drop_last: bool) -> DataLoaderRuntimeConfig:
    if not isinstance(raw, dict):
        raise ValueError("dataloader config must be a mapping")
    values = _coerce_numeric_strings(DataLoaderRuntimeConfig, raw)
    values.setdefault("drop_last", default_drop_last)
    return DataLoaderRuntimeConfig(**values)
