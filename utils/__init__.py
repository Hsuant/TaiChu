"""TaiChu 工具包。"""

# ==================== 基础工具层 ====================
from utils.train_utils import set_seed, get_device, format_time, count_parameters
from utils.model_utils import ModelInitializer, ModelInspector
from utils.logger import LogManager
from utils.checkpoint import CheckpointManager
from utils.visualization import VisualizationManager

# ==================== 配置层 ====================
from utils.config_loader import (
    OptimizerConfig,
    SchedulerConfig,
    DataConfig,
    TrainingConfig,
    EvaluatingConfig,
    EarlyStoppingConfig,
    SwanLabLoggingConfig,
    PretrainConfig,
    load_model_config,
    load_pretrain_config,
)

# ==================== 训练控制层 ====================
from utils.early_stopping import EarlyStopping

# ==================== 评估指标层 ====================
from utils.metrics import (
    SmoothedLossTracker,
    GradientNoiseScale,
    EfficientGradientNoiseScale,
    ModelFlopsUtilization,
    RepresentationStability,
    ValidationMetrics,
    EvaluationMetricsManager,
)

# ==================== 实验跟踪层 ====================
from utils.swanlab_logger import SwanLabLogger


__all__ = [
    # 基础工具
    "set_seed",
    "get_device",
    "format_time",
    "count_parameters",
    "ModelInitializer",
    "ModelInspector",
    "LogManager",
    "CheckpointManager",
    "VisualizationManager",
    # 配置
    "OptimizerConfig",
    "SchedulerConfig",
    "DataConfig",
    "TrainingConfig",
    "EvaluatingConfig",
    "EarlyStoppingConfig",
    "SwanLabLoggingConfig",
    "PretrainConfig",
    "load_model_config",
    "load_pretrain_config",
    # 训练控制
    "EarlyStopping",
    # 评估指标
    "SmoothedLossTracker",
    "GradientNoiseScale",
    "EfficientGradientNoiseScale",
    "ModelFlopsUtilization",
    "RepresentationStability",
    "ValidationMetrics",
    "EvaluationMetricsManager",
    # 实验跟踪
    "SwanLabLogger",
]