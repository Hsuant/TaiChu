"""通用工具函数。

包含随机种子设置、设备获取、时间格式化、参数量统计等。
"""

import os
from datetime import datetime

import math
import random
import numpy as np
import torch

from typing import Callable


def set_seed(seed: int) -> None:
    """设置所有相关库的随机种子，确保可复现性。

    Args:
        seed: 随机种子整数。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(local_rank: int = -1) -> torch.device:
    """获取当前进程使用的设备。

    支持单卡、多卡（DDP）场景。

    Args:
        local_rank: 由 torchrun 设置的局部 rank，-1 表示单卡。

    Returns:
        torch.device 对象。
    """
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f'cuda:{local_rank}')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_distributed_info() -> tuple:
    """获取分布式训练的 rank 和 world_size，单卡时返回 (0, 1)。

    Returns:
        (rank, world_size): 当前进程的全局 rank 和总进程数。
    """
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    elif "LOCAL_RANK" in os.environ or "RANK" in os.environ:
        # 未初始化但环境变量存在（通常由 torchrun 设置）
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        # 不在此处 init_process_group，留给调用方
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def is_main_process(local_rank: int = -1) -> bool:
    """判断当前进程是否为主进程（用于分布式训练时的日志/保存控制）。

    Args:
        local_rank: 由 torchrun 设置的局部 rank，-1 表示单卡。

    Returns:
        True 表示主进程，否则 False。
    """
    return local_rank <= 0


def format_time(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS 字符串。

    Args:
        seconds: 时间秒数。

    Returns:
        格式化后的时间字符串。
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """统计模型的参数量。

    Args:
        model: PyTorch 模型实例。
        trainable_only: 是否仅统计可训练参数。

    Returns:
        参数量（整型）。
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def get_cosine_lr_lambda(
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1
) -> Callable[[int], float]:
    """生成 cosine 衰减 + warmup 的学习率 lambda 函数。

    Args:
        warmup_steps: 预热步数。
        max_steps: 总训练步数。
        min_lr_ratio: 最终学习率相对于初始学习率的比例。

    Returns:
        一个可调用对象，输入当前步数（int），返回缩放因子（float）。
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda

def get_wsd_lr_lambda(
    warmup_steps: int,
    stable_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.0,
    decay_type: str = "cosine",
) -> Callable[[int], float]:
    """生成 WSD（预热-恒定-衰减）学习率缩放因子 lambda 函数。

    WSD 调度将训练分为三个阶段：
    1. 预热阶段：学习率从 0 线性增加到 1.0（相对于峰值学习率）。
    2. 恒定阶段：学习率保持为 1.0，给予模型充分的探索空间。
    3. 衰减阶段：学习率从 1.0 平滑衰减至 min_lr_ratio。

    这种设计已被 DeepSeek-V3 等前沿模型验证，能够在稳定阶段
    维持高学习率跨越鞍点，并在衰减阶段通过极限退火显著降低最终损失。

    Args:
        warmup_steps: 预热步数，学习率从 0 线性增加至 1.0 所需的步数。
        stable_steps: 恒定阶段步数，此期间学习率保持 1.0。
        max_steps: 总训练步数，用于确定衰减阶段的长度。
        min_lr_ratio: 最终学习率相对于峰值学习率的比例，通常设为 0.0 以实现
            完全退火，也可以设为一个小值（如 0.05）以防止学习率绝对为零。
        decay_type: 衰减阶段的衰减方式，目前支持 "cosine"（余弦衰减）与
            "linear"（线性衰减）。默认为 "cosine"。

    Returns:
        一个可调用对象，接收当前步数（int），返回学习率缩放因子（float）。
        缩放因子乘以优化器中的峰值学习率即可得到当前实际学习率。

    Raises:
        ValueError: 如果 decay_type 不是支持的类型。
    """
    # 确保分母不为零，避免除零错误
    warmup_steps = max(0, warmup_steps)
    stable_steps = max(0, stable_steps)
    # 衰减阶段步数：总步数减去预热和恒定阶段步数，至少为 0
    decay_steps = max(0, max_steps - warmup_steps - stable_steps)

    # 预热阶段结束步数（不含）
    warmup_end = warmup_steps
    # 恒定阶段结束步数（不含）
    stable_end = warmup_steps + stable_steps

    # 根据衰减类型选择衰减计算函数
    if decay_type == "cosine":
        def decay_fn(progress: float) -> float:
            """余弦衰减函数，平滑下降。"""
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    elif decay_type == "linear":
        def decay_fn(progress: float) -> float:
            """线性衰减函数，均匀下降。"""
            return 1.0 - progress
    else:
        raise ValueError(f"不支持的衰减类型: {decay_type}，可选 'cosine' 或 'linear'")

    def lr_lambda(current_step: int) -> float:
        """根据当前步数计算学习率缩放因子。

        Args:
            current_step: 当前训练步数，从 0 开始计数。

        Returns:
            学习率缩放因子，范围在 [min_lr_ratio, 1.0] 之间。
        """
        # 转换为浮点数，避免整数运算误差
        step = float(current_step)

        # 阶段 1：线性预热
        if step < warmup_end:
            # 如果预热步数为 0，此分支不会进入，直接跳到恒定阶段
            return step / max(1.0, float(warmup_steps))

        # 阶段 2：恒定学习率
        if step < stable_end:
            return 1.0

        # 阶段 3：学习率衰减
        # 如果当前步数已经超过总步数，则返回最小学习率（确保不会低于最小值）
        if step >= max_steps:
            return min_lr_ratio

        # 衰减进度：0 表示衰减刚开始，1 表示衰减结束
        progress = (step - float(stable_end)) / max(1.0, float(decay_steps))
        # 保证 progress 在 [0, 1] 范围内
        progress = min(max(progress, 0.0), 1.0)

        # 获取衰减系数（范围 0~1）
        decay_coef = decay_fn(progress)
        # 缩放到 [min_lr_ratio, 1.0] 区间
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay_coef

    return lr_lambda

def get_experiment_dir(base_dir: str, experiment_name: str = "", model_name: str = "") -> str:
    """生成唯一的实验目录，若已存在则自动添加后缀 _1, _2...

    Args:
        base_dir: 基础根目录（如 "./experiments"）
        experiment_name: 用户指定的实验名，为空时自动生成
        model_name: 模型名称，用于自动生成默认实验名

    Returns:
        唯一的实验目录绝对路径
    """
    if not experiment_name:
        # 默认名称：模型名 + 时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{model_name}_{timestamp}" if model_name else f"TaiChu_{timestamp}"

    exp_dir = os.path.join(base_dir, experiment_name)
    if not os.path.exists(exp_dir):
        return exp_dir

    # 目录已存在，尝试添加后缀
    counter = 1
    while True:
        new_name = f"{experiment_name}_{counter}"
        new_dir = os.path.join(base_dir, new_name)
        if not os.path.exists(new_dir):
            return new_dir
        counter += 1