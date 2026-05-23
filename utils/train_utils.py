"""通用工具函数。

包含随机种子设置、设备获取、时间格式化、参数量统计等。
"""

import os
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