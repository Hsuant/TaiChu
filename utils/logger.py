"""训练日志管理器。

提供统一的日志记录接口，集成 TensorBoard、控制台输出与文本文件记录。
"""

import logging
import os
import sys
from typing import Any, Dict, Optional, Union

import torch
from torch.utils.tensorboard import SummaryWriter


class LogManager:
    """训练日志管理器。

    封装 TensorBoard SummaryWriter 和 Python 标准 logging 模块，
    支持标量、直方图、文本、超参数、模型图等记录的便捷调用。

    Attributes:
        log_dir: 日志根目录。
        writer: TensorBoard SummaryWriter 实例。
        logger: Python 标准日志记录器。
        step: 当前全局步数（用于 TensorBoard 横轴）。
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        tensorboard: bool = True,
        log_file: Optional[str] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        logger_name: str = "TaiChu",
    ):
        """初始化日志管理器。

        Args:
            log_dir: 日志根目录。
            tensorboard: 是否启用 TensorBoard。
            log_file: 文本日志文件名（不含路径），为 None 则不保存。
            console_level: 控制台日志输出级别。
            file_level: 文件日志输出级别。
            logger_name: Python 日志记录器名称。
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # TensorBoard
        if tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
        else:
            self.writer = None

        # Python logging
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # 控制台 handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(console_level)
        console_fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console.setFormatter(console_fmt)
        self.logger.addHandler(console)

        # 文件 handler
        self.file_handler = None
        if log_file is not None:
            file_path = os.path.join(log_dir, log_file)
            self.file_handler = logging.FileHandler(file_path, encoding="utf-8")
            self.file_handler.setLevel(file_level)
            file_fmt = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            self.file_handler.setFormatter(file_fmt)
            self.logger.addHandler(self.file_handler)

        self.step = 0

    def set_step(self, step: int) -> None:
        """设置全局步数，TensorBoard 记录将使用此值作为横坐标。

        Args:
            step: 当前训练步数。
        """
        self.step = step

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        """记录标量到 TensorBoard 并输出 INFO 日志。

        Args:
            tag: 标签。
            value: 标量值。
            step: 步数，默认使用 self.step。
        """
        if step is None:
            step = self.step
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
        self.logger.info(f"Step {step} - {tag}: {value:.6f}")

    def log_scalars(
        self, main_tag: str, tag_scalar_dict: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """记录多个标量（同一图表多条曲线）。

        Args:
            main_tag: 主标签。
            tag_scalar_dict: 子标签到值的映射。
            step: 步数。
        """
        if step is None:
            step = self.step
        if self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        items = ", ".join(f"{k}={v:.4f}" for k, v in tag_scalar_dict.items())
        self.logger.info(f"Step {step} - {main_tag}: {items}")

    def log_histogram(
        self, tag: str, values: Union[torch.Tensor, list, tuple], step: Optional[int] = None
    ) -> None:
        """记录直方图。

        Args:
            tag: 标签。
            values: 数值张量或可迭代对象。
            step: 步数。
        """
        if step is None:
            step = self.step
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """记录文本。

        Args:
            tag: 标签。
            text: 文本内容。
            step: 步数。
        """
        if step is None:
            step = self.step
        if self.writer is not None:
            self.writer.add_text(tag, text, step)
        self.logger.debug(f"Text log ({tag}): {text}")

    def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """记录超参数及对应指标。

        Args:
            hparams: 超参数字典。
            metrics: 评估指标字典。
        """
        if self.writer is not None:
            self.writer.add_hparams(hparams, metrics)

    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> None:
        """记录模型计算图。

        Args:
            model: PyTorch 模型。
            input_tensor: 样例输入张量。
        """
        if self.writer is not None:
            self.writer.add_graph(model, input_tensor)

    def info(self, msg: str, **kwargs) -> None:
        """输出 INFO 日志。"""
        self.logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        """输出 WARNING 日志。"""
        self.logger.warning(msg, **kwargs)

    def error(self, msg: str, exc_info: bool = False, **kwargs) -> None:
        """输出 ERROR 日志。

        Args:
            msg: 日志消息。
            exc_info: 是否记录异常信息（True 时在日志中输出完整堆栈）。
        """
        self.logger.error(msg, exc_info=exc_info, **kwargs)

    def debug(self, msg: str, **kwargs) -> None:
        """输出 DEBUG 日志。"""
        self.logger.debug(msg, **kwargs)

    def close(self) -> None:
        """关闭所有资源。"""
        if self.writer is not None:
            self.writer.close()
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)