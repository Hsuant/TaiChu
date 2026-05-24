"""SwanLab 实验跟踪工具。

提供与 SwanLab 平台交互的封装类，支持自动记录训练指标、超参数、文本生成样例等。
"""

import os
from typing import Dict, Any, Optional
import swanlab


class SwanLabLogger:
    """SwanLab 日志记录器。

    仅在主进程（global_rank == 0）时生效，避免分布式训练中的重复记录。
    支持配置项目名、实验名、超参数，提供方便的指标记录接口。

    Attributes:
        enabled: 是否启用 SwanLab（主进程且未禁用）。
        run: SwanLab 运行实例。
    """

    def __init__(
        self,
        project: str = "TaiChu-Pretrain",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_dir: str = "./swanlogs",
        disabled: bool = False,
        global_rank: int = 0,
    ):
        """初始化 SwanLab 记录器。

        Args:
            project: 项目名称。
            experiment_name: 实验名称（若不指定则自动生成）。
            config: 超参数字典，将记录到 SwanLab。
            log_dir: SwanLab 日志保存目录。
            disabled: 手动禁用 SwanLab。
            global_rank: 当前进程的全局 rank，仅当 global_rank == 0 时才实际初始化。
        """
        self.enabled = (global_rank == 0) and (not disabled)
        self.run = None

        if not self.enabled:
            return

        # 自动生成实验名（可选）
        if experiment_name is None:
            experiment_name = f"run_{os.getpid()}"

        # 初始化 SwanLab
        self.run = swanlab.init(
            project=project,
            experiment_name=experiment_name,
            config=config or {},
            logdir=log_dir,
        )

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """记录一组标量指标。

        Args:
            metrics: 指标名到值的映射。
            step: 当前步数（可选，SwanLab 会自动记录 step）。
        """
        if not self.enabled:
            return
        swanlab.log(metrics, step=step)

    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """记录文本（如生成样例）。

        Args:
            tag: 文本标签。
            text: 文本内容。
            step: 步数。
        """
        if not self.enabled:
            return
        swanlab.log({tag: swanlab.Text(text, caption=tag)}, step=step)

    def log_config(self, config: Dict[str, Any]) -> None:
        """更新超参数配置（需在 swanlab.init 之前或使用 swanlab.config 更新）。

        Args:
            config: 超参数字典。
        """
        if not self.enabled or self.run is None:
            return
        # SwanLab 支持动态添加配置
        for k, v in config.items():
            swanlab.config[k] = v

    def finish(self) -> None:
        """结束 SwanLab 运行，确保数据上传完成。"""
        if self.enabled and self.run is not None:
            swanlab.finish()