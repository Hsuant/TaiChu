"""可视化辅助工具。

封装 TensorBoard 的高级可视化功能（如模型图、嵌入投影），
也可用于保存注意力权重图、生成训练报告等（按需扩展）。
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np


class VisualizationManager:
    """可视化管理器。

    基于 SummaryWriter 提供模型结构记录、参数分布直方图、
    嵌入投影等可视化功能。可直接用于训练过程中的定期记录。
    """

    def __init__(self, writer: SummaryWriter):
        """初始化。

        Args:
            writer: TensorBoard SummaryWriter 实例。
        """
        self.writer = writer

    def add_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> None:
        """添加模型计算图。

        Args:
            model: 模型实例。
            input_tensor: 样例输入。
        """
        self.writer.add_graph(model, input_tensor)

    def add_weight_histograms(self, model: torch.nn.Module, step: int) -> None:
        """记录模型所有权重的直方图。

        Args:
            model: 模型实例。
            step: 全局步数。
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"weights/{name}", param.data, step)

    def add_gradient_histograms(self, model: torch.nn.Module, step: int) -> None:
        """记录模型所有梯度的直方图。

        Args:
            model: 模型实例。
            step: 全局步数。
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f"gradients/{name}", param.grad, step)

    def add_embedding(
        self,
        features: torch.Tensor,
        metadata: Optional[List[str]] = None,
        label_img: Optional[torch.Tensor] = None,
        global_step: Optional[int] = None,
        tag: str = "embedding",
    ) -> None:
        """添加高维数据的嵌入投影（PCA/t-SNE）。

        Args:
            features: 特征矩阵 (N, D)。
            metadata: 每个点的标签列表。
            label_img: 标签图像（可选）。
            global_step: 全局步数。
            tag: 标签名称。
        """
        self.writer.add_embedding(
            features, metadata=metadata, label_img=label_img,
            global_step=global_step, tag=tag,
        )

    @staticmethod
    def save_attention_plot(
        attention_weights: np.ndarray,
        tokens: List[str],
        filepath: str,
        title: str = "Attention Weights",
    ) -> None:
        """保存注意力权重热力图到文件。

        Args:
            attention_weights: 注意力矩阵 (seq_len, seq_len)。
            tokens: 对应的 token 字符串列表。
            filepath: 保存路径（png）。
            title: 图片标题。
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(attention_weights, cmap="viridis")
        fig.colorbar(cax)
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close(fig)