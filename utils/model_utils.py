"""模型初始化与检测工具。

提供模型权重初始化策略、模型摘要、参数与 FLOPs 统计、梯度监控等功能。
"""

from typing import Optional, Dict
import torch
import torch.nn as nn


class ModelInitializer:
    """模型权重初始化器。

    提供统一的初始化策略，可在模型构建后调用。
    """

    @staticmethod
    def init_weights(module: nn.Module, mean: float = 0.0, std: float = 0.02):
        """递归初始化模块参数。

        线性层使用正态分布 N(mean, std)，偏置置零；嵌入层使用正态分布。

        Args:
            module: 需要初始化的模块。
            mean: 均值。
            std: 标准差。
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=mean, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=mean, std=std)

    @staticmethod
    def apply_to_model(model: nn.Module, mean: float = 0.0, std: float = 0.02):
        """对整个模型应用初始化策略。

        Args:
            model: 模型实例。
            mean: 均值。
            std: 标准差。
        """
        model.apply(lambda m: ModelInitializer.init_weights(m, mean, std))


class ModelInspector:
    """模型检测工具。

    提供参数量统计、FLOPs 估算（可选）、梯度统计、模型摘要等功能。
    """

    @staticmethod
    def get_parameter_count(model: nn.Module, trainable_only: bool = True) -> int:
        """统计模型参数量。

        Args:
            model: 模型实例。
            trainable_only: 是否仅统计可训练参数。

        Returns:
            参数量。
        """
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())

    @staticmethod
    def get_model_size_mb(model: nn.Module, precision: int = 4) -> float:
        """估算模型大小（以 MB 为单位）。

        假设参数为 FP32（4 字节），也可根据 precision 指定（如 2 表示 FP16）。

        Args:
            model: 模型实例。
            precision: 每个参数的字节数（默认为 4）。

        Returns:
            模型大小（MB）。
        """
        num_params = ModelInspector.get_parameter_count(model, trainable_only=False)
        size_bytes = num_params * precision
        return size_bytes / (1024 ** 2)

    @staticmethod
    def gradient_summary(model: nn.Module) -> Dict[str, float]:
        """汇总梯度的统计信息（用于梯度检查）。

        遍历所有可训练参数，返回梯度范数、最大值、最小值。

        Returns:
            包含 'grad_norm', 'grad_max', 'grad_min' 的字典。
        """
        grad_norm = 0.0
        grad_max = float("-inf")
        grad_min = float("inf")
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm(2).item() ** 2
                grad_max = max(grad_max, p.grad.abs().max().item())
                grad_min = min(grad_min, p.grad.abs().min().item())
        grad_norm = grad_norm ** 0.5
        return {"grad_norm": grad_norm, "grad_max": grad_max, "grad_min": grad_min}

    @staticmethod
    def model_summary(model: nn.Module, input_shape: tuple, device: torch.device = torch.device("cpu")) -> str:
        """生成模型结构摘要字符串（使用 torchsummary 风格）。

        注意：此方法需要安装 torchinfo 或手动实现，这里提供基于 torchinfo 的简单封装。
        如果未安装 torchinfo，则降级为打印模型结构。

        Args:
            model: 模型实例。
            input_shape: 输入张量的形状（不含 batch 维度），例如 (2048,) for input_ids。
            device: 设备。

        Returns:
            模型摘要字符串。
        """
        try:
            from torchinfo import summary
            # 创建虚拟输入
            dummy_input = torch.randint(0, 100, (1,) + input_shape).to(device)
            return str(summary(model, input_data=dummy_input, verbose=0))
        except ImportError:
            # 降级：仅打印模型结构
            return str(model)

    @staticmethod
    def estimate_flops(model: nn.Module, input_shape: tuple, device: torch.device = torch.device("cpu")) -> Optional[int]:
        """估算模型的 FLOPs（需要安装 fvcore 或 thop）。

        Args:
            model: 模型实例。
            input_shape: 输入形状（不含 batch 维度）。
            device: 设备。

        Returns:
            FLOPs 估计值，若缺少库则返回 None。
        """
        try:
            from fvcore.nn import FlopCountAnalysis
            dummy_input = torch.randint(0, 100, (1,) + input_shape).to(device)
            flops = FlopCountAnalysis(model, dummy_input)
            return flops.total()
        except ImportError:
            try:
                from thop import profile
                dummy_input = torch.randint(0, 100, (1,) + input_shape).to(device)
                flops, _ = profile(model, inputs=(dummy_input,))
                return flops
            except ImportError:
                print("未安装 fvcore 或 thop，无法估算 FLOPs。")
                return None