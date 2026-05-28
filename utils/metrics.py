# -*- coding: utf-8 -*-
"""LLM 预训练专业评估指标模块。

提供4个指标的封装类：
1. SmoothedLossTracker       - 指数加权平滑损失（EMA Loss）
2. GradientNoiseScale        - 梯度噪声尺度，衡量优化器利用效率
3. ModelFlopsUtilization     - 模型FLOPs利用率，计算硬件效率
4. RepresentationStability   - Token嵌入稳定性，检测表示坍塌
"""

import time
import math
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from torch.utils.hooks import RemovableHandle


class SmoothedLossTracker:
    """指数加权移动平均（EMA）平滑损失。

    平滑损失能够过滤训练过程中的噪声，更清晰反映收敛趋势。

    Attributes:
        beta: 平滑系数，取值范围 (0,1)，越大平滑效果越强。
        smoothed_loss: 当前平滑损失值。
        original_losses: 原始损失值列表（最近100步）。
    """

    def __init__(self, beta: float = 0.9, initial_loss: Optional[float] = None):
        """初始化平滑损失追踪器。

        Args:
            beta: 平滑系数，通常取 0.9 或 0.99。
            initial_loss: 初始平滑损失值（恢复训练时使用）。
        """
        if not (0.0 < beta < 1.0):
            raise ValueError(f"beta must be in (0,1), got {beta}")
        self.beta = beta
        self.smoothed_loss = initial_loss
        self.original_losses: List[float] = []

    def update(self, loss: float) -> float:
        """用新损失更新平滑值。

        Args:
            loss: 当前步的原始损失值。

        Returns:
            更新后的平滑损失值。
        """
        self.original_losses.append(loss)
        if self.smoothed_loss is None:
            self.smoothed_loss = loss
        else:
            self.smoothed_loss = self.beta * self.smoothed_loss + (1 - self.beta) * loss
        return self.smoothed_loss

    def get_metrics(self) -> Dict[str, float]:
        """获取当前平滑损失的各项指标。

        Returns:
            字典，包含：
                - smoothed_loss: 当前平滑损失
                - loss_trend: 最近100步的趋势变化率
                - loss_variance: 最近100步的损失方差
        """
        metrics = {"smoothed_loss": self.smoothed_loss if self.smoothed_loss is not None else 0.0}
        if len(self.original_losses) >= 2:
            window_size = min(100, len(self.original_losses))
            recent = self.original_losses[-window_size:]
            if len(recent) >= 2:
                trend = (recent[-1] - recent[0]) / max(recent[0], 1e-8)
                metrics["loss_trend"] = trend
                mean_loss = sum(recent) / len(recent)
                variance = sum((l - mean_loss) ** 2 for l in recent) / len(recent)
                metrics["loss_variance"] = variance
        else:
            metrics["loss_trend"] = 0.0
            metrics["loss_variance"] = 0.0
        return metrics

    def reset(self) -> None:
        """重置追踪器状态（用于新的训练阶段）。"""
        self.smoothed_loss = None
        self.original_losses = []


class GradientNoiseScale:
    """梯度噪声尺度（Gradient Noise Scale）计算器。

    梯度噪声尺度反映批次梯度噪声与真实梯度的比值，可用于指导动态
    Batch Size 调度。计算公式：GNS = (‖g_batch‖²) / (σ_g²)。

    Attributes:
        window_size: 用于估计梯度方差的滑动窗口大小。
        grad_norms: 梯度范数平方的滑动窗口列表。
        current_gns: 当前估计的梯度噪声尺度。
    """

    def __init__(self, window_size: int = 100):
        """初始化梯度噪声尺度计算器。

        Args:
            window_size: 滑动窗口大小，用于计算梯度方差。
        """
        self.window_size = window_size
        self.grad_norms: List[float] = []
        self.current_gns: Optional[float] = None

    def update(self, model: nn.Module) -> float:
        """基于模型所有可训练参数的梯度更新 GNS 估计。

        注意：调用本方法前必须完成 backward()，否则梯度不存在。

        Args:
            model: 已完成反向传播的模型（含有 .grad）。

        Returns:
            当前估计的梯度噪声尺度值。
        """
        grad_norm_sq = 0.0
        param_grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not param_grads:
            return 0.0
        for grad in param_grads:
            grad_norm_sq += (grad.norm(2).item()) ** 2

        self.grad_norms.append(grad_norm_sq)
        if len(self.grad_norms) > self.window_size:
            self.grad_norms.pop(0)

        if len(self.grad_norms) >= 2:
            mean_norm = sum(self.grad_norms) / len(self.grad_norms)
            grad_norm_variance = sum((n - mean_norm) ** 2 for n in self.grad_norms) / len(self.grad_norms)
            if grad_norm_variance > 1e-12:
                self.current_gns = grad_norm_sq / grad_norm_variance
            else:
                self.current_gns = float("inf")
        else:
            self.current_gns = float("inf")
        return self.current_gns

    def get_metrics(self) -> Dict[str, float]:
        """获取 GNS 及相关统计信息。

        Returns:
            字典，包含：
                - gradient_noise_scale: 当前 GNS
                - grad_norm_sq_mean: 窗口内梯度范数平方的平均值
                - grad_norm_sq_std: 窗口内梯度范数平方的标准差
        """
        metrics = {"gradient_noise_scale": self.current_gns if self.current_gns is not None else 0.0}
        if len(self.grad_norms) > 0:
            mean_val = sum(self.grad_norms) / len(self.grad_norms)
            metrics["grad_norm_sq_mean"] = mean_val
            if len(self.grad_norms) > 1:
                variance = sum((n - mean_val) ** 2 for n in self.grad_norms) / len(self.grad_norms)
                metrics["grad_norm_sq_std"] = math.sqrt(variance)
        return metrics

    def reset(self) -> None:
        """重置追踪器状态。"""
        self.grad_norms = []
        self.current_gns = None


class EfficientGradientNoiseScale(GradientNoiseScale):
    """高效梯度噪声尺度（仅使用归一化层）。

    该变体仅使用 LayerNorm 层的梯度来估计整体 GNS，可大幅降低计算开销。
    适用于 Transformer 等包含大量归一化层的模型。

    Attributes:
        norm_layers: 模型中所有的 LayerNorm 层列表。
        layer_gns: 每层归一化层的梯度贡献字典。
    """

    def __init__(self, model: nn.Module, window_size: int = 100):
        """初始化高效 GNS 计算器。

        Args:
            model: 模型实例，用于收集归一化层。
            window_size: 滑动窗口大小。
        """
        super().__init__(window_size=window_size)
        self.norm_layers = self._collect_norm_layers(model)
        self.layer_gns: Dict[str, float] = {}

    @staticmethod
    def _collect_norm_layers(model: nn.Module) -> List[nn.Module]:
        """递归收集模型中所有 LayerNorm 层。

        Args:
            model: 模型实例。

        Returns:
            LayerNorm 模块列表。
        """
        norm_layers = []
        for module in model.modules():
            if isinstance(module, (nn.LayerNorm,)):
                norm_layers.append(module)
        return norm_layers

    def update(self, model: nn.Module) -> float:
        """仅基于归一化层的梯度更新 GNS 估计。

        Args:
            model: 已完成反向传播的模型。

        Returns:
            当前估计的梯度噪声尺度。
        """
        grad_norm_sq = 0.0
        for layer in self.norm_layers:
            for param in layer.parameters():
                if param.grad is not None:
                    grad_norm_sq += (param.grad.norm(2).item()) ** 2

        self.grad_norms.append(grad_norm_sq)
        if len(self.grad_norms) > self.window_size:
            self.grad_norms.pop(0)

        if len(self.grad_norms) >= 2:
            mean_norm = sum(self.grad_norms) / len(self.grad_norms)
            grad_norm_variance = sum((n - mean_norm) ** 2 for n in self.grad_norms) / len(self.grad_norms)
            if grad_norm_variance > 1e-12:
                self.current_gns = grad_norm_sq / grad_norm_variance
            else:
                self.current_gns = float("inf")
        else:
            self.current_gns = float("inf")

        # 记录各层的梯度贡献
        for idx, layer in enumerate(self.norm_layers):
            layer_grad_sq = sum((p.grad.norm(2).item()) ** 2 for p in layer.parameters() if p.grad is not None)
            self.layer_gns[f"layer_{idx}"] = layer_grad_sq
        return self.current_gns

    def get_metrics(self) -> Dict[str, float]:
        """获取高效 GNS 相关指标。

        Returns:
            字典，包含超类指标以及：
                - num_norm_layers: 归一化层数量
                - max_layer_grad_norm: 各层梯度范数平方的最大值
                - min_layer_grad_norm: 各层梯度范数平方的最小值
        """
        metrics = super().get_metrics()
        metrics["num_norm_layers"] = len(self.norm_layers)
        if self.layer_gns:
            metrics["max_layer_grad_norm"] = max(self.layer_gns.values())
            metrics["min_layer_grad_norm"] = min(self.layer_gns.values())
        else:
            metrics["max_layer_grad_norm"] = 0.0
            metrics["min_layer_grad_norm"] = 0.0
        return metrics


class ModelFlopsUtilization:
    """模型 FLOPs 利用率（MFU）计算器。

    MFU = achieved_FLOPs / (peak_FLOPs × runtime)，衡量硬件计算效率。
    参考业界标准，主流开源 LLM 训练 MFU 通常在 40%-50%。

    Attributes:
        peak_flops: GPU 理论峰值算力（FLOPs/秒）。
        total_flops: 累计 FLOPs 总量。
        start_time: 训练开始时间戳。
        last_step_time: 上一步更新时间戳。
    """

    def __init__(self, peak_flops: float, log_file: Optional[str] = None):
        """初始化 MFU 计算器。

        Args:
            peak_flops: GPU 理论峰值算力，例如 A100 PCIe 为 312e12。
            log_file: 可选输出文件路径，用于持久化 MFU 日志。
        """
        if peak_flops <= 0:
            raise ValueError(f"peak_flops must be positive, got {peak_flops}")
        self.peak_flops = peak_flops
        self.total_flops = 0.0
        self.start_time: Optional[float] = None
        self.last_step_time: Optional[float] = None
        self.log_file = log_file
        if self.log_file:
            import os
            os.makedirs(os.path.dirname(self.log_file) or '.', exist_ok=True)

    def start(self) -> None:
        """开始计时（在训练循环开始前调用）。"""
        self.start_time = time.time()
        self.last_step_time = self.start_time

    def add_flops(self, flops: float) -> None:
        """累加 FLOPs。

        Args:
            flops: 本次前向+反向传播的 FLOPs 值。
        """
        self.total_flops += flops

    def update_step(self, step_flops: float, step: int) -> Dict[str, float]:
        """更新单步的 FLOPs 并计算瞬时/平均 MFU。

        Args:
            step_flops: 本步的 FLOPs 值。
            step: 当前全局步数。

        Returns:
            字典，包含：
                - total_flops: 累计 FLOPs（TFLOPS）
                - elapsed_time: 已运行时间（秒）
                - mfu_average: 平均 MFU
                - mfu_instant: 瞬时 MFU（步间窗口）
        """
        current_time = time.time()
        self.total_flops += step_flops
        metrics = {"total_flops": self.total_flops / 1e12}  # 转换为 TFLOPS

        if self.start_time is not None:
            elapsed = current_time - self.start_time
            metrics["elapsed_time"] = elapsed
            if elapsed > 0:
                avg_mfu = self.total_flops / (self.peak_flops * elapsed)
                metrics["mfu_average"] = avg_mfu

        if self.last_step_time is not None and step > 0:
            step_elapsed = current_time - self.last_step_time
            if step_elapsed > 0:
                instant_mfu = step_flops / (self.peak_flops * step_elapsed)
                metrics["mfu_instant"] = instant_mfu

        self.last_step_time = current_time

        # 每 100 步持久化一次
        if self.log_file and step % 100 == 0:
            self._log_to_file(step, metrics)

        return metrics

    def get_metrics(self) -> Dict[str, float]:
        """获取当前 MFU 状态。

        Returns:
            字典，包含：
                - mfu_average: 平均 MFU
                - total_flops: 累计 FLOPs（TFLOPS）
                - peak_flops: 理论峰值算力（TFLOPs/秒）
        """
        metrics = {
            "mfu_average": 0.0,
            "total_flops": self.total_flops / 1e12,
            "peak_flops": self.peak_flops / 1e12,
        }
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                metrics["mfu_average"] = self.total_flops / (self.peak_flops * elapsed)
        return metrics

    def estimate_model_flops(self, model: nn.Module, seq_len: int, batch_size: int) -> float:
        """粗略估算单次前向+反向传播的 FLOPs。

        估算公式：≈ 6 × 参数量 × 序列长度 × 批次大小。

        Args:
            model: 模型实例。
            seq_len: 序列长度。
            batch_size: 批次大小（微批次累积后的总样本数）。

        Returns:
            估算的 FLOPs 值。
        """
        total_params = sum(p.numel() for p in model.parameters())
        return 6 * total_params * seq_len * batch_size

    def _log_to_file(self, step: int, metrics: Dict[str, float]) -> None:
        """将 MFU 指标写入日志文件（JSON Lines 格式）。"""
        import json
        log_entry = {
            "step": step,
            "mfu_average": metrics.get("mfu_average", 0.0),
            "mfu_instant": metrics.get("mfu_instant", 0.0),
            "total_flops": metrics.get("total_flops", 0.0),
            "elapsed_time": metrics.get("elapsed_time", 0.0),
            "timestamp": time.time(),
        }
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")


class RepresentationStability:
    """Token 嵌入表示稳定性分析器。

    追踪模型最后一层输出的 Token 嵌入变化，计算表示密度和各向同性分数，
    用于检测训练过程中的模式坍塌或不稳定现象。

    Attributes:
        model: 模型实例。
        device: 计算设备。
        hook_handle: 前向钩子句柄。
        saved_embeddings: 最近一次前向传播捕获的嵌入张量。
    """

    def __init__(self, model: nn.Module, device: torch.device, hook_layer: int = -1):
        """初始化表示稳定性分析器。

        Args:
            model: 模型实例（必须是 TaiChuModel 或其子类）。
            device: 计算设备。
            hook_layer: 预留参数（暂未使用，保留接口）。
        """
        self.model = model
        self.device = device
        self.hook_layer = hook_layer
        self.hook_handle: Optional[RemovableHandle] = None
        self.saved_embeddings: Optional[torch.Tensor] = None
        self._register_hook()

    def _register_hook(self) -> None:
        """注册前向钩子以捕获模型输出（通常为 logits 张量）。"""
        self.hook_handle = self.model.register_forward_hook(self._capture_embedding)

    def _capture_embedding(self, module: nn.Module, input: Tuple[torch.Tensor, ...],
                           output: torch.Tensor) -> None:
        """前向钩子回调，保存输出张量副本。

        Args:
            module: 当前模块（未被使用）。
            input: 模块输入（未被使用）。
            output: 模块输出，可能是张量或元组。
        """
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, torch.Tensor):
            self.saved_embeddings = output.detach().clone()

    def update(self, input_ids: torch.Tensor) -> Dict[str, float]:
        """对给定输入进行一次前向传播并计算表示稳定性指标。

        Args:
            input_ids: 输入 Token ID 张量 [batch_size, seq_len]。

        Returns:
            字典，包含：
                - embedding_norm: 嵌入平均范数
                - embedding_std: 嵌入标准差
                - representation_density: Token 嵌入平均余弦相似度
                - representation_density_std: 余弦相似度标准差
                - isotropy_score: 各向同性分数（协方差矩阵最小/最大特征值比）
        """
        metrics = {}
        with torch.no_grad():
            _ = self.model(input_ids.to(self.device))
        if self.saved_embeddings is not None:
            current_emb = self.saved_embeddings
            metrics["embedding_norm"] = current_emb.norm().item() / current_emb.numel() ** 0.5
            metrics["embedding_std"] = current_emb.std().item()
            batch_emb = current_emb[0]  # 取第一个样本 [seq_len, hidden_size]
            if batch_emb.shape[0] > 1:
                # 计算表示密度（平均余弦相似度）
                normed = batch_emb / (batch_emb.norm(dim=-1, keepdim=True) + 1e-8)
                sim_matrix = torch.mm(normed, normed.t())
                mask = torch.ones_like(sim_matrix, dtype=torch.bool).triu(diagonal=1)
                similarity_scores = sim_matrix[mask]
                if similarity_scores.numel() > 0:
                    metrics["representation_density"] = similarity_scores.mean().item()
                    metrics["representation_density_std"] = similarity_scores.std().item()
                else:
                    metrics["representation_density"] = 0.0
                    metrics["representation_density_std"] = 0.0

                # 计算各向同性分数（协方差矩阵条件数）
                cov_matrix = torch.cov(batch_emb.T)
                eigenvalues = torch.linalg.eigvalsh(cov_matrix)
                eigenvalues = torch.clamp(eigenvalues, min=1e-8)
                isotropy = eigenvalues.min() / eigenvalues.max()
                metrics["isotropy_score"] = isotropy.item()
            else:
                metrics["representation_density"] = 0.0
                metrics["representation_density_std"] = 0.0
                metrics["isotropy_score"] = 0.0
        return metrics

    def close(self) -> None:
        """移除钩子，释放资源。"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


class ValidationMetrics:
    """验证阶段评估指标集。

    用于在验证集上计算除 loss/perplexity 之外的多维度指标：
        - next token 准确率 (accuracy@1)
        - 分位数交叉熵 (median ce, 90% ce)
        - 空白提示困惑度 (blank ppl) —— 评估模型在没有有用上下文时的基线行为
    """

    @staticmethod
    def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
        """计算 next token 预测准确率（argmax 与 label 匹配的比例）。

        注意：labels 中可能包含 -100（忽略的 token），计算时会自动跳过。

        Args:
            logits: 模型输出的 logits，形状 [batch_size, seq_len, vocab_size]。
            labels: 标签，形状 [batch_size, seq_len]。

        Returns:
            准确率（0~1 之间）。
        """
        if logits is None or labels is None:
            return 0.0
        # 取每个位置预测概率最高的 token
        preds = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
        # 创建有效位置掩码（忽略 -100）
        valid_mask = labels != -100
        if not valid_mask.any():
            return 0.0
        correct = (preds == labels) & valid_mask
        accuracy = correct.sum().item() / valid_mask.sum().item()
        return accuracy

    @staticmethod
    def compute_percentile_ce(losses_per_token: torch.Tensor, percentiles: List[float] = None) -> Dict[str, float]:
        """计算每个 token 损失的分位数（中位数、90% 分位数等）。

        Args:
            losses_per_token: 每个 token 的交叉熵损失（不取平均），形状 [total_valid_tokens]。
            percentiles: 需要计算的分位数列表，例如 [0.5, 0.9, 0.95]，默认 [0.5, 0.9]。

        Returns:
            字典，键为 f"ce_p{int(p*100)}"，值为对应分位数的损失值。
        """
        if percentiles is None:
            percentiles = [0.5, 0.9]
        if losses_per_token.numel() == 0:
            return {f"ce_p{int(p*100)}": 0.0 for p in percentiles}
        # 排序后计算分位数
        sorted_losses, _ = torch.sort(losses_per_token)
        n = sorted_losses.numel()
        results = {}
        for p in percentiles:
            idx = int(p * (n - 1))
            results[f"ce_p{int(p*100)}"] = sorted_losses[idx].item()
        return results

    @staticmethod
    def compute_blank_ppl(model: nn.Module, tokenizer, blank_text: str = "",
                          max_length: int = 50, device: torch.device = torch.device("cpu")) -> float:
        """计算空白提示的困惑度（Blank PPL）。

        空白提示可以是空字符串或仅包含起始标记的序列，用于评估模型在没有
        有用上下文时的“默认”行为。过高的空白 PPL 可能指示模型过度拟合
        特定模式或训练数据存在偏差。

        Args:
            model: 模型实例（已 eval 模式）。
            tokenizer: tokenizer 实例。
            blank_text: 空白提示文本，默认为空字符串。
            max_length: 生成/评估的最大长度（此处仅计算给定文本的损失）。
            device: 设备。

        Returns:
            困惑度值（perplexity）。
        """
        model.eval()
        # 对空白文本进行编码，不添加特殊 token（或根据 tokenizer 习惯）
        encoding = tokenizer.encode(blank_text, add_special_tokens=True)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)
        # 标签与输入相同（用于计算交叉熵）
        labels = input_ids.clone()
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
        ppl = math.exp(loss.item()) if loss.item() < 100 else float("inf")
        return ppl

    @staticmethod
    def compute_bose_stability(model: nn.Module, tokenizer, num_runs: int = 3,
                               device: torch.device = torch.device("cpu")) -> Dict[str, float]:
        """简化版 BOSE 评估稳定性（通过多次前向传播的损失方差）。

        真实 BOSE 需要构造无指令提示和空白 PPL 基线，这里提供一种轻量实现：
        对同一空白提示多次计算损失，返回标准差作为稳定性指标。

        Args:
            model: 模型实例。
            tokenizer: tokenizer。
            num_runs: 重复次数。
            device: 设备。

        Returns:
            字典，包含 "blank_ppl_mean", "blank_ppl_std"。
        """
        losses = []
        for _ in range(num_runs):
            # 每次重新编码（避免缓存影响）
            encoding = tokenizer.encode("", add_special_tokens=True)
            input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)
            labels = input_ids.clone()
            with torch.no_grad():
                outputs = model(input_ids, labels=labels)
                losses.append(outputs.loss.item())
        mean_loss = sum(losses) / len(losses)
        std_loss = (sum((l - mean_loss) ** 2 for l in losses) / len(losses)) ** 0.5
        mean_ppl = math.exp(mean_loss) if mean_loss < 100 else float("inf")
        std_ppl = std_loss * mean_ppl  # 近似
        return {"blank_ppl_mean": mean_ppl, "blank_ppl_std": std_ppl}


class EvaluationMetricsManager:
    """统一管理所有评估指标的容器。

    该类聚合了平滑损失、梯度噪声尺度、MFU 和表示稳定性四个指标，
    并提供统一的更新和获取接口，便于在训练循环中集成。

    Attributes:
        smoothed_loss: SmoothedLossTracker 实例。
        gns: GradientNoiseScale 实例。
        mfu: ModelFlopsUtilization 实例（需显式初始化）。
        repr_stability: RepresentationStability 实例（需显式初始化）。
    """

    def __init__(self, config: Optional[Dict] = None):
        """初始化评估指标管理器。

        Args:
            config: 可选配置字典，支持以下键：
                - loss_beta: 平滑损失系数，默认 0.9
                - gns_window: GNS 窗口大小，默认 100
        """
        self.smoothed_loss = SmoothedLossTracker(beta=config.get("loss_beta", 0.9) if config else 0.9)
        self.gns = GradientNoiseScale(window_size=config.get("gns_window", 100) if config else 100)
        self.mfu: Optional[ModelFlopsUtilization] = None
        self.repr_stability: Optional[RepresentationStability] = None
        self.config = config or {}

    def initialize_mfu(self, peak_flops: float, log_file: Optional[str] = None) -> None:
        """初始化 MFU 计算器并启动计时。

        Args:
            peak_flops: GPU 理论峰值算力（FLOPs/秒）。
            log_file: 可选日志文件路径。
        """
        self.mfu = ModelFlopsUtilization(peak_flops=peak_flops, log_file=log_file)
        self.mfu.start()

    def initialize_repr_stability(self, model: nn.Module, device: torch.device,
                                  hook_layer: int = -1) -> None:
        """初始化表示稳定性分析器。

        Args:
            model: 模型实例。
            device: 计算设备。
            hook_layer: 钩子层索引（保留，当前未使用）。
        """
        self.repr_stability = RepresentationStability(model, device, hook_layer)

    def update_loss(self, loss: float) -> float:
        """更新平滑损失。

        Args:
            loss: 当前步的原始损失值。

        Returns:
            更新后的平滑损失值。
        """
        return self.smoothed_loss.update(loss)

    def update_gradient(self, model: nn.Module) -> float:
        """更新梯度噪声尺度。

        Args:
            model: 已完成反向传播的模型。

        Returns:
            当前 GNS 值。
        """
        return self.gns.update(model)

    def update_step(self, step_flops: float, step: int) -> Dict[str, float]:
        """更新 MFU 指标。

        Args:
            step_flops: 本步的 FLOPs 估算值。
            step: 当前全局步数。

        Returns:
            包含 MFU 相关指标的字典。
        """
        if self.mfu is not None:
            return self.mfu.update_step(step_flops, step)
        return {}

    def update_repr(self, input_ids: torch.Tensor) -> Dict[str, float]:
        """更新表示稳定性指标。

        Args:
            input_ids: 输入 Token ID 张量。

        Returns:
            表示稳定性相关指标字典。
        """
        if self.repr_stability is not None:
            return self.repr_stability.update(input_ids)
        return {}

    def get_all_metrics(self, include_repr: bool = True) -> Dict[str, float]:
        """获取所有已初始化的指标。

        Args:
            include_repr: 是否包含表示稳定性指标（默认 True，但实际需单独调用 update_repr）。

        Returns:
            聚合的指标字典。
        """
        metrics = {}
        metrics.update(self.smoothed_loss.get_metrics())
        metrics.update(self.gns.get_metrics())
        if self.mfu is not None:
            metrics.update(self.mfu.get_metrics())
        # 表示稳定性指标需通过 update_repr 单独获取，此处不重复包含
        return metrics

    def reset_loss_tracker(self) -> None:
        """重置损失追踪器。"""
        self.smoothed_loss.reset()

    def reset_gradient_tracker(self) -> None:
        """重置梯度追踪器。"""
        self.gns.reset()

    def close(self) -> None:
        """释放资源（钩子等）。"""
        if self.repr_stability is not None:
            self.repr_stability.close()