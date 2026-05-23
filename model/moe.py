"""
混合专家（MoE）模块。

包含：
- 门控网络（Router）及其 Top‑k 选择
- 共享专家（Shared Expert）
- 路由专家（Routed Experts，多个 FFN）
- 负载均衡损失与路由器 z‑loss
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class MoEConfig:
    """MoE 专用配置，仅包含 MoE 相关参数。

    所有字段均有默认值，方便快速实验与论文复现。

    Attributes:
        num_experts: 路由专家数量（不包括共享专家）
        top_k: 每个 token 激活的专家数量
        expert_intermediate_size: 每个路由专家的中间层维度
        num_shared_experts: 共享专家数量
        shared_expert_intermediate_size: 共享专家中间层维度
        load_balancing_loss_weight: 负载均衡损失系数
        router_z_loss_weight: 路由器 z‑loss 系数
    """
    num_experts: int = 8
    top_k: int = 2
    expert_intermediate_size: int = 1536
    num_shared_experts: int = 1
    shared_expert_intermediate_size: int = 3072
    load_balancing_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001


class MoEGate(nn.Module):
    """MoE 门控网络（路由器）。

    将 hidden_states 映射到 num_experts 维度的 logits，
    再通过 softmax 生成路由概率，并进行 Top‑k 选择。

    Attributes:
        weight: 线性投影权重，形状 (hidden_size, num_experts)
        top_k: 每个 token 激活的专家数
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        """初始化门控网络。

        Args:
            hidden_size: 隐藏层维度。
            num_experts: 路由专家总数。
            top_k: 每个 token 激活的专家数量。
        """
        super().__init__()
        self.top_k = top_k
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重，使用较小的标准差以稳定训练。"""
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / self.weight.size(1))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算路由结果。

        Args:
            hidden_states: 输入张量，形状 (batch_size, seq_len, hidden_size)

        Returns:
            topk_indices: 选中的专家索引，形状 (batch_size * seq_len, top_k)
            topk_weights: 对应的权重（已 softmax 归一化），形状 (batch_size * seq_len, top_k)
            router_logits: 原始路由 logits，形状 (batch_size * seq_len, num_experts)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        # 展平为 (tokens, hidden_size)
        flat_input = hidden_states.view(-1, hidden_size)          # (S, D)
        # 计算路由 logits
        logits = F.linear(flat_input, self.weight, bias=None)     # (S, E)
        # softmax 得到概率（用于负载均衡损失计算）
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)    # 用 float32 保证精度
        # Top‑k 选择，返回权重和索引
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)  # (S, K)
        # 对选出的权重重新归一化（保证和为 1）
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        # 权重转回输入 dtype
        topk_weights = topk_weights.type_as(hidden_states)
        return topk_indices, topk_weights, logits


class SharedExpert(nn.Module):
    """共享专家（所有 token 均经过）。

    结构与普通 SwiGLU FFN 相同，但独立于路由专家。

    Attributes:
        gate_proj: 门控投影
        up_proj: 上投影
        down_proj: 下投影
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        """初始化共享专家。

        Args:
            hidden_size: 输入/输出维度。
            intermediate_size: 中间层维度。
        """
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重。"""
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状 (batch_size, seq_len, hidden_size)。

        Returns:
            共享专家输出，形状与输入相同。
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = gate * F.silu(up)
        return self.down_proj(hidden)


class RoutedExpert(nn.Module):
    """单个路由专家（SwiGLU FFN）。

    结构：Linear → SiLU(gate * up) → Linear。

    Attributes:
        gate_proj: 门控投影
        up_proj: 上投影
        down_proj: 下投影
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        """初始化路由专家。

        Args:
            hidden_size: 输入/输出维度。
            intermediate_size: 中间层维度。
        """
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重。"""
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状 (batch_size, seq_len, hidden_size)。

        Returns:
            专家输出，形状与输入相同。
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = gate * F.silu(up)
        return self.down_proj(hidden)


class SparseMoE(nn.Module):
    """稀疏混合专家层。

    整合了门控网络、共享专家（可选）和多个路由专家，
    并计算负载均衡损失与路由器 z‑loss。

    Attributes:
        num_experts: 路由专家数量
        top_k: 每个 token 激活的专家数
        gate: 门控网络
        shared_expert: 共享专家模块（若 num_shared_experts > 0）
        experts: 路由专家列表
        load_balancing_loss_weight: 负载均衡损失系数
        router_z_loss_weight: 路由器 z‑loss 系数
    """

    def __init__(self, hidden_size: int, moe_config: MoEConfig):
        """初始化稀疏 MoE 层。

        Args:
            hidden_size: 隐藏层维度。
            moe_config:
                - num_experts: 路由专家数量。
                - top_k: 每个 token 激活的专家数。
                - expert_intermediate_size: 每个路由专家的中间层维度。
                - num_shared_experts: 共享专家数量（通常为 1）。
                - shared_expert_intermediate_size: 共享专家中间层维度，若未提供则使用 expert_intermediate_size。
                - load_balancing_loss_weight: 负载均衡损失系数。
                - router_z_loss_weight: 路由器 z‑loss 系数。
        """
        super().__init__()
        self.num_experts = moe_config.num_experts
        self.top_k = moe_config.top_k
        self.load_balancing_loss_weight = moe_config.load_balancing_loss_weight
        self.router_z_loss_weight = moe_config.router_z_loss_weight

        # 门控网络
        self.gate = MoEGate(hidden_size, self.num_experts, self.top_k)

        # 共享专家（可选）
        if moe_config.num_shared_experts > 0:
            self.shared_expert = SharedExpert(
                hidden_size,
                moe_config.shared_expert_intermediate_size,
            )
        else:
            self.shared_expert = None

        # 路由专家（每个专家是一个 FFN）
        self.experts = nn.ModuleList([
            RoutedExpert(hidden_size, moe_config.expert_intermediate_size)
            for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """稀疏 MoE 前向传播。

        Args:
            hidden_states: 输入张量，形状 (batch_size, seq_len, hidden_size)

        Returns:
            output: MoE 输出张量，形状 (batch_size, seq_len, hidden_size)
            aux_loss: 辅助损失（负载均衡 + z‑loss），标量
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_tokens = batch_size * seq_len

        # ---- 1. 门控 ----
        topk_indices, topk_weights, router_logits = self.gate(hidden_states)
        # topk_indices: (S, K), topk_weights: (S, K)

        # ---- 2. 计算辅助损失 ----
        aux_loss = self._compute_auxiliary_loss(router_logits, topk_indices)

        # ---- 3. 共享专家输出 ----
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
        else:
            shared_output = 0.0

        # ---- 4. 路由专家计算 ----
        # 将 hidden_states 展平为 (S, D)
        flat_input = hidden_states.view(num_tokens, hidden_size)
        # 初始化输出缓冲区
        output = torch.zeros_like(flat_input)

        # 遍历每个专家，收集属于它的 token
        for expert_idx in range(self.num_experts):
            # 找出该专家处理的 token（所有被选中位置）
            expert_mask = (topk_indices == expert_idx).any(dim=-1)   # (S,) bool
            if not expert_mask.any():
                continue
            # 对应的 token 张量
            expert_input = flat_input[expert_mask]                   # (N_e, D)
            # 专家前向
            expert_output = self.experts[expert_idx](expert_input)   # (N_e, D)
            # 获取这些 token 在该专家上的权重（可能需要按 token 聚合多专家）
            # 权重矩阵 shape: (S, K)，先复制专家掩码
            weights_for_expert = torch.zeros(num_tokens, dtype=flat_input.dtype, device=flat_input.device)
            for k in range(self.top_k):
                k_expert_mask = (topk_indices[:, k] == expert_idx)
                weights_for_expert[k_expert_mask] = topk_weights[k_expert_mask, k]
            weighted_output = expert_output * weights_for_expert[expert_mask].unsqueeze(-1)
            # 累加到输出（同一 token 可能被多个专家处理，累加权重）
            output[expert_mask] += weighted_output

        # 还原形状
        output = output.view(batch_size, seq_len, hidden_size)

        # ---- 5. 合并共享专家输出 ----
        if isinstance(shared_output, torch.Tensor):
            output = output + shared_output

        return output, aux_loss

    def _compute_auxiliary_loss(
        self,
        router_logits: torch.Tensor,   # (S, E)
        topk_indices: torch.Tensor,    # (S, K)
    ) -> torch.Tensor:
        """计算负载均衡损失与路由器 z‑loss。

        负载均衡损失：鼓励每个专家接收的 token 数量均衡。
        z‑loss：鼓励路由器 logits 保持在较小数值，稳定训练。

        Args:
            router_logits: 原始路由 logits，形状 (S, E)
            topk_indices: 选中的专家索引，形状 (S, K)

        Returns:
            total_aux_loss: 辅助损失标量
        """
        num_tokens, num_experts = router_logits.shape

        # --- 负载均衡损失 ---
        # 计算每个专家被选中的次数（soft counting）
        if self.training:
            # 门控概率
            probs = F.softmax(router_logits.float(), dim=-1)       # (S, E)
            # 每个专家在所有 token 上的平均路由概率
            avg_probs = probs.mean(dim=0)                          # (E,)
            # 实际被分发的 token 比例（基于 topk_indices 的 one‑hot 近似）
            # 对每个 token，累加其 topk 索引的 one‑hot
            density = torch.zeros(num_experts, device=router_logits.device)
            for k in range(self.top_k):
                density.scatter_add_(0, topk_indices[:, k], torch.ones(num_tokens, device=router_logits.device))
            density = density / num_tokens                         # (E,)
            # 负载均衡损失 = num_experts * sum(avg_probs * density)
            load_bal_loss = num_experts * (avg_probs * density).sum()
        else:
            load_bal_loss = torch.tensor(0.0, device=router_logits.device)

        # --- 路由器 z‑loss ---
        # 鼓励 logits 的平方和不要太大，源自 DeepSeek-V3
        z_loss = router_logits.float().pow(2).mean()

        total_aux_loss = (
            self.load_balancing_loss_weight * load_bal_loss +
            self.router_z_loss_weight * z_loss
        )
        return total_aux_loss