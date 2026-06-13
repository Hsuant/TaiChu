"""TaiChu 模型超参数配置。

定义模型架构所需的所有超参数，支持从字典或 YAML 文件加载。
"""

from dataclasses import dataclass, fields
from typing import Optional, Dict, Any

from model.moe import MoEConfig


@dataclass
class ModelConfig:
    """TaiChu 语言模型的完整超参数配置。

    包含模型架构、正则化以及可选 MoE（混合专家）模块的所有参数。
    配置对象可直接传递给 TaiChuModel 的构造函数。

    Attributes（按功能分组）:
        - 基础架构: hidden_size, num_layers, num_attention_heads, ...
        - 位置编码: max_position_embeddings, rope_theta
        - 正则化: attention_dropout, hidden_dropout, embedding_dropout
        - 前馈网络类型: ffn_type, intermediate_size, hidden_act
        - 权重绑定: tie_word_embeddings
        - 可选 MoE: moe (MoEConfig 对象，当 ffn_type="moe" 时必须提供)
        - 推理优化: use_flash_attention
    """

    # ==================== 基础架构参数 ====================

    model_name: str = "TaiChu"
    """模型名称，用于日志记录和模型保存。"""

    vocab_size: int = 50304
    """词表大小。通常取 2 的幂次方倍数（如 50304 = 50272 + 32）以优化 GPU 计算。"""

    hidden_size: int = 768
    """隐藏层维度（d_model），Transformer 主干的特征宽度。"""

    num_layers: int = 12
    """Transformer 解码器层的堆叠数量。"""

    num_attention_heads: int = 12
    """多头注意力中的查询头数量。hidden_size 必须能被 num_attention_heads 整除。"""

    num_key_value_heads: int = 0
    """键/值头数量（用于 GQA 分组查询注意力）。

    - 若为 0 或等于 num_attention_heads，则为标准 MHA（多头注意力）。
    - 若小于 num_attention_heads 且为正整数，则启用 GQA，可减少 KV 缓存显存占用。
    - 通常设置为 num_attention_heads 的约数（如 1/4 或 1/2）。
    """

    intermediate_size: int = 3072
    """前馈网络（FFN）中间层维度。对于 SwiGLU/标准 FFN 有效。

    通常设置为 hidden_size 的 4 倍左右。当 ffn_type="moe" 时，此参数不用于路由专家，
    路由专家的中间维度由 moe.expert_intermediate_size 单独控制。
    """

    max_position_embeddings: int = 2048
    """最大支持序列长度。决定了 RoPE 位置编码的预计算范围，
    模型无法处理超过此长度的输入序列。
    """

    rope_theta: float = 10000.0
    """RoPE（旋转位置编码）的基频（theta）参数。

    影响位置编码的长程衰减特性。较大的 theta 值使远程衰减变慢，
    理论上可提升长文本能力，但需要更多训练数据。
    """

    rms_norm_eps: float = 1e-6
    """RMSNorm 归一化层中的 epsilon 值，防止除零错误。"""

    tie_word_embeddings: bool = True
    """是否将输入嵌入层（TokenEmbedding）与输出头（OutputHead）的权重绑定。

    绑定后可减少参数量，且通常能轻微提升训练稳定性。
    """

    # ==================== 前馈网络（FFN）相关 ====================

    ffn_type: str = "swiglu"
    """前馈网络类型，可选值：

    - "swiglu": 使用 SwiGLU 激活函数的 FFN（推荐，多数现代 LLM 采用）。
    - "standard": 标准的两层 FFN，激活函数为 GELU（用于消融实验）。
    - "moe": 使用混合专家（MoE）替换 FFN，此时必须提供 moe 配置。
    """

    hidden_act: str = "silu"
    """激活函数类型，目前仅用于兼容性预留。
    当 ffn_type="swiglu" 时，内部固定使用 SiLU（Swish）。
    """

    # ==================== 正则化参数 ====================

    attention_dropout: float = 0.0
    """注意力权重上的 Dropout 概率。预训练阶段通常设为 0.0；
    微调或小数据场景可适当增加（如 0.1）。"""

    hidden_dropout: float = 0.0
    """残差连接后、FFN 输出上的 Dropout 概率。预训练常为 0.0。"""

    embedding_dropout: float = 0.0
    """嵌入层后的 Dropout 概率。"""

    # ==================== MoE（混合专家）配置参数 ====================
    # 以下参数仅在 ffn_type == "moe" 时生效

    moe_num_experts: int = 8
    """路由专家总数。专家越多模型容量越大，但也会增加计算和通信开销。
    典型值：8, 16, 32, 64。
    仅在 ffn_type="moe" 时生效。"""

    moe_top_k: int = 2
    """每个 token 选择的路由专家数量。较大的 top_k 可提升模型容量，
    但会增加计算量。推荐值为 2（DeepSeek-V3 采用）。
    仅在 ffn_type="moe" 时生效。"""

    moe_expert_intermediate_size: int = 1536
    """每个路由专家内部的 FFN 中间层维度。通常设置为 hidden_size 的 2~4 倍，
    但可以比主模型的 intermediate_size 小，因为多个专家联合提供容量。
    仅在 ffn_type="moe" 时生效。"""

    moe_num_shared_experts: int = 1
    """共享专家数量。共享专家会被所有 token 无差别地经过（类似普通 FFN），
    用于捕捉通用特征。通常设为 0（不使用）或 1。
    仅在 ffn_type="moe" 时生效。"""

    moe_shared_expert_intermediate_size: int = 3072
    """共享专家的 FFN 中间层维度。通常比路由专家的中间层更大，
    因为共享专家需要承担更多通用模式的学习。
    仅在 ffn_type="moe" 时生效。"""

    moe_load_balancing_loss_weight: float = 0.01
    """负载均衡损失系数。该损失鼓励每个专家接收的 token 数量尽量均衡，
    避免某些专家过载而另一些专家被闲置。典型范围：0.001 ~ 0.1。
    仅在 ffn_type="moe" 时生效。"""

    moe_router_z_loss_weight: float = 0.001
    """路由器 z‑loss 系数（源自 DeepSeek-V3），用于惩罚过大的路由 logits，
    稳定门控网络训练。典型范围：0.0001 ~ 0.01。
    仅在 ffn_type="moe" 时生效。"""

    # ==================== 推理优化 ====================

    use_flash_attention: bool = True
    """是否优先使用 FlashAttention 后端进行注意力计算。

    - 若为 True，则通过 ``torch.backends.cuda.sdp_kernel`` 强制启用 FlashAttention，
      可显著降低内存占用并加速训练/推理（需硬件支持，如 Ampere 及以上 GPU）。
    - 若为 False，则回退到 PyTorch 默认的自动后端选择（Memory‑Efficient / Math）。
    """

    def __post_init__(self):
        """配置校验与后处理。

            - 若 num_key_value_heads 为 0，自动设为 num_attention_heads（即 MHA）。
            - 检查 GQA 的整除性。
            - 若启用 MoE，校验 MoE 相关参数的一致性。
            """
        # GQA 处理
        if self.num_key_value_heads <= 0:
            self.num_key_value_heads = self.num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) "
                f"必须能被 num_key_value_heads ({self.num_key_value_heads}) 整除"
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) 必须能被 num_attention_heads ({self.num_attention_heads}) 整除"
            )

        # MoE 参数校验
        if self.ffn_type == "moe":
            if self.moe_num_experts <= 0:
                raise ValueError(f"moe_num_experts 必须为正整数，当前为 {self.moe_num_experts}")
            if not (1 <= self.moe_top_k <= self.moe_num_experts):
                raise ValueError(
                    f"moe_top_k 必须在 [1, moe_num_experts] 范围内，当前为 {self.moe_top_k}"
                )
            if self.moe_expert_intermediate_size <= 0:
                raise ValueError(
                    f"moe_expert_intermediate_size 必须为正整数，当前为 {self.moe_expert_intermediate_size}"
                )
            if self.moe_num_shared_experts < 0:
                raise ValueError(f"moe_num_shared_experts 不能为负数，当前为 {self.moe_num_shared_experts}")
            if self.moe_num_shared_experts > 0 and self.moe_shared_expert_intermediate_size <= 0:
                raise ValueError(
                    f"使用共享专家时 moe_shared_expert_intermediate_size 必须为正，"
                    f"当前为 {self.moe_shared_expert_intermediate_size}"
                )
            if not (0.0 <= self.moe_load_balancing_loss_weight <= 1.0):
                raise ValueError(
                    f"moe_load_balancing_loss_weight 应在 [0,1] 内，当前为 {self.moe_load_balancing_loss_weight}"
                )
            if not (0.0 <= self.moe_router_z_loss_weight <= 1.0):
                raise ValueError(
                    f"moe_router_z_loss_weight 应在 [0,1] 内，当前为 {self.moe_router_z_loss_weight}"
                )
            # 可选警告：专家中间层过小
            if self.moe_expert_intermediate_size < self.hidden_size:
                import warnings
                warnings.warn(
                    f"moe_expert_intermediate_size ({self.moe_expert_intermediate_size}) "
                    f"小于 hidden_size ({self.hidden_size})，可能导致专家网络容量不足。",
                    UserWarning
                )

    @property
    def moe_config(self) -> Optional[MoEConfig]:
        """向后兼容属性：返回 MoEConfig 对象。

        当 ffn_type 为 "moe" 时，根据当前 MoE 平铺参数构造 MoEConfig；
        否则返回 None。
        """
        if self.ffn_type != "moe":
            return None
        return MoEConfig(
            num_experts=self.moe_num_experts,
            top_k=self.moe_top_k,
            expert_intermediate_size=self.moe_expert_intermediate_size,
            num_shared_experts=self.moe_num_shared_experts,
            shared_expert_intermediate_size=self.moe_shared_expert_intermediate_size,
            load_balancing_loss_weight=self.moe_load_balancing_loss_weight,
            router_z_loss_weight=self.moe_router_z_loss_weight,
        )


    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """从字典创建配置对象，自动展平除 moe 外的所有嵌套字典。

        支持以下 YAML 格式：
            1. 平铺字段（新格式）：
               hidden_size: 768
               num_layers: 12
               ...
            2. 任意嵌套结构（旧格式）：
               architecture:
                 hidden_size: 768
                 num_layers: 12
               regularization:
                 attention_dropout: 0.0
               activation:
                 hidden_act: "silu"
                 ffn_type: "swiglu"
               moe:
                 num_experts: 8
                 ...

        Args:
            config_dict: 配置字典（可能包含嵌套块）。

        Returns:
            ModelConfig 实例。
        """
        data = config_dict.copy()

        # 自动展平所有非 moe 的嵌套字典
        nested_keys = [k for k, v in data.items() if isinstance(v, dict) and k != "moe"]
        for key in nested_keys:
            nested_dict = data.pop(key)
            for sub_key, sub_val in nested_dict.items():
                # 嵌套字段优先级高于已存在的同名字段（如有）
                data[sub_key] = sub_val

        # 特殊处理 moe 块：将字段映射为 moe_ 前缀形式
        moe_block = data.pop("moe", None)
        if moe_block is not None and isinstance(moe_block, dict):
            field_names = {f.name for f in fields(cls)}
            for moe_key, moe_val in moe_block.items():
                flat_key = f"moe_{moe_key}"
                if flat_key in field_names:
                    data[flat_key] = moe_val
                else:
                    # 未知字段保留原样
                    data[moe_key] = moe_val

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """将配置对象转换为字典，便于保存为 YAML/JSON。

        该方法会将所有字段导出，同时将平铺的 moe_* 字段重新聚合成一个嵌套的 "moe" 块，
        使得序列化后的 YAML 保持结构清晰（与配置文件风格一致）。

        Returns:
            配置字典，其中 MoE 参数会被嵌套在 "moe" 键下，其他参数保持平铺。
        """
        result: Dict[str, Any] = {}
        # 定义平铺字段到嵌套字段的映射（moe_前缀 -> 去掉前缀）
        moe_field_mapping = {
            "moe_num_experts": "num_experts",
            "moe_top_k": "top_k",
            "moe_expert_intermediate_size": "expert_intermediate_size",
            "moe_num_shared_experts": "num_shared_experts",
            "moe_shared_expert_intermediate_size": "shared_expert_intermediate_size",
            "moe_load_balancing_loss_weight": "load_balancing_loss_weight",
            "moe_router_z_loss_weight": "router_z_loss_weight",
        }
        moe_dict = {}
        # 遍历所有字段
        for field_name, value in self.__dict__.items():
            if field_name.startswith("moe_"):
                # 属于 MoE 相关字段，收集到 moe_dict 中
                if field_name in moe_field_mapping:
                    nested_name = moe_field_mapping[field_name]
                    moe_dict[nested_name] = value
                else:
                    # 未知 moe_ 字段，直接放入根字典并警告
                    result[field_name] = value
            else:
                # 非 MoE 字段直接放入根字典
                # 跳过私有字段（以 _ 开头）
                if not field_name.startswith("_"):
                    result[field_name] = value
        # 如果有任何 MoE 参数被设置，则添加 moe 嵌套块
        if moe_dict:
            result["moe"] = moe_dict
        return result