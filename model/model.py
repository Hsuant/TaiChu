"""TaiChu 语言模型主模块。

组装所有子模块，构成完整的 Decoder-only Transformer。
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embedding import TokenEmbedding
from model.positional_encoding import RoPEPositionEncoding
from model.normalization import RMSNorm
from model.transformer_block import TransformerBlock
from model.output_head import OutputHead


@dataclass
class CausalLMOutput:
    """统一的前向传播返回值。

    Attributes:
        logits: 语言模型输出，形状 (batch, seq_len, vocab_size)
        loss: 交叉熵损失（含 MoE 辅助损失），训练时有效，否则为 None
        past_key_values: 各层的 KV 缓存，仅 use_cache=True 时返回，否则为 None
    """
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None

class TaiChuModel(nn.Module):
    """TaiChu 语言模型。

    架构: TokenEmbedding → RoPE → [TransformerBlock × num_layers] → RMSNorm → OutputHead

    Attributes:
        config: 模型配置数据类实例
        token_embedding: 词嵌入层
        rope: 旋转位置编码
        layers: TransformerBlock 的 ModuleList
        final_norm: 输出前的 RMSNorm
        output_head: 语言模型头
    """

    def __init__(self, config):
        """根据配置初始化模型。

        Args:
            config: 包含所有超参数的数据类，结构对应 YAML 文件
        """
        super().__init__()
        self.config = config

        # 词嵌入
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            dropout=config.embedding_dropout,
        )

        # 旋转位置编码
        head_dim = config.hidden_size // config.num_attention_heads
        self.rope = RoPEPositionEncoding(
            head_dim=head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

        # 若启用了 MoE，则从 config 中提取 MoEConfig（由 ConfigLoader 已解析好的对象）
        moe_config = config.moe_config

        # Transformer 层堆叠
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads
                    if config.num_key_value_heads > 0
                    else config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    ffn_type=config.ffn_type,
                    attention_dropout=config.attention_dropout,
                    hidden_dropout=config.hidden_dropout,
                    rms_norm_eps=config.rms_norm_eps,
                    moe_config=moe_config,
                )
                for _ in range(config.num_layers)
            ]
        )

        # 最终归一化层
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 输出头
        self.output_head = OutputHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            tie_weights=config.tie_word_embeddings,
        )

        # 如果配置要求绑定权重，则让 output_head 共享 embedding 的权重
        if config.tie_word_embeddings:
            self.output_head.linear.weight = self.token_embedding.embedding.weight

        # 初始化所有权重（除了已经初始化过的嵌入和输出头）
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """参数初始化策略。

        对线性层使用正态分布初始化，偏置置零；对嵌入层使用正态分布。

        Args:
            module: 子模块
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ) ->  CausalLMOutput:
        """统一的前向传播入口，支持训练与 KV 缓存推理。

        根据 past_key_values 是否为 None 自动切换模式：
        - 训练 / 完整序列推理：past_key_values=None，可提供 labels 计算损失，
          返回 (logits, loss) 或 (logits,)。
        - KV 缓存增量推理：past_key_values 不为 None，通常输入单个 token，
          返回 (logits, new_past_key_values)，无 loss。

        Args:
            input_ids: token ID 序列，训练/完整序列时形状 (batch, seq_len)，
                       增量推理时形状 (batch, cur_len)（通常 cur_len=1）。
            attention_mask: 可选的注意力掩码，训练时使用。
            labels: 语言模型标签，形状 (batch, seq_len)，仅训练/完整序列时有效。
            past_key_values: 各层上一时刻的 KV 缓存，列表长度 = num_layers，
                             每个元素为 (k, v) 元组或 None。
            use_cache: 是否返回新的 KV 缓存（增量推理时通常为 True）。

        Returns:
            根据调用模式返回不同元组：
            - 训练/完整序列： (logits, loss) 或 (logits, None)
            - 缓存推理： (logits, new_past_key_values)
        """
        batch_size, cur_len = input_ids.shape

        # ========== 1. 计算位置信息与总序列长度 ==========
        if past_key_values is not None and past_key_values[0] is not None:
            # 从缓存中获取已存储的序列长度
            past_len = past_key_values[0][0].size(1)
            total_len = past_len + cur_len
            # 增量推理模式，禁止提供 labels
            if labels is not None:
                raise ValueError("增量推理时 labels 必须为 None")
        else:
            # 完整序列前向（训练或首次推理）
            past_len = 0
            total_len = cur_len

        # ========== 2. 获取 RoPE 的 cos / sin，并截取当前步所需部分 ==========
        cos_full, sin_full = self.rope(total_len)
        # 对于完整序列，取全部位置；对于增量，取最后 cur_len 个位置对应的旋转参数
        cos = cos_full[:, past_len:total_len, :, :]   # (1, cur_len, 1, head_dim)
        sin = sin_full[:, past_len:total_len, :, :]

        # ========== 3. 词嵌入 ==========
        hidden_states = self.token_embedding(input_ids)

        # ========== 4. 逐层传递 Transformer 层 ==========
        new_past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = [] if use_cache else None
        total_aux_loss = 0.0  # MoE 辅助损失累积

        for idx, layer in enumerate(self.layers):
            # 获取本层的旧缓存（如果存在）
            past_kv = past_key_values[idx] if past_key_values is not None else None

            # 调用层的统一 forward（层需支持 past_key_value 参数）
            hidden_states, present_kv, aux_loss = layer(
                hidden_states, cos, sin,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
            )

            # 收集 MoE 辅助损失（仅训练时需要）
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss

            if use_cache:
                assert new_past_key_values is not None
                new_past_key_values.append(present_kv)

        # ========== 5. 最终归一化与输出头 ==========
        hidden_states = self.final_norm(hidden_states)
        logits = self.output_head(hidden_states)  # (batch, cur_len, vocab_size)

        # ========== 6. 计算损失（若提供 labels） ==========
        loss = None
        if labels is not None:
            # 偏移 logits 和 labels，对齐预测目标
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # 将 MoE 辅助损失直接加到总损失上（权重已在损失计算内部处理）
            loss = loss + total_aux_loss

        # ========== 7. 统一返回 CausalLMOutput 对象 ==========
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=new_past_key_values,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """自回归生成文本（使用 KV 缓存加速）。

        该函数将模型切换到 eval 模式，并利用 KV 缓存逐 token 生成。
        首次推理时传入完整 prompt，后续每次只输入最新生成的 token。

        Args:
            input_ids: 起始 token 序列，形状 (batch_size, seq_len)。
            max_new_tokens: 要生成的最大新 token 数。
            temperature: 采样温度，越高越随机。
            top_k: top‑k 采样参数，若为 None 则使用全词表 softmax 采样。

        Returns:
            包含生成结果的完整序列，形状 (batch_size, seq_len + max_new_tokens)。
        """
        self.eval()  # 切换到评估模式（关闭 dropout）
        generated = input_ids
        past_key_values = None  # 初始无缓存

        # 强制使用 eager 模式，确保每次 forward 都不会触发 CUDA Graph 的内存复用冲突
        with torch.compiler.set_stance("force_eager"):
            for _ in range(max_new_tokens):
                # 首次推理输入完整序列，之后只输入最后一个生成的 token
                if past_key_values is None:
                    cur_input = generated
                else:
                    cur_input = generated[:, -1:]

                # 调用统一 forward，启用缓存
                output = self.forward(
                    cur_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = output.logits
                past_key_values = output.past_key_values  # 类型与 past_key_values 一致

                # 取最后一个位置的 logits（用于预测下一个 token）
                next_token_logits = logits[:, -1, :] / temperature

                # 可选的 top‑k 过滤
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")

                # 计算概率并采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

                # 追加到生成序列
                generated = torch.cat([generated, next_token], dim=-1)

                # 防止超出最大位置编码长度（通常不会，因为限制了 max_new_tokens）
                if generated.size(1) > self.config.max_position_embeddings:
                    generated = generated[:, -self.config.max_position_embeddings:]

        return generated

    def configure_optimizers(self, config) -> torch.optim.Optimizer:
        """配置 AdamW 优化器，区分需要权重衰减的参数。

        Args:
            config: 训练配置（包含 learning_rate, weight_decay, beta1, beta2 等）。

        Returns:
            AdamW 优化器实例。
        """
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
        )
        return optimizer

    def get_num_params(self) -> int:
        """返回模型的总参数量（只计算可训练参数）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)