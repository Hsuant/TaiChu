"""TaiChu 语言模型主模块。

组装所有子模块，构成完整的 Decoder-only Transformer。
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embedding import TokenEmbedding
from model.attention import build_attention_mask, normalize_padding_mask
from model.positional_encoding import RoPEPositionEncoding
from model.normalization import RMSNorm
from model.transformer_block import TaiChuBlock
from model.output_head import TaiChuOutputHead


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

        # Transformer 层堆叠
        self.layers = nn.ModuleList([
            TaiChuBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # 最终归一化层
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 输出头
        self.output_head = TaiChuOutputHead(
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
            attention_mask: 可选的二维 Padding Mask，形状
                (batch_size, total_len)。非零/True 表示有效 Token。
            labels: 语言模型标签，形状 (batch, seq_len)，仅训练/完整序列时有效。
            past_key_values: 各层上一时刻的 KV 缓存，列表长度 = num_layers，
                             每个元素为 (k, v) 元组或 None。
            use_cache: 是否返回新的 KV 缓存（增量推理时通常为 True）。

        Returns:
            根据调用模式返回不同元组：
            - 训练/完整序列： (logits, loss) 或 (logits, None)
            - 缓存推理： (logits, new_past_key_values)
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids 必须为二维张量，当前形状为 {tuple(input_ids.shape)}")
        batch_size, cur_len = input_ids.shape
        if cur_len == 0:
            raise ValueError("input_ids 的序列长度不能为 0")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError(
                f"past_key_values 长度应为 {len(self.layers)}，"
                f"当前为 {len(past_key_values)}"
            )
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError(
                f"labels 形状 {tuple(labels.shape)} 必须与 input_ids 形状 "
                f"{tuple(input_ids.shape)} 一致"
            )
        if labels is not None:
            if labels.dtype != torch.long:
                raise ValueError("labels 必须使用 torch.long")
            if labels.device != input_ids.device:
                raise ValueError("labels 与 input_ids 必须位于同一设备")
            if cur_len < 2:
                raise ValueError("训练损失至少需要两个 Token 才能构造 next-token 目标")
            if not (labels[..., 1:] != -100).any():
                raise ValueError("batch 中没有任何可监督的 next-token 标签")
            valid_label_mask = labels != -100
            if valid_label_mask.any() and (
                (labels[valid_label_mask] < 0)
                | (labels[valid_label_mask] >= self.config.vocab_size)
            ).any():
                raise ValueError("labels 中的有效 Token ID 超出词表范围")
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask 仅支持二维 Padding Mask，"
                    f"当前为 {attention_mask.dim()} 维"
                )
            if attention_mask.size(0) != batch_size:
                raise ValueError("attention_mask 的 batch 维度与 input_ids 不一致")
            if labels is not None:
                padding_positions = (attention_mask == 0).to(labels.device)
                if (labels[padding_positions] != -100).any():
                    raise ValueError(
                        "attention_mask 为 0 的位置必须使用 labels=-100"
                    )

        # ========== 1. 计算位置信息与总序列长度 ==========
        first_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = (
            past_key_values[0] if past_key_values else None
        )
        if first_cache is not None:
            # 从缓存中获取已存储的序列长度
            # 缓存 k 的布局为 (batch, num_kv_heads, total_len, head_dim)，
            # 序列长度位于第 2 维。
            past_len = first_cache[0].size(2)
            total_len = past_len + cur_len
            for layer_idx, cache in enumerate(past_key_values or []):
                if cache is None:
                    raise ValueError(
                        f"past_key_values[{layer_idx}] 为 None，但其他层缓存存在"
                    )
                cache_k, cache_v = cache
                if cache_k.dim() != 4 or cache_v.dim() != 4:
                    raise ValueError("KV cache 的 K/V 必须为四维张量")
                if cache_k.size(2) != past_len or cache_v.size(2) != past_len:
                    raise ValueError("各层 KV cache 的序列长度必须一致")
                if cache_k.size(0) != batch_size or cache_v.size(0) != batch_size:
                    raise ValueError("KV cache 的 batch 维度与当前输入不一致")
            # 增量推理模式，禁止提供 labels
            if labels is not None:
                raise ValueError("增量推理时 labels 必须为 None")
        else:
            if past_key_values is not None and any(
                cache is not None for cache in past_key_values
            ):
                raise ValueError("past_key_values[0] 为 None，但其他层缓存存在")
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
        mask = build_attention_mask(
            q_len=cur_len,
            total_len=total_len,
            batch_size=batch_size,
            attention_mask=attention_mask,
            device=hidden_states.device,
        )

        # ========== 4. 逐层传递 Transformer 层 ==========
        new_past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = [] if use_cache else None
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)  # MoE 辅助损失累积

        for idx, layer in enumerate(self.layers):
            # 获取本层的旧缓存（如果存在）
            past_kv = past_key_values[idx] if past_key_values is not None else None

            # 调用层的统一 forward（层需支持 past_key_value 参数）
            hidden_states, present_kv, aux_loss = layer(
                hidden_states, cos, sin,
                mask=mask,
                past_key_value=past_kv,
                use_cache=use_cache,
            )

            # 收集 MoE 辅助损失（仅训练时需要）
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
            # 将 MoE 辅助损失直接加到总损失上（权重已在损失计算内部处理），
            # 并保持与主损失相同的数据类型
            loss = loss + total_aux_loss.to(loss.dtype)

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
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """自回归生成文本（使用 KV 缓存加速）。

        该函数将模型切换到 eval 模式，并利用 KV 缓存逐 token 生成。
        首次推理时传入完整 prompt，后续每次只输入最新生成的 token。

        Args:
            input_ids: 起始 token 序列，形状 (batch_size, seq_len)。
            max_new_tokens: 要生成的最大新 token 数。
            temperature: 采样温度，越高越随机；<= 0 时退化为贪心解码。
            top_k: top‑k 采样参数，为 None 时不过滤。
            top_p: nucleus 采样阈值，保留累积概率不超过该值的最小 token 集合，
                   为 None 或 1.0 时不过滤。
            eos_token_id: 结束符 token id，对应行生成结束后停止推进，
                   整批全部完成时提前结束。
            attention_mask: 可选形状 ``(batch, seq_len)`` 的 padding mask，
                    仅接受 bool 或整数张量，非零/True 表示有效 Token。
                    增量解码时会自动延长。

        Returns:
            包含生成结果的完整序列，形状 (batch_size, seq_len + 实际生成数)。
        """
        if input_ids.dim() != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids 必须为非空的二维张量")
        if input_ids.size(1) > self.config.max_position_embeddings:
            raise ValueError(
                f"输入长度 {input_ids.size(1)} 超过最大上下文 "
                f"{self.config.max_position_embeddings}"
            )
        if input_ids.size(1) + max_new_tokens > self.config.max_position_embeddings:
            raise ValueError(
                f"输入长度 {input_ids.size(1)} + max_new_tokens {max_new_tokens} "
                f"超过最大上下文 {self.config.max_position_embeddings}"
            )
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens 不能为负数")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        if top_p is not None and not (0.0 < top_p <= 1.0):
            raise ValueError("top_p 必须在 (0, 1] 范围内")
        if eos_token_id is not None and not (0 <= eos_token_id < self.config.vocab_size):
            raise ValueError("eos_token_id 超出词表范围")

        key_attention_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            key_attention_mask = normalize_padding_mask(
                attention_mask,
                batch_size=input_ids.size(0),
                total_len=input_ids.size(1),
                device=input_ids.device,
            )

        was_training = self.training
        self.eval()
        try:
            generated = input_ids
            past_key_values = None
            finished = torch.zeros(
                (input_ids.size(0), 1), dtype=torch.bool, device=input_ids.device
            )

            # 强制使用 eager 模式，确保每次 forward 都不会触发 CUDA Graph 的内存复用冲突
            with torch.compiler.set_stance("force_eager"):
                for _ in range(max_new_tokens):
                    # 首次推理输入完整序列，之后只输入最后一个生成的 token
                    cur_input = generated if past_key_values is None else generated[:, -1:]

                    output = self.forward(
                        cur_input,
                        attention_mask=key_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = output.logits
                    past_key_values = output.past_key_values
                    next_token_logits = logits[:, -1, :]

                    if temperature > 0:
                        filtered_logits = self._filter_logits_for_sampling(
                            next_token_logits / temperature,
                            top_k=top_k,
                            top_p=top_p,
                        )
                        probs = F.softmax(filtered_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = next_token_logits.argmax(dim=-1, keepdim=True)

                    # 达到结束符的行停止推进，后续统一填充结束符直至整批完成
                    if eos_token_id is not None:
                        finished = finished | (next_token == eos_token_id)
                        next_token = torch.where(
                            finished,
                            torch.full_like(next_token, eos_token_id),
                            next_token,
                        )

                    generated = torch.cat([generated, next_token], dim=-1)
                    if key_attention_mask is not None:
                        current_mask: torch.Tensor = key_attention_mask
                        key_attention_mask = F.pad(
                            current_mask,
                            (0, 1),
                            value=True,
                        )

                    if eos_token_id is not None and finished.all():
                        break
            return generated
        finally:
            self.train(was_training)

    @staticmethod
    def _filter_logits_for_sampling(
        logits: torch.Tensor,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """按 Top-K 与 Top-P 过滤二维 logits，每一行至少保留一个候选。"""
        filtered_logits = logits.clone()
        if top_k is not None:
            top_k_eff = min(top_k, filtered_logits.size(-1))
            top_values, _ = torch.topk(filtered_logits, top_k_eff)
            filtered_logits = filtered_logits.masked_fill(
                filtered_logits < top_values[:, [-1]],
                float("-inf"),
            )

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(filtered_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            remove = remove.scatter(-1, sorted_idx, remove)
            filtered_logits = filtered_logits.masked_fill(remove, float("-inf"))
        return filtered_logits

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
