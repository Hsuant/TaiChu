"""TaiChu 语言模型主模块。

组装所有子模块，构成完整的 Decoder-only Transformer。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embedding import TokenEmbedding
from model.attention import build_attention_mask, normalize_padding_mask
from model.positional_encoding import RoPEPositionEncoding
from model.normalization import RMSNorm
from model.transformer_block import TaiChuBlock
from model.output_head import TaiChuOutputHead


KVCache = List[Optional[Tuple[torch.Tensor, torch.Tensor]]]


@dataclass
class CausalLMOutput:
    """TaiChu causal language model unified forward output.

    Attributes:
        logits: Language modeling logits with shape
            ``(batch, output_seq_len, vocab_size)``. ``None`` when not requested.
        loss: Final optimization objective, including CE and auxiliary losses.
        loss_sum: Sum of valid next-token CE. Never includes auxiliary losses.
        num_loss_tokens: Number of valid next-token targets.
        token_losses: Detached per-token CE in flattened valid-token layout.
        correct_predictions: Number of correct valid next-token predictions.
        aux_loss: Combined auxiliary optimization loss, excluding CE.
        loss_components: Named primary and auxiliary loss components.
        past_key_values: Updated autoregressive cache, or ``None`` when disabled.
    """
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    loss_sum: Optional[torch.Tensor] = None
    num_loss_tokens: Optional[torch.Tensor] = None
    token_losses: Optional[torch.Tensor] = None
    correct_predictions: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    loss_components: Optional[Dict[str, torch.Tensor]] = None
    past_key_values: Optional[Any] = None

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
        segment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
        use_cache: bool = False,
        *,
        return_logits: bool = True,
        validate_inputs: bool = True,
        return_token_statistics: bool = False,
    ) -> CausalLMOutput:
        """统一的前向传播入口，支持训练与 KV 缓存推理。

        根据 past_key_values 是否为 None 自动切换模式：
        - 训练 / 完整序列推理：past_key_values=None，可提供 labels 计算损失。
        - KV 缓存增量推理：past_key_values 为完整的逐层 Cache，通常输入单个
          token，返回新的 Cache，且不计算 loss。

        ``past_key_values=None`` 表示没有缓存对象；长度与层数一致且全为
        ``None`` 的列表表示空 Cache。空列表不是合法的 Cache 对象。

        Args:
            input_ids: token ID 序列，训练/完整序列时形状 (batch, seq_len)，
                       增量推理时形状 (batch, cur_len)（通常 cur_len=1）。
            attention_mask: 可选的二维 Padding Mask，形状
                (batch_size, total_len)。非零/True 表示有效 Token。
            segment_ids: 可选的二维 Attention Group ID，形状
                (batch_size, seq_len)。不同 Group 之间不可互相 Attention。
            position_ids: 可选的位置编号，形状 (batch_size, seq_len)。
                Packing 后可按 Attention Group 重置位置。
            labels: 语言模型标签，形状 (batch, seq_len)，仅训练/完整序列时有效。
            past_key_values: 各层上一时刻的 KV 缓存，列表长度 = num_layers，
                             每个元素为 (k, v) 元组或 None。
            use_cache: 是否返回新的 KV 缓存（增量推理时通常为 True）。
            return_logits: 是否向调用方返回完整 logits。提供 labels 时即使设为
                False，内部仍会计算完整 logits 用于损失计算。
            validate_inputs: 是否执行需要设备同步的标签范围与 padding 校验。
                可信数据管线可在热路径关闭；默认保持完整校验。
            return_token_statistics: 是否返回逐 token CE 与 argmax 正确数。

        Returns:
            所有模式均返回 :class:`CausalLMOutput`。未计算或未请求的统计字段为
            ``None``；训练时 ``loss_sum`` 仅包含 Token CE，``aux_loss``
            独立保存辅助优化目标。
        """
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids 必须为 torch.Tensor")
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids 必须为二维张量，当前形状为 {tuple(input_ids.shape)}")
        batch_size, cur_len = input_ids.shape
        if batch_size == 0 or cur_len == 0:
            raise ValueError("input_ids 的 batch 和序列长度必须非空")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError(
                f"past_key_values 长度应为 {len(self.layers)}，"
                f"当前为 {len(past_key_values)}"
            )
        if labels is not None and not isinstance(labels, torch.Tensor):
            raise TypeError("labels 必须为 torch.Tensor 或 None")
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
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError("attention_mask 必须为 torch.Tensor 或 None")
            if attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask 仅支持二维 Padding Mask，"
                    f"当前为 {attention_mask.dim()} 维"
                )
            if attention_mask.size(0) != batch_size:
                raise ValueError("attention_mask 的 batch 维度与 input_ids 不一致")
        for name, value in (
            ("segment_ids", segment_ids),
            ("position_ids", position_ids),
        ):
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} 必须为 torch.Tensor 或 None")
            if value.shape != input_ids.shape:
                raise ValueError(
                    f"{name} 形状 {tuple(value.shape)} 必须与 input_ids "
                    f"形状 {tuple(input_ids.shape)} 一致"
                )
            if value.dtype != torch.long:
                raise ValueError(f"{name} 必须使用 torch.long")
            if value.device != input_ids.device:
                raise ValueError(f"{name} 与 input_ids 必须位于同一设备")

        # ========== 1. 计算位置信息与总序列长度 ==========
        caches = list(past_key_values) if past_key_values is not None else []
        populated_caches = [cache is not None for cache in caches]
        if any(populated_caches) and not all(populated_caches):
            raise ValueError("past_key_values 中的缓存必须全部存在或全部为 None")

        if any(populated_caches):
            first_cache = next(cache for cache in caches if cache is not None)
            if (
                not isinstance(first_cache, (tuple, list))
                or len(first_cache) != 2
                or not all(isinstance(tensor, torch.Tensor) for tensor in first_cache)
            ):
                raise ValueError("KV cache 的每一层必须为 (k, v) Tensor 元组")

            cache_k, cache_v = first_cache
            if cache_k.dim() != 4 or cache_v.dim() != 4:
                raise ValueError("KV cache 的 K/V 必须为四维张量")
            if cache_k.shape != cache_v.shape:
                raise ValueError("KV cache 的 K/V 形状必须一致")

            expected_head_dim = (
                self.config.hidden_size // self.config.num_attention_heads
            )
            if (
                cache_k.size(1) != self.config.num_key_value_heads
                or cache_k.size(3) != expected_head_dim
            ):
                raise ValueError("KV cache 的 head 维度与模型配置不一致")
            if cache_k.size(0) != batch_size:
                raise ValueError("KV cache 的 batch 维度与当前输入不一致")
            if cache_k.size(2) <= 0:
                raise ValueError("KV cache 的序列长度必须为正整数")

            past_len = cache_k.size(2)
            total_len = past_len + cur_len
            for layer_idx, cache in enumerate(caches):
                assert cache is not None
                if (
                    not isinstance(cache, (tuple, list))
                    or len(cache) != 2
                    or not all(isinstance(tensor, torch.Tensor) for tensor in cache)
                ):
                    raise ValueError(
                        f"past_key_values[{layer_idx}] 必须为 (k, v) Tensor 元组"
                    )
                cache_k, cache_v = cache
                if cache_k.dim() != 4 or cache_v.dim() != 4:
                    raise ValueError("KV cache 的 K/V 必须为四维张量")
                if cache_k.shape != cache_v.shape:
                    raise ValueError("KV cache 的 K/V 形状必须一致")
                if cache_k.shape != first_cache[0].shape:
                    raise ValueError("各层 KV cache 的形状必须一致")
                if (
                    cache_k.device != input_ids.device
                    or cache_v.device != input_ids.device
                ):
                    raise ValueError("KV cache 的 device 与当前输入不一致")
                if cache_v.device != cache_k.device:
                    raise ValueError("KV cache 的 K/V device 必须一致")
                if cache_v.dtype != cache_k.dtype:
                    raise ValueError("KV cache 的 K/V dtype 必须一致")
                if (
                    cache_k.dtype != first_cache[0].dtype
                    or cache_v.dtype != first_cache[1].dtype
                ):
                    raise ValueError("各层 KV cache 的 dtype 必须一致")
            if labels is not None:
                raise ValueError("增量推理时 labels 必须为 None")
            if segment_ids is not None or position_ids is not None:
                raise ValueError(
                    "增量推理暂不支持 segment_ids 或 position_ids"
                )
        else:
            # 完整序列前向（训练或首次推理）
            past_len = 0
            total_len = cur_len
            if labels is not None and cur_len < 2:
                raise ValueError("训练损失至少需要两个 Token 才能构造 next-token 目标")

        if attention_mask is not None and attention_mask.size(1) != total_len:
            raise ValueError(
                "attention_mask 的序列长度必须等于当前输入长度与 Cache 长度之和，"
                f"期望 {total_len}，当前为 {attention_mask.size(1)}"
            )

        if labels is not None and validate_inputs:
            if not labels[..., 1:].ne(-100).any():
                raise ValueError(
                    "batch 中没有任何可监督的 next-token 标签；"
                    "训练数据管线应过滤该 Batch"
                )
            valid_label_mask = labels != -100
            if valid_label_mask.any() and (
                (labels[valid_label_mask] < 0)
                | (labels[valid_label_mask] >= self.config.vocab_size)
            ).any():
                raise ValueError("labels 中的有效 Token ID 超出词表范围")
            if attention_mask is not None:
                padding_positions = (attention_mask == 0).to(labels.device)
                if (labels[padding_positions] != -100).any():
                    raise ValueError(
                        "attention_mask 为 0 的位置必须使用 labels=-100"
                    )

        # ========== 2. 获取 RoPE 的 cos / sin，并截取当前步所需部分 ==========
        if position_ids is None:
            cos_full, sin_full = self.rope(total_len)
            # 对于完整序列，取全部位置；对于增量，取最后 cur_len 个位置对应的旋转参数
            cos = cos_full[:, past_len:total_len, :, :]
            sin = sin_full[:, past_len:total_len, :, :]
        else:
            max_position = int(position_ids.max().item())
            if max_position >= self.config.max_position_embeddings:
                raise ValueError(
                    f"position_ids 超出模型最大位置 {self.config.max_position_embeddings}"
                )
            if int(position_ids.min().item()) < 0:
                raise ValueError("position_ids 不能包含负数")
            cos = self.rope.cos_cached[0][position_ids]
            sin = self.rope.sin_cached[0][position_ids]

        # ========== 3. 词嵌入 ==========
        hidden_states = self.token_embedding(input_ids)
        mask = build_attention_mask(
            q_len=cur_len,
            total_len=total_len,
            batch_size=batch_size,
            attention_mask=attention_mask,
            device=hidden_states.device,
            segment_ids=segment_ids,
        )

        # ========== 4. 逐层传递 Transformer 层 ==========
        new_past_key_values: Optional[KVCache] = [] if use_cache else None
        moe_aux_component_sums: Dict[str, torch.Tensor] = {}

        for idx, layer in enumerate(self.layers):
            # 获取本层的旧缓存（如果存在）
            past_kv = past_key_values[idx] if past_key_values is not None else None

            # 调用层的统一 forward（层需支持 past_key_value 参数）
            hidden_states, present_kv, layer_aux_components = layer(
                hidden_states, cos, sin,
                mask=mask,
                past_key_value=past_kv,
                use_cache=use_cache,
            )

            if labels is not None and layer.is_moe:
                for name, component in layer_aux_components.items():
                    if not isinstance(component, torch.Tensor):
                        raise TypeError(
                            f"TransformerBlock[{idx}] 的 {name} 必须为 Tensor"
                        )
                    if component.dim() != 0 or not component.is_floating_point():
                        raise ValueError(
                            f"TransformerBlock[{idx}] 的 {name} 必须为浮点标量"
                        )
                    moe_aux_component_sums[name] = (
                        component
                        if name not in moe_aux_component_sums
                        else moe_aux_component_sums[name] + component
                    )

            if use_cache:
                if present_kv is None:
                    raise RuntimeError(
                        f"TransformerBlock[{idx}] 未在 use_cache=True 时返回 Cache"
                    )
                assert new_past_key_values is not None
                new_past_key_values.append(present_kv)

        aux_loss = None
        for component in moe_aux_component_sums.values():
            aux_loss = component if aux_loss is None else aux_loss + component

        # ========== 5. 最终归一化与输出头 ==========
        hidden_states = self.final_norm(hidden_states)
        logits = (
            self.output_head(hidden_states)
            if labels is not None or return_logits
            else None
        )
        returned_logits = logits if return_logits else None

        # ========== 6. 计算损失（若提供 labels） ==========
        loss = None
        loss_sum = None
        num_loss_tokens = None
        token_losses = None
        correct_predictions = None
        loss_components = None
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = shift_labels.ne(-100)
            num_loss_tokens = valid_mask.sum(dtype=torch.long)
            assert logits is not None
            shift_logits = logits[..., :-1, :].contiguous()
            flat_shift_logits = shift_logits.float().reshape(
                -1,
                shift_logits.size(-1),
            )
            flat_shift_labels = shift_labels.reshape(-1)
            if return_token_statistics:
                per_token_losses = F.cross_entropy(
                    flat_shift_logits,
                    flat_shift_labels,
                    ignore_index=-100,
                    reduction="none",
                )
                flat_valid_mask = valid_mask.view(-1)
                token_losses = per_token_losses[flat_valid_mask].detach()
                loss_sum = per_token_losses.sum()
                correct_predictions = (
                    shift_logits.argmax(dim=-1).eq(shift_labels) & valid_mask
                ).sum(dtype=torch.long).detach()
            else:
                cross_entropy_sum = F.cross_entropy(
                    flat_shift_logits,
                    flat_shift_labels,
                    ignore_index=-100,
                    reduction="sum",
                )
                loss_sum = cross_entropy_sum
            ce_loss = loss_sum / num_loss_tokens.clamp_min(1).to(loss_sum.dtype)
            loss = ce_loss
            loss_components = {"ce_loss": ce_loss}
            if aux_loss is not None:
                loss = loss + aux_loss.to(dtype=loss.dtype)
                loss_components.update(moe_aux_component_sums)
                loss_components["moe_aux_loss"] = aux_loss

        # ========== 7. 统一返回 CausalLMOutput 对象 ==========
        return CausalLMOutput(
            logits=returned_logits,
            loss=loss,
            loss_sum=loss_sum,
            num_loss_tokens=num_loss_tokens,
            token_losses=token_losses,
            correct_predictions=correct_predictions,
            aux_loss=aux_loss,
            loss_components=loss_components,
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
