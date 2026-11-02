# TaiChu 训练与验证指标审计

> 本文是 [`AUDIT_REPORT.md`](AUDIT_REPORT.md) 的指标与损失专项附录。
> 总体问题状态、测试证据和剩余风险以审计总报告为准。

审计日期：2026-10-26  
环境：Python 3.13、PyTorch 2.10、CPU only

## 1. 审计结论

模型主损失的计算方式正确：logits 使用 `[:, :-1]`，labels 使用 `[:, 1:]`，
交叉熵忽略 `-100`，loss 对有效 next-token 目标取平均。数据管线不提前移位
labels，并在 padding 位置写入 `-100`。

验证指标已完成以下修正：

| 指标 | 原状态 | 当前处理 |
|---|---|---|
| `val_loss` | 原本已按有效 Token 加权 | 统一由共享 token 统计函数计算 |
| `val_perplexity` | `loss >= 100` 时人为返回 `inf` | 非有限或溢出时返回 `None` |
| `val_bpb` | 缺失 | 新增，排除特殊 Token 后按 UTF-8 字节计算 |
| `val_accuracy` | 无有效 Token 时返回 `0.0` | 无定义时返回 `None` |
| CE 分位数 | 空集返回 `0.0`，训练器使用固定范围近似 | 使用精确线性分位数 |
| 空白提示 PPL | 空序列返回 `inf`，单 Token 会异常 | 改为可选显式 BOS → EOS，无定义时 `None` |
| CPU DDP 聚合 | 不执行 all-reduce | Gloo/NCCL 均聚合 |
| 分布式分位数 | 固定 0 到 20 的直方图 | 汇总本地逐 Token 损失后精确计算 |
| 训练吞吐 | 使用最大序列长度估算 | 使用实际 padded Token 数 |
| MFU | CPU 环境使用虚构的 330 TFLOPS | 峰值未定义时禁用并显示 `N/A` |
| 梯度指标 | 标为 Gradient Noise Scale，但公式不是标准 GNS | 改名为梯度范数平方/方差比 |

## 2. 指标定义

设第 `i` 个有效 next-token 目标的交叉熵为 `ce_i`，有效目标数为 `N`：

```text
val_loss = sum(ce_i) / N
val_perplexity = exp(val_loss)
val_accuracy = correct_next_token_count / N
```

BPB 仅统计非特殊 Token 的目标：

```text
content_nll_nats = sum(ce_i for i in content_targets)
bits = content_nll_nats / ln(2)
val_bpb = bits / utf8_bytes(decoded_content_targets)
```

PPL 只在相同 tokenizer 下可直接比较。跨 tokenizer 或词表配置比较时应使用 BPB。

## 3. 空前缀指标

“空白字符串 PPL”不再作为默认指标。当前 tokenizer 对空字符串编码为空序列，
没有 next-token 目标，PPL 数学上未定义。

需要空请求诊断时，使用显式的两 Token 序列：

```text
[BOS, EOS]
```

指标只包含一次转移：

```text
empty_prefix_eos_ce = -log P(EOS | BOS)
empty_prefix_eos_ppl = exp(empty_prefix_eos_ce)
```

该指标默认关闭。BOS/EOS 不存在、两者相同或模型输出非有限时返回 `None`，
不会返回 `inf`，也不会参与最佳模型选择或早停。

## 4. MoE 辅助损失

主语言模型损失与 MoE 辅助损失分开计算后再相加。

负载均衡损失参考 Switch-style 形式：

```text
load_balance = num_experts * sum_i(mean_router_prob_i * assignment_fraction_i)
assignment_fraction_i = selected_count_i / (num_tokens * top_k)
```

路由器 z-loss 使用：

```text
z_loss = mean(logsumexp(router_logits, dim=experts)^2)
```

之前的实现使用 `mean(router_logits^2)`，不等价于标准 router z-loss，已修正。

## 5. 指标完整性

当前默认验证集包含：

- token 加权 loss
- perplexity
- bits-per-byte
- next-token top-1 accuracy
- 逐 Token CE 的 p50/p90/p95
- 有效 Token 数、目标字节数和批次数
- 可选 BOS → EOS 空前缀诊断

尚未包含、需要独立评测框架的能力：

- MMLU、C-Eval、代码、数学等下游 benchmark
- 指令跟随、事实性和安全性评测
- 实际 GPU 上的 MFU、FlashAttention 和 DDP 性能

## 6. 局限

- 当前环境为 CPU，未验证 CUDA AMP、GPU FlashAttention 和多卡 NCCL。
- 训练期间分布式分位数使用 `all_gather_object` 汇总逐 Token 损失；结果精确，
  但验证集极大时会增加 CPU 内存和通信开销。
- MFU 使用 `6 × 参数量 × 实际 padded Token 数` 的近似公式，不包含注意力
  通信和所有算子开销，应视为估算值。
- `GradientNoiseScale` 类名为兼容保留；当前日志已改名为
  `grad_norm_sq_variance_ratio`，不再宣称它是标准 per-example GNS。
