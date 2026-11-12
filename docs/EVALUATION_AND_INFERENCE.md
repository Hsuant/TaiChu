# 评估与推理

本文统一说明预训练评估、原始文本续写和终端 Chat。三者使用同一模型实现，但输入协议
和结论边界不同。

## 1. 分工

| 入口 | 输入协议 | 回答的问题 |
|---|---|---|
| `eval.eval_pretrain_model` | 验证集 Causal LM Batch | Loss、PPL、BPB、Accuracy 与固定 Prompt 生成 |
| `eval.pretrain_probe` | 原始文本前缀 | 基座 Next-Token 分布如何续写 |
| `python -m chat` | Chat Template 多轮消息 | 会话协议、上下文管理和生成工程是否工作 |

Chat 入口不会把预训练基座变成指令模型。它验证的是推理与交互基础设施，不改变权重能力。

## 2. 验证指标

### Loss

模型内部只做一次 shift：

```text
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
loss = CrossEntropy(shift_logits, shift_labels, ignore_index=-100)
```

Data Layer 保持 `labels` 与 `input_ids` 对齐，不提前移位。Packed 文档边界的目标会被
屏蔽，避免跨文档 Next-Token 监督。

### Perplexity

```text
ppl = exp(loss)
```

只在相同 Tokenizer 下可比。跨 Tokenizer 比较应使用 BPB。

### Bits per UTF-8 Byte

```text
bpb = total_nll_nats / (ln(2) × UTF-8_bytes_without_special_tokens)
```

分子是有效目标 Token 的交叉熵总和（Nats），分母是对应目标文本的 UTF-8 字节数并排除
特殊 Token。该指标减少 Tokenization 粒度变化造成的不可比性。

### Accuracy

有效 Next-Token 中 Top-1 预测与目标相同的比例。

### CE 分位数

评估器保留逐 Token Cross Entropy，并输出精确线性分位数：

- `val_ce_p50`
- `val_ce_p90`
- `val_ce_p95`
- `val_ce_max_exact`

实现使用 Chunk 与精确选择，避免把超大验证集全部常驻内存。

## 3. 独立评估

```powershell
python -m eval.eval_pretrain_model `
  --pretrain_config configs/train_pretrain_80m.yaml `
  --exp_dir <experiment-dir> `
  --artifact best `
  --generate `
  --num_generate_tokens 32
```

参数：

| 参数 | 含义 |
|---|---|
| `--pretrain_config` | 验证数据、批大小、Prompt 和日志配置 |
| `--exp_dir` | 实验根目录 |
| `--artifact best` | 读取 `best_models/best_model.pt` |
| `--artifact final` | 读取 `final_models/final_model.pt` |
| `--checkpoint` | 直接指定其他最小推理产物 |
| `--generate` | 执行固定 Prompt 生成诊断 |

输出：

```text
<experiment-dir>/evaluation/best/
├── eval_metrics.json
├── eval.log
└── generated_samples.json
```

`final` 产物写入 `evaluation/final/`。评估器会校验模型结构来自产物内嵌配置，
不要求另传模型 YAML。

## 4. 原始文本续写

```powershell
python -m eval.pretrain_probe `
  --config configs/pretrain_probe_80m.yaml `
  --prompt "清晨，阳光透过窗户洒进房间，" `
  --max-new-tokens 64
```

支持：

- 多条 `--prompt`
- 文本文件 `--prompts-file`
- JSONL 文件
- 交互模式
- JSONL 输出
- `--json`
- 流式与计时

建议：

1. 固定 Seed 和 Prefix，比较不同 Checkpoint。
2. 用 `temperature=0` 或贪心观察确定性续写。
3. 默认采样只用于多样性观察，不能当作能力结论。
4. 将生成结果与验证集 PPL/BPB 一起归档。

## 5. 终端 Chat

```powershell
python -m chat --config configs/chat_80m.yaml
```

单次请求：

```powershell
python -m chat `
  --config configs/chat_80m.yaml `
  --prompt "用三句话解释什么是语言模型。" `
  --max-new-tokens 96
```

交互命令：

| 命令 | 行为 |
|---|---|
| `/help` | 显示命令 |
| `/clear` | 清空会话 |
| `/system <文本>` | 更新系统提示并清空历史 |
| `/stats` | 显示上一轮 Token、耗时与吞吐 |
| `/exit`、`/quit` | 退出 |

上下文策略：

- `context_tokens: null` 使用模型 `max_position_embeddings`。
- 输入预算为 `context_tokens - max_new_tokens`。
- 超限时删除最早的完整 user/assistant 轮次。
- 系统提示与最新用户消息始终保留。
- 最新一轮单独超限时拒绝请求。
- 生成失败或 `Ctrl+C` 时回滚未完成轮次。

## 6. 最小推理产物

```text
best_models/best_model.pt
final_models/final_model.pt
```

包含：

- `model_state_dict`
- `model_config`
- `global_step`
- `metrics`
- `artifact_variant`
- `stop_reason`
- `format_version`
- `created_at`

不包含 Optimizer、Scheduler、Scaler、RNG、DataState 或 SwanLab 状态。完整训练恢复
必须使用 `checkpoints/` 中的文件。

## 7. 结果解释边界

- Loss/PPL/BPB 衡量验证集 Next-Token 拟合，不衡量指令遵循。
- 生成样本是诊断材料，不是 Benchmark。
- 80M 预训练基座可能重复、漂移或产生事实错误。
- 在相同 Tokenizer、验证集和 Prompt 下比较模型，禁止拼接不同设置的数字。
- 没有公开原始结果的指标用“待补充”表示。

## 8. 尚未实现

- MMLU、C-Eval、CMMLU、HumanEval、GSM8K 等下游 Benchmark。
- 指令遵循、安全对齐、工具调用和 Agent 评估。
- Base Model 与 Chat Model 的独立评测矩阵。
- 自动上传评估结果到模型卡或实验平台。
