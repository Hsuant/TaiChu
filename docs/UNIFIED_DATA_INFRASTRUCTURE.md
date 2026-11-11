# TaiChu 统一 LLM 训练数据基础设施

## 1. 设计结论

TaiChu 的训练数据路径统一为：

```text
Raw Dataset
  -> Schema Validation
  -> Task Preprocessing Adapter
  -> Tokenization / Mask / Metadata
  -> Versioned Binary Storage
  -> Indexed mmap Reader
  -> Sampler / Distributed Sampler
  -> Optional Dataset Mixer
  -> Task-selected Packing
  -> Task-selected Batch Builder
  -> Model Input Adapter
  -> Training Engine
```

统一的是存储、索引、读取、分片、采样、混流、恢复和监控。不同训练任务保留独立的
Sample Schema、Preprocessing Adapter、Packing 策略、Batch Builder 和 Model Input
Adapter。

项目不再维护旧 `dataset.pretrain` 加载器、旧 flat data 配置、旧 Parquet
训练热路径或兼容模式。当前只保留一条数据基础设施主路径。

## 2. LLM 生命周期任务矩阵

| 阶段或任务 | 样本单位 | 核心语义 | Pair / Group | 在线数据 | 主要输入契约 | Packing |
|---|---|---|---|---|---|---|
| Pretraining | Document / Packed Sequence | 全 Token 因果语言建模 | 否 | 否 | `PretrainingSample` | Stream |
| Continued Pretraining | Document / Packed Sequence | 领域继续预训练 | 否 | 否 | `PretrainingSample` | Stream |
| SFT | Multi-turn Conversation | Assistant-only Loss、Prompt Mask | 否 | 否 | `SFTSequenceSample` | Sample-isolated |
| Instruction Tuning | Conversation | 与 SFT 相同的数据语义 | 否 | 否 | `SFTSequenceSample` | Sample-isolated |
| Preference / DPO / IPO / KTO | Prompt + Chosen + Rejected | Pair 不可拆分、Preference Label | Pair | 否 | `PreferenceSample` | 关闭 |
| Reward Model | Prompt + Response | Sequence Reward / Ranking Label | 可选 | 可扩展 | `RewardSample` | 关闭 |
| RLHF / PPO | Trajectory | Reward、Advantage、Old/Ref LogProb、Value | Candidate | 是 | `TrajectorySample` | 关闭 |
| GRPO | Candidate Group | Group ID、候选保留、Advantage | Candidate Group | 是 | `TrajectorySample` | 关闭 |
| Critic | Trajectory | Value Target、Response Mask | 否 | 是 | `TrajectorySample` | 关闭 |
| Multimodal | Text + Image Ref/Tensor | 模态引用与 Token 对齐 | 否 | 可扩展 | `MultimodalSample` | 由模型能力决定 |

DPO、IPO、KTO、PPO、GRPO 和 RLHF 当前实现的是数据接口与 Batch 契约，不包含这些
算法的优化目标、Reference Model 管理或采样环境。

## 3. 数据表示

### 3.1 公共字段

所有 Sample 继承 `CommonSample`：

```python
class CommonSample:
    sample_id: str
    dataset_id: str
    source_id: str
    metadata: dict
```

`sample_type` 和 `array_fields` 是类级契约。`array_fields` 中声明的字段写入二进制
数组区，其余字段写入 JSON Metadata。

### 3.2 任务 Schema

| Schema | 关键字段 |
|---|---|
| `PretrainingSample` | `input_ids`, `labels`, `attention_mask`, `document_ids` |
| `SFTSequenceSample` | `input_ids`, `labels`, `loss_mask`, `segment_ids`, `messages` |
| `PreferenceSample` | `prompt_input_ids`, `chosen_input_ids`, `rejected_input_ids`, 两侧 Mask |
| `RewardSample` | `prompt_input_ids`, `response_input_ids`, `reward_label`, `ranking_label` |
| `TrajectorySample` | `rewards`, `advantages`, `old_log_probs`, `reference_log_probs`, `values` |
| `MultimodalSample` | `input_ids`, `image_refs`, `image_pixel_values`, `image_shape` |

Schema 不把 Preference、Reward 或 Trajectory 压缩成普通 `input_ids/labels`。任务语义
不会因存储格式而丢失。

## 4. 数据生命周期

### 4.1 Offline

```text
JSONL / JSONL.GZ / Parquet
  -> schema validation
  -> task adapter
  -> optional cleaning and exact deduplication
  -> tokenization
  -> task metadata and masks
  -> binary shard + index
  -> manifest + content hash
```

Offline Preparation 入口：

```powershell
python -m dataset.offline.prepare `
  --config configs/prepare_pretrain_data.yaml
```

### 4.2 Runtime

```text
manifest.json
  -> BinaryDatasetReader
  -> IndexedTokenDataset
  -> DistributedSampler / RandomSampler
  -> DataLoader workers + prefetch
  -> packing collator
  -> batch builder
  -> model input adapter
```

训练热路径不读取原始 JSONL，不执行 Tokenization，不做文本清洗，也不解析普通 Parquet。

### 4.3 Online

RLHF、PPO 和 GRPO 可在运行时接入：

1. 离线 Binary Dataset。
2. 当前 Policy 动态生成的 Sample。
3. `ExperienceBuffer` 中的 Rollout。

`HybridSampleSource` 负责组合这三类来源。在线样本直接使用
`TrajectorySample`，与离线轨迹共享序列化和 Batch 契约。

## 5. 模块边界

```text
dataset/
├── schema/          # Common + task-specific sample contracts
├── storage/         # versioned manifest + mmap binary reader/writer
├── offline/
│   ├── adapters/    # task semantics and masks
│   ├── prepare.py   # offline orchestration + atomic output
│   ├── pretraining/ # config, dedup, segmentation, output, reporting, orchestration
│   ├── readers.py   # raw JSONL / GZ / Parquet input
│   └── cleaning.py  # shared cleaning and deduplication
├── sampling/        # random, distributed, weighted, mixer, shuffle buffer
├── packing/         # stream and sample-isolated strategies
├── batching/        # task batch builders, padding, segment alignment
├── adapters/        # model input contracts
├── runtime/         # DataLoader, online buffer, resume state
├── metrics/         # data pipeline counters
├── config.py
└── tasks.py         # extensible task component registry
```

依赖方向：

```text
schema
  <- storage
  <- offline
  <- sampling
  <- packing
  <- batching
  <- adapters
  <- runtime
  <- trainer
```

底层模块不导入 Runtime 或 Trainer。Runtime 通过 `tasks.py` 注册表选择任务组件，
不在一个 Dataset 或 DataLoader 中写任务分支。

## 6. 存储格式

每个 Split 是一个数据集目录：

```text
train/
├── manifest.json
├── data-00000.tcbin
├── data-00000.tcidx
├── data-00001.tcbin
└── data-00001.tcidx
```

`tcbin` 保存：

- Magic、格式版本。
- JSON Metadata 区。
- 按字段保存的 `int32`、`int64` 或 `float32` 一维数组。

`tcidx` 保存固定宽度 Record Offset。Reader 通过只读 `mmap` 和 offset 随机访问，
不需要把全部 Sample 对象常驻 Python 内存。

`manifest.json` 记录：

- `dataset_id`、`sample_type`、`schema_version`。
- 数据集版本、父版本和创建时间。
- 每个 Shard 的样本数、大小与 SHA-256。
- 总样本数和 Manifest Content Hash。

Manifest 和 Shard 校验失败会直接报错，不允许静默训练错误数据。

## 7. Runtime 数据流

### 7.1 Pretraining / CPT

```text
PretrainingSample
  -> PretrainingPacker
  -> PackedSequence
  -> PretrainingBatchBuilder
  -> {input_ids, attention_mask, labels, segment_ids, position_ids}
  -> Segment-aware Causal LM
```

`document_ids` 与 `segment_ids` 随 Packed Sequence 保留。Batch Builder 将文档边界
转换为 Attention Group ID，并按 Group 重置 `position_ids`。模型在 SDPA 前构造
`causal AND same_group AND valid_key` 的块对角布尔 Mask，因此不同文档及同一长文档
的不同滑动窗口不能仅因物理拼接而互相 Attention。

### 7.2 SFT

```text
messages
  -> rendered user / assistant / tool spans
  -> assistant-only loss_mask
  -> SFTSequenceSample
  -> SampleIsolatedPacker
  -> SFTBatchBuilder
  -> {input_ids, attention_mask, labels, loss_mask, segment_ids}
```

Prompt Mask、Assistant-only Loss、EOS Mask、Conversation Boundary 和 Segment ID 在
离线阶段或 Packing 阶段生成。`labels` 在数据层不做因果移位。

### 7.3 Preference

```text
PreferenceSample(pair)
  -> pair-aware sampler
  -> PreferenceBatchBuilder
  -> chosen sequence + rejected sequence + preference label
```

一个 Preference Sample 就是一个不可拆分的 Pair 单元。Batch Builder 会拒绝重复
`pair_id`，并同时返回 Prompt、完整 Chosen/Rejected 序列、Response 序列与 Mask。

### 7.4 Reward / Trajectory

```text
RewardSample       -> prompt + response + reward/ranking
TrajectorySample   -> prompt + response + reward/advantage/logprob/value
```

Batch Builder 保留 Padding Mask、Response Mask、Group ID、Candidate Index 和
Stop Reason。Loss 函数从 Batch 中读取任务字段，不从 `input_ids` 猜测任务语义。

## 8. 单卡与多卡

### 单卡

```text
BinaryDatasetReader
  -> RandomSampler
  -> DataLoader
```

`shuffle` 和 `seed` 决定 Epoch 内顺序。`set_epoch(epoch)` 产生可复现的新排列。

### 多卡

```text
Rank 0..N-1
  -> identical manifest + seed + world_size
  -> DistributedSampler (rank-strided)
  -> optional world-size padding
  -> one DataLoader per rank
```

只有 Sampler、World Size 和 Batch 分发依赖 GPU 数量。Schema、Storage、Packing 和
任务语义不依赖 GPU 数量。

所有 Rank 必须使用相同的 `dataset version`、`seed`、`world_size` 和 Batch 配置。

## 9. 配置

训练配置示例：

- `configs/data_pretraining_unified.yaml`
- `configs/data_sft_unified.yaml`
- `configs/data_preference_unified.yaml`
- `configs/data_reward_unified.yaml`
- `configs/data_trajectory_unified.yaml`

配置类在 `dataset/config.py`，加载时会：

- 检查未知字段。
- 检查必填字段和类型。
- 检查 Storage Format、Sampler、Packing 和 Batching Strategy。
- 检查 Dataset Path、Shard Size 和 Sequence Length。

旧 flat 配置不会自动迁移，也不存在兼容别名。

## 10. Public API

```python
from tokenizers import Tokenizer
from dataset.config import UnifiedDataConfig
from dataset.runtime import UnifiedDataLoaderFactory

data_config = UnifiedDataConfig.from_yaml(
    "configs/data_pretraining_unified.yaml"
)
tokenizer = Tokenizer.from_file("taichu_tokenizer/tokenizer.json")

factory = UnifiedDataLoaderFactory(
    data_config,
    pad_token_id=tokenizer.token_to_id(data_config.pad_token),
)
loader = factory.create(split="train")
loader.set_epoch(0)

for batch in loader:
    model_inputs = factory.input_adapter()(batch)
    train_step(model_inputs)
```

离线准备 API：

```python
from dataset.offline.pretraining import (
    PretrainingDataPreparation,
    load_pretraining_config,
)

config = load_pretraining_config("configs/prepare_pretrain_data.yaml")
manifest = PretrainingDataPreparation(config).run()
```

## 11. Checkpoint / Resume

`DataState` 保存：

```text
dataset_id
dataset_version
epoch
batch_index
consumed_samples
```

`ResumableBatchIterator` 在恢复时校验 Dataset Identity，设置 Sampler Epoch 并跳过已完成
Batch。Checkpoint 中应同时保存 Trainer 的 Global Step 和 `DataState`。

## 12. 扩展规则

### 新增 Alignment 算法

如果数据语义仍是 Prompt + Chosen + Rejected：

1. 复用 `PreferenceSample`。
2. 新增 Loss / Trainer。
3. 如需新的采样约束，新增 Sampler。
4. 在 Preference Batch 上读取已有 Pair 字段。

如果算法需要新字段：

1. 新增或扩展任务 Schema。
2. 新增 Preprocessing Adapter。
3. 新增 Batch Builder 和 Model Input Adapter。
4. 在 `tasks.py` 注册任务。
5. 不修改 Storage 或 Reader。

### 新增模型架构

通常只需要：

1. 修改 Model Input Adapter。
2. 如果 Position ID、Segment ID 或 Modality ID 约束变化，修改 Batch Builder。
3. 如果 Token 或 Image Tensor 存储字段变化，扩展 Schema 与 Storage DType。

Storage、Manifest、Sampler、Distributed Sampler、Mixer 和 Checkpoint 不应因模型架构
变化而重写。

## 13. 职责问答

| 问题 | 答案 |
|---|---|
| 所有任务共享什么 | Schema 基类、Binary Storage、Manifest、Reader、Sampler、Mixer、Prefetch、Checkpoint、Metrics |
| 任务专用是什么 | Schema 字段、Preprocessing Adapter、Packing、Batch Builder、Model Input Adapter、Loss Metadata |
| Offline 做什么 | 校验、清洗、格式化、Tokenization、Mask、Metadata、版本化写入 |
| Online 做什么 | Rollout、Experience Buffer、动态 Candidate、模型生成字段 |
| 哪些依赖 GPU 数量 | Distributed Sampler、World-size Padding、Batch 分发 |
| 哪些依赖任务 | Sample 语义、Loss Mask、Packing、Batch Builder、Input Adapter |
| 新 Alignment 算法改哪里 | 优先复用 Preference/Trajectory Schema，新增 Loss、Sampler 或 Input Adapter |
| 新模型架构改哪里 | Model Input Adapter，必要时 Batch Alignment 和 Schema DType |

## 14. 验证范围

自动化测试覆盖：

- Common / Pretraining / SFT / Preference / Reward / Trajectory / Multimodal Schema。
- Binary Shard、Manifest、Checksum 和 mmap 随机访问。
- 多 Shard 数据集读取。
- 多轮 Conversation 的 Assistant-only Mask。
- Preference Pair 与 Pair-aware Batch。
- Reward / Trajectory Batch 契约。
- Pretraining Stream Packing、块对角 Attention 和 SFT Sample-isolated Packing。
- 单卡 Sampler、多 Rank 分片、Epoch 可复现性。
- 统一离线 JSONL 到 Binary 数据集。
- 配置严格校验和单一 Runtime 路径。

当前完整 unittest 回归为 `204 tests, 2 skipped`。

CPU 端到端验证还覆盖了：

- `500` 条 JSONL -> Unified Binary Dataset。
- 预训练 `79` 步、验证、生成诊断和 Checkpoint。
- 从 `checkpoint-30` 恢复并在校验 Dataset ID/Version 后继续到第 `79` 步。

本地 CPU 微基准：

```powershell
python benchmarks/benchmark_unified_data.py `
  --config tests/test_train/pretrain_sim.yaml `
  --batches 20 --batch-size 4 --num-workers 0
```

当前审计环境的一次结果为约 `1,094 samples/s`、`70,033 tokens/s`。该数字只用于
验证监控链路可用，不作为 GPU 或多 Worker 生产吞吐结论。

训练前 next-token 与模型正确性预检：

```powershell
python benchmarks/preflight_next_token.py `
  --model-config configs/TaiChu_Small_80M.yaml `
  --report reports/next_token_pipeline_report_80m.json
```

该工具会输出并保存：

- 离线数据生成结果与 Binary 读取结果。
- `input_ids`、`labels`、`attention_mask` 的逐位置对比。
- `logits[:, :-1]` 与 `labels[:, 1:]` 的 next-token 对齐。
- 模型 loss 与手动 CrossEntropy 的数值差。
- Padding 不变性、未来 Token 因果性、梯度与优化器单步更新。
- SFT Assistant-only Mask 和 Stream Packing 文档边界。

当前 `segment_ids` / `document_ids` 已进入模型 Attention。不同 Group 使用独立因果
块，Packing 后不会发生跨文档 Attention；增量解码路径仍使用普通 KV Cache 因果
语义，不接收 `segment_ids` 或自定义 `position_ids`。

仍未验证：

- GPU 数据吞吐和 NCCL 多节点同步。
- 超大语料下的 mmap Page Cache、Shard 数量与 Worker 压测。
- DPO、IPO、KTO、PPO、GRPO、RLHF 的完整训练算法。
- 真实多模态 Tensor 解码和模型侧对齐。
