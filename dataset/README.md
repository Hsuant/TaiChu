# Dataset Package

`dataset/` 是 TaiChu 的统一训练数据基础设施。当前主路径：

```text
Raw records
  → task adapter / cleaning
  → typed sample
  → tcbin + tcidx + manifest
  → mmap reader
  → sampler
  → packing
  → batch builder
  → model input
```

## Public API

```python
from dataset import (
    MultimodalSample,
    PreferenceSample,
    PretrainingSample,
    RewardSample,
    SFTSequenceSample,
    TrajectorySample,
    UnifiedDataConfig,
    UnifiedDataLoaderFactory,
)
```

数据契约分为：

| Schema | 当前用途 |
|---|---|
| `PretrainingSample` | 预训练 / CPT，已接入训练 |
| `SFTSequenceSample` | SFT 数据与 Batch 契约，训练器未接入 |
| `PreferenceSample` | DPO/IPO/KTO 数据契约，算法未实现 |
| `RewardSample` | Reward Model 数据契约，Reward Head 未实现 |
| `TrajectorySample` | PPO/GRPO/RLHF 在线与离线轨迹契约，算法未实现 |
| `MultimodalSample` | 多模态数据契约，模型训练未接入 |

Schema 存在不等于训练算法已实现。

## 模块

```text
dataset/
├── schema/       # CommonSample 与任务样本
├── storage/      # tcbin/tcidx、Manifest、mmap
├── offline/      # 抽样、读取、清洗、Adapter、准备管线
├── sampling/     # Random / Distributed / Mixer / Shuffle
├── packing/      # Stream / Sample-isolated
├── batching/     # Padding 与任务 Batch Builder
├── adapters/     # Model Input Adapter
├── runtime/      # DataLoader、Online Buffer、Resume
└── metrics/      # 数据管线统计
```

## Commands

```powershell
python -m dataset.offline.sample_sources --help
python -m dataset.offline.prepare --help
python -m trainer.train_pretrain --help
```

在包含测试目录的维护工作区：

```powershell
python -m pytest -q
```

## 验证边界

- Binary Storage、Manifest、mmap、Sampler、Packing、Batch 与 Runtime Resume 有自动化测试。
- `tcbin` 支持 `int16`、`uint16`、`int32`、`int64`、`float32`。
- `segment_ids` 已进入模型 Attention；增量解码不接受 Segment Mask。
- 当前公开训练只使用 Pretraining / CPT 路径。
- GPU 多 Worker 吞吐、超大 Shard 数量和 mmap Page Cache 仍需目标环境压测。

详细文档：

- [统一数据基础设施](../docs/UNIFIED_DATA_INFRASTRUCTURE.md)
- [预训练数据管线](../docs/PRETRAINING_DATA_PIPELINE.md)
- [Offline Preparation](offline/README.md)
- [Data Directory](../data/README.md)
