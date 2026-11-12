# Offline Preparation

`dataset.offline` 把原始记录转换为统一 Binary Dataset，并生成可用于恢复和审计的
Manifest。

## 文档级预训练

```text
JSONL / JSONL.GZ / Parquet
  → RecordSourceReader
  → NFC 清洗
  → 精确去重
  → 13-gram MinHash / LSH
  → 精确 Jaccard 复核
  → Cluster 级 Split
  → 全文档 Tokenization
  → 2,046 Token 滑窗 + 128 Overlap
  → tcbin / tcidx / Manifest
```

```powershell
python -m dataset.offline.prepare `
  --config configs/prepare_pretrain_data.yaml
```

这是当前预训练与 CPT 的规范入口。处理内容包括：

- 文档级 BOS/EOS。
- 窗口重叠区 `labels=-100`。
- Train / Validation / Diagnostic 的重复 Cluster 隔离。
- 来源、语言、Token、窗口和泄漏统计。
- 原子输出与可恢复 Staging。

## 来源抽样

文档级准备前，可先按 GB 权重抽取原始来源：

```powershell
python -m dataset.offline.sample_sources `
  --config configs/sample_pretrain_80m.yaml
```

先做路径与预算检查：

```powershell
python -m dataset.offline.sample_sources `
  --config configs/sample_pretrain_80m.yaml `
  --dry-run
```

抽样是确定性的，并输出：

```text
sampling_manifest.json
train/<source>.jsonl
validation/<source>.jsonl
```

最终 Split 不以抽样阶段结果为准；文档级准备会重新执行去重和 Cluster 隔离。

## 通用任务准备

```powershell
python -m dataset.offline.prepare `
  --config configs/prepare_sft_data.yaml
```

支持：

- SFT / Instruction Tuning
- Preference / DPO / IPO / KTO 数据契约
- Reward 数据契约
- Trajectory / PPO / GRPO 数据契约

除 Pretraining 外，这些任务目前只有数据准备与 Batch 契约，不代表对应训练算法已实现。

## 输出

```text
data/unified/taichu_80m/
├── dataset_manifest.json
├── train/
│   ├── manifest.json
│   ├── data-00000.tcbin
│   └── data-00000.tcidx
├── validation/
│   ├── manifest.json
│   ├── data-00000.tcbin
│   └── data-00000.tcidx
└── diagnostic/
```

## 实现结构

```text
dataset/offline/
├── readers.py              # JSONL / GZ / Parquet
├── cleaning.py             # 通用清洗
├── sample_sources.py       # GB 加权确定性抽样
├── prepare.py              # 统一准备路由
├── pretraining/
│   ├── config.py
│   ├── hashing.py
│   ├── fingerprinting.py
│   ├── deduplication.py
│   ├── segmentation.py
│   ├── output.py
│   ├── reporting.py
│   └── pipeline.py
└── adapters/               # SFT / Preference / Reward / Trajectory
```

## 注意事项

- 配置中的路径来自作者本机，执行前必须替换。
- `--restart` 只处理不完整 Staging，不覆盖已完成的正式输出。
- 现有数据配置与训练配置的 Dataset ID 不一致，复现前应统一。
- 生产语料和输出目录不应提交。

详细说明：

- [Pre-training Data Pipeline](../../docs/PRETRAINING_DATA_PIPELINE.md)
- [Unified Data Infrastructure](../../docs/UNIFIED_DATA_INFRASTRUCTURE.md)
- [Configuration](../../docs/CONFIGURATION.md)
