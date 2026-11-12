# Data Directory

`data/` 保存评估样例和生产数据目录，不属于 Python Package。

## 目录约定

```text
data/
├── eval/
│   ├── eval_tokenizer.jsonl
│   ├── eval_wiki_ch.jsonl
│   └── eval_wiki_en.jsonl
├── raw/          # 原始语料，默认不提交
├── processed/    # 中间清洗结果，默认不提交
└── unified/      # tcbin/tcidx 与 Manifest，默认不提交
```

当前仓库只提交 `data/eval/` 下的评估样例。过去文档中的 `data/samples/`
目录已删除，不再作为快速开始或测试数据来源。

## 数据主路径

```text
data/raw
  → dataset.offline.sample_sources
  → 来源抽样 JSONL + sampling_manifest.json
  → dataset.offline.prepare
  → data/unified/<dataset>/<split>/
  → trainer.train_pretrain
```

## 命令

先检查来源路径与 GB 预算：

```powershell
python -m dataset.offline.sample_sources `
  --config configs/sample_pretrain_80m.yaml `
  --dry-run
```

执行抽样：

```powershell
python -m dataset.offline.sample_sources `
  --config configs/sample_pretrain_80m.yaml
```

执行文档级准备：

```powershell
python -m dataset.offline.prepare `
  --config configs/prepare_pretrain_data.yaml
```

## 输出结构

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

实际 Shard 数量和 Split 以配置与 `dataset_manifest.json` 为准。

## 重要说明

- `configs/sample_pretrain_80m.yaml` 与 `configs/prepare_pretrain_data.yaml`
  含有作者本机数据路径，公开复现前必须替换。
- `prepare_pretrain_data.yaml` 的 `dataset_id` 与当前 80M 训练配置不一致，
  准备前应同时修正并保持数据身份一致。
- 生产语料、模型权重、实验输出和检查点均不应提交。
- `data/eval/*.jsonl` 只用于 Tokenizer / 语言评估，不代表预训练数据规模。

相关文档：

- [预训练数据管线](../docs/PRETRAINING_DATA_PIPELINE.md)
- [统一数据基础设施](../docs/UNIFIED_DATA_INFRASTRUCTURE.md)
- [Offline Preparation](../dataset/offline/README.md)
- [配置指南](../docs/CONFIGURATION.md)
