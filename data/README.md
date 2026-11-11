# Data Directory

`data/` 保存样例数据和生产数据，不属于 Python 包。

## 目录约定

```text
data/
├── samples/
│   ├── pretrain/
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   └── generic/
│       └── sample.jsonl
├── eval/
│   ├── eval_tokenizer.jsonl
│   ├── eval_wiki_ch.jsonl
│   └── eval_wiki_en.jsonl
├── raw/          # 生产原始数据，不提交
├── processed/    # 清洗/抽样后的文本，不提交
└── unified/      # 版本化 Binary Shards 与 Manifest，不提交
```

## 样例数据

- `samples/pretrain/train.jsonl`：小规模预训练训练样例。
- `samples/pretrain/val.jsonl`：小规模验证样例。
- `samples/generic/sample.jsonl`：通用 JSONL 格式样例。
- `eval/*.jsonl`：Tokenizer 和语言评估样例。

样例数据只用于测试和快速启动，不代表正式训练的数据规模或质量。

## 生产数据

推荐流程：

```text
data/raw
  -> dataset.offline.prepare
data/unified
  -> trainer.train_pretrain
```

`data/raw/`、`data/processed/` 和 `data/unified/` 已加入 `.gitignore`。

规范准备命令：

```powershell
python -m dataset.offline.prepare `
  --config configs/prepare_pretrain_data.yaml
```

`dataset.offline.prepare` 会按 `task_type` 选择文档级预训练管线或通用任务适配器。

最终 Binary Shard、Manifest、Sample Schema、Packing 和配置约束见
[`../dataset/offline/README.md`](../dataset/offline/README.md)。

详细规模和配比见：

- [`../configs/SMALL_MODEL_CONFIGS.md`](../configs/SMALL_MODEL_CONFIGS.md)
- [`../dataset/ARCHITECTURE.md`](../dataset/ARCHITECTURE.md)
- [`../docs/UNIFIED_DATA_INFRASTRUCTURE.md`](../docs/UNIFIED_DATA_INFRASTRUCTURE.md)
