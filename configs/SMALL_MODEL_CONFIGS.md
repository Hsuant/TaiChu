# TaiChu 80M / 150M / 190M 配置说明

本文是 [`../docs/AUDIT_REPORT.md`](../docs/AUDIT_REPORT.md) 的小模型配置附录。
所有参数均按当前仓库实现和 32768 Tokenizer 实测。

## 模型规格

| 规格 | 参数量 | Hidden | Layers | Q/KV Heads | FFN | 目标 Token |
|---|---:|---:|---:|---|---:|---:|
| 80M | 80,757,248 | 512 | 16 | 8 / 2 | 2176 | 4.0B |
| 150M | 149,057,280 | 768 | 18 | 12 / 4 | 2304 | 5.2B |
| 190M | 191,394,816 | 1024 | 14 | 16 / 4 | 2816 | 7.0B |

均使用：

- Decoder-only、SwiGLU、RMSNorm、RoPE、GQA
- 输入输出 Embedding 权重绑定
- `vocab_size=32768`
- `max_position_embeddings=2048`
- BF16 训练

## 配置对应关系

| 模型 | Model YAML | Training YAML |
|---|---|---|
| 80M | [`TaiChu_Small_80M.yaml`](./TaiChu_Small_80M.yaml) | [`train_small_80m.yaml`](./train_small_80m.yaml) |
| 150M | [`TaiChu_Small_150M.yaml`](./TaiChu_Small_150M.yaml) | [`train_small_150m.yaml`](./train_small_150m.yaml) |
| 190M | [`TaiChu_Small_190M.yaml`](./TaiChu_Small_190M.yaml) | [`train_small_190m.yaml`](./train_small_190m.yaml) |

## 数据规模预算

当前混合样例语料实测约 4.5 bytes/token。下表保守按 4 bytes/token 计算，
GB 为十进制 GB，不是 GiB。

| 模型 | 过滤后文本 | 过滤前原始数据 | Binary Train Shards | Binary Validation Shards |
|---|---:|---:|---:|---:|
| 80M | 约 16GB | 20-24GB | 7-10GB | 0.2-0.4GB |
| 150M | 约 21GB | 26-32GB | 9-15GB | 0.3-0.5GB |
| 190M | 约 28GB | 35-42GB | 12-20GB | 0.4-0.6GB |

建议额外保留 20-30% 原始数据用于去重、过滤和语言配比调整。
验证集必须与训练集文档级隔离，并建议至少包含 5-10M Token。

推荐 Token 配比：

| 数据类型 | Token 占比 |
|---|---:|
| 中文网页、百科、新闻 | 35-40% |
| 英文网页、百科 | 20-25% |
| 代码与技术文档 | 10-15% |
| 数学与逻辑 | 5-10% |
| 书籍、论文、高质量长文 | 10% |
| 对话与结构化文本 | 3-5% |

## 数据目录

训练配置默认读取：

```text
./data/unified/taichu_80m/train
./data/unified/taichu_80m/validation

./data/unified/taichu_150m/train
./data/unified/taichu_150m/validation

./data/unified/taichu_190m/train
./data/unified/taichu_190m/validation
```

准备统一 Binary Dataset：

```powershell
python -m dataset.offline.prepare `
  --config configs/prepare_pretrain_data.yaml
```

训练集和验证集在同一次运行中生成。切换模型规模时修改
`output_dir` 与 `splits`，并保持 Tokenizer、序列长度和 Packing 参数一致。
目录职责和样例数据说明见 [`../data/README.md`](../data/README.md)。

## 训练 batch

三个配置都按约 0.5M padded Token/优化步设计：

| 模型 | Micro Batch | 梯度累积 | Token/Step | 预计步数 |
|---|---:|---:|---:|---:|
| 80M | 8 | 32 | 524,288 | 7,630 |
| 150M | 4 | 64 | 524,288 | 9,919 |
| 190M | 2 | 128 | 524,288 | 13,352 |

多卡训练时将梯度累积除以 world size，以保持全局 Token/步不变。

## 硬件建议

| 模型 | 最低显存 | 建议硬件 |
|---|---:|---|
| 80M | 12-16GB | 1 x RTX 4090 24GB |
| 150M | 24GB | 1 x RTX 4090 24GB 或 A100 40GB |
| 190M | 24GB 较紧 | A100 40GB 或 2 x 24GB |

12-16GB 设备训练 150M/190M 前应增加 activation checkpointing。
当前仓库尚未提供该配置开关。

## 启动训练

```powershell
python -m trainer.train_pretrain `
  --model_config configs/TaiChu_Small_150M.yaml `
  --pretrain_config configs/train_small_150m.yaml
```
