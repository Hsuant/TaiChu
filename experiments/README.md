# 实验目录

新实验统一使用以下结构：

```text
experiments/
├── 80m/
│   ├── pretrain/
│   │   └── 4b_v2/
│   └── sft/
│       └── unified_2epoch_v1/
├── 150m/
│   ├── pretrain/
│   │   └── 5_2b_v1/
│   └── sft/
│       └── unified_2epoch_v1/
└── 190m/
    ├── pretrain/
    │   └── 7b_v1/
    └── sft/
        └── unified_2epoch_v1/
```

每个 Run 目录包含：

```text
<experiment>/
├── config/
│   ├── model.yaml
│   └── training.yaml
├── metadata.json
├── checkpoints/
├── best_models/
├── final_models/
├── logs/
└── evaluation/
```

- `config/` 与 `metadata.json` 保存启动时解析后的不可变配置快照。
- `checkpoints/` 保存完整恢复状态。
- `best_models/` 与 `final_models/` 保存最小推理产物。
- `logs/` 与 `evaluation/` 分别保存运行日志和独立评估结果。

`80m_4B/`、`80m_8B/` 等既有目录属于历史实验，保留原位用于恢复和审计；新实验不再写入这些旧路径。
