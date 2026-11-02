# Dataset Package

`dataset/` 是 TaiChu 的统一 LLM 训练数据基础设施。当前只有一条主路径：

```text
Raw records
  -> task preprocessing adapter
  -> binary shards + manifest
  -> mmap reader
  -> sampler
  -> task packing and batching
  -> model input adapter
```

## Public Imports

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

## Commands

```powershell
python -m dataset.offline.prepare --help
python -m trainer.train_pretrain --help
python -m unittest tests.test_unified_schema_storage `
  tests.test_unified_offline_preparation `
  tests.test_unified_runtime_pipeline -v
```

完整架构、任务矩阵、扩展规则和验收范围见
[`docs/UNIFIED_DATA_INFRASTRUCTURE.md`](../docs/UNIFIED_DATA_INFRASTRUCTURE.md)。
