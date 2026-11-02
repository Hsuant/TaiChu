# Offline Preparation

Raw records are converted to unified binary datasets here.

```text
JSONL / JSONL.GZ / Parquet
  -> RecordSourceReader
  -> task adapter
  -> optional cleaning/dedup
  -> Typed Sample
  -> Binary Shard + Manifest
```

## Commands

```powershell
python -m dataset.offline.prepare `
  --config configs/prepare_pretrain_data.yaml
```

Output:

```text
data/unified/taichu_80m/
├── dataset_manifest.json
├── train/
│   ├── manifest.json
│   ├── data-00000.tcbin
│   └── data-00000.tcidx
└── validation/
    ├── manifest.json
    ├── data-00000.tcbin
    └── data-00000.tcidx
```

Task Adapters:

- `PretrainingPreprocessor`
- `SFTPreprocessor`
- `PreferencePreprocessor`
- `RewardPreprocessor`
- `TrajectoryPreprocessor`

All adapters share `TextTokenizer`, validation, metadata and Binary Storage. Each adapter
owns only task-specific formatting and mask semantics.
