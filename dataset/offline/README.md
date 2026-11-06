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

For production pretraining, sample the configured source mixture first:

```powershell
python -m dataset.offline.sample_sources `
  --config configs/sample_pretrain_mix.yaml
```

Use the size-specific configs for the 80M, 150M and 190M presets. Sampling is
deterministic for a fixed seed and writes a per-source manifest with selected
bytes, train records and validation records. Point the preparation config's
`splits` entries at the generated `train` and `validation` directories.
Records outside `min_chars`/`max_chars` are removed before sampling. Both the
length-statistics pass and the sampling pass show byte-based progress bars.
Thresholds can be overridden from the command line:

```powershell
python -m dataset.offline.sample_sources `
  --config configs/sample_pretrain_190m.yaml `
  --min-chars 32 `
  --max-chars 100000
```

Use `--no-progress` for non-interactive logs. The manifest reports
`short_records`, `long_records`, `eligible_records` and `eligible_gb`.

Sampling first builds a compact temporary index of eligible record offsets. The
sampling pass seeks only selected records, so source JSON is parsed once and
eligible data is not read a second time. If Windows denies the final directory
rename, the script retries and falls back to moving files within the same volume.
A completed staging directory is retained on commit failure; rerunning the same
command finalizes it without sampling again.

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
