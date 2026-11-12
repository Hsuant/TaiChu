# Pretraining Data Pipeline

> 当前配置：80M 原始抽样目标 18 GB，训练目标 4.0B Token。
> 本文的性能数字是维护者本机历史测量，只用于解释瓶颈与容量规划，不是当前机器基准。

## Pipeline

```text
raw JSONL / GZ / Parquet
  -> validation
  -> NFC cleaning with structure preserved
  -> whole-document exact dedup
  -> 13-gram MinHash + LSH candidates
  -> exact Jaccard verification
  -> connected duplicate clusters
  -> stable cluster-level train / validation / diagnostic split
  -> whole-document tokenization
  -> 2046-token windows, stride 1918, overlap 128
  -> one independent PretrainingSample per window
  -> Binary Shard + Manifest
```

The command is:

```powershell
python -m dataset.offline.prepare `
  --config configs/prepare_pretrain_data.yaml
```

The SQLite staging database stores document text and exact digests. MinHash
signatures are used to populate LSH rows but are not retained per document, and
complete 13-gram sets are recomputed from stored text only when an LSH candidate
needs exact Jaccard verification. This keeps the implementation disk-backed
without writing the much larger per-document fingerprint payloads. It supports
restart from an incomplete scan because document IDs and LSH/duplicate edges are
persisted. Rerunning after a split-write failure rebuilds the split output from
the completed document index.

Document fingerprinting runs in bounded process-pool batches. The scanner
reports byte-level progress for each source file, plus committed document count
and pending batch count, so multi-GB shards do not appear idle.

## Performance Model

The scan is CPU-bound, not storage-bound. On the local 66.2 MB FineWeb2
validation shard:

- raw sequential file reads reached about 1.87 GB/s;
- the original scan path measured about 0.53 MB/s;
- the optimized scan path measured 4.8-4.9 MB/s;
- full preparation after all optimizations took 83-88 s, including a 60-66 s
  tokenization/write phase at about 250-273 documents/s.

Fingerprinting, raw-token counting and split tokenization run in bounded process
pools. Band keys are computed once per document, exact Jaccard uses the already
sorted NumPy shingle arrays, and split verification plus manifest aggregation
read metadata without decoding token arrays.

Split output runs document-ID ranges in parallel. Each worker owns its tokenizer
with `RAYON_NUM_THREADS=1`, performs Tokenization, segmentation and part-shard
writing, and returns token counts. Part manifests omit per-part checksums and
fsync; final shards are merged in deterministic order, hashed once, and described
by the final split manifest. `output_workers` controls this stage independently
from scan workers.

Stored pretraining windows omit the redundant all-ones `attention_mask` and
all-zero `document_ids` arrays. Trusted tokenizer output uses signed int16 when
all token IDs and labels fit its range. On the configured 32K tokenizer these
arrays fit, reducing each token from 8 bytes to 4 bytes without changing the
in-memory training contract.

The small-shard result is a warm-cache microbenchmark, not a full-corpus
projection. A 20-shard run over 18.01 GB and about 5.15 million documents took
about 5 hours 27 minutes to scan, or approximately 0.91 MB/s overall. That run
used the old 2 MiB SQLite page cache, so it does not measure the new cache,
page-size and query changes.

On a 2,000-document sample, omitting per-document shingle/signature payloads
reduced staging from 37.6 MB to 10.8 MB for 8.28 MB of cleaned UTF-8 text. For
that historical 18.01 GB raw mix, the projection was roughly 24 GB of staging
plus roughly 79 GB of final binary output at peak. Re-measure free space and
output size for the current corpus before a full run.

`pending_batches` is the bounded in-flight queue, normally
`2 * processing_workers`. A full queue therefore means work is saturating the
workers, not that the scanner is stuck. Set `processing_workers` no higher than
the number of physical CPU cores; process-level Tokenizers parallelism can
otherwise oversubscribe the machine.

The progress postfix also reports cumulative `candidates`, exact Jaccard
`jaccard` checks, and the current per-document `max_candidates`. Stable candidate
counts with declining byte throughput point to SQLite random I/O. Rapid growth
in `candidates` or `max_candidates` points to hot LSH buckets or a source with
many near duplicates.

Source composition also changes byte throughput. In a 1,000-document sample,
`automath_arxiv_en` averaged 53.6 KB and 18,802 tokens per document, while
`pes2o_en` averaged only 1.4 KB per document. Long-document sources spend much
more CPU per input byte, and short-document sources pay per-document overhead
more often. Compare MB/s within the same source rather than only across the
whole run.

The staging database uses exclusive locking, a 1 GiB page cache, the largest
available memory map, and 32 KiB pages for newly created stores. The redundant
`text_sha` index is removed because exact deduplication uses the in-memory digest
map. Existing staging databases keep their original page size, but the larger
cache, memory map, index removal and merged 16-band lookup apply when the scan is
resumed.

Cluster assignment has separate `clusters_assigned` and `clusters_complete`
checkpoints. The duplicate-cluster artifact is written to a temporary file and
atomically replaced, and cluster reporting reads only document IDs that occur in
`duplicate_edges` instead of sorting the complete document table.

A 100,000-document output benchmark measured 2,958.6 documents/s with 16
single-thread output workers and 128 scheduling parts, about 11.2 times the
previous 265 documents/s. This is a local parallel-output measurement rather than
a full-corpus completion claim.

Resume and finalization report dedicated progress stages:
`restore dedup index`, `resolve clusters`, `assign clusters`, and
`write duplicate clusters`. Before the last stage begins yielding rows, the
command explicitly prints that SQLite is reading and sorting duplicate
documents, so a long sort does not appear as a silent hang.

## Sliding Windows

For `N` document tokens:

```text
0 .. 2046
1918 .. 3964
3836 .. 5000
...
```

The final window is always retained, including tails shorter than 32 tokens.
`min_fragment_tokens=32` is diagnostic metadata only and never a deletion rule.

Special-token and label rules:

| Window | Input | Supervised labels |
|---|---|---|
| Single | BOS + content + EOS | content + EOS |
| First of many | BOS + content | all content |
| Middle | content | 128 overlap tokens are `-100`; new content is supervised |
| Last | content + EOS | 128 overlap tokens are `-100`; new content + EOS |

The model applies `labels[..., 1:]` against `logits[..., :-1, :]`. Since every
BOS label is `-100`, each content token is a next-token target exactly once.

Each stored window represents exactly one source document and is not padded.
The redundant all-ones `attention_mask` and all-zero `document_ids` arrays are
therefore omitted from storage. Packing and batch construction already recreate
these defaults from sample boundaries.

## Packing And Attention

The Packer physically concatenates samples while retaining `segment_ids` and
`document_ids`. The Batch Builder emits:

- `segment_ids`: independent Attention Groups.
- `position_ids`: reset to zero at every Group boundary.
- `attention_mask`: padding visibility.
- `labels`: next-token targets, with cross-document targets masked as `-100`.

The model constructs:

```text
allowed(i, j) =
    j <= i
    AND segment_id[i] == segment_id[j]
    AND segment_id[j] >= 0
    AND attention_mask[j] != 0
```

This is a block-diagonal causal mask. Different source documents, and different
sliding windows of the same source document, remain independent even when packed
into one physical sequence.

## Split Isolation

Every exact/near-duplicate cluster receives one split from a stable hash. All
members and every window inherit that split. The pipeline verifies:

```text
parent_document_leaks = 0
near_duplicate_cluster_leaks = 0
```

The generated `duplicate_clusters.jsonl` retains canonical IDs, source mappings,
detection reason and exact Jaccard similarity.

## Manifest Counts

The root `dataset_manifest.json` separately reports:

- original input tokens,
- cleaned tokens before deduplication,
- exact and near-duplicate removed tokens,
- final unique content tokens,
- window input tokens,
- overlap tokens,
- BOS/EOS tokens,
- supervised loss tokens,
- recovered tokens beyond the v1 2046-token boundary,
- source and language shares plus deltas against the v1 reference.

The configured inputs do not carry a v1 truncation manifest or v1-to-original
document lookup IDs. The manifest therefore reports recovery of all content
beyond the old 2046 boundary, while explicitly marking v1 truncated-sample
lookup as not applicable rather than claiming an unmeasured recovery rate.

## Validation Scope

Automated tests cover:

- 1, 31, 32, 2045, 2046, 2047, 2048, 3964, 5000 and 8000 content tokens,
- exact `[0,2046)`, `[1918,3964)`, `[3836,5000)` windows,
- exactly-once content supervision and legal tail preservation,
- BOS only on first windows and EOS only on final windows,
- exact and exact-Jaccard-verified near-duplicate clustering,
- deterministic split isolation,
- JSONL to Binary integration with the bundled tokenizer,
- block-diagonal Attention and Group-local position IDs.

Full-corpus deduplication, the configured production sampling volume and GPU
throughput have not been run in this workspace. Those require the production raw
cache and are reported as operational next steps, not test results.
