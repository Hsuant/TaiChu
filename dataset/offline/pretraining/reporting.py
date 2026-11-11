"""Pretraining dataset manifest and statistics."""
from __future__ import annotations

import subprocess
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ...storage import BinaryDatasetReader
from ..adapters import TextTokenizer
from .config import PretrainingConfig
from .segmentation import sliding_window_bounds


V1_SOURCE_SHARES = {
    "fineweb2_zh": 0.1904,
    "industrycorpus_zh": 0.1882,
    "baike_wiki_zh": 0.1434,
    "cci_hq_zh": 0.1007,
    "fineweb_en": 0.0954,
    "fineweb_edu_en": 0.0931,
    "baike_wiki_en": 0.0907,
    "pes2o_en": 0.0480,
    "automath_code_en": 0.0435,
    "automath_arxiv_en": 0.0066,
}
V1_LANGUAGE_SHARES = {
    "ch": 0.6227,
    "en": 0.3338,
    "code": 0.0435,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _source_counter() -> dict[str, Any]:
    return {
        "raw_documents": 0,
        "cleaned_documents": 0,
        "canonical_documents": 0,
        "duplicate_documents": 0,
        "raw_tokens": 0,
        "cleaned_tokens": 0,
        "unique_tokens": 0,
        "segments": 0,
        "loss_tokens": 0,
        "recovered_tokens_beyond_2046": 0,
        "split_segments": {},
        "split_loss_tokens": {},
    }


def build_manifest(
    *,
    config: PretrainingConfig,
    tokenizer: TextTokenizer,
    tokenizer_version: str,
    database,
    document_count: int,
    split_manifests: Mapping[str, Any],
    leakage: Mapping[str, Any],
    started_at: str,
    staging: Path,
    processing_version: str,
) -> dict[str, Any]:
    sources: dict[str, dict[str, Any]] = defaultdict(_source_counter)
    languages: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "raw_tokens": 0,
            "cleaned_tokens": 0,
            "unique_tokens": 0,
            "loss_tokens": 0,
        }
    )
    exact_docs = near_docs = exact_tokens = near_tokens = 0
    raw_tokens = cleaned_tokens = unique_tokens = 0
    recovered_tokens = long_documents = tail_fragments = 0
    canonical_documents = 0

    for row in database.execute(
        """
        SELECT id, source_name, language, raw_tokens, cleaned_tokens,
               canonical_id, text_sha
        FROM documents ORDER BY id
        """
    ):
        document_id = int(row["id"])
        canonical_id = int(row["canonical_id"])
        source = sources[str(row["source_name"])]
        language = languages[str(row["language"])]
        raw_count = int(row["raw_tokens"])
        clean_count = int(row["cleaned_tokens"])
        source["raw_documents"] += 1
        source["cleaned_documents"] += 1
        source["raw_tokens"] += raw_count
        source["cleaned_tokens"] += clean_count
        language["raw_tokens"] += raw_count
        language["cleaned_tokens"] += clean_count
        raw_tokens += raw_count
        cleaned_tokens += clean_count
        if document_id != canonical_id:
            source["duplicate_documents"] += 1
            canonical = database.execute(
                "SELECT text_sha FROM documents WHERE id = ?",
                (canonical_id,),
            ).fetchone()
            if bytes(row["text_sha"]) == bytes(canonical["text_sha"]):
                exact_docs += 1
                exact_tokens += clean_count
            else:
                near_docs += 1
                near_tokens += clean_count
            continue
        canonical_documents += 1
        source["canonical_documents"] += 1
        source["unique_tokens"] += clean_count
        language["unique_tokens"] += clean_count
        unique_tokens += clean_count
        recovered = max(
            0,
            clean_count - config.segmentation.max_segment_content_tokens,
        )
        source["recovered_tokens_beyond_2046"] += recovered
        recovered_tokens += recovered
        if clean_count > config.segmentation.max_segment_content_tokens:
            long_documents += 1
            bounds = sliding_window_bounds(
                clean_count,
                max_content_tokens=config.segmentation.max_segment_content_tokens,
                overlap_tokens=config.segmentation.overlap_tokens,
            )
            final_new = (
                bounds[-1][1]
                - bounds[-1][0]
                - config.segmentation.overlap_tokens
            )
            if final_new < config.segmentation.min_fragment_tokens:
                tail_fragments += 1

    for split in split_manifests:
        with BinaryDatasetReader(staging / split) as reader:
            for metadata in reader.iter_metadata():
                source = sources[str(metadata["source_id"])]
                language = languages[str(metadata["metadata"]["language"])]
                loss_tokens = int(metadata["metadata"]["loss_token_count"])
                source["segments"] += 1
                source["loss_tokens"] += loss_tokens
                source["split_segments"][split] = (
                    source["split_segments"].get(split, 0) + 1
                )
                source["split_loss_tokens"][split] = (
                    source["split_loss_tokens"].get(split, 0) + loss_tokens
                )
                language["loss_tokens"] += loss_tokens

    split_totals = {
        key: sum(int(payload["stats"][key]) for payload in split_manifests.values())
        for key in (
            "segments",
            "input_tokens",
            "loss_tokens",
            "overlap_tokens",
            "special_tokens",
        )
    }
    storage = {
        split: {
            "samples": int(payload["sample_count"]),
            "binary_storage_bytes": int(payload["stats"]["storage_bytes"]),
            "binary_storage_gb": float(payload["stats"]["storage_gb"]),
            "logical_input_gb": float(payload["stats"]["logical_input_gb"]),
            "loss_target_gb": float(payload["stats"]["loss_target_gb"]),
        }
        for split, payload in split_manifests.items()
    }
    total_storage_bytes = sum(
        value["binary_storage_bytes"] for value in storage.values()
    )
    source_shares = {
        name: value["unique_tokens"] / unique_tokens
        for name, value in sources.items()
    } if unique_tokens else {}
    language_shares = {
        name: value["unique_tokens"] / unique_tokens
        for name, value in languages.items()
    } if unique_tokens else {}
    errors = [
        {
            "source_id": row["source_name"],
            "source_path": row["source_path"],
            "record_index": row["record_index"],
            "stage": row["stage"],
            "reason": row["reason"],
        }
        for row in database.execute(
            """
            SELECT source_name, source_path, record_index, stage, reason
            FROM errors ORDER BY id
            """
        )
    ]
    return {
        "schema_version": 2,
        "dataset_id": config.dataset_id,
        "task_type": config.task_type,
        "version": config.version,
        "processing_version": processing_version,
        "created_at": _now(),
        "started_at": started_at,
        "git_commit": _git_commit(),
        "tokenizer": {
            "path": str(config.tokenizer_path),
            "version": tokenizer_version,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "auto_add_special_tokens": False,
        },
        "config": _config_snapshot(config),
        "splits": dict(split_manifests),
        "statistics": {
            "documents": {
                "cleaned_documents": document_count,
                "canonical_documents": canonical_documents,
                "exact_duplicate_documents_removed": exact_docs,
                "near_duplicate_documents_removed": near_docs,
            },
            "tokens": {
                "original_input_tokens": raw_tokens,
                "cleaned_tokens_before_dedup": cleaned_tokens,
                "exact_duplicate_tokens_removed": exact_tokens,
                "near_duplicate_tokens_removed": near_tokens,
                "final_unique_content_tokens": unique_tokens,
                "window_input_tokens": split_totals["input_tokens"],
                "overlap_tokens": split_totals["overlap_tokens"],
                "special_tokens": split_totals["special_tokens"],
                "padding_tokens": 0,
                "final_supervised_loss_tokens": split_totals["loss_tokens"],
            },
            "recovery": {
                "documents_over_2046_tokens": long_documents,
                "tokens_beyond_2046_recovered": recovered_tokens,
                "legal_tail_fragments_preserved": tail_fragments,
                "v1_truncated_sample_id_recovery": (
                    "not_applicable: inputs do not provide a v1 truncation manifest"
                ),
            },
            "segments": {
                "total": split_totals["segments"],
                "overlap_share_of_input_tokens": (
                    split_totals["overlap_tokens"] / split_totals["input_tokens"]
                    if split_totals["input_tokens"]
                    else 0.0
                ),
            },
                "storage": {
                    "splits": storage,
                    "total_binary_storage_bytes": total_storage_bytes,
                    "total_binary_storage_gb": (
                        total_storage_bytes / 1_000_000_000
                    ),
                    "definition": (
                        "binary_storage_gb is the actual tcbin+tcidx size; "
                        "logical_input_gb is window_input_tokens*4 bytes; "
                        "loss_target_gb is supervised_tokens*4 bytes"
                    ),
                },
            "sources": dict(sources),
            "languages": dict(languages),
            "source_token_shares": source_shares,
            "language_token_shares": language_shares,
            "v1_reference_source_shares": V1_SOURCE_SHARES,
            "v1_reference_language_shares": V1_LANGUAGE_SHARES,
            "source_share_delta_vs_v1": {
                name: share - V1_SOURCE_SHARES.get(name, 0.0)
                for name, share in source_shares.items()
            },
            "language_share_delta_vs_v1": {
                name: share - V1_LANGUAGE_SHARES.get(name, 0.0)
                for name, share in language_shares.items()
            },
            "errors": errors,
            "invalid_document_count": len(errors),
        },
        "leakage": dict(leakage),
        "artifacts": {"duplicate_clusters": "duplicate_clusters.jsonl"},
    }


def _config_snapshot(config: PretrainingConfig) -> dict[str, Any]:
    return {
        "dataset_id": config.dataset_id,
        "task_type": config.task_type,
        "version": config.version,
        "tokenizer_path": str(config.tokenizer_path),
        "output_dir": str(config.output_dir),
        "text_column": config.text_column,
        "pad_token": config.pad_token,
        "bos_token": config.bos_token,
        "eos_token": config.eos_token,
        "invalid_policy": config.invalid_policy,
        "shard_size": config.shard_size,
        "output_workers": config.output_workers,
        "segmentation": asdict(config.segmentation),
        "stride_tokens": config.segmentation.stride_tokens,
        "dedup": asdict(config.dedup),
        "splits": asdict(config.splits),
        "cleaning": asdict(config.cleaning),
        "sources": [
            {
                "name": source.name,
                "language": source.language,
                "paths": [str(path) for path in source.paths],
            }
            for source in config.sources
        ],
    }


__all__ = ["build_manifest"]
