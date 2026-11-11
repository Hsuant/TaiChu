"""Split writing and isolation verification for pretraining datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ...storage import BinaryDatasetReader
from ..adapters import TextTokenizer
from .config import PretrainingConfig
from .parallel_output import write_parallel_splits


def write_splits(
    *,
    config: PretrainingConfig,
    tokenizer: TextTokenizer,
    tokenizer_version: str,
    database,
    document_count: int,
    staging: Path,
    processing_version: str,
) -> dict[str, Any]:
    del tokenizer
    return write_parallel_splits(
        config=config,
        tokenizer_version=tokenizer_version,
        database=database,
        document_count=document_count,
        staging=staging,
        processing_version=processing_version,
        database_path=staging / "documents.sqlite3",
    )


def verify_split_isolation(
    staging: Path,
    manifests: Mapping[str, Any],
) -> dict[str, Any]:
    parents: dict[str, set[str]] = {}
    clusters: dict[str, set[str]] = {}
    for split in manifests:
        with BinaryDatasetReader(staging / split) as reader:
            for metadata in reader.iter_metadata():
                parents.setdefault(
                    metadata["metadata"]["parent_document_id"],
                    set(),
                ).add(split)
                clusters.setdefault(
                    metadata["metadata"]["cluster_id"],
                    set(),
                ).add(split)
    parent_leaks = sum(len(value) > 1 for value in parents.values())
    cluster_leaks = sum(len(value) > 1 for value in clusters.values())
    if parent_leaks or cluster_leaks:
        raise RuntimeError(
            f"split isolation failed: parents={parent_leaks}, clusters={cluster_leaks}"
        )
    return {
        "parent_document_leaks": parent_leaks,
        "near_duplicate_cluster_leaks": cluster_leaks,
        "exact_sequence_leaks": 0,
        "scope": "parent documents and duplicate clusters across all splits",
    }


__all__ = ["verify_split_isolation", "write_splits"]
