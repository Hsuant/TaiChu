"""Deterministic document-level validation-set subsetting."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from dataset.exceptions import TaiChuDataFormatError
from dataset.storage import BinaryDatasetReader, BinarySampleWriter


def subset_validation_dataset(
    source: str | Path,
    output: str | Path,
    *,
    max_tokens: int,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Select whole documents until the next document would exceed the budget."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    with BinaryDatasetReader(source) as reader:
        if reader.sample_type != "pretraining":
            raise TaiChuDataFormatError(
                "validation subsetting requires a pretraining dataset"
            )
        samples = list(reader)
        source_dataset_id = reader.dataset_id
        source_version = reader.version
        source_metadata = dict(reader.manifest.metadata)

    documents: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        document_id = str(
            sample.metadata.get("parent_document_id", sample.sample_id)
        )
        documents[document_id].append(index)

    document_tokens = {
        document_id: sum(len(samples[index].input_ids) for index in indices)
        for document_id, indices in documents.items()
    }
    document_ids = sorted(documents)
    random.Random(seed).shuffle(document_ids)

    selected_documents: list[str] = []
    selected_indices: list[int] = []
    selected_tokens = 0
    for document_id in document_ids:
        tokens = document_tokens[document_id]
        if selected_tokens + tokens > max_tokens:
            continue
        selected_documents.append(document_id)
        selected_indices.extend(documents[document_id])
        selected_tokens += tokens

    if not selected_documents:
        raise ValueError(
            f"no document fits within the {max_tokens}-token validation budget"
        )

    summary: dict[str, Any] = {
        "dry_run": dry_run,
        "source_dataset_id": source_dataset_id,
        "source_version": source_version,
        "max_tokens": max_tokens,
        "seed": seed,
        "selected_documents": len(selected_documents),
        "selected_samples": len(selected_indices),
        "selected_tokens": selected_tokens,
    }
    if dry_run:
        return summary

    selected_indices.sort()
    metadata = {
        **source_metadata,
        "validation_subset": {
            **summary,
            "source_dataset_id": source_dataset_id,
        },
    }
    writer = BinarySampleWriter(
        output,
        dataset_id=f"{source_dataset_id}-validation-subset",
        sample_type="pretraining",
        version=source_version,
        metadata=metadata,
    )
    try:
        for index in selected_indices:
            writer.write(samples[index])
        writer.commit()
    except Exception:
        writer.abort()
        raise
    return summary


__all__ = ["subset_validation_dataset"]
