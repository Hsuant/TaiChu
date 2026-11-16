"""Split writing and isolation verification for pretraining datasets."""
from __future__ import annotations

import json
import random
import shutil
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from ...storage import BinaryDatasetReader, BinarySampleWriter
from ...storage.manifest import DatasetManifest
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


def _load_mixture_weights(path: str | Path) -> dict[str, float]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("mixture manifest needs a non-empty sources list")
    weights: dict[str, float] = {}
    for source in raw_sources:
        if not isinstance(source, Mapping):
            raise ValueError("mixture manifest source entries must be mappings")
        name = str(source.get("name", ""))
        if not name:
            raise ValueError("mixture manifest source name cannot be empty")
        value = source.get("target_bytes", source.get("weight_gb"))
        weight = float(value)
        if weight <= 0:
            raise ValueError(
                f"mixture manifest source {name!r} needs a positive target"
            )
        weights[name] = weight
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


def _allocate_token_quotas(
    weights: Mapping[str, float],
    target_tokens: int,
) -> dict[str, int]:
    if target_tokens <= 0:
        raise ValueError("selection target_tokens must be positive")
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("selection source weights must have positive total")
    raw = {
        name: target_tokens * float(weight) / total_weight
        for name, weight in weights.items()
    }
    quotas = {name: int(value) for name, value in raw.items()}
    remaining = target_tokens - sum(quotas.values())
    order = sorted(
        raw,
        key=lambda name: (-(raw[name] - quotas[name]), name),
    )
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


def _storage_bytes(root: Path, manifest: DatasetManifest) -> int:
    return sum(
        shard.bytes + (root / shard.index_path).stat().st_size
        for shard in manifest.shards
    )


def select_pretraining_split(
    source: str | Path,
    output: str | Path,
    *,
    target_tokens: int,
    mixture_manifest_path: str | Path,
    dataset_id: str,
    version: str,
    metric: str = "input_tokens",
    seed: int = 42,
    shard_size: int = 50_000,
    dry_run: bool = False,
) -> tuple[DatasetManifest | None, dict[str, Any]]:
    """Select pretraining windows to source token quotas.

    Selection is deterministic at window granularity. The final source may
    overshoot by one window, so callers can validate the reported tolerance
    before publishing the dataset.
    """
    if metric != "input_tokens":
        raise ValueError("pretraining token selection supports input_tokens only")
    if seed < 0:
        raise ValueError("selection seed cannot be negative")
    weights = _load_mixture_weights(mixture_manifest_path)
    quotas = _allocate_token_quotas(weights, target_tokens)

    source_path = Path(source)
    output_path = Path(output)
    with BinaryDatasetReader(source_path) as reader:
        if reader.sample_type != "pretraining":
            raise ValueError("token selection requires a pretraining dataset")
        indices_by_source: dict[str, list[int]] = defaultdict(list)
        input_counts = array("I")
        loss_counts = array("I")
        content_counts = array("I")
        overlap_counts = array("I")
        special_counts = bytearray()
        available: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "samples": 0,
                "content_tokens": 0,
                "input_tokens": 0,
                "loss_tokens": 0,
                "overlap_tokens": 0,
                "special_tokens": 0,
            }
        )
        for index in range(len(reader)):
            metadata = reader.metadata_at(index)
            source_name = str(metadata.get("source_id", ""))
            if not source_name:
                raise ValueError(f"sample {index} has no source_id")
            sample_metadata = metadata.get("metadata", {})
            input_tokens = int(sample_metadata["input_token_count"])
            loss_tokens = int(sample_metadata["loss_token_count"])
            content_tokens = int(sample_metadata["segment_token_count"])
            overlap_tokens = int(sample_metadata["overlap_token_count"])
            special_tokens = (
                int(bool(sample_metadata["has_bos"]))
                + int(bool(sample_metadata["has_eos"]))
            )
            input_counts.append(input_tokens)
            loss_counts.append(loss_tokens)
            content_counts.append(content_tokens)
            overlap_counts.append(overlap_tokens)
            special_counts.append(special_tokens)
            indices_by_source[source_name].append(index)
            available[source_name]["samples"] += 1
            available[source_name]["content_tokens"] += content_tokens
            available[source_name]["input_tokens"] += input_tokens
            available[source_name]["loss_tokens"] += loss_tokens
            available[source_name]["overlap_tokens"] += overlap_tokens
            available[source_name]["special_tokens"] += special_tokens

        candidate_sources = set(indices_by_source)
        target_sources = set(weights)
        missing = sorted(target_sources - candidate_sources)
        unexpected = sorted(candidate_sources - target_sources)
        if missing or unexpected:
            raise ValueError(
                "candidate/target source mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )

        deficits = {
            source_name: quotas[source_name]
            - available[source_name]["input_tokens"]
            for source_name in weights
            if available[source_name]["input_tokens"] < quotas[source_name]
        }
        if deficits and not dry_run:
            details = ", ".join(
                f"{name}: deficit={deficit:,}"
                for name, deficit in sorted(deficits.items())
            )
            raise ValueError(
                f"candidate dataset cannot satisfy token quotas: {details}"
            )

        selected = bytearray(len(reader))
        actual: dict[str, dict[str, int]] = {}
        for source_index, source_name in enumerate(sorted(weights)):
            available_tokens = available[source_name]["input_tokens"]
            quota = quotas[source_name]
            if available_tokens < quota:
                continue
            indices = indices_by_source[source_name]
            random.Random(seed + source_index * 1_000_003).shuffle(indices)
            source_tokens = 0
            source_loss_tokens = 0
            source_content_tokens = 0
            source_overlap_tokens = 0
            source_special_tokens = 0
            sample_count = 0
            for index in indices:
                if source_tokens >= quota:
                    break
                source_tokens += input_counts[index]
                source_loss_tokens += loss_counts[index]
                source_content_tokens += content_counts[index]
                source_overlap_tokens += overlap_counts[index]
                source_special_tokens += special_counts[index]
                sample_count += 1
                selected[index] = 1
            actual[source_name] = {
                "samples": sample_count,
                "content_tokens": source_content_tokens,
                "input_tokens": source_tokens,
                "loss_tokens": source_loss_tokens,
                "overlap_tokens": source_overlap_tokens,
                "special_tokens": source_special_tokens,
            }

        actual_total = sum(item["input_tokens"] for item in actual.values())
        actual_loss = sum(item["loss_tokens"] for item in actual.values())
        report: dict[str, Any] = {
            "dry_run": dry_run,
            "feasible": not deficits,
            "metric": metric,
            "target_tokens": target_tokens,
            "seed": seed,
            "mixture_manifest_path": str(Path(mixture_manifest_path)),
            "source_weights": weights,
            "source_quotas": quotas,
            "source_available": dict(available),
            "source_deficits": deficits,
            "source_actual": actual,
            "source_input_token_shares": {
                name: item["input_tokens"] / actual_total
                for name, item in actual.items()
            } if actual_total else {},
            "source_loss_token_shares": {
                name: item["loss_tokens"] / actual_loss
                for name, item in actual.items()
            } if actual_loss else {},
            "selected_samples": sum(item["samples"] for item in actual.values()),
            "actual_total_input_tokens": actual_total,
            "actual_total_loss_tokens": actual_loss,
            "actual_total_overlap_tokens": sum(
                item["overlap_tokens"] for item in actual.values()
            ),
            "actual_total_special_tokens": sum(
                item["special_tokens"] for item in actual.values()
            ),
            "overshoot_tokens": actual_total - target_tokens,
        }
        if deficits:
            return None, report
        if actual_total < target_tokens:
            raise RuntimeError("token selection produced fewer tokens than target")
        if dry_run:
            return None, report
        if output_path.exists():
            raise FileExistsError(f"selection output already exists: {output_path}")

        writer = BinarySampleWriter(
            output_path,
            dataset_id=f"{dataset_id}:train",
            sample_type="pretraining",
            version=version,
            shard_size=shard_size,
            metadata={
                **dict(reader.manifest.metadata),
                "selection": report,
            },
        )
        try:
            for index, sample in enumerate(reader):
                if selected[index]:
                    writer.write(sample, trusted_arrays=True)
            manifest = writer.commit()
        except Exception:
            writer.abort()
            raise

    report["storage_bytes"] = _storage_bytes(output_path, manifest)
    report["storage_gb"] = report["storage_bytes"] / 1_000_000_000
    return manifest, report


def apply_training_selection(
    *,
    config: PretrainingConfig,
    staging: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.selection is None:
        raise ValueError("training selection config is required")
    source = staging / "train"
    selected = staging / ".train.selected"
    manifest, report = select_pretraining_split(
        source,
        selected,
        target_tokens=config.selection.target_tokens,
        mixture_manifest_path=config.selection.mixture_manifest_path,
        dataset_id=config.dataset_id,
        version=config.version,
        metric=config.selection.metric,
        seed=config.selection.seed,
        shard_size=config.shard_size,
    )
    assert manifest is not None
    shutil.rmtree(source)
    selected.replace(source)
    input_tokens = int(report["actual_total_input_tokens"])
    loss_tokens = int(report["actual_total_loss_tokens"])
    overlap_tokens = int(report["actual_total_overlap_tokens"])
    special_tokens = int(report["actual_total_special_tokens"])
    samples = int(report["selected_samples"])
    documents = 0
    previous_parent = None
    with BinaryDatasetReader(source) as reader:
        for metadata in reader.iter_metadata():
            parent = metadata["metadata"]["parent_document_id"]
            if parent != previous_parent:
                documents += 1
                previous_parent = parent
    stats = {
        "documents": documents,
        "segments": samples,
        "content_tokens": sum(
            int(item["content_tokens"])
            for item in report["source_actual"].values()
        ),
        "input_tokens": input_tokens,
        "loss_tokens": loss_tokens,
        "overlap_tokens": overlap_tokens,
        "special_tokens": special_tokens,
        "storage_bytes": int(report["storage_bytes"]),
        "storage_gb": float(report["storage_gb"]),
        "logical_input_gb": input_tokens * 4 / 1_000_000_000,
        "loss_target_gb": loss_tokens * 4 / 1_000_000_000,
    }
    entry = {
        "manifest": "train/manifest.json",
        "sample_count": manifest.sample_count,
        "content_hash": manifest.content_hash,
        "stats": stats,
    }
    return entry, report


__all__ = [
    "apply_training_selection",
    "select_pretraining_split",
    "verify_split_isolation",
    "write_splits",
]
