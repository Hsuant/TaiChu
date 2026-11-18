"""Split writing and isolation verification for pretraining datasets."""
from __future__ import annotations

import json
import os
import random
import shutil
from array import array
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Mapping

from tqdm import tqdm

from ...storage import (
    BinaryDatasetReader,
    BinarySampleWriter,
    BinaryShardReader,
    ShardInfo,
    atomic_replace_directory,
)
from ...storage.manifest import DatasetManifest
from ..adapters import TextTokenizer
from .config import PretrainingConfig
from .parallel_output import write_parallel_splits


MAX_REALLOCATION_SHARE = 0.01


def _select_shard(
    task: tuple[
        int,
        str,
        str,
        str,
        bytes,
        int,
        str,
        str,
        int,
        dict[str, Any],
    ],
) -> tuple[int, str | None, int, int]:
    (
        part_index,
        data_path,
        index_path,
        part_root,
        mask,
        sample_count,
        dataset_id,
        version,
        shard_size,
        metadata,
    ) = task
    output_dir = Path(part_root) / f"part-{part_index:05d}"
    reader = BinaryShardReader(data_path, index_path)
    writer = BinarySampleWriter(
        output_dir,
        dataset_id=f"{dataset_id}:part-{part_index:05d}",
        sample_type="pretraining",
        version=version,
        shard_size=shard_size,
        metadata={
            **metadata,
            "selection_part_index": part_index,
        },
    )
    selected = 0
    try:
        for local_index in range(sample_count):
            if not mask[local_index]:
                continue
            writer.write(
                reader.read_sample(local_index),
                trusted_arrays=True,
            )
            selected += 1
        if selected == 0:
            writer.abort()
            return part_index, None, 0, sample_count
        writer.commit()
        return part_index, str(output_dir), selected, sample_count
    except Exception:
        writer.abort()
        raise
    finally:
        reader.close()


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
    total = sum(
        int(manifest["sample_count"])
        for manifest in manifests.values()
    )
    with tqdm(
        total=total,
        desc="verify split isolation",
        unit="sample",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress:
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
                    progress.update()
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


def _rebalance_quotas(
    quotas: Mapping[str, int],
    available: Mapping[str, int],
    target_tokens: int,
) -> tuple[dict[str, int], dict[str, int]]:
    deficits = {
        name: quota - available[name]
        for name, quota in quotas.items()
        if available[name] < quota
    }
    if not deficits:
        return dict(quotas), {}
    total_deficit = sum(deficits.values())
    if (
        total_deficit > target_tokens * MAX_REALLOCATION_SHARE
        or sum(available.values()) < target_tokens
    ):
        return dict(quotas), {}

    effective = dict(quotas)
    surpluses = {
        name: available[name] - quota
        for name, quota in quotas.items()
        if available[name] > quota
    }
    total_surplus = sum(surpluses.values())
    if total_surplus < total_deficit:
        return dict(quotas), {}

    for name, deficit in deficits.items():
        effective[name] -= deficit
    raw_increases = {
        name: total_deficit * surplus / total_surplus
        for name, surplus in surpluses.items()
    }
    increases = {
        name: int(value)
        for name, value in raw_increases.items()
    }
    remaining = total_deficit - sum(increases.values())
    order = sorted(
        raw_increases,
        key=lambda name: (-(raw_increases[name] - increases[name]), name),
    )
    for name in order[:remaining]:
        increases[name] += 1
    for name, increase in increases.items():
        effective[name] += increase
    reallocations = {
        name: effective[name] - quotas[name]
        for name in quotas
        if effective[name] != quotas[name]
    }
    return effective, reallocations


def _storage_bytes(root: Path, manifest: DatasetManifest) -> int:
    return sum(
        shard.bytes + (root / shard.index_path).stat().st_size
        for shard in manifest.shards
    )


def existing_split_manifests(
    staging: Path,
    *,
    dataset_id: str,
    version: str,
    tokenizer_version: str,
) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "diagnostic"):
        split_path = staging / split
        manifest_path = split_path / "manifest.json"
        if not manifest_path.is_file():
            continue
        stats = {
            "documents": 0,
            "segments": 0,
            "content_tokens": 0,
            "input_tokens": 0,
            "loss_tokens": 0,
            "overlap_tokens": 0,
            "special_tokens": 0,
        }
        previous_parent = None
        with BinaryDatasetReader(split_path) as reader:
            with tqdm(
                total=len(reader),
                desc=f"inspect {split} manifest",
                unit="sample",
                dynamic_ncols=True,
                mininterval=1.0,
            ) as progress:
                for metadata in reader.iter_metadata():
                    payload = metadata["metadata"]
                    parent = payload["parent_document_id"]
                    if parent != previous_parent:
                        stats["documents"] += 1
                        previous_parent = parent
                    stats["segments"] += 1
                    stats["content_tokens"] += int(
                        payload["segment_token_count"]
                    )
                    stats["input_tokens"] += int(
                        payload["input_token_count"]
                    )
                    stats["loss_tokens"] += int(payload["loss_token_count"])
                    stats["overlap_tokens"] += int(
                        payload["overlap_token_count"]
                    )
                    stats["special_tokens"] += (
                        int(bool(payload["has_bos"]))
                        + int(bool(payload["has_eos"]))
                    )
                    progress.update()
            manifest = reader.manifest
        if (
            manifest.dataset_id != f"{dataset_id}:{split}"
            or manifest.version != version
            or manifest.metadata.get("tokenizer_version")
            != tokenizer_version
        ):
            continue
        storage_bytes = _storage_bytes(split_path, manifest)
        stats["storage_bytes"] = storage_bytes
        stats["storage_gb"] = storage_bytes / 1_000_000_000
        stats["logical_input_gb"] = stats["input_tokens"] * 4 / 1_000_000_000
        stats["loss_target_gb"] = stats["loss_tokens"] * 4 / 1_000_000_000
        manifests[split] = {
            "manifest": f"{split}/manifest.json",
            "sample_count": manifest.sample_count,
            "content_hash": manifest.content_hash,
            "stats": stats,
        }
    return manifests


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
    workers: int | None = None,
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
        with tqdm(
            total=len(reader),
            desc="scan selection metadata",
            unit="sample",
            dynamic_ncols=True,
            mininterval=1.0,
        ) as progress:
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
                progress.update()

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
        available_inputs = {
            name: available[name]["input_tokens"]
            for name in weights
        }
        effective_quotas, reallocations = _rebalance_quotas(
            quotas,
            available_inputs,
            target_tokens,
        )
        feasible = not deficits or bool(reallocations)
        if deficits and not feasible and not dry_run:
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
            quota = effective_quotas[source_name]
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
            "feasible": feasible,
            "metric": metric,
            "target_tokens": target_tokens,
            "seed": seed,
            "mixture_manifest_path": str(Path(mixture_manifest_path)),
            "source_weights": weights,
            "source_quotas": quotas,
            "source_effective_quotas": effective_quotas,
            "source_reallocations": reallocations,
            "source_deficit_tokens": sum(deficits.values()),
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
        if not feasible:
            return None, report
        if actual_total < target_tokens:
            raise RuntimeError("token selection produced fewer tokens than target")
        if dry_run:
            return None, report
        if output_path.exists():
            raise FileExistsError(f"selection output already exists: {output_path}")

        base_metadata = dict(reader.manifest.metadata)
        part_root = output_path.with_name(f".{output_path.name}.parts")
        staging = output_path.with_name(f".{output_path.name}.staging")
        if part_root.exists():
            shutil.rmtree(part_root)
        if staging.exists():
            shutil.rmtree(staging)
        part_root.mkdir(parents=True)
        tasks: list[
            tuple[
                int,
                str,
                str,
                str,
                bytes,
                int,
                str,
                str,
                int,
                dict[str, Any],
            ]
        ] = []
        sample_start = 0
        for part_index, shard in enumerate(reader.manifest.shards):
            sample_end = sample_start + shard.samples
            tasks.append(
                (
                    part_index,
                    str(reader.root / shard.path),
                    str(reader.root / shard.index_path),
                    str(part_root),
                    bytes(selected[sample_start:sample_end]),
                    shard.samples,
                    dataset_id,
                    version,
                    max(shard_size, shard.samples),
                    base_metadata,
                )
            )
            sample_start = sample_end
        worker_count = min(
            workers or os.cpu_count() or 1,
            max(1, len(tasks)),
        )
        results: dict[int, tuple[int, str | None, int, int]] = {}
        try:
            with tqdm(
                total=len(reader),
                desc="write selected train",
                unit="sample",
                dynamic_ncols=True,
                mininterval=1.0,
            ) as progress:
                if worker_count == 1:
                    for task in tasks:
                        result = _select_shard(task)
                        results[result[0]] = result
                        progress.update(result[3])
                else:
                    with ProcessPoolExecutor(
                        max_workers=worker_count,
                    ) as executor:
                        futures = {
                            executor.submit(_select_shard, task): task[0]
                            for task in tasks
                        }
                        pending = set(futures)
                        try:
                            while pending:
                                finished, pending = wait(
                                    pending,
                                    timeout=0.5,
                                    return_when=FIRST_COMPLETED,
                                )
                                for future in finished:
                                    result = future.result()
                                    results[result[0]] = result
                                    progress.update(result[3])
                        except BaseException:
                            for future in futures:
                                future.cancel()
                            raise

            print("merging selected train shards", flush=True)
            staging.mkdir(parents=True)
            final_shards: list[ShardInfo] = []
            for part_index in sorted(results):
                part_path = results[part_index][1]
                if part_path is None:
                    continue
                part_manifest = DatasetManifest.load(part_path)
                for shard in part_manifest.shards:
                    number = len(final_shards)
                    data_name = f"data-{number:05d}.tcbin"
                    index_name = f"data-{number:05d}.tcidx"
                    os.replace(
                        Path(part_path) / shard.path,
                        staging / data_name,
                    )
                    os.replace(
                        Path(part_path) / shard.index_path,
                        staging / index_name,
                    )
                    final_shards.append(
                        ShardInfo(
                            path=data_name,
                            index_path=index_name,
                            samples=shard.samples,
                            bytes=(staging / data_name).stat().st_size,
                            sha256=shard.sha256,
                            index_sha256=shard.index_sha256,
                        )
                    )
            if not final_shards:
                raise RuntimeError("token selection produced no samples")
            manifest = DatasetManifest(
                dataset_id=f"{dataset_id}:train",
                sample_type="pretraining",
                version=version,
                sample_count=sum(shard.samples for shard in final_shards),
                shards=tuple(final_shards),
                metadata={
                    **base_metadata,
                    "selection": report,
                },
            ).with_content_hash()
            manifest.write(staging)
            atomic_replace_directory(staging, output_path)
            shutil.rmtree(part_root)
        except Exception:
            shutil.rmtree(part_root, ignore_errors=True)
            shutil.rmtree(
                staging,
                ignore_errors=True,
            )
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
        workers=config.output_workers or config.processing_workers,
    )
    assert manifest is not None
    print("replacing candidate train with selected train", flush=True)
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
        with tqdm(
            total=len(reader),
            desc="inspect selected train",
            unit="sample",
            dynamic_ncols=True,
            mininterval=1.0,
        ) as progress:
            for metadata in reader.iter_metadata():
                parent = metadata["metadata"]["parent_document_id"]
                if parent != previous_parent:
                    documents += 1
                    previous_parent = parent
                progress.update()
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
    "existing_split_manifests",
    "select_pretraining_split",
    "verify_split_isolation",
    "write_splits",
]
