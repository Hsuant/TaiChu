"""Parallel document-range tokenization and binary shard writing."""
from __future__ import annotations

import hashlib
import multiprocessing
import os
import shutil
import sqlite3
from array import array
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Mapping

from tqdm import tqdm

from ...storage import BinarySampleWriter
from ...storage.manifest import DatasetManifest, ShardInfo
from .config import PretrainingConfig
from .segmentation import build_segment_samples
from .tokenization import (
    get_worker_tokenizer,
    initialize_tokenizer_worker,
    tokenize_batch,
)


_WORKER_CONFIG: PretrainingConfig | None = None
_WORKER_DATABASE: Path | None = None
_WORKER_PART_ROOT: Path | None = None
_WORKER_BOS_TOKEN_ID = 0
_WORKER_EOS_TOKEN_ID = 0
_WORKER_PROGRESS_QUEUE: Any | None = None


@dataclass(frozen=True, slots=True)
class PartResult:
    part_index: int
    start: int
    end: int
    token_counts: bytes
    split_paths: dict[str, str]
    stats: dict[str, dict[str, int]]


def _new_stats() -> dict[str, dict[str, int]]:
    return {
        split: {
            "documents": 0,
            "segments": 0,
            "content_tokens": 0,
            "input_tokens": 0,
            "loss_tokens": 0,
            "overlap_tokens": 0,
            "special_tokens": 0,
        }
        for split in ("train", "validation", "diagnostic")
    }


def initialize_parallel_output_worker(
    config: PretrainingConfig,
    database_path: str,
    part_root: str,
    tokenizer_version: str,
    processing_version: str,
    progress_queue: Any,
) -> None:
    global _WORKER_CONFIG, _WORKER_DATABASE, _WORKER_PART_ROOT
    global _WORKER_BOS_TOKEN_ID, _WORKER_EOS_TOKEN_ID
    global _WORKER_PROGRESS_QUEUE
    _WORKER_CONFIG = config
    _WORKER_DATABASE = Path(database_path)
    _WORKER_PART_ROOT = Path(part_root)
    _WORKER_PROGRESS_QUEUE = progress_queue
    initialize_tokenizer_worker(
        str(config.tokenizer_path),
        config.pad_token,
        config.bos_token,
        config.eos_token,
    )
    worker_tokenizer = get_worker_tokenizer()
    if (
        worker_tokenizer.bos_token_id is None
        or worker_tokenizer.eos_token_id is None
    ):
        raise RuntimeError("output worker tokenizer is missing BOS or EOS")
    _WORKER_BOS_TOKEN_ID = int(worker_tokenizer.bos_token_id)
    _WORKER_EOS_TOKEN_ID = int(worker_tokenizer.eos_token_id)
    _WORKER_TOKENIZER_VERSION = tokenizer_version
    _WORKER_PROCESSING_VERSION = processing_version


_WORKER_TOKENIZER_VERSION = ""
_WORKER_PROCESSING_VERSION = ""


def _process_part(part_index: int, start: int, end: int) -> PartResult:
    if (
        _WORKER_CONFIG is None
        or _WORKER_DATABASE is None
        or _WORKER_PART_ROOT is None
    ):
        raise RuntimeError("parallel output worker was not initialized")

    config = _WORKER_CONFIG
    database = sqlite3.connect(
        _WORKER_DATABASE.resolve().as_uri() + "?mode=ro&immutable=1",
        uri=True,
        timeout=30,
    )
    database.row_factory = sqlite3.Row
    writers: dict[str, BinarySampleWriter] = {}
    split_paths: dict[str, str] = {}
    stats = _new_stats()
    token_counts = array("I")

    def writer_for(split: str) -> BinarySampleWriter:
        writer = writers.get(split)
        if writer is not None:
            return writer
        output = _WORKER_PART_ROOT / f"part-{part_index:05d}" / split
        writer = BinarySampleWriter(
            output,
            dataset_id=f"{config.dataset_id}:{split}:part-{part_index:05d}",
            sample_type="pretraining",
            version=config.version,
            shard_size=config.shard_size,
            sync_on_close=False,
            checksums_on_commit=False,
            metadata={
                "task_type": config.task_type,
                "split": split,
                "part_index": part_index,
                "processing_version": _WORKER_PROCESSING_VERSION,
                "tokenizer_path": str(config.tokenizer_path),
                "tokenizer_version": _WORKER_TOKENIZER_VERSION,
                "segmentation": asdict(config.segmentation),
            },
        )
        writers[split] = writer
        split_paths[split] = str(output)
        return writer

    def consume(rows: list[sqlite3.Row]) -> None:
        texts = [str(row["text"]) for row in rows]
        token_rows = tokenize_batch(texts)
        for row, token_ids in zip(rows, token_rows, strict=True):
            document_id = int(row["id"])
            token_counts.append(len(token_ids))
            if document_id != int(row["canonical_id"]):
                continue
            split = str(row["split"])
            writer = writer_for(split)
            samples = build_segment_samples(
                token_ids,
                dataset_id=config.dataset_id,
                parent_document_id=str(row["parent_document_id"]),
                source_id=str(row["source_name"]),
                language=str(row["language"]),
                split=split,
                tokenizer_version=_WORKER_TOKENIZER_VERSION,
                bos_token_id=_WORKER_BOS_TOKEN_ID,
                eos_token_id=_WORKER_EOS_TOKEN_ID,
                segmentation=config.segmentation,
                processing_version=_WORKER_PROCESSING_VERSION,
            )
            stats[split]["documents"] += 1
            stats[split]["content_tokens"] += len(token_ids)
            for sample in samples:
                writer.write(sample, trusted_arrays=True)
                meta = sample.metadata
                stats[split]["segments"] += 1
                stats[split]["input_tokens"] += int(meta["input_token_count"])
                stats[split]["loss_tokens"] += int(meta["loss_token_count"])
                stats[split]["overlap_tokens"] += int(meta["overlap_token_count"])
                stats[split]["special_tokens"] += (
                    int(meta["has_bos"]) + int(meta["has_eos"])
                )

    try:
        cursor = database.execute(
            """
            SELECT id, parent_document_id, source_name, language, text,
                   split, canonical_id
            FROM documents
            WHERE id >= ? AND id < ?
            ORDER BY id
            """,
            (start, end),
        )
        batch: list[sqlite3.Row] = []
        batch_characters = 0
        for row in cursor:
            batch.append(row)
            batch_characters += len(str(row["text"]))
            if len(batch) >= config.fingerprint_batch_size or batch_characters >= 4_000_000:
                completed = len(batch)
                consume(batch)
                if _WORKER_PROGRESS_QUEUE is not None:
                    _WORKER_PROGRESS_QUEUE.put((part_index, completed))
                batch = []
                batch_characters = 0
        if batch:
            completed = len(batch)
            consume(batch)
            if _WORKER_PROGRESS_QUEUE is not None:
                _WORKER_PROGRESS_QUEUE.put((part_index, completed))

        for split, writer in writers.items():
            writer.commit()
        expected = end - start
        if len(token_counts) != expected:
            raise RuntimeError(
                f"part {part_index} processed {len(token_counts)} documents, "
                f"expected {expected}"
            )
        return PartResult(
            part_index=part_index,
            start=start,
            end=end,
            token_counts=token_counts.tobytes(),
            split_paths=split_paths,
            stats=stats,
        )
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise
    finally:
        database.close()


def _final_split_metadata(
    config: PretrainingConfig,
    *,
    split: str,
    tokenizer_version: str,
    processing_version: str,
) -> dict[str, Any]:
    return {
        "task_type": config.task_type,
        "split": split,
        "processing_version": processing_version,
        "tokenizer_path": str(config.tokenizer_path),
        "tokenizer_version": tokenizer_version,
        "segmentation": asdict(config.segmentation),
    }


def _merge_parts(
    *,
    config: PretrainingConfig,
    staging: Path,
    results: Mapping[int, PartResult],
    tokenizer_version: str,
    processing_version: str,
) -> dict[str, Any]:
    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    manifests: dict[str, Any] = {}
    for split in ("train", "validation", "diagnostic"):
        split_dir = staging / split
        split_dir.mkdir(parents=True, exist_ok=True)
        shards: list[ShardInfo] = []
        stats = _new_stats()[split]
        for part_index in sorted(results):
            result = results[part_index]
            part_path = result.split_paths.get(split)
            if part_path is None:
                continue
            part_manifest = DatasetManifest.load(part_path)
            for shard in part_manifest.shards:
                number = len(shards)
                data_name = f"data-{number:05d}.tcbin"
                index_name = f"data-{number:05d}.tcidx"
                os.replace(
                    Path(part_path) / shard.path,
                    split_dir / data_name,
                )
                os.replace(
                    Path(part_path) / shard.index_path,
                    split_dir / index_name,
                )
                data_path = split_dir / data_name
                index_path = split_dir / index_name
                shards.append(
                    ShardInfo(
                        path=data_name,
                        index_path=index_name,
                        samples=shard.samples,
                        bytes=data_path.stat().st_size,
                        sha256=sha256_file(data_path),
                        index_sha256=sha256_file(index_path),
                    )
                )
            for key in stats:
                stats[key] += int(result.stats[split][key])

        if not shards:
            split_dir.rmdir()
            continue
        manifest = DatasetManifest(
            dataset_id=f"{config.dataset_id}:{split}",
            sample_type="pretraining",
            version=config.version,
            sample_count=sum(shard.samples for shard in shards),
            shards=tuple(shards),
            metadata=_final_split_metadata(
                config,
                split=split,
                tokenizer_version=tokenizer_version,
                processing_version=processing_version,
            ),
        ).with_content_hash()
        manifest.write(split_dir)
        storage_bytes = sum(
            shard.bytes + (split_dir / shard.index_path).stat().st_size
            for shard in shards
        )
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


def _persist_token_counts(
    database: sqlite3.Connection,
    results: Mapping[int, PartResult],
) -> None:
    total = sum(result.end - result.start for result in results.values())
    with tqdm(
        total=total,
        desc="persist token counts",
        unit="doc",
        dynamic_ncols=True,
    ) as progress:
        updates: list[tuple[int, int]] = []
        for part_index in sorted(results):
            result = results[part_index]
            counts = array("I")
            counts.frombytes(result.token_counts)
            updates.extend(
                (int(count), result.start + offset)
                for offset, count in enumerate(counts)
            )
            progress.update(len(counts))
            if len(updates) >= 100_000:
                database.executemany(
                    "UPDATE documents SET cleaned_tokens = ? WHERE id = ?",
                    updates,
                )
                database.commit()
                updates.clear()
        if updates:
            database.executemany(
                "UPDATE documents SET cleaned_tokens = ? WHERE id = ?",
                updates,
            )
        database.commit()


def write_parallel_splits(
    *,
    config: PretrainingConfig,
    tokenizer_version: str,
    database: sqlite3.Connection,
    document_count: int,
    staging: Path,
    processing_version: str,
    database_path: Path,
) -> dict[str, Any]:
    for split in ("train", "validation", "diagnostic"):
        split_path = staging / split
        writer_staging = split_path.with_name(f".{split_path.name}.staging")
        for path in (split_path, writer_staging):
            if path.exists():
                shutil.rmtree(path)

    part_root = staging / ".parallel-splits"
    if part_root.exists():
        shutil.rmtree(part_root)
    part_root.mkdir(parents=True)

    database.commit()
    requested_workers = config.output_workers or config.processing_workers
    workers = min(requested_workers, os.cpu_count() or 1)
    part_count = min(max(workers, workers * 8), max(1, document_count))
    range_size = max(1, (document_count + part_count - 1) // part_count)
    ranges = [
        (index, start, min(start + range_size, document_count))
        for index, start in enumerate(range(0, document_count, range_size))
    ]
    results: dict[int, PartResult] = {}

    previous_rayon_threads = os.environ.get("RAYON_NUM_THREADS")
    os.environ["RAYON_NUM_THREADS"] = "1"
    progress_queue = multiprocessing.Queue()
    try:
        with tqdm(
            total=document_count,
            desc="tokenize+shard",
            unit="doc",
            dynamic_ncols=True,
        ) as progress:
            reported_by_part = {part_index: 0 for part_index, _, _ in ranges}
            completed_parts: set[int] = set()

            def flush_progress() -> None:
                while True:
                    try:
                        part_index, completed = progress_queue.get_nowait()
                    except Empty:
                        return
                    if part_index in completed_parts:
                        continue
                    progress.update(completed)
                    reported_by_part[part_index] += completed

            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=initialize_parallel_output_worker,
                initargs=(
                    config,
                    str(database_path),
                    str(part_root),
                    tokenizer_version,
                    processing_version,
                    progress_queue,
                ),
            ) as executor:
                futures = {
                    executor.submit(_process_part, part_index, start, end): (
                        part_index,
                        start,
                        end,
                    )
                    for part_index, start, end in ranges
                }
                try:
                    pending = set(futures)
                    while pending:
                        finished, pending = wait(
                            pending,
                            timeout=0.5,
                            return_when=FIRST_COMPLETED,
                        )
                        flush_progress()
                        for future in finished:
                            result = future.result()
                            results[result.part_index] = result
                            remaining = (
                                result.end
                                - result.start
                                - reported_by_part[result.part_index]
                            )
                            if remaining:
                                progress.update(remaining)
                            completed_parts.add(result.part_index)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise

        _persist_token_counts(database, results)
        manifests = _merge_parts(
            config=config,
            staging=staging,
            results=results,
            tokenizer_version=tokenizer_version,
            processing_version=processing_version,
        )
        shutil.rmtree(part_root)
        return manifests
    except BaseException:
        shutil.rmtree(part_root, ignore_errors=True)
        for split in ("train", "validation", "diagnostic"):
            shutil.rmtree(staging / split, ignore_errors=True)
        raise
    finally:
        progress_queue.close()
        if previous_rayon_threads is None:
            os.environ.pop("RAYON_NUM_THREADS", None)
        else:
            os.environ["RAYON_NUM_THREADS"] = previous_rayon_threads


__all__ = ["write_parallel_splits"]
