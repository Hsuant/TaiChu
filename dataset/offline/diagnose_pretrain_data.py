#!/usr/bin/env python3
"""Summarize token, source, duplicate, and packing statistics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from ..storage import BinaryDatasetReader


ProgressCallback = Callable[[int, int], None]


def _percentile(sorted_values: list[int], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(
        len(sorted_values) - 1,
        max(0, int(math.ceil(fraction * len(sorted_values))) - 1),
    )
    return float(sorted_values[index])


def _sample_indices(total: int, sample_size: int, seed: int) -> list[int]:
    if total < 0:
        raise ValueError("total sample count cannot be negative")
    if sample_size < 0:
        raise ValueError("sample_size cannot be negative")
    if sample_size == 0 or sample_size >= total:
        indices = list(range(total))
    else:
        indices = sorted(random.Random(seed).sample(range(total), sample_size))
    return indices


def _source_name(source_id: str) -> str:
    name = Path(source_id).name if source_id else "<unknown>"
    return name or "<unknown>"


def _language(source_id: str) -> str:
    name = _source_name(source_id).lower()
    if "_ch" in name or "_zh" in name or "chinese" in name:
        return "ch"
    if "code" in name:
        return "code"
    if "_en" in name or "english" in name:
        return "en"
    return "unknown"


def _hash_token_ids(input_ids: Iterable[int]) -> bytes:
    values = array("I", (int(token) for token in input_ids))
    if sys.byteorder != "little":
        values.byteswap()
    return hashlib.blake2b(values.tobytes(), digest_size=16).digest()


def _parse_micro_batch_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("micro batch sizes must be positive integers")
    return sizes


def _packing_stats(
    lengths: list[int],
    max_seq_length: int,
    micro_batch_size: int,
) -> dict[str, Any]:
    usable = len(lengths) - len(lengths) % micro_batch_size
    content_tokens = 0
    padded_sequences = 0
    for start in range(0, usable, micro_batch_size):
        batch_tokens = sum(lengths[start : start + micro_batch_size])
        content_tokens += batch_tokens
        padded_sequences += max(1, math.ceil(batch_tokens / max_seq_length))
    padded_tokens = padded_sequences * max_seq_length
    return {
        "micro_batch_size": micro_batch_size,
        "raw_samples": usable,
        "content_tokens": content_tokens,
        "packed_sequences": padded_sequences,
        "padded_tokens": padded_tokens,
        "padding_tokens": padded_tokens - content_tokens,
        "padding_ratio": (
            (padded_tokens - content_tokens) / padded_tokens
            if padded_tokens
            else 0.0
        ),
        "packing_efficiency": (
            content_tokens / padded_tokens if padded_tokens else 0.0
        ),
        "padded_tokens_per_raw_sample": (
            padded_tokens / usable if usable else 0.0
        ),
    }


def _optimizer_step_stats(
    lengths: list[int],
    *,
    max_seq_length: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
) -> dict[str, Any]:
    micro_batches = gradient_accumulation_steps * world_size
    global_raw_samples = micro_batch_size * micro_batches
    windows = len(lengths) // global_raw_samples
    content_tokens = 0
    padded_sequences = 0
    padding_tokens = 0
    for window in range(windows):
        window_start = window * global_raw_samples
        for micro_index in range(micro_batches):
            start = window_start + micro_index * micro_batch_size
            batch_tokens = sum(lengths[start : start + micro_batch_size])
            sequences = max(1, math.ceil(batch_tokens / max_seq_length))
            padded = sequences * max_seq_length
            content_tokens += batch_tokens
            padded_sequences += sequences
            padding_tokens += padded - batch_tokens
    return {
        "world_size": world_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "micro_batch_size": micro_batch_size,
        "micro_batches_per_optimizer_step": micro_batches,
        "raw_samples_per_optimizer_step": global_raw_samples,
        "nominal_tokens_per_optimizer_step": (
            global_raw_samples * max_seq_length
        ),
        "sampled_optimizer_steps": windows,
        "mean_content_tokens_per_optimizer_step": (
            content_tokens / windows if windows else 0.0
        ),
        "mean_padded_tokens_per_optimizer_step": (
            padded_sequences * max_seq_length / windows if windows else 0.0
        ),
        "mean_packed_sequences_per_optimizer_step": (
            padded_sequences / windows if windows else 0.0
        ),
        "padding_ratio": (
            padding_tokens / (padded_sequences * max_seq_length)
            if padded_sequences
            else 0.0
        ),
    }


def diagnose_pretrain_dataset(
    path: str | Path,
    *,
    sample_size: int = 100_000,
    seed: int = 42,
    max_seq_length: int = 2048,
    micro_batch_sizes: tuple[int, ...] = (2, 4, 8, 16),
    configured_micro_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    world_size: int = 2,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if max_seq_length < 2:
        raise ValueError("max_seq_length must be >= 2")
    if configured_micro_batch_size <= 0:
        raise ValueError("configured_micro_batch_size must be positive")
    if gradient_accumulation_steps <= 0 or world_size <= 0:
        raise ValueError("gradient accumulation and world size must be positive")

    with BinaryDatasetReader(path) as reader:
        indices = _sample_indices(len(reader), sample_size, seed)
        lengths: list[int] = []
        source_tokens: Counter[str] = Counter()
        source_samples: Counter[str] = Counter()
        source_lengths: dict[str, list[int]] = defaultdict(list)
        language_tokens: Counter[str] = Counter()
        duplicate_hashes: set[bytes] = set()
        text_characters = 0
        samples_with_text_characters = 0
        samples_with_labels = 0
        samples_with_attention_mask = 0
        samples_with_document_boundaries = 0
        document_boundaries = 0

        for offset, index in enumerate(indices):
            sample = reader[index]
            length = len(sample.input_ids)
            lengths.append(length)
            source = _source_name(sample.source_id)
            source_samples[source] += 1
            source_tokens[source] += length
            source_lengths[source].append(length)
            language_tokens[_language(sample.source_id)] += length
            duplicate_hashes.add(_hash_token_ids(sample.input_ids))
            if sample.labels is not None:
                samples_with_labels += 1
            if sample.attention_mask is not None:
                samples_with_attention_mask += 1
            if sample.document_ids:
                boundaries = sum(
                    previous != current
                    for previous, current in zip(
                        sample.document_ids,
                        sample.document_ids[1:],
                    )
                )
                if boundaries:
                    samples_with_document_boundaries += 1
                document_boundaries += boundaries
            value = sample.metadata.get("text_characters")
            if isinstance(value, int) and value >= 0:
                text_characters += value
                samples_with_text_characters += 1
            if progress is not None and (offset + 1) % 25_000 == 0:
                progress(offset + 1, len(indices))

        if not lengths:
            raise ValueError(f"dataset {path!s} produced no samples")

        sorted_lengths = sorted(lengths)
        sampled_tokens = sum(lengths)
        total_sampled = len(lengths)
        mean_tokens = sampled_tokens / total_sampled
        estimated_total_tokens = mean_tokens * len(reader)
        duplicate_count = total_sampled - len(duplicate_hashes)
        unique_sources = source_samples.most_common()
        source_rows = []
        for source, count in unique_sources[:50]:
            lengths_for_source = sorted(source_lengths[source])
            source_rows.append({
                "source": source,
                "samples": count,
                "tokens": source_tokens[source],
                "token_share": source_tokens[source] / sampled_tokens,
                "mean_tokens": source_tokens[source] / count,
                "length_p50": _percentile(lengths_for_source, 0.50),
                "length_p90": _percentile(lengths_for_source, 0.90),
                "under_512_ratio": (
                    sum(length < 512 for length in lengths_for_source) / count
                ),
                "exact_max_length_ratio": (
                    sum(length == max_seq_length for length in lengths_for_source)
                    / count
                ),
            })
        packing = {
            str(size): _packing_stats(lengths, max_seq_length, size)
            for size in micro_batch_sizes
        }
        optimizer_step = _optimizer_step_stats(
            lengths,
            max_seq_length=max_seq_length,
            micro_batch_size=configured_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            world_size=world_size,
        )
        sample_fraction = total_sampled / len(reader)
        warnings = []
        truncation_rate = sum(length == max_seq_length for length in lengths) / total_sampled
        duplicate_rate = duplicate_count / total_sampled
        top_source_share = (
            source_rows[0]["token_share"] if source_rows else 0.0
        )
        likely_padding_ratio = optimizer_step["padding_ratio"]

        if truncation_rate > 0.10:
            warnings.append(
                f"{truncation_rate:.1%} of sampled documents are exactly "
                f"{max_seq_length} tokens and may be truncated."
            )
        if duplicate_rate > 0.01:
            warnings.append(
                f"Exact duplicate rate is {duplicate_rate:.2%} in the sample."
            )
        if likely_padding_ratio > 0.20:
            warnings.append(
                f"Packing padding is {likely_padding_ratio:.1%} for the configured "
                "optimizer step; padded token targets are overstated."
            )
        if (
            optimizer_step["nominal_tokens_per_optimizer_step"]
            > optimizer_step["mean_padded_tokens_per_optimizer_step"] * 1.5
        ):
            warnings.append(
                "The nominal raw-sample token count overstates actual packed "
                "tokens by more than 50%; compute the schedule from measured "
                "padded-token throughput."
            )
        if mean_tokens < 512:
            warnings.append(
                f"Mean document length is only {mean_tokens:.1f} tokens; "
                "cross-document packing will be frequent."
            )
        if sum(length < 512 for length in lengths) / total_sampled > 0.50:
            warnings.append(
                "More than half of sampled documents are shorter than 512 tokens; "
                "many training sequences contain multiple unrelated documents."
            )
        if top_source_share > 0.50:
            warnings.append(
                f"Top source accounts for {top_source_share:.1%} of sampled tokens."
            )

        return {
            "path": str(Path(path)),
            "dataset_id": reader.dataset_id,
            "dataset_version": reader.version,
            "manifest": {
                "sample_count": len(reader),
                "shard_count": len(reader.manifest.shards),
                "storage_bytes": sum(shard.bytes for shard in reader.manifest.shards),
                "metadata": reader.manifest.metadata,
            },
            "sample": {
                "requested_samples": sample_size,
                "sampled_samples": total_sampled,
                "sample_fraction": sample_fraction,
                "seed": seed,
            },
            "tokens": {
                "sampled_tokens": sampled_tokens,
                "mean_tokens": mean_tokens,
                "estimated_total_tokens": estimated_total_tokens,
                "estimated_tokens_per_parameter_80m": estimated_total_tokens / 80_757_248,
                "text_characters": text_characters,
                "tokens_per_text_character": (
                    sampled_tokens / text_characters if text_characters else None
                ),
                "text_character_coverage": (
                    samples_with_text_characters / total_sampled
                ),
            },
            "lengths": {
                "min": sorted_lengths[0],
                "p10": _percentile(sorted_lengths, 0.10),
                "p50": _percentile(sorted_lengths, 0.50),
                "p90": _percentile(sorted_lengths, 0.90),
                "p95": _percentile(sorted_lengths, 0.95),
                "p99": _percentile(sorted_lengths, 0.99),
                "max": sorted_lengths[-1],
                "under_128_ratio": sum(length < 128 for length in lengths) / total_sampled,
                "under_512_ratio": sum(length < 512 for length in lengths) / total_sampled,
                "exact_max_length_ratio": truncation_rate,
            },
            "structure": {
                "samples_with_labels": samples_with_labels,
                "samples_with_attention_mask": samples_with_attention_mask,
                "samples_with_document_boundaries": samples_with_document_boundaries,
                "document_boundaries": document_boundaries,
                "document_boundaries_per_sample": document_boundaries / total_sampled,
            },
            "duplicates": {
                "sampled_unique": len(duplicate_hashes),
                "sampled_duplicates": duplicate_count,
                "duplicate_rate": duplicate_rate,
            },
            "languages": {
                language: {
                    "tokens": tokens,
                    "token_share": tokens / sampled_tokens,
                }
                for language, tokens in language_tokens.most_common()
            },
            "sources": {
                "unique_sources": len(source_samples),
                "top_by_tokens": source_rows,
            },
            "packing": packing,
            "configured_optimizer_step": optimizer_step,
            "warnings": warnings,
        }


def compare_dataset_overlap(
    source_path: str | Path,
    compare_path: str | Path,
    *,
    source_sample_size: int = 100_000,
    compare_sample_size: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    """Measure exact sample ID and token-sequence overlap between two splits."""
    with (
        BinaryDatasetReader(source_path) as source_reader,
        BinaryDatasetReader(compare_path) as compare_reader,
    ):
        source_indices = _sample_indices(
            len(source_reader),
            source_sample_size,
            seed,
        )
        compare_indices = _sample_indices(
            len(compare_reader),
            compare_sample_size,
            seed + 1,
        )
        source_ids: set[str] = set()
        source_hashes: set[bytes] = set()
        for index in source_indices:
            sample = source_reader[index]
            source_ids.add(sample.sample_id)
            source_hashes.add(_hash_token_ids(sample.input_ids))

        compare_ids: set[str] = set()
        compare_hashes: set[bytes] = set()
        for index in compare_indices:
            sample = compare_reader[index]
            compare_ids.add(sample.sample_id)
            compare_hashes.add(_hash_token_ids(sample.input_ids))

        sample_id_overlap = source_ids & compare_ids
        token_overlap = source_hashes & compare_hashes
        return {
            "source_path": str(Path(source_path)),
            "compare_path": str(Path(compare_path)),
            "source": {
                "sample_count": len(source_reader),
                "sampled": len(source_indices),
                "sample_fraction": len(source_indices) / len(source_reader),
                "unique_sample_ids": len(source_ids),
                "unique_token_sequences": len(source_hashes),
            },
            "compare": {
                "sample_count": len(compare_reader),
                "sampled": len(compare_indices),
                "sample_fraction": len(compare_indices) / len(compare_reader),
                "unique_sample_ids": len(compare_ids),
                "unique_token_sequences": len(compare_hashes),
            },
            "sample_id_overlap": len(sample_id_overlap),
            "sample_id_overlap_rate": (
                len(sample_id_overlap) / min(len(source_ids), len(compare_ids))
                if source_ids and compare_ids
                else 0.0
            ),
            "token_sequence_overlap": len(token_overlap),
            "token_sequence_overlap_rate": (
                len(token_overlap) / min(len(source_hashes), len(compare_hashes))
                if source_hashes and compare_hashes
                else 0.0
            ),
        }


def _print_progress(done: int, total: int) -> None:
    print(f"diagnosed {done}/{total} samples", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose a binary pretraining dataset without training"
    )
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--micro-batch-sizes", default="2,4,8,16")
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--compare-path", type=Path)
    parser.add_argument("--compare-sample-size", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = diagnose_pretrain_dataset(
        args.path,
        sample_size=args.sample_size,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        micro_batch_sizes=_parse_micro_batch_sizes(args.micro_batch_sizes),
        configured_micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        world_size=args.world_size,
        progress=_print_progress,
    )
    if args.compare_path:
        report["comparison"] = compare_dataset_overlap(
            args.path,
            args.compare_path,
            source_sample_size=args.sample_size,
            compare_sample_size=args.compare_sample_size,
            seed=args.seed,
        )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["compare_dataset_overlap", "diagnose_pretrain_dataset"]
