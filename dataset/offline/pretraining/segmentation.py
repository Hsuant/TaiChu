"""Whole-document sliding windows with exact causal-LM supervision."""
from __future__ import annotations

import hashlib

from ...schema import PretrainingSample
from .config import SegmentationConfig
from .hashing import cluster_id


def sliding_window_bounds(
    num_tokens: int,
    *,
    max_content_tokens: int = 2046,
    overlap_tokens: int = 128,
) -> list[tuple[int, int]]:
    if num_tokens < 1:
        raise ValueError("num_tokens must be positive")
    if max_content_tokens < 1:
        raise ValueError("max_content_tokens must be positive")
    if not 0 <= overlap_tokens < max_content_tokens:
        raise ValueError("overlap_tokens must be in [0, max_content_tokens)")
    stride = max_content_tokens - overlap_tokens
    windows: list[tuple[int, int]] = []
    start = 0
    while start < num_tokens:
        end = min(start + max_content_tokens, num_tokens)
        windows.append((start, end))
        if end == num_tokens:
            break
        start += stride
    return windows


def build_segment_samples(
    token_ids: list[int],
    *,
    dataset_id: str,
    parent_document_id: str,
    source_id: str,
    language: str,
    split: str,
    tokenizer_version: str,
    bos_token_id: int,
    eos_token_id: int,
    segmentation: SegmentationConfig,
    processing_version: str,
) -> list[PretrainingSample]:
    if not token_ids:
        raise ValueError("token_ids cannot be empty")
    bounds = sliding_window_bounds(
        len(token_ids),
        max_content_tokens=segmentation.max_segment_content_tokens,
        overlap_tokens=segmentation.overlap_tokens,
    )
    group_id = cluster_id(parent_document_id)
    samples: list[PretrainingSample] = []
    for segment_index, (start, end) in enumerate(bounds):
        content = token_ids[start:end]
        is_first = segment_index == 0
        is_last = segment_index == len(bounds) - 1
        overlap = 0 if is_first else min(
            segmentation.overlap_tokens,
            len(content),
        )
        input_ids = (
            ([bos_token_id] if is_first else [])
            + content
            + ([eos_token_id] if is_last else [])
        )
        labels = (
            ([-100] if is_first else [])
            + [-100] * overlap
            + list(content[overlap:])
            + ([eos_token_id] if is_last else [])
        )
        if len(input_ids) != len(labels):
            raise RuntimeError("segment labels are not input-aligned")
        if len(input_ids) > segmentation.max_sequence_length:
            raise RuntimeError("segment exceeds configured sequence length")
        new_tokens = len(content) - overlap
        flags = []
        if len(token_ids) < segmentation.min_fragment_tokens:
            flags.append("short_document")
        if len(token_ids) > segmentation.max_segment_content_tokens:
            flags.append("long_document")
        if is_last and not is_first and new_tokens < segmentation.min_fragment_tokens:
            flags.append("legal_tail_fragment")
        samples.append(
            PretrainingSample._from_trusted(
                sample_id=f"{parent_document_id}:segment-{segment_index:05d}",
                dataset_id=dataset_id,
                source_id=source_id,
                input_ids=input_ids,
                labels=labels,
                attention_mask=None,
                document_ids=[],
                metadata={
                    "parent_document_id": parent_document_id,
                    "source_id": source_id,
                    "language": language,
                    "split": split,
                    "cluster_id": group_id,
                    "segment_index": segment_index,
                    "start_token": start,
                    "end_token": end,
                    "document_token_count": len(token_ids),
                    "segment_token_count": len(content),
                    "overlap_token_count": overlap,
                    "new_token_count": new_tokens,
                    "attention_group_id": hashlib.sha256(
                        f"{parent_document_id}:{segment_index}".encode("utf-8")
                    ).hexdigest()[:24],
                    "has_bos": is_first,
                    "has_eos": is_last,
                    "input_token_count": len(input_ids),
                    "loss_token_count": sum(label != -100 for label in labels),
                    "tokenizer_version": tokenizer_version,
                    "processing_version": processing_version,
                    "quality_flags": flags,
                    "fragment_suspect": (
                        len(token_ids) < segmentation.min_fragment_tokens
                    ),
                },
            )
        )
    return samples


__all__ = ["build_segment_samples", "sliding_window_bounds"]
