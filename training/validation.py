"""Exact next-token validation metrics shared by training and evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class TokenStatistics:
    nll_nats: float
    token_count: int
    correct_count: int
    token_losses: torch.Tensor
    bpb_nll_nats: float
    bpb_bytes: int
    max_token_loss: float = 0.0


def compute_perplexity(loss: float | None) -> float | None:
    if loss is None:
        return None
    value = float(loss)
    if not math.isfinite(value) or value < 0.0:
        return None
    try:
        result = math.exp(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def compute_bits_per_byte(
    total_nll_nats: float,
    total_utf8_bytes: int,
) -> float | None:
    if total_utf8_bytes <= 0 or not math.isfinite(total_nll_nats):
        return None
    if total_nll_nats < 0.0:
        raise ValueError("total_nll_nats cannot be negative")
    result = total_nll_nats / (math.log(2.0) * total_utf8_bytes)
    return result if math.isfinite(result) else None


def compute_percentile_ce(
    losses_per_token: torch.Tensor,
    percentiles: Iterable[float] = (0.5, 0.9),
) -> dict[str, float | None]:
    requested = list(percentiles)
    if any(not 0.0 <= value <= 1.0 for value in requested):
        raise ValueError("percentiles must be in [0, 1]")
    keys = {value: f"ce_p{int(value * 100)}" for value in requested}
    if losses_per_token.numel() == 0:
        return {key: None for key in keys.values()}
    losses = losses_per_token.detach().reshape(-1)
    if not torch.isfinite(losses).all():
        raise ValueError("per-token losses must be finite")
    values = losses.to(device="cpu", dtype=torch.float64).numpy()
    last_index = values.size - 1
    needed = sorted(
        {
            index
            for percentile in requested
            for index in (
                math.floor(percentile * last_index),
                math.ceil(percentile * last_index),
            )
        }
    )
    partitioned = np.partition(values, needed)
    result: dict[str, float | None] = {}
    for percentile in requested:
        position = percentile * last_index
        lower_index = math.floor(position)
        upper_index = math.ceil(position)
        lower = float(partitioned[lower_index])
        if lower_index == upper_index:
            value = lower
        else:
            upper = float(partitioned[upper_index])
            value = lower + (upper - lower) * (position - lower_index)
        result[keys[percentile]] = value
    return result


def _special_token_ids(tokenizer) -> set[int]:
    if tokenizer is None:
        return set()
    if hasattr(tokenizer, "all_special_ids"):
        return {int(token_id) for token_id in tokenizer.all_special_ids}
    if hasattr(tokenizer, "get_added_tokens_decoder"):
        return {
            int(token_id)
            for token_id in tokenizer.get_added_tokens_decoder()
        }
    if hasattr(tokenizer, "get_added_vocab"):
        return {
            int(token_id)
            for token_id in tokenizer.get_added_vocab().values()
        }
    if hasattr(tokenizer, "get_vocab"):
        return {
            int(token_id)
            for token, token_id in tokenizer.get_vocab().items()
            if (
                (token.startswith("<|") and token.endswith("|>"))
                or token in {"<s>", "</s>", "<unk>", "<pad>"}
            )
        }
    return set()


def _bpb_for_targets(
    targets: torch.Tensor,
    losses: torch.Tensor,
    tokenizer,
) -> tuple[float, int]:
    if tokenizer is None or targets.numel() == 0:
        return 0.0, 0
    special_ids = _special_token_ids(tokenizer)
    if special_ids:
        special = torch.tensor(
            sorted(special_ids),
            dtype=targets.dtype,
            device=targets.device,
        )
        content = ~torch.isin(targets, special)
        targets = targets[content]
        losses = losses[content]
    if targets.numel() == 0:
        return 0.0, 0
    decoded = tokenizer.decode(
        targets.tolist(),
        skip_special_tokens=True,
    )
    return float(losses.double().sum().item()), len(decoded.encode("utf-8"))


def compute_token_statistics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer=None,
    *,
    loss_chunk_size: int = 256,
) -> TokenStatistics:
    """Compute CE, accuracy, exact losses, and BPB from full logits."""
    if logits.dim() != 3 or labels.dim() != 2:
        raise ValueError("logits must be 3D and labels must be 2D")
    if tuple(labels.shape) != tuple(logits.shape[:2]):
        raise ValueError("labels shape must match logits batch and sequence axes")
    if logits.size(1) < 2:
        return TokenStatistics(
            0.0,
            0,
            0,
            torch.empty(0, dtype=torch.float64),
            0.0,
            0,
        )
    if loss_chunk_size <= 0:
        raise ValueError("loss_chunk_size must be positive")

    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    valid = shift_labels != -100
    token_count = int(valid.sum().item())
    if token_count == 0:
        return TokenStatistics(
            0.0,
            0,
            0,
            torch.empty(0, dtype=torch.float64),
            0.0,
            0,
        )

    nll = 0.0
    correct = 0
    bpb_nll = 0.0
    bpb_bytes = 0
    max_loss = 0.0
    collected: list[torch.Tensor] = []
    row_losses: list[list[torch.Tensor]] = [
        [] for _ in range(shift_labels.size(0))
    ]
    for start in range(0, shift_labels.size(1), loss_chunk_size):
        end = min(start + loss_chunk_size, shift_labels.size(1))
        chunk_logits = shift_logits[:, start:end, :]
        chunk_labels = shift_labels[:, start:end]
        chunk_valid = valid[:, start:end]
        losses = F.cross_entropy(
            chunk_logits.reshape(-1, chunk_logits.size(-1)).float(),
            chunk_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(chunk_labels.shape)
        selected = losses[chunk_valid]
        if selected.numel() == 0:
            continue
        if not torch.isfinite(selected).all():
            raise ValueError("token losses must be finite")
        nll += float(selected.sum().item())
        correct += int(
            (
                (chunk_logits.argmax(dim=-1) == chunk_labels)
                & chunk_valid
            ).sum().item()
        )
        max_loss = max(max_loss, float(selected.max().item()))
        collected.append(selected.detach().to(device="cpu", dtype=torch.float64))
        for row in range(chunk_labels.size(0)):
            row_valid = chunk_valid[row]
            if row_valid.any():
                row_losses[row].append(
                    losses[row][row_valid].detach().to(
                        device="cpu",
                        dtype=torch.float64,
                    )
                )
    for row in range(shift_labels.size(0)):
        row_valid = valid[row]
        if row_valid.any():
            losses_for_row = torch.cat(row_losses[row])
            row_nll, row_bytes = _bpb_for_targets(
                shift_labels[row][row_valid],
                losses_for_row,
                tokenizer,
            )
            bpb_nll += row_nll
            bpb_bytes += row_bytes

    return TokenStatistics(
        nll_nats=nll,
        token_count=token_count,
        correct_count=correct,
        token_losses=(
            torch.cat(collected)
            if collected
            else torch.empty(0, dtype=torch.float64)
        ),
        bpb_nll_nats=bpb_nll,
        bpb_bytes=bpb_bytes,
        max_token_loss=max_loss,
    )


def compute_token_statistics_from_losses(
    token_losses: torch.Tensor,
    labels: torch.Tensor,
    *,
    correct_count: int,
    tokenizer=None,
) -> TokenStatistics:
    """Compute statistics from model-provided per-token CE values."""
    if labels.dim() != 2 or labels.size(1) < 2:
        raise ValueError("labels must be a rank-2 sequence with length >= 2")
    shift_labels = labels[..., 1:]
    valid = shift_labels != -100
    selected = token_losses.detach().reshape(-1)
    token_count = int(valid.sum().item())
    if selected.numel() != token_count:
        raise ValueError(
            "token_losses length must equal the number of valid targets"
        )
    if token_count == 0:
        return TokenStatistics(
            0.0,
            0,
            0,
            torch.empty(0, dtype=torch.float64),
            0.0,
            0,
        )
    if not torch.isfinite(selected).all():
        raise ValueError("token losses must be finite")

    bpb_nll = 0.0
    bpb_bytes = 0
    offset = 0
    for row in range(shift_labels.size(0)):
        row_mask = valid[row]
        count = int(row_mask.sum().item())
        if count:
            row_losses = selected[offset : offset + count]
            offset += count
            row_nll, row_bytes = _bpb_for_targets(
                shift_labels[row][row_mask],
                row_losses,
                tokenizer,
            )
            bpb_nll += row_nll
            bpb_bytes += row_bytes

    return TokenStatistics(
        nll_nats=float(selected.sum().item()),
        token_count=token_count,
        correct_count=int(correct_count),
        token_losses=selected.to(device="cpu", dtype=torch.float32),
        bpb_nll_nats=bpb_nll,
        bpb_bytes=bpb_bytes,
        max_token_loss=float(selected.max().item()),
    )


def compute_empty_prefix_eos_metrics(
    model: nn.Module,
    *,
    bos_token_id: int | None,
    eos_token_id: int | None,
    device: torch.device,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "empty_prefix_eos_ce": None,
        "empty_prefix_eos_ppl": None,
    }
    if (
        bos_token_id is None
        or eos_token_id is None
        or bos_token_id == eos_token_id
        or bos_token_id < 0
        or eos_token_id < 0
    ):
        return result

    was_training = model.training
    input_ids = torch.tensor(
        [[bos_token_id, eos_token_id]],
        dtype=torch.long,
        device=device,
    )
    model.eval()
    try:
        with torch.no_grad():
            loss = model(input_ids, labels=input_ids).loss
    except (RuntimeError, ValueError):
        return result
    finally:
        model.train(was_training)
    if loss is None:
        return result
    value = float(loss.item())
    if not math.isfinite(value) or value < 0.0:
        return result
    result["empty_prefix_eos_ce"] = value
    result["empty_prefix_eos_ppl"] = compute_perplexity(value)
    return result


__all__ = [
    "TokenStatistics",
    "compute_bits_per_byte",
    "compute_empty_prefix_eos_metrics",
    "compute_percentile_ce",
    "compute_perplexity",
    "compute_token_statistics",
    "compute_token_statistics_from_losses",
]
