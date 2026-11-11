"""Deterministic, testable next-token sampling."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from chat.config import SamplingConfig


def sample_next_token(
    logits: torch.Tensor,
    *,
    generated_token_ids: list[int],
    config: SamplingConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample one token from a one-dimensional logits vector."""

    if logits.dim() != 1:
        raise ValueError("logits must be one-dimensional")
    scores = logits.clone()

    if config.repetition_penalty > 1.0 and generated_token_ids:
        token_ids = torch.tensor(
            sorted(set(generated_token_ids)),
            dtype=torch.long,
            device=logits.device,
        )
        selected = scores.index_select(0, token_ids)
        selected = torch.where(
            selected < 0,
            selected * config.repetition_penalty,
            selected / config.repetition_penalty,
        )
        scores.index_copy_(0, token_ids, selected)

    if config.temperature == 0:
        return scores.argmax(dim=-1)

    scores = scores / config.temperature
    if config.top_k is not None:
        top_k = min(config.top_k, scores.numel())
        threshold = torch.topk(scores, top_k).values[-1]
        scores = scores.masked_fill(scores < threshold, float("-inf"))

    if config.top_p is not None and config.top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_scores, dim=-1), dim=-1)
        remove = cumulative > config.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        remove = torch.zeros_like(remove).scatter(0, sorted_indices, remove)
        scores = scores.masked_fill(remove, float("-inf"))

    probabilities = F.softmax(scores, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)[0]


__all__ = ["sample_next_token"]
