"""Batch builder protocol."""
from __future__ import annotations

from typing import Any, Protocol

import torch


class BatchBuilder(Protocol):
    def __call__(self, samples: list[Any]) -> dict[str, torch.Tensor | list[Any]]: ...


__all__ = ["BatchBuilder"]
