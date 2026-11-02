"""Model input adapter protocol."""
from __future__ import annotations

from typing import Any, Protocol


class ModelInputAdapter(Protocol):
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]: ...


def select_keys(batch: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {name: batch[name] for name in names if name in batch}


__all__ = ["ModelInputAdapter", "select_keys"]
