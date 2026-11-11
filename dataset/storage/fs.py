"""Small filesystem primitives for atomic dataset publication."""
from __future__ import annotations

import time
from pathlib import Path


def atomic_replace_directory(
    source: str | Path,
    target: str | Path,
    *,
    attempts: int = 6,
) -> None:
    """Publish a completed directory without exposing partial output."""
    source_path = Path(source)
    target_path = Path(target)
    if not source_path.is_dir():
        raise FileNotFoundError(f"staging directory does not exist: {source_path}")
    if target_path.exists():
        raise FileExistsError(f"target already exists: {target_path}")
    delay = 0.05
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            source_path.replace(target_path)
            return
        except OSError as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(delay)
                delay = min(delay * 2, 1.0)
    assert last_error is not None
    raise last_error


__all__ = ["atomic_replace_directory"]
