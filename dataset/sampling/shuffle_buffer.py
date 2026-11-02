"""Bounded deterministic shuffle buffer for streaming samples."""
from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


def shuffle_buffer(
    items: Iterable[T],
    *,
    size: int,
    seed: int,
) -> Iterator[T]:
    if size < 0:
        raise ValueError("shuffle buffer size cannot be negative")
    if size <= 1:
        yield from items
        return
    rng = random.Random(seed)
    buffer: list[T] = []
    for item in items:
        if len(buffer) < size:
            buffer.append(item)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = item
    rng.shuffle(buffer)
    yield from buffer


__all__ = ["shuffle_buffer"]
