"""Storage-facing protocols shared by offline and online data sources."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..schema import CommonSample


@runtime_checkable
class SampleWriter(Protocol):
    def write(self, sample: CommonSample) -> None: ...

    def commit(self): ...

    def abort(self) -> None: ...


@runtime_checkable
class IndexedSampleReader(Protocol):
    dataset_id: str
    sample_type: str
    version: str

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> CommonSample: ...

    def __iter__(self) -> Iterator[CommonSample]: ...


PathLike = str | Path


__all__ = [
    "IndexedSampleReader",
    "PathLike",
    "SampleWriter",
]
