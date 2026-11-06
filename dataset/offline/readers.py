"""Streaming raw-record readers for offline preparation."""
from __future__ import annotations

import gzip
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import orjson

from ..exceptions import TaiChuDataFormatError


ProgressCallback = Callable[[int], None]


class RecordSourceReader:
    @staticmethod
    def resolve(paths: Iterable[str | Path]) -> tuple[Path, ...]:
        files: set[Path] = set()
        for raw in paths:
            path = Path(raw).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"preparation source does not exist: {path}")
            if path.is_file():
                RecordSourceReader._check(path)
                files.add(path.resolve())
                continue
            files.update(
                item.resolve()
                for item in path.rglob("*")
                if item.is_file() and RecordSourceReader._supported(item)
            )
        if not files:
            raise FileNotFoundError("no supported preparation sources found")
        return tuple(sorted(files, key=lambda item: str(item).casefold()))

    @staticmethod
    def _supported(path: Path) -> bool:
        return path.suffix.lower() == ".parquet" or path.name.lower().endswith(
            (".jsonl", ".jsonl.gz")
        )

    @staticmethod
    def _check(path: Path) -> None:
        if not RecordSourceReader._supported(path):
            raise TaiChuDataFormatError(
                f"unsupported preparation format {path.suffix!r}; "
                "use JSONL, JSONL.GZ or Parquet"
            )

    def read(
        self,
        path: str | Path,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> Iterator[dict[str, Any]]:
        source = Path(path)
        self._check(source)
        if source.suffix.lower() == ".parquet":
            yield from self._read_parquet(source, progress_callback)
        else:
            yield from self._read_jsonl(source, progress_callback)

    @staticmethod
    def _read_jsonl(
        path: Path,
        progress_callback: ProgressCallback | None,
    ) -> Iterator[dict[str, Any]]:
        opener = gzip.open if path.name.lower().endswith(".gz") else open
        with opener(path, "rb") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    if progress_callback is not None:
                        progress_callback(len(line))
                    continue
                try:
                    record = orjson.loads(line)
                except orjson.JSONDecodeError as exc:
                    raise TaiChuDataFormatError(
                        f"invalid JSONL in {path} at line {line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    raise TaiChuDataFormatError(
                        f"JSONL record in {path} at line {line_number} must be an object"
                    )
                yield record
                if progress_callback is not None:
                    progress_callback(len(line))

    @staticmethod
    def _read_parquet(
        path: Path,
        progress_callback: ProgressCallback | None,
    ) -> Iterator[dict[str, Any]]:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise ImportError("pyarrow is required to read Parquet sources") from exc
        with parquet.ParquetFile(path) as handle:
            for batch in handle.iter_batches(use_threads=False):
                records = batch.to_pylist()
                if not records:
                    continue
                step = max(1, batch.nbytes // len(records))
                for record in records:
                    yield record
                    if progress_callback is not None:
                        progress_callback(step)


__all__ = ["RecordSourceReader"]
