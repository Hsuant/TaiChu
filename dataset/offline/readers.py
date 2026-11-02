"""Streaming raw-record readers for offline preparation."""
from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ..exceptions import TaiChuDataFormatError

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

    def read(self, path: str | Path) -> Iterator[dict[str, Any]]:
        source = Path(path)
        self._check(source)
        if source.suffix.lower() == ".parquet":
            yield from self._read_parquet(source)
        else:
            yield from self._read_jsonl(source)

    @staticmethod
    def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
        opener = gzip.open if path.name.lower().endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TaiChuDataFormatError(
                        f"invalid JSONL in {path} at line {line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    raise TaiChuDataFormatError(
                        f"JSONL record in {path} at line {line_number} must be an object"
                    )
                yield record

    @staticmethod
    def _read_parquet(path: Path) -> Iterator[dict[str, Any]]:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise ImportError("pyarrow is required to read Parquet sources") from exc
        with parquet.ParquetFile(path) as handle:
            for batch in handle.iter_batches(use_threads=False):
                yield from batch.to_pylist()


__all__ = ["RecordSourceReader"]
