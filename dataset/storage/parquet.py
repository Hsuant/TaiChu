"""Parquet-backed sample storage for offline and runtime data paths."""
from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator, Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from ..exceptions import TaiChuDataFormatError
from ..schema import (
    CommonSample,
    get_sample_class,
    sample_from_storage,
    sample_to_storage,
)
from .base import IndexedSampleReader, PathLike
from .fs import atomic_replace_directory
from .manifest import (
    DatasetManifest,
    MANIFEST_NAME,
    STORAGE_FORMAT,
    build_manifest,
)


FLOAT_ARRAY_FIELDS = frozenset(
    {
        "image_pixel_values",
        "rewards",
        "advantages",
        "old_log_probs",
        "reference_log_probs",
        "values",
    }
)


def _pyarrow_modules():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for Parquet dataset storage"
        ) from exc
    return pa, pq


def _field_array_type(
    sequences: list[list[Any]] | None,
    *,
    field_name: str,
):
    pa, _ = _pyarrow_modules()
    if field_name in FLOAT_ARRAY_FIELDS:
        return pa.list_(pa.float32())
    if not sequences:
        return pa.list_(pa.int32())
    first = next(
        (
            values
            for values in sequences
            if values is not None and len(values) > 0
        ),
        None,
    )
    if first is None:
        return pa.list_(pa.int32())
    if all(isinstance(value, Integral) and not isinstance(value, bool) for value in first):
        minimum = min(int(value) for value in first)
        maximum = max(int(value) for value in first)
        if -(2**31) <= minimum and maximum < 2**31:
            return pa.list_(pa.int32())
        return pa.list_(pa.int64())
    if all(isinstance(value, Real) and not isinstance(value, bool) for value in first):
        return pa.list_(pa.float32())
    raise TaiChuDataFormatError(
        "Parquet array fields currently support finite integer or float sequences"
    )


def _schema_for_rows(
    array_fields: tuple[str, ...],
    rows: list[dict[str, Any]],
):
    pa, _ = _pyarrow_modules()
    fields = [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("dataset_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("sample_type", pa.string(), nullable=False),
        pa.field("metadata_json", pa.string(), nullable=False),
        pa.field("input_token_count", pa.int64(), nullable=False),
        pa.field("loss_token_count", pa.int64(), nullable=False),
    ]
    for name in array_fields:
        sequences = [
            row[name]
            for row in rows
            if row.get(name) is not None
        ]
        fields.append(
            pa.field(
                name,
                _field_array_type(sequences, field_name=name),
            )
        )
    return pa.schema(fields)


def _non_padding_input_tokens(
    input_ids: list[int],
    attention_mask: list[int] | None,
) -> int:
    if attention_mask is None:
        return len(input_ids)
    if len(attention_mask) != len(input_ids):
        raise TaiChuDataFormatError(
            "attention_mask must align with input_ids for Parquet storage"
        )
    return sum(int(value) != 0 for value in attention_mask)


class ParquetSampleWriter:
    """Write typed samples into checksummed Parquet shards atomically."""

    def __init__(
        self,
        output_dir: PathLike,
        *,
        dataset_id: str,
        sample_type: str,
        version: str = "1",
        parent_version: str | None = None,
        shard_size: int = 50_000,
        metadata: Mapping[str, Any] | None = None,
        sync_on_close: bool = True,
        checksums_on_commit: bool = True,
        row_group_size: int = 1_000,
        compression: str = "zstd",
    ) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if row_group_size <= 0:
            raise ValueError("row_group_size must be positive")
        if not dataset_id or not sample_type or not version:
            raise ValueError("dataset_id, sample_type and version must be non-empty")
        self.output_dir = Path(output_dir)
        self.staging_dir = self.output_dir.with_name(
            f".{self.output_dir.name}.staging"
        )
        if self.output_dir.exists():
            raise FileExistsError(f"output dataset already exists: {self.output_dir}")
        if self.staging_dir.exists():
            raise FileExistsError(
                f"staging directory already exists: {self.staging_dir}"
            )
        self.dataset_id = dataset_id
        self.sample_type = sample_type
        self.version = version
        self.parent_version = parent_version
        self.shard_size = int(shard_size)
        self.metadata = dict(metadata or {})
        self.sync_on_close = bool(sync_on_close)
        self.checksums_on_commit = bool(checksums_on_commit)
        self.row_group_size = int(row_group_size)
        self.compression = compression
        self._schema = None
        self._array_fields: tuple[str, ...] = ()
        self._rows: list[dict[str, Any]] = []
        self._shard_files: list[tuple[Path, int]] = []
        self._writer = None
        self._writer_path: Path | None = None
        self._current_samples = 0
        self._closed = False
        self.staging_dir.mkdir(parents=True)

    def _initialize_schema(self, sample: CommonSample) -> None:
        self._array_fields = tuple(sample.array_fields)

    def _row_for_sample(self, sample: CommonSample) -> dict[str, Any]:
        metadata, arrays = sample_to_storage(sample)
        input_ids = list(arrays.get("input_ids", []))
        attention_mask = arrays.get("attention_mask")
        labels = arrays.get("labels")
        input_token_count = _non_padding_input_tokens(
            input_ids,
            list(attention_mask) if attention_mask is not None else None,
        )
        loss_token_count = (
            sum(int(value) != -100 for value in labels)
            if labels is not None
            else len(input_ids)
        )
        stored_metadata = metadata.get("metadata")
        if isinstance(stored_metadata, dict):
            stored_metadata = dict(stored_metadata)
            stored_metadata.setdefault("input_token_count", input_token_count)
            stored_metadata.setdefault("loss_token_count", loss_token_count)
            metadata["metadata"] = stored_metadata
        row: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "dataset_id": sample.dataset_id,
            "source_id": sample.source_id,
            "sample_type": sample.sample_type,
            "metadata_json": json.dumps(
                metadata,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "input_token_count": input_token_count,
            "loss_token_count": loss_token_count,
        }
        for name in self._array_fields:
            value = arrays.get(name)
            row[name] = list(value) if value is not None else None
        return row

    def write(self, sample: CommonSample, *, trusted_arrays: bool = False) -> None:
        del trusted_arrays
        if self._closed:
            raise RuntimeError("writer is closed")
        if sample.sample_type != self.sample_type:
            raise TaiChuDataFormatError(
                f"writer expects {self.sample_type!r}, got {sample.sample_type!r}"
            )
        if not self._array_fields:
            self._initialize_schema(sample)
        self._rows.append(self._row_for_sample(sample))
        self._current_samples += 1
        if len(self._rows) >= self.row_group_size:
            self._flush_rows()
        if self._current_samples >= self.shard_size:
            self._close_shard()

    def _flush_rows(self) -> None:
        if not self._rows:
            return
        pa, pq = _pyarrow_modules()
        if self._schema is None:
            self._schema = _schema_for_rows(
                self._array_fields,
                self._rows,
            )
        if self._writer is None:
            number = len(self._shard_files)
            self._writer_path = self.staging_dir / f"data-{number:05d}.parquet"
            self._writer = pq.ParquetWriter(
                self._writer_path,
                self._schema,
                compression=self.compression,
            )
        table = pa.Table.from_pylist(
            self._rows,
            schema=self._schema,
        )
        self._writer.write_table(table, row_group_size=len(self._rows))
        self._rows = []

    def _close_shard(self) -> None:
        if self._current_samples == 0:
            return
        self._flush_rows()
        assert self._writer is not None
        assert self._writer_path is not None
        self._writer.close()
        if self.sync_on_close:
            with self._writer_path.open("rb+") as stream:
                os.fsync(stream.fileno())
        self._shard_files.append((self._writer_path, self._current_samples))
        self._writer = None
        self._writer_path = None
        self._current_samples = 0
        self._schema = None
        self._rows = []

    def commit(self) -> DatasetManifest:
        if self._closed:
            raise RuntimeError("writer is closed")
        self._close_shard()
        if not self._shard_files:
            self.abort()
            raise ValueError("cannot commit a dataset with no samples")
        manifest = build_manifest(
            dataset_id=self.dataset_id,
            sample_type=self.sample_type,
            version=self.version,
            shard_files=self._shard_files,
            root=self.staging_dir,
            metadata=self.metadata,
            parent_version=self.parent_version,
            checksums=self.checksums_on_commit,
        )
        manifest.write(self.staging_dir)
        atomic_replace_directory(self.staging_dir, self.output_dir)
        self._closed = True
        return manifest

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.close()
        self._writer = None
        self._writer_path = None
        self._current_samples = 0
        self._rows = []
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
        self._closed = True

    def __enter__(self) -> "ParquetSampleWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._closed:
            if exc_type is None:
                self.commit()
            else:
                self.abort()


class ParquetShardReader:
    """Random-access reader over one Parquet shard."""

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        _, pq = _pyarrow_modules()
        if not self.path.is_file():
            raise FileNotFoundError(f"Parquet shard does not exist: {self.path}")
        self._file = pq.ParquetFile(self.path)
        self._count = int(self._file.metadata.num_rows)
        self._row_group_starts: list[int] = []
        cursor = 0
        for group_index in range(self._file.num_row_groups):
            self._row_group_starts.append(cursor)
            cursor += int(
                self._file.metadata.row_group(group_index).num_rows
            )
        if cursor != self._count:
            raise TaiChuDataFormatError(
                f"Parquet row groups do not match row count for {self.path}"
            )
        self._cached_group = -1
        self._cached_table = None
        self._cached_metadata_group = -1
        self._cached_metadata_table = None

    def __len__(self) -> int:
        return self._count

    def _group_for(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        low, high = 0, len(self._row_group_starts) - 1
        while low < high:
            middle = (low + high + 1) // 2
            if self._row_group_starts[middle] <= index:
                low = middle
            else:
                high = middle - 1
        return low, index - self._row_group_starts[low]

    def _table_for_group(self, group_index: int):
        if self._cached_group != group_index:
            self._cached_table = self._file.read_row_group(group_index)
            self._cached_group = group_index
        return self._cached_table

    def _metadata_table_for_group(self, group_index: int):
        if self._cached_metadata_group != group_index:
            self._cached_metadata_table = self._file.read_row_group(
                group_index,
                columns=[
                    "metadata_json",
                    "input_token_count",
                    "loss_token_count",
                ],
            )
            self._cached_metadata_group = group_index
        return self._cached_metadata_table

    def _metadata_at(self, group_index: int, row_index: int) -> dict[str, Any]:
        table = self._metadata_table_for_group(group_index)
        value = table.column("metadata_json")[row_index].as_py()
        try:
            metadata = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise TaiChuDataFormatError(
                "invalid Parquet sample metadata"
            ) from exc
        if not isinstance(metadata, dict):
            raise TaiChuDataFormatError(
                "Parquet sample metadata must be an object"
            )
        metadata["_storage_input_tokens"] = int(
            table.column("input_token_count")[row_index].as_py()
        )
        metadata["_storage_loss_tokens"] = int(
            table.column("loss_token_count")[row_index].as_py()
        )
        return metadata

    def read_metadata(self, index: int) -> dict[str, Any]:
        group_index, row_index = self._group_for(index)
        return self._metadata_at(group_index, row_index)

    def read_sample(self, index: int) -> CommonSample:
        group_index, row_index = self._group_for(index)
        metadata = self._metadata_at(group_index, row_index)
        sample_type = metadata.get("_sample_type")
        cls = get_sample_class(str(sample_type))
        table = self._table_for_group(group_index)
        arrays: dict[str, list[Any]] = {}
        for name in cls.array_fields:
            if name not in table.column_names:
                raise TaiChuDataFormatError(
                    f"Parquet shard is missing array column {name!r}"
                )
            value = table.column(name)[row_index].as_py()
            if value is not None:
                arrays[name] = list(value)
        return sample_from_storage(metadata, arrays)

    def iter_metadata(self) -> Iterator[dict[str, Any]]:
        for group_index in range(self._file.num_row_groups):
            table = self._file.read_row_group(
                group_index,
                columns=[
                    "metadata_json",
                    "input_token_count",
                    "loss_token_count",
                ],
            )
            metadata_values = table.column("metadata_json").to_pylist()
            input_counts = table.column("input_token_count").to_pylist()
            loss_counts = table.column("loss_token_count").to_pylist()
            for value, input_count, loss_count in zip(
                metadata_values,
                input_counts,
                loss_counts,
                strict=True,
            ):
                try:
                    metadata = json.loads(value)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise TaiChuDataFormatError(
                        "invalid Parquet sample metadata"
                    ) from exc
                if not isinstance(metadata, dict):
                    raise TaiChuDataFormatError(
                        "Parquet sample metadata must be an object"
                    )
                metadata["_storage_input_tokens"] = int(input_count)
                metadata["_storage_loss_tokens"] = int(loss_count)
                yield metadata

    def close(self) -> None:
        self._cached_table = None
        self._cached_metadata_table = None
        self._file.close()

    def __enter__(self) -> "ParquetShardReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class ParquetDatasetReader(IndexedSampleReader):
    """Manifest-backed random-access reader over Parquet shards."""

    def __init__(
        self,
        path: PathLike,
        *,
        checksums: bool = False,
    ) -> None:
        self.path = Path(path)
        self.checksums = bool(checksums)
        self.root = self.path if self.path.is_dir() else self.path.parent
        manifest_path = (
            self.path if self.path.is_file() else self.root / MANIFEST_NAME
        )
        self.manifest = DatasetManifest.load(manifest_path)
        if self.manifest.format != STORAGE_FORMAT:
            raise TaiChuDataFormatError(
                f"expected Parquet manifest, got {self.manifest.format!r}"
            )
        if self.checksums:
            self.manifest.verify(self.root, checksums=True)
        self.dataset_id = self.manifest.dataset_id
        self.sample_type = self.manifest.sample_type
        self.version = self.manifest.version
        self._shards = [
            ParquetShardReader(self.root / shard.path)
            for shard in self.manifest.shards
        ]
        self._starts: list[int] = []
        cursor = 0
        for shard in self._shards:
            self._starts.append(cursor)
            cursor += len(shard)
        if cursor != self.manifest.sample_count:
            self.close()
            raise TaiChuDataFormatError(
                "manifest sample_count does not match Parquet shards"
            )

    def __len__(self) -> int:
        return self.manifest.sample_count

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        low, high = 0, len(self._starts) - 1
        while low < high:
            middle = (low + high + 1) // 2
            if self._starts[middle] <= index:
                low = middle
            else:
                high = middle - 1
        return low, index - self._starts[low]

    def __getitem__(self, index: int) -> CommonSample:
        shard_index, local_index = self._locate(index)
        return self._shards[shard_index].read_sample(local_index)

    def metadata_at(self, index: int) -> dict[str, Any]:
        shard_index, local_index = self._locate(index)
        return self._shards[shard_index].read_metadata(local_index)

    def iter_metadata(self) -> Iterator[dict[str, Any]]:
        for shard in self._shards:
            yield from shard.iter_metadata()

    def __iter__(self) -> Iterator[CommonSample]:
        for shard in self._shards:
            for index in range(len(shard)):
                yield shard.read_sample(index)

    def verify(self, *, checksums: bool = True) -> None:
        self.manifest.verify(self.root, checksums=checksums)

    def close(self) -> None:
        for shard in getattr(self, "_shards", ()):
            shard.close()

    def __getstate__(self) -> dict[str, Any]:
        return {"path": str(self.path), "checksums": self.checksums}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(
            state["path"],
            checksums=bool(state.get("checksums", False)),
        )

    def __enter__(self) -> "ParquetDatasetReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = [
    "ParquetDatasetReader",
    "ParquetSampleWriter",
    "ParquetShardReader",
]
