"""Versioned binary shards with mmap-backed random access."""
from __future__ import annotations

import json
import mmap
import os
import shutil
import struct
import sys
from array import array
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..exceptions import TaiChuDataFormatError
from ..schema import CommonSample, sample_from_storage, sample_to_storage
from .base import IndexedSampleReader, PathLike
from .fs import atomic_replace_directory
from .manifest import DatasetManifest, MANIFEST_NAME, build_manifest


DATA_MAGIC = b"TCDB"
INDEX_MAGIC = b"TCDI"
STORAGE_VERSION = 1
HEADER = struct.Struct("<4sH")
RECORD_HEADER = struct.Struct("<IH")
ARRAY_HEADER = struct.Struct("<HBI")
INDEX_ENTRY = struct.Struct("<Q")

DTYPE_CODES = {
    1: "int32",
    2: "int64",
    3: "float32",
    4: "uint16",
    5: "int16",
}
ARRAY_TYPECODES = {1: "i", 2: "q", 3: "f", 4: "H", 5: "h"}
ARRAY_WIDTHS = {1: 4, 2: 8, 3: 4, 4: 2, 5: 2}


def _encode_array(
    values: list[Any],
    name: str,
    *,
    trusted_integers: bool = False,
) -> tuple[int, bytes]:
    if not values:
        return 1, b""
    if not trusted_integers and any(isinstance(item, bool) for item in values):
        raise TaiChuDataFormatError(f"array {name!r} must contain only numbers")
    if trusted_integers:
        try:
            values_array = array("h", values)
            if sys.byteorder != "little":
                values_array.byteswap()
            return 5, values_array.tobytes()
        except (OverflowError, TypeError, ValueError):
            pass
    try:
        values_array = array("i", values)
        dtype = 1
    except OverflowError:
        try:
            values_array = array("q", values)
            dtype = 2
        except (OverflowError, TypeError) as exc:
            raise TaiChuDataFormatError(
                f"array {name!r} contains integers outside the int64 range"
            ) from exc
    except (TypeError, ValueError):
        try:
            values_array = array("f", values)
            dtype = 3
        except (OverflowError, TypeError, ValueError) as exc:
            raise TaiChuDataFormatError(
                f"array {name!r} must contain only numbers"
            ) from exc
    if dtype in (1, 2):
        if sys.byteorder != "little":
            values_array.byteswap()
        return dtype, values_array.tobytes()
    if sys.byteorder != "little":
        values_array.byteswap()
    return dtype, values_array.tobytes()


def _read_array(buffer: mmap.mmap, offset: int, dtype: int, count: int) -> list:
    typecode = ARRAY_TYPECODES.get(dtype)
    if typecode is None:
        raise TaiChuDataFormatError(f"unknown binary array dtype code {dtype}")
    width = ARRAY_WIDTHS[dtype]
    end = offset + count * width
    if end > len(buffer):
        raise TaiChuDataFormatError("binary array extends beyond shard boundary")
    values = array(typecode)
    values.frombytes(buffer[offset:end])
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


class BinarySampleWriter:
    """Write typed samples into checksummed binary shards atomically."""

    def __init__(
        self,
        output_dir: PathLike,
        *,
        dataset_id: str,
        sample_type: str,
        version: str = "1",
        parent_version: str | None = None,
        shard_size: int = 50_000,
        metadata: dict[str, Any] | None = None,
        sync_on_close: bool = True,
        checksums_on_commit: bool = True,
    ) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if not dataset_id or not sample_type or not version:
            raise ValueError("dataset_id, sample_type and version must be non-empty")
        self.output_dir = Path(output_dir)
        self.staging_dir = self.output_dir.with_name(f".{self.output_dir.name}.staging")
        if self.output_dir.exists():
            raise FileExistsError(f"output dataset already exists: {self.output_dir}")
        if self.staging_dir.exists():
            raise FileExistsError(f"staging directory already exists: {self.staging_dir}")
        self.dataset_id = dataset_id
        self.sample_type = sample_type
        self.version = version
        self.parent_version = parent_version
        self.shard_size = int(shard_size)
        self.metadata = dict(metadata or {})
        self.sync_on_close = bool(sync_on_close)
        self.checksums_on_commit = bool(checksums_on_commit)
        self._data_stream = None
        self._index_stream = None
        self._data_path: Path | None = None
        self._index_path: Path | None = None
        self._shard_samples = 0
        self._shard_files: list[tuple[Path, Path, int]] = []
        self._total_samples = 0
        self._closed = False
        self.staging_dir.mkdir(parents=True)

    def _open_shard(self) -> None:
        number = len(self._shard_files)
        self._data_path = self.staging_dir / f"data-{number:05d}.tcbin"
        self._index_path = self.staging_dir / f"data-{number:05d}.tcidx"
        self._data_stream = self._data_path.open("w+b")
        self._index_stream = self._index_path.open("w+b")
        self._data_stream.write(HEADER.pack(DATA_MAGIC, STORAGE_VERSION))
        self._index_stream.write(HEADER.pack(INDEX_MAGIC, STORAGE_VERSION))
        self._shard_samples = 0

    def write(self, sample: CommonSample, *, trusted_arrays: bool = False) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        if sample.sample_type != self.sample_type:
            raise TaiChuDataFormatError(
                f"writer expects {self.sample_type!r}, got {sample.sample_type!r}"
            )
        if self._data_stream is None:
            self._open_shard()
        assert self._data_stream is not None
        assert self._index_stream is not None

        metadata, arrays = sample_to_storage(sample)
        metadata_bytes = json.dumps(
            metadata,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        offset = self._data_stream.tell()
        self._data_stream.write(RECORD_HEADER.pack(len(metadata_bytes), len(arrays)))
        self._data_stream.write(metadata_bytes)
        for name in sorted(arrays):
            name_bytes = name.encode("utf-8")
            dtype, payload = _encode_array(
                arrays[name],
                name,
                trusted_integers=trusted_arrays,
            )
            self._data_stream.write(
                ARRAY_HEADER.pack(len(name_bytes), dtype, len(arrays[name]))
            )
            self._data_stream.write(name_bytes)
            self._data_stream.write(payload)
        self._index_stream.write(INDEX_ENTRY.pack(offset))
        self._shard_samples += 1
        self._total_samples += 1
        if self._shard_samples >= self.shard_size:
            self._close_shard()

    def _close_shard(self) -> None:
        if self._data_stream is None:
            return
        assert self._index_stream is not None
        assert self._data_path is not None
        assert self._index_path is not None
        for stream in (self._data_stream, self._index_stream):
            stream.flush()
            if self.sync_on_close:
                os.fsync(stream.fileno())
            stream.close()
        self._shard_files.append(
            (self._data_path, self._index_path, self._shard_samples)
        )
        self._data_stream = None
        self._index_stream = None
        self._data_path = None
        self._index_path = None
        self._shard_samples = 0

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
        for stream in (self._data_stream, self._index_stream):
            if stream is not None:
                stream.close()
        self._data_stream = None
        self._index_stream = None
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
        self._closed = True

    def __enter__(self) -> "BinarySampleWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._closed:
            if exc_type is None:
                self.commit()
            else:
                self.abort()


class BinaryShardReader:
    """Read one binary shard through read-only memory maps."""

    def __init__(self, data_path: PathLike, index_path: PathLike) -> None:
        self.data_path = Path(data_path)
        self.index_path = Path(index_path)
        self._data_file = self.data_path.open("rb")
        self._index_file = self.index_path.open("rb")
        self._data = mmap.mmap(self._data_file.fileno(), 0, access=mmap.ACCESS_READ)
        self._index = mmap.mmap(self._index_file.fileno(), 0, access=mmap.ACCESS_READ)
        if len(self._data) < HEADER.size or len(self._index) < HEADER.size:
            self.close()
            raise TaiChuDataFormatError("truncated binary storage header")
        data_magic, data_version = HEADER.unpack_from(self._data, 0)
        index_magic, index_version = HEADER.unpack_from(self._index, 0)
        if data_magic != DATA_MAGIC or index_magic != INDEX_MAGIC:
            self.close()
            raise TaiChuDataFormatError("invalid binary storage magic")
        if data_version != STORAGE_VERSION or index_version != STORAGE_VERSION:
            self.close()
            raise TaiChuDataFormatError("unsupported binary storage version")
        remainder = len(self._index) - HEADER.size
        if remainder % INDEX_ENTRY.size:
            self.close()
            raise TaiChuDataFormatError("invalid binary index length")
        self._count = remainder // INDEX_ENTRY.size

    def __len__(self) -> int:
        return self._count

    def read_sample(self, index: int) -> CommonSample:
        metadata, position, array_count = self._read_metadata_record(index)
        arrays: dict[str, list] = {}
        for _ in range(array_count):
            try:
                name_length, dtype, count = ARRAY_HEADER.unpack_from(
                    self._data,
                    position,
                )
            except struct.error as exc:
                raise TaiChuDataFormatError("truncated binary array header") from exc
            position += ARRAY_HEADER.size
            name_end = position + name_length
            if name_end > len(self._data):
                raise TaiChuDataFormatError("binary array name extends beyond shard")
            name = self._data[position:name_end].decode("utf-8")
            position = name_end
            arrays[name] = _read_array(self._data, position, dtype, count)
            position += count * ARRAY_WIDTHS[dtype]
        return sample_from_storage(metadata, arrays)

    def read_metadata(self, index: int) -> dict[str, Any]:
        metadata, _, _ = self._read_metadata_record(index)
        return metadata

    def _read_metadata_record(
        self,
        index: int,
    ) -> tuple[dict[str, Any], int, int]:
        if not 0 <= index < self._count:
            raise IndexError(index)
        (offset,) = INDEX_ENTRY.unpack_from(
            self._index,
            HEADER.size + index * INDEX_ENTRY.size,
        )
        try:
            metadata_length, array_count = RECORD_HEADER.unpack_from(self._data, offset)
        except struct.error as exc:
            raise TaiChuDataFormatError("truncated binary record header") from exc
        position = offset + RECORD_HEADER.size
        metadata_end = position + metadata_length
        if metadata_end > len(self._data):
            raise TaiChuDataFormatError("binary metadata extends beyond shard")
        try:
            metadata = json.loads(self._data[position:metadata_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TaiChuDataFormatError("invalid binary sample metadata") from exc
        return metadata, metadata_end, array_count

    def close(self) -> None:
        for value in ("_data", "_index"):
            buffer = getattr(self, value, None)
            if buffer is not None:
                buffer.close()
                setattr(self, value, None)
        for value in ("_data_file", "_index_file"):
            stream = getattr(self, value, None)
            if stream is not None:
                stream.close()
                setattr(self, value, None)

    def __enter__(self) -> "BinaryShardReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class BinaryDatasetReader(IndexedSampleReader):
    """Manifest-backed random-access reader over multiple binary shards."""

    def __init__(
        self,
        path: PathLike,
        *,
        checksums: bool = False,
    ) -> None:
        self.path = Path(path)
        self.checksums = bool(checksums)
        self.root = self.path if self.path.is_dir() else self.path.parent
        manifest_path = self.path if self.path.is_file() else self.root / MANIFEST_NAME
        self.manifest = DatasetManifest.load(manifest_path)
        if self.checksums:
            self.manifest.verify(self.root, checksums=True)
        self.dataset_id = self.manifest.dataset_id
        self.sample_type = self.manifest.sample_type
        self.version = self.manifest.version
        self._shards = [
            BinaryShardReader(
                self.root / shard.path,
                self.root / shard.index_path,
            )
            for shard in self.manifest.shards
        ]
        self._starts: list[int] = []
        cursor = 0
        for shard in self._shards:
            self._starts.append(cursor)
            cursor += len(shard)
        if cursor != self.manifest.sample_count:
            self.close()
            raise TaiChuDataFormatError("manifest sample_count does not match shards")

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
        for index in range(len(self)):
            yield self.metadata_at(index)

    def __iter__(self) -> Iterator[CommonSample]:
        for index in range(len(self)):
            yield self[index]

    def verify(self, *, checksums: bool = True) -> None:
        self.manifest.verify(self.root, checksums=checksums)

    def close(self) -> None:
        for shard in getattr(self, "_shards", ()):
            shard.close()

    def __getstate__(self) -> dict[str, Any]:
        """Serialize identity only; mmap handles must be reopened per worker."""
        return {
            "path": str(self.path),
            "checksums": self.checksums,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(
            state["path"],
            checksums=bool(state.get("checksums", False)),
        )

    def __enter__(self) -> "BinaryDatasetReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class BinaryStorageInfo:
    data_magic: bytes = DATA_MAGIC
    index_magic: bytes = INDEX_MAGIC
    version: int = STORAGE_VERSION
    dtypes: tuple[str, ...] = tuple(DTYPE_CODES.values())


__all__ = [
    "BinaryDatasetReader",
    "BinarySampleWriter",
    "BinaryShardReader",
    "BinaryStorageInfo",
    "DATA_MAGIC",
    "INDEX_MAGIC",
    "STORAGE_VERSION",
]
