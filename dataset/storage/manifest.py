"""Versioned, checksummed dataset manifest contracts."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..exceptions import TaiChuDataFormatError


MANIFEST_NAME = "manifest.json"
STORAGE_FORMAT = "taichu-binary-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(shards: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for shard in shards:
        digest.update(str(shard["path"]).encode("utf-8"))
        digest.update(str(shard["sha256"]).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ShardInfo:
    path: str
    index_path: str
    samples: int
    bytes: int
    sha256: str
    index_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "index_path": self.index_path,
            "samples": self.samples,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "index_sha256": self.index_sha256,
        }


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    sample_type: str
    version: str
    sample_count: int
    shards: tuple[ShardInfo, ...]
    schema_version: int = 1
    format: str = STORAGE_FORMAT
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    content_hash: str = ""
    parent_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.sample_type or not self.version:
            raise ValueError("dataset_id, sample_type and version must be non-empty")
        if self.sample_count < 0 or self.sample_count != sum(
            shard.samples for shard in self.shards
        ):
            raise ValueError("sample_count must equal the sum of shard sample counts")
        if self.format != STORAGE_FORMAT:
            raise ValueError(f"unsupported manifest format {self.format!r}")
        if not isinstance(self.metadata, dict):
            raise TypeError("manifest metadata must be a dictionary")

    def with_content_hash(self) -> "DatasetManifest":
        payload = self.to_dict(include_hash=False)
        digest = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        return DatasetManifest(
            dataset_id=self.dataset_id,
            sample_type=self.sample_type,
            version=self.version,
            sample_count=self.sample_count,
            shards=self.shards,
            schema_version=self.schema_version,
            format=self.format,
            created_at=self.created_at,
            content_hash=digest,
            parent_version=self.parent_version,
            metadata=self.metadata,
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "format": self.format,
            "dataset_id": self.dataset_id,
            "sample_type": self.sample_type,
            "version": self.version,
            "created_at": self.created_at,
            "sample_count": self.sample_count,
            "shards": [shard.to_dict() for shard in self.shards],
            "parent_version": self.parent_version,
            "metadata": self.metadata,
        }
        if include_hash:
            result["content_hash"] = self.content_hash
        return result

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DatasetManifest":
        allowed = {
            "schema_version",
            "format",
            "dataset_id",
            "sample_type",
            "version",
            "created_at",
            "sample_count",
            "shards",
            "content_hash",
            "parent_version",
            "metadata",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise TaiChuDataFormatError(f"unknown manifest keys: {sorted(unknown)}")
        try:
            shards = tuple(ShardInfo(**item) for item in raw.get("shards", []))
            manifest = cls(
                dataset_id=str(raw["dataset_id"]),
                sample_type=str(raw["sample_type"]),
                version=str(raw["version"]),
                sample_count=int(raw["sample_count"]),
                shards=shards,
                schema_version=int(raw.get("schema_version", 1)),
                format=str(raw.get("format", STORAGE_FORMAT)),
                created_at=str(raw.get("created_at", "")),
                content_hash=str(raw.get("content_hash", "")),
                parent_version=raw.get("parent_version"),
                metadata=dict(raw.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TaiChuDataFormatError(f"invalid dataset manifest: {exc}") from exc
        if not manifest.content_hash:
            raise TaiChuDataFormatError("dataset manifest is missing content_hash")
        if not manifest.created_at:
            raise TaiChuDataFormatError("dataset manifest is missing created_at")
        expected = manifest.with_content_hash().content_hash
        if manifest.content_hash != expected:
            raise TaiChuDataFormatError("dataset manifest content_hash does not match")
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        manifest_path = Path(path)
        if manifest_path.is_dir():
            manifest_path = manifest_path / MANIFEST_NAME
        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                raw = json.load(stream)
        except json.JSONDecodeError as exc:
            raise TaiChuDataFormatError(
                f"invalid manifest JSON: {manifest_path}"
            ) from exc
        if not isinstance(raw, dict):
            raise TaiChuDataFormatError("dataset manifest root must be an object")
        return cls.from_dict(raw)

    def verify(self, root: str | Path, *, checksums: bool = True) -> None:
        dataset_root = Path(root)
        if dataset_root.is_file():
            dataset_root = dataset_root.parent
        for shard in self.shards:
            data_path = dataset_root / shard.path
            index_path = dataset_root / shard.index_path
            if not data_path.is_file() or not index_path.is_file():
                raise FileNotFoundError(f"missing storage shard {shard.path}")
            if checksums:
                if _sha256_file(data_path) != shard.sha256:
                    raise TaiChuDataFormatError(f"checksum mismatch: {shard.path}")
                if _sha256_file(index_path) != shard.index_sha256:
                    raise TaiChuDataFormatError(
                        f"index checksum mismatch: {shard.index_path}"
                    )

    def write(self, directory: str | Path) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        target = root / MANIFEST_NAME
        temporary = root / f".{MANIFEST_NAME}.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        return target


def build_manifest(
    *,
    dataset_id: str,
    sample_type: str,
    version: str,
    shard_files: list[tuple[Path, Path, int]],
    root: Path,
    metadata: dict[str, Any] | None = None,
    parent_version: str | None = None,
    checksums: bool = True,
) -> DatasetManifest:
    shards = []
    for data_path, index_path, samples in shard_files:
        shards.append(
            ShardInfo(
                path=data_path.relative_to(root).as_posix(),
                index_path=index_path.relative_to(root).as_posix(),
                samples=samples,
                bytes=data_path.stat().st_size,
                sha256=_sha256_file(data_path) if checksums else "",
                index_sha256=_sha256_file(index_path) if checksums else "",
            )
        )
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        sample_type=sample_type,
        version=version,
        sample_count=sum(shard.samples for shard in shards),
        shards=tuple(shards),
        parent_version=parent_version,
        metadata=dict(metadata or {}),
    )
    manifest = manifest.with_content_hash()
    if not manifest.content_hash:
        raise RuntimeError("content fingerprint was not generated")
    return manifest


__all__ = [
    "DatasetManifest",
    "MANIFEST_NAME",
    "STORAGE_FORMAT",
    "ShardInfo",
    "build_manifest",
]
