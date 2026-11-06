#!/usr/bin/env python3
"""Deterministically sample JSONL sources to a GB-based pretraining mixture."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import struct
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO

import orjson
import yaml
from tqdm import tqdm


GB = 1_000_000_000
ELIGIBLE_ENTRY = struct.Struct("<BQI")


@dataclass(frozen=True, slots=True)
class SamplingSource:
    name: str
    language: str
    patterns: tuple[str, ...]
    weight_gb: float


@dataclass(frozen=True, slots=True)
class SourceSamplingConfig:
    target_total_gb: float
    output_dir: Path
    sources: tuple[SamplingSource, ...]
    seed: int = 42
    validation_fraction: float = 0.02
    min_chars: int = 16
    max_chars: int = 200_000


@dataclass(slots=True)
class SourceStats:
    records: int = 0
    input_bytes: int = 0
    short_records: int = 0
    long_records: int = 0
    eligible_records: int = 0
    eligible_bytes: int = 0
    selected_records: int = 0
    selected_bytes: int = 0
    selected_text_bytes: int = 0
    train_records: int = 0
    train_bytes: int = 0
    validation_records: int = 0
    validation_bytes: int = 0


def load_sampling_config(path: str | Path) -> SourceSamplingConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("sampling config root must be a mapping")
    allowed = {
        "target_total_gb",
        "output_dir",
        "seed",
        "validation_fraction",
        "min_chars",
        "max_chars",
        "sources",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown sampling config keys: {sorted(unknown)}")
    raw_sources = raw.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("sampling config needs a non-empty sources list")
    sources = []
    names = set()
    for item in raw_sources:
        if not isinstance(item, dict):
            raise ValueError("each sampling source must be a mapping")
        source_allowed = {"name", "language", "patterns", "weight_gb"}
        source_unknown = set(item) - source_allowed
        if source_unknown:
            raise ValueError(
                f"unknown sampling source keys: {sorted(source_unknown)}"
            )
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("sampling source name must be non-empty")
        if name in names:
            raise ValueError(f"duplicate sampling source name: {name}")
        names.add(name)
        patterns = item.get("patterns")
        if (
            not isinstance(patterns, list)
            or not patterns
            or any(not isinstance(pattern, str) for pattern in patterns)
        ):
            raise ValueError(f"source {name!r} needs a non-empty patterns list")
        sources.append(
            SamplingSource(
                name=name,
                language=str(item.get("language", "")),
                patterns=tuple(patterns),
                weight_gb=float(item["weight_gb"]),
            )
        )
    config = SourceSamplingConfig(
        target_total_gb=float(raw["target_total_gb"]),
        output_dir=Path(raw["output_dir"]),
        sources=tuple(sources),
        seed=int(raw.get("seed", 42)),
        validation_fraction=float(raw.get("validation_fraction", 0.02)),
        min_chars=int(raw.get("min_chars", 16)),
        max_chars=int(raw.get("max_chars", 200_000)),
    )
    validate_config(config)
    return config


def validate_config(config: SourceSamplingConfig) -> None:
    if config.target_total_gb <= 0:
        raise ValueError("target_total_gb must be positive")
    if not 0 <= config.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if config.min_chars < 0 or config.max_chars < config.min_chars:
        raise ValueError("min_chars/max_chars must satisfy 0 <= min <= max")
    if any(source.weight_gb <= 0 for source in config.sources):
        raise ValueError("source weight_gb values must be positive")
    if not config.sources:
        raise ValueError("at least one sampling source is required")


def resolve_source_files(source: SamplingSource) -> tuple[Path, ...]:
    files: set[Path] = set()
    for raw_pattern in source.patterns:
        matches = [
            Path(item)
            for item in glob.glob(raw_pattern, recursive=True)
        ]
        if not matches:
            raise FileNotFoundError(
                f"source {source.name!r} pattern matched no files: {raw_pattern}"
            )
        for match in matches:
            if match.is_file() and match.name.lower().endswith(".jsonl"):
                files.add(match.resolve())
            elif match.is_dir():
                files.update(
                    item.resolve()
                    for item in match.rglob("*.jsonl")
                    if item.is_file()
                )
    if not files:
        raise FileNotFoundError(
            f"source {source.name!r} has no supported JSONL files"
        )
    return tuple(sorted(files, key=lambda item: str(item).casefold()))


def _sampling_key(seed: int, source: str, record_index: int) -> bytes:
    return hashlib.blake2b(
        f"{seed}:{source}:{record_index}".encode("utf-8"),
        digest_size=16,
    ).digest()


def _record_sample_value(key: bytes) -> float:
    return int.from_bytes(key[:8], "big") / 2**64


def _validation_value(key: bytes) -> float:
    return int.from_bytes(key[8:], "big") / 2**64


def _read_jsonl_record(
    path: Path,
    line: bytes,
    line_number: int,
) -> dict[str, Any]:
    try:
        record = orjson.loads(line)
    except orjson.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL in {path} at line {line_number}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"JSONL record in {path} at line {line_number} is not an object")
    if not isinstance(record.get("text"), str):
        raise ValueError(
            f"JSONL record in {path} at line {line_number} has no text string"
        )
    return record


def _target_bytes(
    config: SourceSamplingConfig,
    source: SamplingSource,
    target_total_gb: float,
) -> int:
    total_weight = sum(item.weight_gb for item in config.sources)
    return round(target_total_gb * GB * source.weight_gb / total_weight)


def sampling_plan(
    config: SourceSamplingConfig,
    *,
    target_total_gb: float,
) -> list[dict[str, Any]]:
    plan = []
    for source in config.sources:
        files = resolve_source_files(source)
        input_bytes = sum(path.stat().st_size for path in files)
        target_bytes = _target_bytes(config, source, target_total_gb)
        if target_bytes > input_bytes:
            raise ValueError(
                f"source {source.name!r} needs {target_bytes / GB:.3f}GB "
                f"but only has {input_bytes / GB:.3f}GB"
            )
        plan.append(
            {
                "name": source.name,
                "language": source.language,
                "weight_gb": source.weight_gb,
                "target_gb": target_bytes / GB,
                "input_gb": input_bytes / GB,
                "sample_fraction": target_bytes / input_bytes,
                "files": [str(path) for path in files],
            }
        )
    return plan


def _open_output(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("wb")


def _replace_with_retry(source: Path, target: Path) -> None:
    delay = 0.25
    for attempt in range(5):
        try:
            os.replace(source, target)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 2.0)


def _merge_staging_tree(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    entries = sorted(
        source.iterdir(),
        key=lambda item: (item.name == "sampling_manifest.json", item.name.casefold()),
    )
    for entry in entries:
        destination = target / entry.name
        if entry.is_dir():
            if destination.exists() and not destination.is_dir():
                raise FileExistsError(
                    f"cannot merge staging directory over file: {destination}"
                )
            try:
                _replace_with_retry(entry, destination)
            except OSError:
                _merge_staging_tree(entry, destination)
            continue
        if destination.exists():
            if (
                destination.is_file()
                and destination.stat().st_size == entry.stat().st_size
            ):
                entry.unlink()
                continue
            raise FileExistsError(
                f"cannot overwrite existing staging output: {destination}"
            )
        _replace_with_retry(entry, destination)
    source.rmdir()


def _commit_staging(staging: Path, output_dir: Path) -> None:
    manifest = staging / "sampling_manifest.json"
    if not manifest.is_file():
        raise RuntimeError(
            f"sampling staging is incomplete and has no manifest: {staging}"
        )
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"sampling output is not a directory: {output_dir}")
        _merge_staging_tree(staging, output_dir)
        return
    try:
        _replace_with_retry(staging, output_dir)
    except OSError:
        _merge_staging_tree(staging, output_dir)


def _read_sampling_manifest(output_dir: Path) -> dict[str, Any]:
    with (output_dir / "sampling_manifest.json").open(
        "r",
        encoding="utf-8",
    ) as stream:
        return json.load(stream)


def _progress_bar(total: int, description: str, show_progress: bool):
    return tqdm(
        total=total,
        desc=description,
        unit="B",
        unit_scale=True,
        unit_divisor=1000,
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )


def _scan_source(
    source: SamplingSource,
    *,
    files: tuple[Path, ...],
    config: SourceSamplingConfig,
    eligible_index: BinaryIO,
    show_progress: bool,
) -> SourceStats:
    if len(files) > 255:
        raise ValueError(
            f"source {source.name!r} has more than 255 files; "
            "reduce the source glob scope"
        )
    input_bytes = sum(path.stat().st_size for path in files)
    stats = SourceStats()
    index_buffer = bytearray()
    with _progress_bar(
        input_bytes,
        f"scan   {source.name}",
        show_progress,
    ) as progress:
        for file_index, path in enumerate(files):
            offset = 0
            with path.open("rb") as stream:
                for line_number, line in enumerate(stream, start=1):
                    raw_bytes = len(line)
                    if line.strip():
                        record = _read_jsonl_record(path, line, line_number)
                        stats.records += 1
                        stats.input_bytes += raw_bytes
                        text_chars = len(record["text"])
                        if text_chars < config.min_chars:
                            stats.short_records += 1
                        elif text_chars > config.max_chars:
                            stats.long_records += 1
                        else:
                            stats.eligible_records += 1
                            stats.eligible_bytes += raw_bytes
                            index_buffer.extend(
                                ELIGIBLE_ENTRY.pack(
                                    file_index,
                                    offset,
                                    raw_bytes,
                                )
                            )
                            if len(index_buffer) >= 1 << 20:
                                eligible_index.write(index_buffer)
                                index_buffer.clear()
                    offset += raw_bytes
                    progress.update(raw_bytes)
    if index_buffer:
        eligible_index.write(index_buffer)
    eligible_index.flush()
    return stats


def _sample_source(
    source: SamplingSource,
    *,
    files: tuple[Path, ...],
    target_bytes: int,
    config: SourceSamplingConfig,
    staging: Path,
    show_progress: bool,
) -> dict[str, Any]:
    with tempfile.TemporaryFile(mode="w+b") as eligible_index:
        stats = _scan_source(
            source,
            files=files,
            config=config,
            eligible_index=eligible_index,
            show_progress=show_progress,
        )
        if stats.eligible_bytes == 0:
            raise ValueError(
                f"source {source.name!r} has no records within "
                f"{config.min_chars}..{config.max_chars} characters"
            )
        if target_bytes > stats.eligible_bytes:
            raise ValueError(
                f"source {source.name!r} needs {target_bytes / GB:.3f}GB after "
                f"length filtering but only has {stats.eligible_bytes / GB:.3f}GB"
            )
        sample_fraction = target_bytes / stats.eligible_bytes
        train_path = staging / "train" / f"{source.name}.jsonl"
        validation_path = staging / "validation" / f"{source.name}.jsonl"
        eligible_index.seek(0)
        source_streams = [path.open("rb") for path in files]
        try:
            with (
                _open_output(train_path) as train_stream,
                _open_output(validation_path) as validation_stream,
                _progress_bar(
                    stats.eligible_bytes,
                    f"sample {source.name}",
                    show_progress,
                ) as progress,
            ):
                record_index = 0
                chunk_size = ELIGIBLE_ENTRY.size * 8192
                while chunk := eligible_index.read(chunk_size):
                    if len(chunk) % ELIGIBLE_ENTRY.size:
                        raise RuntimeError(
                            f"invalid eligible-record index for {source.name!r}"
                        )
                    for position in range(0, len(chunk), ELIGIBLE_ENTRY.size):
                        file_index, offset, raw_bytes = ELIGIBLE_ENTRY.unpack_from(
                            chunk,
                            position,
                        )
                        key = _sampling_key(
                            config.seed,
                            source.name,
                            record_index,
                        )
                        record_index += 1
                        try:
                            if _record_sample_value(key) >= sample_fraction:
                                continue
                            source_stream = source_streams[file_index]
                            source_stream.seek(offset)
                            line = source_stream.read(raw_bytes)
                            if len(line) != raw_bytes:
                                raise RuntimeError(
                                    f"short read from {files[file_index]} "
                                    f"at offset {offset}"
                                )
                            if not line.endswith(b"\n"):
                                line += b"\n"
                            if _validation_value(key) < config.validation_fraction:
                                stream_out = validation_stream
                                stats.validation_records += 1
                                stats.validation_bytes += raw_bytes
                            else:
                                stream_out = train_stream
                                stats.train_records += 1
                                stats.train_bytes += raw_bytes
                            stream_out.write(line)
                            record = _read_jsonl_record(
                                files[file_index],
                                line,
                                record_index,
                            )
                            stats.selected_records += 1
                            stats.selected_bytes += raw_bytes
                            stats.selected_text_bytes += len(
                                record["text"].encode("utf-8")
                            )
                        finally:
                            progress.update(raw_bytes)
        finally:
            for stream in source_streams:
                stream.close()
    return {
        "name": source.name,
        "language": source.language,
        "weight_gb": source.weight_gb,
        "target_bytes": target_bytes,
        "target_gb": target_bytes / GB,
        "sample_fraction": sample_fraction,
        "min_chars": config.min_chars,
        "max_chars": config.max_chars,
        "input_files": [str(path) for path in files],
        "input_gb": stats.input_bytes / GB,
        "eligible_gb": stats.eligible_bytes / GB,
        "selected_gb": stats.selected_bytes / GB,
        "selected_text_gb": stats.selected_text_bytes / GB,
        "train_gb": stats.train_bytes / GB,
        "validation_gb": stats.validation_bytes / GB,
        **asdict(stats),
    }


def run_sampling(
    config: SourceSamplingConfig,
    *,
    target_total_gb: float | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
    min_chars: int | None = None,
    max_chars: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    if min_chars is not None or max_chars is not None:
        config = replace(
            config,
            min_chars=config.min_chars if min_chars is None else int(min_chars),
            max_chars=config.max_chars if max_chars is None else int(max_chars),
        )
        validate_config(config)
    target_total_gb = (
        config.target_total_gb if target_total_gb is None else float(target_total_gb)
    )
    if target_total_gb <= 0:
        raise ValueError("target_total_gb must be positive")
    output_dir = config.output_dir if output_dir is None else Path(output_dir)
    plan = sampling_plan(config, target_total_gb=target_total_gb)
    summary = {
        "schema_version": 1,
        "seed": config.seed,
        "validation_fraction": config.validation_fraction,
        "min_chars": config.min_chars,
        "max_chars": config.max_chars,
        "target_total_gb": target_total_gb,
        "output_dir": str(output_dir),
        "sources": plan,
    }
    if dry_run:
        return summary

    staging = output_dir.with_name(f".{output_dir.name}.staging")
    if output_dir.exists():
        if staging.exists() and (staging / "sampling_manifest.json").is_file():
            _commit_staging(staging, output_dir)
            return _read_sampling_manifest(output_dir)
        raise FileExistsError(f"sampling output already exists: {output_dir}")
    if staging.exists():
        if (staging / "sampling_manifest.json").is_file():
            _commit_staging(staging, output_dir)
            return _read_sampling_manifest(output_dir)
        raise FileExistsError(
            f"sampling staging is incomplete and cannot be resumed: {staging}"
        )
    staging.mkdir(parents=True)
    finalizing = False
    try:
        source_results = []
        for source, source_plan in zip(config.sources, plan, strict=True):
            source_results.append(
                _sample_source(
                    source,
                    files=tuple(Path(path) for path in source_plan["files"]),
                    target_bytes=round(source_plan["target_gb"] * GB),
                    config=config,
                    staging=staging,
                    show_progress=show_progress,
                )
            )
        actual_bytes = sum(item["selected_bytes"] for item in source_results)
        summary["sources"] = source_results
        summary["actual_total_gb"] = actual_bytes / GB
        summary["actual_total_text_gb"] = sum(
            item["selected_text_bytes"] for item in source_results
        ) / GB
        summary["train_records"] = sum(
            item["train_records"] for item in source_results
        )
        summary["validation_records"] = sum(
            item["validation_records"] for item in source_results
        )
        summary["short_records"] = sum(
            item["short_records"] for item in source_results
        )
        summary["long_records"] = sum(
            item["long_records"] for item in source_results
        )
        summary["eligible_records"] = sum(
            item["eligible_records"] for item in source_results
        )
        summary["eligible_gb"] = sum(
            item["eligible_bytes"] for item in source_results
        ) / GB
        with (staging / "sampling_manifest.json").open(
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        finalizing = True
        _commit_staging(staging, output_dir)
    except Exception:
        if staging.exists() and not finalizing:
            shutil.rmtree(staging)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample raw JSONL sources into a deterministic GB mixture"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-gb", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-chars", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="show_progress",
        help="disable tqdm progress bars",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_sampling_config(args.config)
    summary = run_sampling(
        config,
        target_total_gb=args.target_gb,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        show_progress=args.show_progress,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "SourceSamplingConfig",
    "SamplingSource",
    "load_sampling_config",
    "run_sampling",
    "sampling_plan",
]
