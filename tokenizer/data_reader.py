"""Streaming JSON and JSONL readers used by Byte-Level BPE training."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Sequence

from tokenizer.config import DataConfig

logger = logging.getLogger(__name__)


@dataclass
class ReaderStats:
    records_read: int = 0
    records_emitted: int = 0
    invalid_records: int = 0
    missing_text: int = 0
    empty_text: int = 0
    truncated_text: int = 0
    chars_emitted: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


class JsonlRecordReader:
    """Read one JSON value per line without retaining prior records."""

    def __init__(self, path: Path, stats: ReaderStats):
        self.path = path
        self.stats = stats

    def __iter__(self) -> Iterator[Any]:
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    self.stats.invalid_records += 1
                    if self.stats.invalid_records <= 5:
                        logger.warning("跳过无效 JSON: %s:%d", self.path, line_number)
                    continue
                self.stats.records_read += 1
                yield record


class JsonRecordReader:
    """Read a JSON array incrementally or an explicit object/prefix."""

    def __init__(
        self,
        path: Path,
        stats: ReaderStats,
        json_prefix: Optional[str],
        allow_full_json_load: bool,
    ):
        self.path = path
        self.stats = stats
        self.json_prefix = json_prefix
        self.allow_full_json_load = allow_full_json_load

    def __iter__(self) -> Iterator[Any]:
        first = self._first_non_whitespace()
        if first == "[":
            for record in self._iter_array():
                self.stats.records_read += 1
                yield record
            return

        if self.json_prefix:
            try:
                import ijson
            except ImportError:
                ijson = None
            if ijson is not None:
                with self.path.open("rb") as handle:
                    for record in ijson.items(handle, self.json_prefix):
                        self.stats.records_read += 1
                        yield record
                return

        if not self.allow_full_json_load:
            raise ValueError(
                f"{self.path} 不是顶层 JSON 数组；为避免将整个对象读入内存，"
                "请安装 ijson 并设置 data.json_prefix，"
                "或对小文件显式设置 data.allow_full_json_load: true"
            )

        with self.path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        records = _resolve_prefix(document, self.json_prefix)
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list):
            raise ValueError(f"{self.path} 的 data.json_prefix 未解析出数组或对象")
        for record in records:
            self.stats.records_read += 1
            yield record

    def _first_non_whitespace(self) -> str:
        with self.path.open("r", encoding="utf-8") as handle:
            while True:
                char = handle.read(1)
                if not char or not char.isspace():
                    return char

    def _iter_array(self) -> Iterator[Any]:
        decoder = json.JSONDecoder()
        with self.path.open("r", encoding="utf-8") as handle:
            buffer = ""
            position = 0
            started = False
            eof = False
            while True:
                if not started:
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                    if position >= len(buffer):
                        chunk = handle.read(1024 * 1024)
                        if not chunk:
                            raise ValueError(f"{self.path} 是空 JSON 文件")
                        buffer = chunk
                        position = 0
                        continue
                    if buffer[position] != "[":
                        raise ValueError(f"{self.path} 不是顶层 JSON 数组")
                    position += 1
                    started = True

                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position >= len(buffer):
                    if eof:
                        return
                    chunk = handle.read(1024 * 1024)
                    if chunk:
                        buffer = buffer[position:] + chunk
                        position = 0
                        continue
                    eof = True
                    continue
                if buffer[position] == "]":
                    return

                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    chunk = handle.read(1024 * 1024)
                    if chunk:
                        buffer = buffer[position:] + chunk
                        position = 0
                        continue
                    eof = True
                    continue
                yield value
                position = end
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) and buffer[position] == ",":
                    position += 1


class TextRecordIterator:
    """Yield only the configured text field and keep bounded diagnostics."""

    def __init__(
        self,
        config: DataConfig,
        on_progress: Optional[Callable[[ReaderStats], None]] = None,
    ):
        self.config = config
        self.on_progress = on_progress
        self.stats = ReaderStats()

    def __iter__(self) -> Iterator[str]:
        if self.config.num_workers != 1:
            logger.warning(
                "data.num_workers=%d 已忽略：项目内 Python 多进程会让每个进程重复载入"
                "解释器和库，实测显著放大 RSS。Tokenizers 原生并行负责预分词。",
                self.config.num_workers,
            )
        for _ in range(self.config.epoch):
            for path in self.config.files:
                for record in self._iter_records(path):
                    text = self._extract_text(record)
                    if text is None:
                        continue
                    self.stats.records_emitted += 1
                    self.stats.chars_emitted += len(text)
                    if self.on_progress is not None:
                        self.on_progress(self.stats)
                    yield text

    def _iter_records(self, path: Path) -> Iterator[Any]:
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            return iter(JsonlRecordReader(path, self.stats))
        return iter(
            JsonRecordReader(
                path,
                self.stats,
                self.config.json_prefix,
                self.config.allow_full_json_load,
            )
        )

    def _extract_text(self, record: Any) -> Optional[str]:
        if not isinstance(record, dict):
            self.stats.missing_text += 1
            return None
        text = record.get(self.config.text_key)
        if not isinstance(text, str):
            self.stats.missing_text += 1
            return None
        if self.config.skip_empty and not text.strip():
            self.stats.empty_text += 1
            return None
        if self.config.max_text_chars and len(text) > self.config.max_text_chars:
            text = text[: self.config.max_text_chars]
            self.stats.truncated_text += 1
        return text


class CorpusIterator(TextRecordIterator):
    """Backward-compatible name for the unified text iterator."""

    def __init__(
        self,
        files: Sequence[str | Path],
        epoch: int = 1,
        text_key: str = "text",
        max_text_length: Optional[int] = None,
        num_workers: int = 1,
        skip_empty: bool = True,
        chunk_size: int = 1000,
        **kwargs: Any,
    ):
        del chunk_size
        paths = tuple(Path(path).expanduser().resolve() for path in files)
        config = DataConfig(
            files=paths,
            epoch=epoch,
            text_key=text_key,
            max_text_chars=max_text_length,
            skip_empty=skip_empty,
            num_workers=num_workers,
            json_prefix=kwargs.get("json_prefix"),
            allow_full_json_load=kwargs.get("allow_full_json_load", False),
        )
        super().__init__(config, on_progress=kwargs.get("on_progress"))


def _resolve_prefix(document: Any, prefix: Optional[str]) -> Any:
    if not prefix:
        return document
    current = document
    for part in prefix.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"JSON 前缀不存在: {prefix}")
        current = current[part]
    return current
