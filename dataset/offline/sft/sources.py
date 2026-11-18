"""Format-specific raw SFT source adapters.

Every adapter returns canonical message records for the generic SFT selection
pipeline. Dataset-specific field parsing stays in this module so the core
selection and Parquet preparation logic remains source-agnostic.
"""
from __future__ import annotations

import gzip
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import ijson
import pyarrow.parquet as parquet

from .config import RawSourceConfig


_ROLE_MAP = {
    "assistant": "assistant",
    "gpt": "assistant",
    "human": "user",
    "user": "user",
    "system": "system",
    "tool": "tool",
}

_CHINESE_CATEGORY_PREFIXES = (
    ("stem_zh/", "zh_stem"),
    ("exam/", "zh_exam"),
    ("zhihu/", "zh_general"),
    ("xhs/", "zh_general"),
    ("human_value/", "zh_general"),
    ("ruozhiba/", "zh_general"),
    ("coig/", "zh_general"),
)

_CHINESE_MATH_SOURCES = {
    "EduChat-Math",
    "meta-math/GSM8K_zh",
    "gavinluo/applied_math",
    "Haijian/Advanced-Math",
    "logi_qa/logi-qa",
}

_MOSS_TURN = re.compile(
    r"\[(Human|MOSS)\]:\s*(.*?)(?=\[Human\]:|\[MOSS\]:|$)",
    flags=re.DOTALL,
)
_MOSS_TAG = re.compile(r"<eoh>|<eoa>")


def _resolve_files(
    path: Path,
    *,
    parquet_only: bool = False,
    extra_suffixes: tuple[str, ...] = (),
) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"raw SFT source does not exist: {path}")
    if parquet_only:
        files = sorted(path.rglob("*.parquet"))
    else:
        suffixes = (".jsonl", ".jsonl.gz", *extra_suffixes)
        files = sorted(
            item
            for item in path.rglob("*")
            if item.is_file()
            and any(item.name.lower().endswith(suffix) for suffix in suffixes)
        )
    if not files:
        raise FileNotFoundError(f"raw SFT source has no data files: {path}")
    return files


def _iter_parquet(
    files: list[Path],
    *,
    columns: list[str] | None = None,
    batch_size: int = 8192,
) -> Iterator[dict[str, Any]]:
    for path in files:
        handle = parquet.ParquetFile(path)
        for batch in handle.iter_batches(
            batch_size=batch_size,
            columns=columns,
            use_threads=True,
        ):
            yield from batch.to_pylist()


def _iter_dataset_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream records from the raw formats supported by canonical adapters."""

    name = path.name.lower()
    if name.endswith(".parquet"):
        yield from _iter_parquet([path])
        return
    if name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        return
    if name.endswith(".jsonl"):
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        return
    if name.endswith(".json"):
        with path.open("rb") as stream:
            yield from ijson.items(stream, "item")
        return
    raise ValueError(f"unsupported raw SFT file format: {path}")


def _decode_conversations(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return value if isinstance(value, list) else []
    return []


def _conversation_messages(
    raw: Any,
    *,
    system: str | None = None,
) -> list[dict[str, str]]:
    messages = []
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system.strip()})
    for item in _decode_conversations(raw):
        if not isinstance(item, dict):
            continue
        human = item.get("human")
        assistant = item.get("assistant")
        if isinstance(human, str) and human.strip():
            messages.append({"role": "user", "content": human.strip()})
        if isinstance(assistant, str) and assistant.strip():
            messages.append({"role": "assistant", "content": assistant.strip()})
        if isinstance(human, str) or isinstance(assistant, str):
            continue
        role = _ROLE_MAP.get(str(item.get("from") or item.get("role") or ""))
        content = item.get("value", item.get("content"))
        if role is None or not isinstance(content, str) or not content.strip():
            continue
        messages.append({"role": role, "content": content.strip()})
    return messages


def _record_messages(
    record: dict[str, Any],
    *,
    messages_key: str | None = None,
) -> list[dict[str, str]]:
    keys = (messages_key,) if messages_key else (
        "messages",
        "conversations",
        "conversation",
    )
    for key in keys:
        if key in record:
            return _conversation_messages(record.get(key))
    return []


def _moss_messages(plain_text: Any) -> list[dict[str, str]]:
    if not isinstance(plain_text, str):
        return []
    messages = []
    for speaker, content in _MOSS_TURN.findall(plain_text):
        text = _MOSS_TAG.sub("", content).strip()
        if not text:
            continue
        messages.append(
            {
                "role": "user" if speaker == "Human" else "assistant",
                "content": text,
            }
        )
    return messages


def _has_all_roles(
    messages: list[dict[str, str]],
    roles: set[str],
) -> bool:
    return roles.issubset({message["role"] for message in messages})


def _text_sample(messages: list[dict[str, str]], size: int = 4000) -> str:
    return "\n".join(message["content"] for message in messages)[:size].lower()


def _classify_openhermes(
    source: str,
    category: str,
    messages: list[dict[str, str]],
) -> str:
    source_lower = source.lower()
    category_lower = category.lower()
    text = _text_sample(messages)
    if (
        "glaive" in source_lower
        or "code" in source_lower
        or "code" in category_lower
        or any(token in text for token in ("def ", "python", "javascript", "java code"))
    ):
        return "en_code"
    if "metamath" in source_lower or "math" in source_lower:
        return "en_math"
    if source_lower in {"cogstackmed", "camelai"}:
        return "en_science"
    if any(
        token in category_lower or token in source_lower
        for token in ("biology", "physics", "chemistry", "medical", "science")
    ):
        return "en_science"
    if any(token in text for token in ("theorem", "equation", "\\frac", "solve for")):
        return "en_math"
    return "en_general"


def _classify_bespoke(messages: list[dict[str, str]]) -> str:
    text = _text_sample(messages)
    if any(
        token in text
        for token in ("python", "function", "algorithm", "programming", "codeforces")
    ):
        return "en_code"
    if any(
        token in text
        for token in ("biology", "chemistry", "physics", "scientific")
    ):
        return "en_science"
    return "en_math"


def _classify_cot_type(value: str) -> str:
    lowered = value.lower()
    if "code" in lowered:
        return "en_code"
    if "math" in lowered:
        return "en_math"
    if any(
        token in lowered
        for token in ("science", "biology", "chemistry", "physics")
    ):
        return "en_science"
    return "en_general"


class RawSFTAdapter(ABC):
    """Stream canonical SFT records from one raw source."""

    group: str

    def __init__(self, config: RawSourceConfig) -> None:
        self.config = config
        self.path = config.path
        self.group = config.group
        self.options = dict(config.options)

    @abstractmethod
    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Yield sample_id, source_id, category and messages."""


class ChineseDeepSeekAdapter(RawSFTAdapter):
    def iter_records(self) -> Iterator[dict[str, Any]]:
        min_score = int(self.options.get("min_score", 8))
        index = 0
        for path in _resolve_files(self.path):
            with path.open("rb") as stream:
                for line in stream:
                    record = json.loads(line)
                    score = int(record.get("score") or 0)
                    if score < min_score:
                        index += 1
                        continue
                    source_id = str(record.get("repo_name") or self.config.name)
                    category = (
                        "zh_math"
                        if source_id in _CHINESE_MATH_SOURCES
                        else "zh_general"
                    )
                    for prefix, value in _CHINESE_CATEGORY_PREFIXES:
                        if source_id.startswith(prefix):
                            category = value
                            break
                    instruction = str(record.get("instruction") or "").strip()
                    extra_input = str(record.get("input") or "").strip()
                    output = str(record.get("output") or "").strip()
                    user_content = (
                        f"{instruction}\n\n{extra_input}"
                        if extra_input
                        else instruction
                    )
                    messages = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": output},
                    ]
                    yield {
                        "sample_id": f"zh-deepseek-{index:08d}",
                        "source_id": source_id,
                        "source_group": self.group,
                        "category": category,
                        "messages": messages,
                        "quality_score": score,
                    }
                    index += 1


class AlpacaZhAdapter(RawSFTAdapter):
    def iter_records(self) -> Iterator[dict[str, Any]]:
        if not self.path.is_file():
            raise FileNotFoundError(f"alpaca-zh source does not exist: {self.path}")
        with self.path.open("r", encoding="utf-8") as stream:
            records = json.load(stream)
        if not isinstance(records, list):
            raise ValueError("alpaca-zh source must contain a JSON list")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            instruction = str(record.get("instruction") or "").strip()
            extra_input = str(record.get("input") or "").strip()
            output = str(record.get("output") or "").strip()
            if not instruction or not output:
                continue
            user_content = (
                f"{instruction}\n\n{extra_input}"
                if extra_input
                else instruction
            )
            yield {
                "sample_id": f"alpaca-zh-{index:08d}",
                "source_id": "shibing624/alpaca-zh",
                "source_group": self.group,
                "category": "zh_general",
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output},
                ],
            }


class OpenHermesAdapter(RawSFTAdapter):
    def iter_records(self) -> Iterator[dict[str, Any]]:
        files = _resolve_files(self.path)
        index = 0
        for path in files:
            if path.suffix.lower() != ".json":
                continue
            with path.open("rb") as stream:
                for record in ijson.items(stream, "item"):
                    messages = _conversation_messages(record.get("conversations"))
                    if not _has_all_roles(messages, {"user", "assistant"}):
                        index += 1
                        continue
                    source_id = str(
                        record.get("source")
                        or record.get("category")
                        or "openhermes"
                    )
                    category = _classify_openhermes(
                        source_id,
                        str(record.get("category") or ""),
                        messages,
                    )
                    yield {
                        "sample_id": f"openhermes-{index:08d}",
                        "source_id": source_id,
                        "source_group": self.group,
                        "category": category,
                        "messages": messages,
                    }
                    index += 1


class OpenThoughtsAdapter(RawSFTAdapter):
    def iter_records(self) -> Iterator[dict[str, Any]]:
        index = 0
        columns = [
            "problem",
            "deepseek_reasoning",
            "deepseek_solution",
            "domain",
            "source",
        ]
        for record in _iter_parquet(
            _resolve_files(self.path, parquet_only=True),
            columns=columns,
        ):
            problem = str(record.get("problem") or "").strip()
            reasoning = str(record.get("deepseek_reasoning") or "").strip()
            solution = str(record.get("deepseek_solution") or "").strip()
            if not problem or not reasoning or not solution:
                index += 1
                continue
            domain = str(record.get("domain") or "").lower()
            if domain == "math":
                category = "en_math"
            elif domain == "code":
                category = "en_code"
            elif domain in {"biology", "physics", "chemistry"}:
                category = "en_science"
            else:
                category = "en_general"
            assistant = (
                f"<|think|>\n{reasoning}\n<|/think|>\n\n{solution}"
            )
            yield {
                "sample_id": f"open-thoughts-{index:08d}",
                "source_id": f"open_thoughts:{record.get('source') or domain}",
                "source_group": self.group,
                "category": category,
                "messages": [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": assistant},
                ],
            }
            index += 1


class OpenR1Adapter(RawSFTAdapter):
    def iter_records(self) -> Iterator[dict[str, Any]]:
        index = 0
        columns = ["messages", "source", "uuid", "correctness_count"]
        for record in _iter_parquet(
            _resolve_files(self.path, parquet_only=True),
            columns=columns,
        ):
            if int(record.get("correctness_count") or 0) <= 0:
                index += 1
                continue
            messages = _conversation_messages(record.get("messages"))
            if not _has_all_roles(messages, {"user", "assistant"}):
                index += 1
                continue
            source_id = f"open_r1:{record.get('source') or 'unknown'}"
            yield {
                "sample_id": str(record.get("uuid") or f"open-r1-{index:08d}"),
                "source_id": source_id,
                "source_group": self.group,
                "category": "en_math",
                "messages": messages,
            }
            index += 1


class BespokeStratosAdapter(RawSFTAdapter):
    def iter_records(self) -> Iterator[dict[str, Any]]:
        index = 0
        for record in _iter_parquet(
            _resolve_files(self.path, parquet_only=True),
            columns=["system", "conversations"],
        ):
            messages = _conversation_messages(
                record.get("conversations"),
                system=record.get("system"),
            )
            if not _has_all_roles(messages, {"user", "assistant"}):
                index += 1
                continue
            yield {
                "sample_id": f"bespoke-{index:08d}",
                "source_id": "bespoke-stratos-17k",
                "source_group": self.group,
                "category": _classify_bespoke(messages),
                "messages": messages,
            }
            index += 1


class CanonicalMessagesAdapter(RawSFTAdapter):
    """Read Parquet, JSONL, or JSON records into canonical message arrays."""

    def iter_records(self) -> Iterator[dict[str, Any]]:
        category = str(self.options.get("category", "en_general"))
        messages_key = self.options.get("messages_key")
        min_user_turns = int(self.options.get("min_user_turns", 1))
        files = _resolve_files(
            self.path,
            extra_suffixes=(".json", ".parquet"),
        )
        include_patterns = self.options.get("include_patterns")
        if include_patterns:
            if not isinstance(include_patterns, list) or not all(
                isinstance(pattern, str) for pattern in include_patterns
            ):
                raise ValueError(
                    "canonical_messages include_patterns must be a string list"
                )
            files = [
                path
                for path in files
                if any(
                    fnmatch(path.name, pattern)
                    or fnmatch(
                        path.relative_to(self.path).as_posix(),
                        pattern,
                    )
                    for pattern in include_patterns
                )
            ]
            if not files:
                raise FileNotFoundError(
                    f"raw SFT source has no files matching {include_patterns}: "
                    f"{self.path}"
                )
        index = 0
        for path in files:
            for record in _iter_dataset_records(path):
                messages = _record_messages(
                    record,
                    messages_key=(
                        str(messages_key) if messages_key is not None else None
                    ),
                )
                if (
                    not _has_all_roles(messages, {"user", "assistant"})
                    or sum(message["role"] == "user" for message in messages)
                    < min_user_turns
                ):
                    index += 1
                    continue
                yield {
                    "sample_id": (
                        f"{self.config.name}-{path.stem}-{index:08d}"
                    ),
                    "source_id": f"{self.config.name}:{path.stem}",
                    "source_group": self.group,
                    "category": category,
                    "messages": messages,
                }
                index += 1


class MossSftAdapter(RawSFTAdapter):
    """Parse MOSS plain_text conversations delimited by <eoh>/<eoa>."""

    def iter_records(self) -> Iterator[dict[str, Any]]:
        category = str(self.options.get("category", "zh_general"))
        min_user_turns = int(self.options.get("min_user_turns", 1))
        files = _resolve_files(
            self.path,
            extra_suffixes=(".json",),
        )
        for path in files:
            for index, record in enumerate(_iter_dataset_records(path)):
                messages = _moss_messages(record.get("plain_text"))
                if (
                    not _has_all_roles(messages, {"user", "assistant"})
                    or sum(message["role"] == "user" for message in messages)
                    < min_user_turns
                ):
                    continue
                yield {
                    "sample_id": f"{self.config.name}-{path.stem}-{index:08d}",
                    "source_id": f"{self.config.name}:{path.stem}",
                    "source_group": self.group,
                    "category": category,
                    "messages": messages,
                }


class S1KAdapter(RawSFTAdapter):
    def iter_records(self) -> Iterator[dict[str, Any]]:
        index = 0
        columns = [
            "question",
            "cot_type",
            "source_type",
            "deepseek_thinking_trajectory",
            "deepseek_attempt",
            "deepseek_grade",
        ]
        for record in _iter_parquet(
            _resolve_files(self.path, parquet_only=True),
            columns=columns,
        ):
            if str(record.get("deepseek_grade") or "").lower() != "yes":
                index += 1
                continue
            question = str(record.get("question") or "").strip()
            reasoning = str(
                record.get("deepseek_thinking_trajectory") or ""
            ).strip()
            attempt = str(record.get("deepseek_attempt") or "").strip()
            if not question or not reasoning or not attempt:
                index += 1
                continue
            assistant = f"<|think|>\n{reasoning}\n<|/think|>\n\n{attempt}"
            yield {
                "sample_id": f"s1k-{index:04d}",
                "source_id": f"s1k:{record.get('source_type') or 'unknown'}",
                "source_group": self.group,
                "category": _classify_cot_type(str(record.get("cot_type") or "")),
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": assistant},
                ],
            }
            index += 1


ADAPTERS = {
    "alpaca_zh": AlpacaZhAdapter,
    "canonical_messages": CanonicalMessagesAdapter,
    "chinese_deepseek": ChineseDeepSeekAdapter,
    "moss_sft": MossSftAdapter,
    "openhermes": OpenHermesAdapter,
    "open_thoughts": OpenThoughtsAdapter,
    "open_r1": OpenR1Adapter,
    "bespoke_stratos": BespokeStratosAdapter,
    "s1k": S1KAdapter,
}


def create_raw_adapter(config: RawSourceConfig) -> RawSFTAdapter:
    try:
        adapter_type = ADAPTERS[config.adapter]
    except KeyError as exc:
        raise ValueError(
            f"unknown SFT raw adapter {config.adapter!r}; "
            f"expected one of {sorted(ADAPTERS)}"
        ) from exc
    return adapter_type(config)


__all__ = ["RawSFTAdapter", "create_raw_adapter"]
