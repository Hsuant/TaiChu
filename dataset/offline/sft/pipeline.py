"""通用 SFT source 与 unified Parquet Dataset 原子构建流程。

处理顺序：

1. 读取 canonical messages，或通过隔离的 raw source adapter 转换原始格式。
2. 校验 role 和 conversation 结构。
3. 用 tokenizer 批量检查完整会话长度。
4. 对完整 conversation 做稳定内容去重。
5. 按可选 category quota 做确定性的 bottom-k 选择。
6. 写入 source train/validation JSONL。
7. 复用 ``UnifiedDataPreparation`` 生成 unified Parquet Dataset。
8. 全部成功后再原子发布整个 SFT 目录。

配额、去重、长度校验和 Parquet Dataset 构建保持数据源无关；平台字段只在
``sources.py`` 中解析。
"""
from __future__ import annotations

import hashlib
import heapq
import json
import shutil
import struct
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from tqdm import tqdm

from ...storage.fs import atomic_replace_directory
from ...schema import SFTSequenceSample
from ...storage import ParquetSampleWriter
from ..readers import RecordSourceReader
from .config import SFTBuildConfig
from .policy import SFTContentPolicy, SFTContentRejected
from .sources import create_raw_adapter


@dataclass(frozen=True, slots=True)
class Candidate:
    """已经通过结构、长度和配额候选检查的一条 SFT 样本。"""

    category: str
    sample_id: str
    source_id: str
    source_group: str
    messages: list[dict[str, str]]
    token_count: int
    content_key: bytes
    selection_key: int
    token_offset: int
    supervised_tokens: int
    assistant_tokens: int
    has_reasoning: bool
    reasoning_mode: str


@dataclass(frozen=True, slots=True)
class CandidateDraft:
    """A cheap pre-tokenization candidate used by quota prefilters."""

    category: str
    sample_id: str
    source_id: str
    source_group: str
    messages: list[dict[str, str]]
    content_key: bytes
    selection_key: int
    has_reasoning: bool
    assistant_texts: tuple[str, ...]
    reasoning_mode: str


@dataclass(slots=True)
class PreparationCounters:
    char_rejected: int = 0
    dropped_system_messages: int = 0
    content_rejected: Counter[str] = field(default_factory=Counter)


def _normalize_messages(raw: Any) -> list[dict[str, str]]:
    """把外部 messages 规范化为 system/user/assistant/tool 消息列表。

    空消息、未知 role 和非字符串 content 会被忽略。最终至少需要一条 user
    和一条 assistant，否则记录不能构成可训练的 SFT 样本。
    """
    if not isinstance(raw, list):
        return []
    messages = []
    for item in raw:
        if not isinstance(item, dict):
            return []
        role = str(item.get("role", ""))
        content = item.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        if isinstance(content, str) and content.strip():
            content = content.strip()
            messages.append({"role": role, "content": content})
    roles = {message["role"] for message in messages}
    if "user" not in roles or "assistant" not in roles:
        return []
    return messages


def _content_key(seed: int, messages: list[dict[str, str]]) -> bytes:
    """生成与 sample_id 无关的稳定内容指纹，用于跨输入精确去重。"""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(seed).encode("ascii"))
    for message in messages:
        digest.update(message["role"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(message["content"].encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.digest()


def _selection_key(seed: int, identity: str) -> int:
    """生成确定性选择键，避免输入文件顺序影响 bottom-k 配额结果。"""
    digest = hashlib.blake2b(
        f"{seed}:{identity}".encode("utf-8"),
        digest_size=8,
    )
    return int.from_bytes(digest.digest(), "big")


def _token_summary(values: list[int]) -> dict[str, int]:
    if not values:
        return {"p50": 0, "p90": 0, "p95": 0, "max": 0}
    ordered = sorted(values)

    def percentile(value: float) -> int:
        return ordered[int((len(ordered) - 1) * value)]

    return {
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _is_multi_turn(messages: list[dict[str, str]]) -> bool:
    """Return whether a conversation contains at least two user/assistant turns."""

    return (
        sum(message["role"] == "user" for message in messages) > 1
        and sum(message["role"] == "assistant" for message in messages) > 1
    )


def _render_conversation(
    messages: list[dict[str, str]],
    *,
    assistant_only_loss: bool = True,
    train_on_eos: bool = True,
) -> tuple[str, list[tuple[int, int]]]:
    """Render the exact training sequence and assistant-supervised spans."""
    parts = []
    spans = []
    cursor = 0
    for message in messages:
        prefix = f"<|taichu_start|>{message['role']}\n"
        parts.append(prefix)
        cursor += len(prefix)
        content = message["content"]
        content_start = cursor
        parts.append(content)
        cursor += len(content)
        train_content = not assistant_only_loss or message["role"] == "assistant"
        if train_content:
            spans.append((content_start, cursor))
        suffix = "<|taichu_end|>\n"
        suffix_start = cursor
        parts.append(suffix)
        cursor += len(suffix)
        if train_content and train_on_eos:
            spans.append(
                (suffix_start, suffix_start + len("<|taichu_end|>"))
            )
    return "".join(parts), spans


def _mask_from_offsets(
    offsets: list[tuple[int, int]],
    spans: list[tuple[int, int]],
) -> list[int]:
    """Map character-level assistant spans to token-level loss positions."""
    mask = [0] * len(offsets)
    span_index = 0
    for token_index, (start, _) in enumerate(offsets):
        while span_index < len(spans) and start >= spans[span_index][1]:
            span_index += 1
        if (
            span_index < len(spans)
            and spans[span_index][0] <= start < spans[span_index][1]
        ):
            mask[token_index] = 1
    return mask


class _TokenCache:
    """Append-only compact cache for selected SFT token IDs and loss masks."""

    _HEADER = struct.Struct("<I")

    def __init__(self, path: Path, *, write: bool = True) -> None:
        self.path = path
        self.stream = path.open("w+b" if write else "rb")
        self.offset = 0

    def append(self, input_ids: list[int], loss_mask: list[int]) -> int:
        offset = self.offset
        ids = array("i", input_ids)
        masks = bytes(loss_mask)
        payload = self._HEADER.pack(len(input_ids)) + ids.tobytes() + masks
        self.stream.write(payload)
        self.offset += len(payload)
        return offset

    def read(self, offset: int) -> tuple[list[int], list[int]]:
        self.stream.seek(offset)
        header = self.stream.read(self._HEADER.size)
        if len(header) != self._HEADER.size:
            raise ValueError("truncated SFT token cache header")
        (count,) = self._HEADER.unpack(header)
        ids_buffer = array("i")
        ids_buffer.frombytes(self.stream.read(count * ids_buffer.itemsize))
        masks = list(self.stream.read(count))
        if len(ids_buffer) != count or len(masks) != count:
            raise ValueError("truncated SFT token cache record")
        return ids_buffer.tolist(), masks

    def close(self) -> None:
        self.stream.close()


def _batched(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """把流式记录切成固定大小 batch，供 tokenizer 批量编码。"""
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class _Bucket:
    """按 category 保存 selection-key 最小的 target 条记录。

    heap 中保存负 key，因此堆顶是当前最大 key。新记录只有在 key 更小时才能
    替换堆顶，从而得到确定性的 bottom-k 样本，同时避免保存全部候选。
    """

    def __init__(self, target: int) -> None:
        self.target = target
        self.heap: list[tuple[int, int, Candidate]] = []
        self.serial = 0

    def would_accept(self, key: int) -> bool:
        """在 tokenize 前快速判断该候选是否可能进入配额。"""
        return len(self.heap) < self.target or key < -self.heap[0][0]

    def offer(self, candidate: Candidate) -> None:
        """插入候选并在超过配额时淘汰当前最大 selection key。"""
        if len(self.heap) >= self.target:
            heapq.heappop(self.heap)
        self.serial += 1
        heapq.heappush(
            self.heap,
            (-candidate.selection_key, self.serial, candidate),
        )

    def selected(self) -> list[Candidate]:
        """返回当前 bucket 中保留的最终候选集合。"""
        return [item[2] for item in self.heap]


class SFTDataPreparation:
    """构建 canonical SFT JSONL 和 unified Parquet Dataset。"""

    def __init__(self, config: SFTBuildConfig) -> None:
        """加载 tokenizer 和通用记录读取器。"""
        self.config = config
        if not config.tokenizer_path.is_file():
            raise FileNotFoundError(f"tokenizer does not exist: {config.tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(config.tokenizer_path))
        self.reader = RecordSourceReader()
        self.content_policy = SFTContentPolicy(
            drop_system=config.drop_system,
        )

    def run(self, *, restart: bool = False) -> dict[str, Any]:
        """执行完整构建并原子发布输出目录。

        staging 同时承载 source 和 unified。只有两步都成功后才替换正式
        output_dir，因此失败不会暴露半成品数据集。
        """
        output = self.config.output_dir
        staging = output.with_name(f".{output.name}.staging")
        if output.exists():
            raise FileExistsError(f"SFT output already exists: {output}")
        if staging.exists():
            if not restart:
                raise FileExistsError(f"SFT staging already exists: {staging}")
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        token_cache_path = staging / ".token-cache.bin"
        try:
            report, train, validation = self._build_source(
                staging,
                token_cache_path,
            )
            self._build_dataset(
                staging,
                train,
                validation,
                token_cache_path,
                report,
            )
            token_cache_path.unlink(missing_ok=True)
            (staging / "build_manifest.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            atomic_replace_directory(staging, output)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return report

    def _records(
        self,
    ) -> Iterator[tuple[str, str, str, str, list[dict[str, str]]]]:
        """流式产生 (category, sample_id, source_id, group, messages)。"""
        index = 0
        for source in self.config.raw_sources:
            adapter = create_raw_adapter(source)
            for record in adapter.iter_records():
                messages = _normalize_messages(record.get("messages"))
                if not messages:
                    index += 1
                    continue
                category = str(record.get("category") or "general")
                source_id = str(record.get("source_id") or source.name)
                sample_id = str(
                    record.get("sample_id")
                    or f"{self.config.dataset_id}-{source.name}-{index:08d}"
                )
                source_group = str(
                    record.get("source_group") or source.group
                )
                index += 1
                yield category, sample_id, source_id, source_group, messages
        canonical_files = (
            self.reader.resolve(self.config.inputs)
            if self.config.inputs
            else ()
        )
        for path in canonical_files:
            for record in self.reader.read(path):
                messages = _normalize_messages(record.get("messages"))
                if not messages:
                    index += 1
                    continue
                category = str(record.get("category") or "general")
                source_id = str(record.get("source_id") or path.stem)
                source_group = str(
                    record.get("source_group") or "canonical"
                )
                sample_id = str(
                    record.get("sample_id")
                    or f"{self.config.dataset_id}-{path.stem}-{index:08d}"
                )
                index += 1
                yield category, sample_id, source_id, source_group, messages

    def _draft_batch(
        self,
        values: list[tuple[str, str, str, str, list[dict[str, str]]]],
        counters: PreparationCounters,
    ) -> list[CandidateDraft]:
        """Apply cheap role/character filters before expensive tokenization."""
        drafts = []
        for category, sample_id, source_id, source_group, messages in values:
            has_source_reasoning = self.content_policy.has_reasoning(messages)
            keep_reasoning = (
                has_source_reasoning
                and _selection_key(
                    self.config.seed + 3,
                    f"{source_id}:{sample_id}",
                )
                / float(1 << 64)
                < self.config.reasoning_sample_ratio
            )
            try:
                content = self.content_policy.apply(
                    messages,
                    strip_reasoning=(
                        has_source_reasoning and not keep_reasoning
                    ),
                )
            except SFTContentRejected as exc:
                counters.content_rejected[exc.reason] += 1
                continue
            counters.dropped_system_messages += (
                content.dropped_system_messages
            )
            messages = content.messages
            role_chars = Counter()
            for message in messages:
                role_chars[message["role"]] += len(message["content"])
            char_count = sum(role_chars.values())
            if not self.config.min_chars <= char_count <= self.config.max_chars:
                counters.char_rejected += 1
                continue
            if role_chars["user"] < self.config.min_user_chars:
                counters.char_rejected += 1
                continue
            content_key = _content_key(self.config.seed, messages)
            selection_key = _selection_key(
                self.config.seed,
                f"{source_id}:{sample_id}",
            )
            drafts.append(
                CandidateDraft(
                    category=category,
                    sample_id=sample_id,
                    source_id=source_id,
                    source_group=source_group,
                    messages=messages,
                    content_key=content_key,
                    selection_key=selection_key,
                    has_reasoning=content.has_reasoning,
                    assistant_texts=content.assistant_texts,
                    reasoning_mode=(
                        "think" if content.has_reasoning else "no_think"
                    ),
                )
            )
        return drafts

    def _tokenize_drafts(
        self,
        drafts: list[CandidateDraft],
        token_cache: _TokenCache,
        counters: PreparationCounters,
    ) -> tuple[list[Candidate], int]:
        if not drafts:
            return [], 0
        rendered = [
            _render_conversation(
                draft.messages,
                assistant_only_loss=self.config.assistant_only_loss,
                train_on_eos=self.config.train_on_eos,
            )
            for draft in drafts
        ]
        encoded = self.tokenizer.encode_batch(
            [text for text, _ in rendered],
            add_special_tokens=False,
        )
        assistant_texts = [
            text
            for draft in drafts
            for text in draft.assistant_texts
        ]
        assistant_encodings = self.tokenizer.encode_batch(
            assistant_texts,
            add_special_tokens=False,
        )
        accepted = []
        rejected = 0
        assistant_offset = 0
        for draft, (_, spans), encoding in zip(
            drafts,
            rendered,
            encoded,
            strict=True,
        ):
            counts = [
                len(assistant_encodings[index].ids)
                for index in range(
                    assistant_offset,
                    assistant_offset + len(draft.assistant_texts),
                )
            ]
            assistant_offset += len(draft.assistant_texts)
            if sum(counts) < self.config.min_assistant_tokens:
                counters.content_rejected["assistant_too_short"] += 1
                continue
            if any(
                count > self.config.max_assistant_tokens
                for count in counts
            ):
                counters.content_rejected["assistant_too_long"] += 1
                continue
            if len(encoding.ids) > self.config.max_seq_length:
                rejected += 1
                continue
            loss_mask = _mask_from_offsets(encoding.offsets, spans)
            if not any(loss_mask):
                rejected += 1
                continue
            token_offset = token_cache.append(encoding.ids, loss_mask)
            accepted.append(
                Candidate(
                    category=draft.category,
                    sample_id=draft.sample_id,
                    source_id=draft.source_id,
                    source_group=draft.source_group,
                    messages=draft.messages,
                    token_count=len(encoding.ids),
                    content_key=draft.content_key,
                    selection_key=draft.selection_key,
                    token_offset=token_offset,
                    supervised_tokens=sum(loss_mask),
                    assistant_tokens=sum(counts),
                    has_reasoning=draft.has_reasoning,
                    reasoning_mode=draft.reasoning_mode,
                )
            )
        return accepted, rejected

    def _build_source(
        self,
        staging: Path,
        token_cache_path: Path,
    ) -> tuple[dict[str, Any], list[Candidate], list[Candidate]]:
        """完成过滤、去重、配额、train/validation 划分和 source 写入。"""
        quotas = self.config.quotas
        # quotas=None 表示不限制类别规模；配置 quotas 时，每个 category 独立
        # 维护 bottom-k，避免某一类数据因文件扫描顺序抢占其他类别配额。
        buckets = (
            {name: _Bucket(count) for name, count in quotas.items()}
            if quotas is not None
            else None
        )
        source_quotas = self.config.source_quotas
        source_buckets = (
            {name: _Bucket(count) for name, count in source_quotas.items()}
            if source_quotas is not None
            else None
        )
        offered_keys: set[bytes] = set()
        unlimited_selected: dict[bytes, Candidate] = {}
        input_records = 0
        counters = PreparationCounters()
        length_rejected = 0
        quota_rejected = 0
        duplicate_records = 0
        token_cache = _TokenCache(token_cache_path)
        batches = tqdm(
            _batched(self._records(), self.config.preprocess_batch_size),
            desc="[SFT] scan + tokenize",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in batches:
            input_records += len(batch)
            drafts = self._draft_batch(batch, counters)
            eligible = []
            for draft in drafts:
                if draft.content_key in offered_keys:
                    duplicate_records += 1
                    continue
                category_bucket = (
                    None if buckets is None else buckets.get(draft.category)
                )
                if buckets is not None and category_bucket is None:
                    quota_rejected += 1
                    continue
                if (
                    category_bucket is not None
                    and not category_bucket.would_accept(draft.selection_key)
                ):
                    quota_rejected += 1
                    continue
                group_bucket = (
                    None
                    if source_buckets is None
                    else source_buckets.get(draft.source_group)
                )
                if source_buckets is not None and group_bucket is None:
                    quota_rejected += 1
                    continue
                if (
                    group_bucket is not None
                    and not group_bucket.would_accept(draft.selection_key)
                ):
                    quota_rejected += 1
                    continue
                eligible.append(draft)
            candidates, batch_length_rejected = self._tokenize_drafts(
                eligible,
                token_cache,
                counters,
            )
            length_rejected += batch_length_rejected
            batches.set_postfix(
                records=input_records,
                content_rejected=sum(counters.content_rejected.values()),
                length_rejected=length_rejected,
                duplicates=duplicate_records,
                quota_rejected=quota_rejected,
            )
            for candidate in candidates:
                if candidate.content_key in offered_keys:
                    duplicate_records += 1
                    continue
                category_bucket = (
                    None
                    if buckets is None
                    else buckets.get(candidate.category)
                )
                group_bucket = (
                    None
                    if source_buckets is None
                    else source_buckets.get(candidate.source_group)
                )
                if (
                    category_bucket is not None
                    and not category_bucket.would_accept(candidate.selection_key)
                ):
                    quota_rejected += 1
                    continue
                if (
                    group_bucket is not None
                    and not group_bucket.would_accept(candidate.selection_key)
                ):
                    quota_rejected += 1
                    continue
                offered_keys.add(candidate.content_key)
                if category_bucket is not None:
                    category_bucket.offer(candidate)
                if group_bucket is not None:
                    group_bucket.offer(candidate)
                if category_bucket is None and group_bucket is None:
                    unlimited_selected[candidate.content_key] = candidate
        token_cache.close()
        candidate_pool = dict(unlimited_selected)
        for bucket in (*(buckets or {}).values(), *(source_buckets or {}).values()):
            for candidate in bucket.selected():
                candidate_pool[candidate.content_key] = candidate
        records = self._select_with_constraints(
            list(candidate_pool.values()),
            category_quotas=quotas,
            source_quotas=source_quotas,
            unlimited=unlimited_selected,
        )
        train, validation = self._stratified_split(records)
        self._write_split(train, staging / "source" / "train" / "sft.jsonl")
        self._write_split(
            validation,
            staging / "source" / "validation" / "sft.jsonl",
        )
        category_counts = Counter(candidate.category for candidate in records)
        source_counts = Counter(candidate.source_id for candidate in records)
        source_group_counts = Counter(
            candidate.source_group for candidate in records
        )
        source_group_tokens = Counter()
        category_tokens = Counter()
        for candidate in records:
            source_group_tokens[candidate.source_group] += (
                candidate.supervised_tokens
            )
            category_tokens[candidate.category] += candidate.supervised_tokens
        total_supervised_tokens = sum(category_tokens.values())
        reasoning_samples = sum(
            int(candidate.has_reasoning) for candidate in records
        )
        reasoning_mode_counts = Counter(
            candidate.reasoning_mode for candidate in records
        )
        multi_turn_count = sum(
            int(_is_multi_turn(candidate.messages))
            for candidate in records
        )
        underfilled = {
            name: count - category_counts.get(name, 0)
            for name, count in (quotas or {}).items()
            if category_counts.get(name, 0) < count
        }
        underfilled_sources = {
            name: count - source_group_counts.get(name, 0)
            for name, count in (source_quotas or {}).items()
            if source_group_counts.get(name, 0) < count
        }
        report = {
            "dataset_id": self.config.dataset_id,
            "version": self.config.version,
            "profile": self.config.profile,
            "seed": self.config.seed,
            "max_seq_length": self.config.max_seq_length,
            "input_records": input_records,
            "char_rejected_records": counters.char_rejected,
            "content_rejected_records": sum(
                counters.content_rejected.values()
            ),
            "content_rejected_by_reason": dict(counters.content_rejected),
            "dropped_system_messages": counters.dropped_system_messages,
            "length_rejected_records": length_rejected,
            "quota_rejected_records": quota_rejected,
            "duplicate_records": duplicate_records,
            "selected_samples": len(records),
            "train_samples": len(train),
            "validation_samples": len(validation),
            "total_input_tokens": sum(
                candidate.token_count for candidate in records
            ),
            "total_supervised_tokens": total_supervised_tokens,
            "assistant_token_distribution": _token_summary(
                [candidate.assistant_tokens for candidate in records]
            ),
            "reasoning_sample_count": reasoning_samples,
            "reasoning_sample_ratio": (
                reasoning_samples / len(records) if records else 0.0
            ),
            "reasoning_mode_counts": dict(reasoning_mode_counts),
            "multi_turn_samples": multi_turn_count,
            "multi_turn_ratio": (
                multi_turn_count / len(records) if records else 0.0
            ),
            "train_multi_turn_samples": sum(
                int(_is_multi_turn(candidate.messages))
                for candidate in train
            ),
            "validation_multi_turn_samples": sum(
                int(_is_multi_turn(candidate.messages))
                for candidate in validation
            ),
            "category_counts": dict(category_counts),
            "train_category_counts": dict(
                Counter(candidate.category for candidate in train)
            ),
            "validation_category_counts": dict(
                Counter(candidate.category for candidate in validation)
            ),
            "source_counts": dict(source_counts.most_common()),
            "source_group_counts": dict(source_group_counts),
            "source_token_shares": {
                name: tokens / total_supervised_tokens
                for name, tokens in source_group_tokens.items()
            }
            if total_supervised_tokens
            else {},
            "category_token_shares": {
                name: tokens / total_supervised_tokens
                for name, tokens in category_tokens.items()
            }
            if total_supervised_tokens
            else {},
            "underfilled_categories": underfilled,
            "underfilled_source_groups": underfilled_sources,
        }
        return report, train, validation

    def _stratified_split(
        self,
        records: list[Candidate],
    ) -> tuple[list[Candidate], list[Candidate]]:
        categories: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in records:
            categories[candidate.category].append(candidate)
        validation_keys: set[bytes] = set()
        ratio = self.config.validation_ratio
        for category, candidates in categories.items():
            if ratio <= 0 or len(candidates) < 2:
                continue
            candidates = sorted(
                candidates,
                key=lambda item: _selection_key(
                    self.config.seed + 2,
                    f"{category}:{item.source_id}:{item.sample_id}",
                ),
            )
            count = max(1, round(len(candidates) * ratio))
            count = min(count, len(candidates) - 1)
            validation_keys.update(
                candidate.content_key for candidate in candidates[:count]
            )
        train = [
            candidate
            for candidate in records
            if candidate.content_key not in validation_keys
        ]
        validation = [
            candidate
            for candidate in records
            if candidate.content_key in validation_keys
        ]
        return train, validation

    @staticmethod
    def _select_with_constraints(
        candidates: list[Candidate],
        *,
        category_quotas: dict[str, int] | None,
        source_quotas: dict[str, int] | None,
        unlimited: dict[bytes, Candidate],
    ) -> list[Candidate]:
        """Apply category and source-group caps to the union of bucket winners."""
        if category_quotas is None and source_quotas is None:
            return sorted(
                unlimited.values(),
                key=lambda item: item.selection_key,
            )
        category_remaining = dict(category_quotas or {})
        source_remaining = dict(source_quotas or {})
        selected = []
        for candidate in sorted(
            candidates,
            key=lambda item: item.selection_key,
        ):
            if (
                category_quotas is not None
                and category_remaining.get(candidate.category, 0) <= 0
            ):
                continue
            if (
                source_quotas is not None
                and source_remaining.get(candidate.source_group, 0) <= 0
            ):
                continue
            if category_quotas is not None:
                category_remaining[candidate.category] -= 1
            if source_quotas is not None:
                source_remaining[candidate.source_group] -= 1
            selected.append(candidate)
        return selected

    def _build_dataset(
        self,
        staging: Path,
        train: list[Candidate],
        validation: list[Candidate],
        token_cache_path: Path,
        report: dict[str, Any],
    ) -> None:
        """Write tokens produced during selection directly into Parquet."""
        token_cache = _TokenCache(token_cache_path, write=False)
        split_manifests = {}
        try:
            for split, records in (
                ("train", train),
                ("validation", validation),
            ):
                if not records:
                    continue
                writer = ParquetSampleWriter(
                    staging / split,
                    dataset_id=f"{self.config.dataset_id}:{split}",
                    sample_type="sft",
                    version=self.config.version,
                    shard_size=self.config.shard_size,
                    metadata={
                        "task_type": "sft",
                        "split": split,
                        "storage_format": "parquet",
                        "profile": self.config.profile,
                        "tokenizer_path": str(self.config.tokenizer_path),
                    },
                )
                progress = tqdm(
                    records,
                    total=len(records),
                    desc=f"[SFT] write {split}",
                    unit="sample",
                    dynamic_ncols=True,
                )
                for candidate in progress:
                    input_ids, loss_mask = token_cache.read(
                        candidate.token_offset
                    )
                    labels = [
                        token if mask else -100
                        for token, mask in zip(
                            input_ids,
                            loss_mask,
                            strict=True,
                        )
                    ]
                    writer.write(
                        SFTSequenceSample(
                            sample_id=candidate.sample_id,
                            dataset_id=self.config.dataset_id,
                            source_id=candidate.source_id,
                            input_ids=input_ids,
                            labels=labels,
                            loss_mask=loss_mask,
                            metadata={
                                "category": candidate.category,
                                "source_group": candidate.source_group,
                                "reasoning_mode": candidate.reasoning_mode,
                            },
                        ),
                        trusted_arrays=True,
                    )
                    progress.set_postfix(
                        tokens=candidate.token_count,
                    )
                manifest = writer.commit()
                split_manifests[split] = {
                    "manifest": (Path(split) / "manifest.json").as_posix(),
                    "sample_count": manifest.sample_count,
                    "content_hash": manifest.content_hash,
                    "stats": {
                        "source_records": len(records),
                        "cleaned_records": len(records),
                        "duplicate_records": 0,
                        "rejected_records": 0,
                        "written_records": manifest.sample_count,
                        "supervised_tokens": sum(
                            candidate.supervised_tokens for candidate in records
                        ),
                        "source_groups": dict(
                            Counter(
                                candidate.source_group
                                for candidate in records
                            )
                        ),
                    },
                }
        finally:
            token_cache.close()
        dataset_manifest = {
            "schema_version": 1,
            "dataset_id": self.config.dataset_id,
            "version": self.config.version,
            "task_type": "sft",
            "sample_type": "sft",
            "splits": split_manifests,
            "statistics": report,
        }
        (staging / "dataset_manifest.json").write_text(
            json.dumps(dataset_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_split(records: list[Candidate], output: Path) -> None:
        """按稳定排序结果写出 canonical messages JSONL。"""
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            progress = tqdm(
                records,
                total=len(records),
                desc=f"[SFT] source {output.parent.name}",
                unit="sample",
                dynamic_ncols=True,
            )
            for record in progress:
                payload = {
                    "sample_id": record.sample_id,
                    "source_id": record.source_id,
                    "source_group": record.source_group,
                    "category": record.category,
                    "reasoning_mode": record.reasoning_mode,
                    "messages": record.messages,
                }
                stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["SFTDataPreparation"]
