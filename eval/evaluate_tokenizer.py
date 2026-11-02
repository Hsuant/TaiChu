#!/usr/bin/env python3
"""Build a bounded tokenizer evaluation set and report detailed quality metrics.

The source corpus is expected to contain ``ch`` and ``en`` directories with
JSONL files whose records have a string ``text`` field. Sampling uses random
byte offsets inside deterministic file segments, so only a small, stratified
portion of very large files is read.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import Levenshtein
except ImportError:  # pragma: no cover - requirements include this package
    Levenshtein = None


DEFAULT_SOURCE_ROOT = Path(r"E:\Academic\Datasets\TaiChu\Pretrain")
DEFAULT_EVAL_FILE = Path("data/eval/eval_tokenizer.jsonl")
DEFAULT_TOKENIZER = Path("taichu_tokenizer")
DEFAULT_REPORT = Path("data/eval/tokenizer_quality_report.json")

LANGUAGE_DIRS = ("ch", "en")
JSONL_SUFFIXES = {".jsonl"}

CJK_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
LATIN_RE = re.compile(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]")
LATIN_WORD_RE = re.compile(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]+(?:['-][A-Za-z]+)*")
NUMBER_RE = re.compile(r"\d+")

DIAGNOSTIC_CASES = (
    ("zh", "人工智能正在改变软件开发、科学研究与日常生活。"),
    ("en", "Tokenization quality affects training efficiency and generation quality."),
    ("mixed", "TaiChu tokenizer 能处理中英混合 mixed text 与标点！"),
    (
        "code",
        "def tokenize(text: str) -> list[int]:\n"
        "    return tokenizer.encode(text, add_special_tokens=False)\n",
    ),
    ("numbers", "2026-10-27  3.1415926535  12345678901234567890"),
    ("whitespace", "  leading\n\nindented\tcolumn  \n"),
    ("emoji", "Unicode: 中文 😀🚀，English éé."),
    ("fullwidth", "ＡＢＣ１２３，。！？"),
)


@dataclass(frozen=True)
class SourceFile:
    path: Path
    language: str
    source: str
    size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a bounded tokenizer evaluation set and evaluate a tokenizer."
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER,
        help="Tokenizer directory or Hugging Face model identifier.",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DEFAULT_EVAL_FILE,
        help="Evaluation JSONL path.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root containing ch/ and en/ source directories.",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Sample source files and write the evaluation JSONL before evaluation.",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Replace a non-empty evaluation JSONL.",
    )
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=24,
        help="Number of sampled documents per source file.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=3000,
        help="Maximum retained characters per sampled document.",
    )
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--skip-speed",
        action="store_true",
        help="Skip the repeated throughput benchmark.",
    )
    parser.add_argument(
        "--speed-repeats",
        type=int,
        default=3,
        help="Encoding repetitions used for the throughput benchmark.",
    )
    return parser.parse_args()


def discover_source_files(source_root: Path) -> List[SourceFile]:
    files: List[SourceFile] = []
    for language in LANGUAGE_DIRS:
        directory = source_root / language
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing source directory: {directory}")
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in JSONL_SUFFIXES:
                source = re.sub(r"_\d{4}$", "", path.stem)
                files.append(SourceFile(path, language, source, path.stat().st_size))
    if not files:
        raise FileNotFoundError(f"No JSONL files found under {source_root}")
    return files


def parse_jsonl_record(raw: bytes) -> Optional[Dict[str, Any]]:
    try:
        record = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    text = record.get("text")
    if not isinstance(text, str):
        return None
    return record


def sample_source_file(
    source_file: SourceFile,
    samples: int,
    max_chars: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if source_file.size <= 0:
        return [], {"requested": samples, "emitted": 0, "truncated": 0, "invalid": 0}

    records: List[Dict[str, Any]] = []
    stats = {"requested": samples, "emitted": 0, "truncated": 0, "invalid": 0}
    bucket_size = source_file.size / samples

    with source_file.path.open("rb") as handle:
        for bucket in range(samples):
            bucket_start = int(bucket * bucket_size)
            bucket_end = max(bucket_start + 1, int((bucket + 1) * bucket_size))
            record = None
            offset = 0
            for _ in range(6):
                offset = rng.randrange(bucket_start, bucket_end)
                handle.seek(offset)
                if offset > 0:
                    handle.readline()
                raw = handle.readline()
                record = parse_jsonl_record(raw) if raw else None
                if record is not None:
                    break
            if record is None:
                stats["invalid"] += 1
                continue

            original_text = record["text"]
            if not original_text.strip():
                stats["invalid"] += 1
                continue
            text = original_text[:max_chars]
            truncated = len(text) < len(original_text)
            stats["truncated"] += int(truncated)
            stats["emitted"] += 1
            records.append(
                {
                    "id": f"{source_file.language}:{source_file.path.stem}:{offset}",
                    "language": source_file.language,
                    "source": source_file.source,
                    "source_file": str(source_file.path),
                    "byte_offset": offset,
                    "original_chars": len(original_text),
                    "truncated": truncated,
                    "text": text,
                }
            )
    return records, stats


def sample_corpus(
    source_root: Path,
    samples_per_file: int,
    max_chars: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    source_files = discover_source_files(source_root)
    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []
    per_file: List[Dict[str, Any]] = []
    seen_text_hashes: set[str] = set()
    duplicates = 0

    for source_file in source_files:
        sampled, stats = sample_source_file(
            source_file, samples_per_file, max_chars, rng
        )
        kept = 0
        for record in sampled:
            digest = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
            if digest in seen_text_hashes:
                duplicates += 1
                continue
            seen_text_hashes.add(digest)
            records.append(record)
            kept += 1
        stats["kept"] = kept
        stats["duplicate"] = stats["emitted"] - kept
        stats["path"] = str(source_file.path)
        stats["language"] = source_file.language
        stats["source"] = source_file.source
        per_file.append(stats)

    summary = {
        "source_root": str(source_root.resolve()),
        "seed": seed,
        "samples_per_file": samples_per_file,
        "max_chars": max_chars,
        "source_files": len(source_files),
        "documents": len(records),
        "characters": sum(len(item["text"]) for item in records),
        "truncated_documents": sum(int(item["truncated"]) for item in records),
        "duplicates_removed": duplicates,
        "per_file": per_file,
    }
    return records, summary


def write_eval_jsonl(
    path: Path,
    records: Sequence[Dict[str, Any]],
    force: bool,
) -> None:
    if path.exists() and path.stat().st_size > 0 and not force:
        raise FileExistsError(
            f"Evaluation file is non-empty: {path}. Use --force-prepare to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def load_eval_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Evaluation data is missing or empty: {path}. Run with --prepare first."
        )
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("text"), str) or not record["text"]:
                raise ValueError(f"Invalid text at {path}:{line_number}")
            record.setdefault("language", "unknown")
            record.setdefault("source", "unknown")
            records.append(record)
    if not records:
        raise ValueError(f"No evaluation records loaded from {path}")
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_to_unicode() -> Dict[int, str]:
    values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    characters = values[:]
    next_code = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            characters.append(256 + next_code)
            next_code += 1
    return dict(zip(values, (chr(code) for code in characters)))


BYTE_DECODER = {character: byte for byte, character in bytes_to_unicode().items()}


def token_bytes(token: str) -> bytes:
    try:
        return bytes(BYTE_DECODER[character] for character in token)
    except KeyError:
        return token.encode("utf-8")


def new_accumulator() -> Dict[str, Any]:
    return {
        "documents": 0,
        "characters": 0,
        "utf8_bytes": 0,
        "tokens": 0,
        "cjk_chars": 0,
        "latin_chars": 0,
        "digit_chars": 0,
        "whitespace_chars": 0,
        "punctuation_chars": 0,
        "non_ascii_chars": 0,
        "newline_chars": 0,
        "tab_chars": 0,
        "latin_words": 0,
        "lexical_units": 0,
        "exact_roundtrips": 0,
        "edit_distance": 0,
        "unknown_tokens": 0,
        "documents_with_newlines": 0,
        "newline_documents_exact": 0,
    }


def add_document(
    accumulator: Dict[str, Any],
    text: str,
    ids: Sequence[int],
    decoded: str,
    unknown_id: Optional[int],
) -> None:
    cjk = 0
    latin = 0
    digits = 0
    whitespace = 0
    punctuation = 0
    non_ascii = 0
    newlines = 0
    tabs = 0
    for character in text:
        if CJK_RE.match(character):
            cjk += 1
        if LATIN_RE.match(character):
            latin += 1
        if character.isdigit():
            digits += 1
        if character.isspace():
            whitespace += 1
        if unicodedata.category(character).startswith("P"):
            punctuation += 1
        if ord(character) > 127:
            non_ascii += 1
        if character == "\n":
            newlines += 1
        elif character == "\t":
            tabs += 1

    latin_words = len(LATIN_WORD_RE.findall(text))
    exact = text == decoded
    unknown_tokens = 0
    if unknown_id is not None:
        unknown_tokens = sum(int(token_id == unknown_id) for token_id in ids)

    distance = 0
    if not exact and Levenshtein is not None:
        distance = Levenshtein.distance(text, decoded)

    accumulator["documents"] += 1
    accumulator["characters"] += len(text)
    accumulator["utf8_bytes"] += len(text.encode("utf-8"))
    accumulator["tokens"] += len(ids)
    accumulator["cjk_chars"] += cjk
    accumulator["latin_chars"] += latin
    accumulator["digit_chars"] += digits
    accumulator["whitespace_chars"] += whitespace
    accumulator["punctuation_chars"] += punctuation
    accumulator["non_ascii_chars"] += non_ascii
    accumulator["newline_chars"] += newlines
    accumulator["tab_chars"] += tabs
    accumulator["latin_words"] += latin_words
    accumulator["lexical_units"] += cjk + latin_words + len(NUMBER_RE.findall(text))
    accumulator["exact_roundtrips"] += int(exact)
    accumulator["edit_distance"] += distance
    accumulator["unknown_tokens"] += unknown_tokens
    if newlines:
        accumulator["documents_with_newlines"] += 1
        accumulator["newline_documents_exact"] += int(exact)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def finalize_accumulator(accumulator: Dict[str, Any]) -> Dict[str, Any]:
    tokens = accumulator["tokens"]
    characters = accumulator["characters"]
    utf8_bytes = accumulator["utf8_bytes"]
    result = dict(accumulator)
    result.update(
        {
            "chars_per_token": safe_div(characters, tokens),
            "bytes_per_token": safe_div(utf8_bytes, tokens),
            "tokens_per_char": safe_div(tokens, characters),
            "tokens_per_utf8_byte": safe_div(tokens, utf8_bytes),
            "tokens_per_latin_word": safe_div(tokens, accumulator["latin_words"]),
            "tokens_per_lexical_unit": safe_div(tokens, accumulator["lexical_units"]),
            "cjk_chars_per_token": safe_div(accumulator["cjk_chars"], tokens),
            "latin_chars_per_token": safe_div(accumulator["latin_chars"], tokens),
            "digit_chars_per_token": safe_div(accumulator["digit_chars"], tokens),
            "whitespace_char_share": safe_div(accumulator["whitespace_chars"], characters),
            "non_ascii_char_share": safe_div(accumulator["non_ascii_chars"], characters),
            "unknown_token_rate": safe_div(accumulator["unknown_tokens"], tokens),
            "exact_roundtrip_rate": safe_div(
                accumulator["exact_roundtrips"], accumulator["documents"]
            ),
            "cer": safe_div(accumulator["edit_distance"], characters),
            "newline_document_exact_rate": safe_div(
                accumulator["newline_documents_exact"],
                accumulator["documents_with_newlines"],
            ),
        }
    )
    return result


def weighted_percentile(
    frequencies: Dict[int, int], values: Dict[int, int], percentile: float
) -> float:
    pairs = sorted(
        ((values[token_id], count) for token_id, count in frequencies.items()),
        key=lambda item: item[0],
    )
    total = sum(count for _, count in pairs)
    if total == 0:
        return 0.0
    threshold = total * percentile / 100.0
    cumulative = 0
    for value, count in pairs:
        cumulative += count
        if cumulative >= threshold:
            return float(value)
    return float(pairs[-1][0])


def token_statistics(
    tokenizer: Any,
    frequencies: Dict[int, int],
    special_ids: set[int],
) -> Dict[str, Any]:
    length_by_id: Dict[int, int] = {}
    bytes_covered: set[int] = set()
    vocabulary_lengths: List[int] = []
    for token_id in range(len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if not isinstance(token, str):
            continue
        raw = token_bytes(token)
        length_by_id[token_id] = len(raw)
        vocabulary_lengths.append(len(raw))
        if len(raw) == 1:
            bytes_covered.add(raw[0])

    observed_ids = set(frequencies)
    top_tokens = []
    for token_id, count in collections.Counter(frequencies).most_common(20):
        token = tokenizer.convert_ids_to_tokens(token_id)
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        top_tokens.append(
            {
                "id": token_id,
                "count": count,
                "token": token,
                "decoded": decoded[:80],
                "bytes": len(token_bytes(token)) if isinstance(token, str) else None,
            }
        )

    vocabulary_lengths.sort()

    def vocab_percentile(percentile: int) -> int:
        if not vocabulary_lengths:
            return 0
        index = min(
            len(vocabulary_lengths) - 1,
            max(0, math.ceil(percentile / 100.0 * len(vocabulary_lengths)) - 1),
        )
        return vocabulary_lengths[index]

    return {
        "model_vocab_size": tokenizer.vocab_size,
        "total_tokens_with_added": len(tokenizer),
        "observed_vocab_size": len(observed_ids),
        "observed_non_special_vocab": len(observed_ids.difference(special_ids)),
        "vocab_utilization": safe_div(len(observed_ids), len(tokenizer)),
        "unused_vocab_rate": 1.0 - safe_div(len(observed_ids), len(tokenizer)),
        "single_byte_token_count": len(bytes_covered),
        "full_byte_coverage": bytes_covered == set(range(256)),
        "missing_bytes": sorted(set(range(256)).difference(bytes_covered))[:32],
        "weighted_token_bytes": {
            "mean": safe_div(
                sum(length_by_id.get(token_id, 0) * count for token_id, count in frequencies.items()),
                sum(frequencies.values()),
            ),
            "p50": weighted_percentile(frequencies, length_by_id, 50),
            "p90": weighted_percentile(frequencies, length_by_id, 90),
            "p99": weighted_percentile(frequencies, length_by_id, 99),
            "max": max((length_by_id.get(token_id, 0) for token_id in observed_ids), default=0),
        },
        "vocabulary_token_bytes": {
            "mean": safe_div(sum(vocabulary_lengths), len(vocabulary_lengths)),
            "p50": vocab_percentile(50),
            "p90": vocab_percentile(90),
            "p99": vocab_percentile(99),
            "max": max(vocabulary_lengths, default=0),
        },
        "top_tokens": top_tokens,
    }


def distribution_metrics(frequencies: Dict[int, int]) -> Dict[str, float]:
    if not frequencies:
        return {"token_entropy_bits": 0.0, "normalized_entropy": 0.0, "gini": 0.0}
    counts = sorted(frequencies.values())
    total = sum(counts)
    entropy = -sum(
        (count / total) * math.log2(count / total) for count in counts
    )
    maximum = math.log2(len(counts)) if len(counts) > 1 else 1.0
    weighted_sum = sum(index * count for index, count in enumerate(counts, 1))
    gini = (2.0 * weighted_sum / (len(counts) * total)) - (
        (len(counts) + 1) / len(counts)
    )
    return {
        "token_entropy_bits": entropy,
        "normalized_entropy": safe_div(entropy, maximum),
        "gini": gini,
    }


def reservoir_occurrences(
    records: Sequence[Dict[str, Any]],
    pattern: re.Pattern[str],
    limit: int,
    rng: random.Random,
) -> List[str]:
    reservoir: List[str] = []
    seen = 0
    for record in records:
        for match in pattern.finditer(record["text"]):
            value = match.group(0)
            seen += 1
            if len(reservoir) < limit:
                reservoir.append(value)
            else:
                index = rng.randrange(seen)
                if index < limit:
                    reservoir[index] = value
    return reservoir


def fragmentation_metrics(
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed + 1)
    cjk_chars = reservoir_occurrences(records, re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF]"), 10000, rng)
    latin_words = reservoir_occurrences(records, LATIN_WORD_RE, 10000, rng)

    def measure(values: Sequence[str]) -> Dict[str, Any]:
        if not values:
            return {
                "samples": 0,
                "single_piece_rate": 0.0,
                "pieces_per_unit": 0.0,
                "p95_pieces": 0.0,
                "max_pieces": 0,
            }
        encodings = tokenizer(
            list(values), add_special_tokens=False, padding=False, truncation=False
        )
        pieces = sorted(len(ids) for ids in encodings.input_ids)
        p95 = pieces[min(len(pieces) - 1, math.ceil(0.95 * len(pieces)) - 1)]
        return {
            "samples": len(values),
            "single_piece_rate": safe_div(sum(piece == 1 for piece in pieces), len(pieces)),
            "pieces_per_unit": safe_div(sum(pieces), len(pieces)),
            "p95_pieces": p95,
            "max_pieces": pieces[-1],
        }

    return {
        "cjk_character": measure(cjk_chars),
        "latin_word": measure(latin_words),
    }


def special_token_metrics(tokenizer: Any) -> Dict[str, Any]:
    names = ("bos_token", "eos_token", "unk_token", "pad_token")
    results: Dict[str, Any] = {}
    for name in names:
        token = getattr(tokenizer, name, None)
        if not isinstance(token, str):
            results[name] = {"token": None, "ok": False}
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        results[name] = {
            "token": token,
            "id": token_id,
            "encode_ok": encoded == [token_id],
            "decode_ok": decoded == token,
            "ok": encoded == [token_id] and decoded == token,
        }
    added_vocab = tokenizer.get_added_vocab()
    ids = sorted(added_vocab.values())
    results["added_vocab"] = added_vocab
    results["added_ids"] = ids
    results["added_ids_contiguous_from_zero"] = ids == list(range(len(ids)))
    results["all_standard_tokens_ok"] = all(
        results[name]["ok"] for name in names
    )
    return results


def chat_template_metrics(tokenizer: Any) -> Dict[str, Any]:
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return {"available": False, "ok": False}
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "用一句话解释 tokenizer。"},
        {
            "role": "assistant",
            "content": "它把文本映射为离散 token。",
            "reasoning_content": "先说明映射目标。",
        },
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    return {
        "available": True,
        "characters": len(rendered),
        "tokens": len(ids),
        "exact_roundtrip": rendered == decoded,
        "ok": rendered == decoded,
    }


def diagnostic_metrics(tokenizer: Any) -> List[Dict[str, Any]]:
    results = []
    for name, text in DIAGNOSTIC_CASES:
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        byte_count = len(text.encode("utf-8"))
        results.append(
            {
                "case": name,
                "characters": len(text),
                "utf8_bytes": byte_count,
                "tokens": len(ids),
                "bytes_per_token": safe_div(byte_count, len(ids)),
                "exact_roundtrip": text == decoded,
            }
        )
    return results


def speed_metrics(
    tokenizer: Any, texts: Sequence[str], repeats: int
) -> Dict[str, float]:
    tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)
    total_tokens = 0
    total_chars = 0
    total_bytes = 0
    started = time.perf_counter()
    for _ in range(repeats):
        encodings = tokenizer(
            texts, add_special_tokens=False, padding=False, truncation=False
        )
        total_tokens += sum(len(ids) for ids in encodings.input_ids)
        total_chars += sum(len(text) for text in texts)
        total_bytes += sum(len(text.encode("utf-8")) for text in texts)
    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "documents_per_second": safe_div(len(texts) * repeats, elapsed),
        "characters_per_second": safe_div(total_chars, elapsed),
        "utf8_megabytes_per_second": safe_div(total_bytes, elapsed * 1024 * 1024),
        "tokens_per_second": safe_div(total_tokens, elapsed),
    }


def evaluate(
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    skip_speed: bool,
    speed_repeats: int,
    seed: int,
) -> Dict[str, Any]:
    texts = [record["text"] for record in records]
    encodings = tokenizer(
        texts, add_special_tokens=False, padding=False, truncation=False
    )
    ids_list = encodings.input_ids
    decoded_texts = tokenizer.batch_decode(
        ids_list, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    unknown_id = getattr(tokenizer, "unk_token_id", None)
    accumulators: Dict[str, Dict[str, Any]] = {"all": new_accumulator()}
    for record in records:
        language = f"language:{record['language']}"
        source = f"source:{record['source']}"
        accumulators.setdefault(language, new_accumulator())
        accumulators.setdefault(source, new_accumulator())

    frequencies: collections.Counter[int] = collections.Counter()
    for record, ids, decoded in zip(records, ids_list, decoded_texts):
        add_document(accumulators["all"], record["text"], ids, decoded, unknown_id)
        add_document(
            accumulators[f"language:{record['language']}"],
            record["text"],
            ids,
            decoded,
            unknown_id,
        )
        add_document(
            accumulators[f"source:{record['source']}"],
            record["text"],
            ids,
            decoded,
            unknown_id,
        )
        frequencies.update(ids)

    special_ids = {
        token_id
        for token_id in tokenizer.get_added_vocab().values()
    }
    global_metrics = finalize_accumulator(accumulators["all"])
    global_metrics.update(distribution_metrics(dict(frequencies)))
    global_metrics["fragmentation"] = fragmentation_metrics(tokenizer, records, seed)

    groups = {
        "language": {
            key.split(":", 1)[1]: finalize_accumulator(value)
            for key, value in accumulators.items()
            if key.startswith("language:")
        },
        "source": {
            key.split(":", 1)[1]: finalize_accumulator(value)
            for key, value in accumulators.items()
            if key.startswith("source:")
        },
    }

    special_metrics = special_token_metrics(tokenizer)
    chat_metrics = chat_template_metrics(tokenizer)
    token_metrics = token_statistics(tokenizer, dict(frequencies), special_ids)
    diagnostics = diagnostic_metrics(tokenizer)
    performance = (
        {}
        if skip_speed
        else speed_metrics(tokenizer, texts, max(1, speed_repeats))
    )

    language_groups = groups["language"]
    zh_metrics = language_groups.get("ch", {})
    en_metrics = language_groups.get("en", {})
    correctness_ok = (
        global_metrics["exact_roundtrip_rate"] == 1.0
        and global_metrics["unknown_token_rate"] == 0.0
        and special_metrics["all_standard_tokens_ok"]
        and special_metrics["added_ids_contiguous_from_zero"]
        and token_metrics["full_byte_coverage"]
    )
    compression_status = {
        "overall_bytes_per_token": rating(
            global_metrics["bytes_per_token"], (3.0, 4.0)
        ),
        "chinese_chars_per_token": rating(
            zh_metrics.get("chars_per_token", 0.0), (1.0, 1.25)
        ),
        "english_tokens_per_word": inverse_rating(
            en_metrics.get("tokens_per_latin_word", 0.0), (1.5, 1.8)
        ),
    }
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tokenizer": {
            "path": str(getattr(tokenizer, "name_or_path", "")),
            "class": tokenizer.__class__.__name__,
            "model_max_length": tokenizer.model_max_length,
            "normalizer": str(tokenizer.backend_tokenizer.normalizer),
            "pre_tokenizer": str(tokenizer.backend_tokenizer.pre_tokenizer),
        },
        "global": global_metrics,
        "groups": groups,
        "token_statistics": token_metrics,
        "special_tokens": special_metrics,
        "chat_template": chat_metrics,
        "diagnostics": diagnostics,
        "performance": performance,
        "assessment": {
            "correctness": "pass" if correctness_ok else "review",
            "compression": compression_status,
            "notes": [
                "Samples come from the same pretraining corpus and measure fit, not "
                "generalization to unseen text.",
                "Compression ratings use broad engineering thresholds and are not "
                "substitutes for downstream model loss.",
                "Vocabulary utilization is sample-size dependent; compare tokenizers "
                "only on the same evaluation file.",
            ],
        },
    }


def rating(value: float, thresholds: Tuple[float, float]) -> str:
    if value >= thresholds[1]:
        return "strong"
    if value >= thresholds[0]:
        return "reasonable"
    return "weak"


def inverse_rating(value: float, thresholds: Tuple[float, float]) -> str:
    if value <= 0:
        return "unavailable"
    if value <= thresholds[0]:
        return "strong"
    if value <= thresholds[1]:
        return "reasonable"
    return "weak"


def summarize_report(report: Dict[str, Any], eval_path: Path) -> str:
    global_metrics = report["global"]
    zh = report["groups"]["language"].get("ch", {})
    en = report["groups"]["language"].get("en", {})
    fragmentation = global_metrics["fragmentation"]
    lines = [
        "Tokenizer quality evaluation",
        f"data: {eval_path}",
        (
            f"documents={global_metrics['documents']} "
            f"chars={global_metrics['characters']} "
            f"tokens={global_metrics['tokens']}"
        ),
        (
            f"overall: chars/token={global_metrics['chars_per_token']:.3f} "
            f"bytes/token={global_metrics['bytes_per_token']:.3f} "
            f"CER={global_metrics['cer']:.6f}"
        ),
        (
            f"ch: chars/token={zh.get('chars_per_token', 0):.3f} "
            f"tokens/lexical-unit={zh.get('tokens_per_lexical_unit', 0):.3f}"
        ),
        (
            f"en: tokens/word={en.get('tokens_per_latin_word', 0):.3f} "
            f"bytes/token={en.get('bytes_per_token', 0):.3f}"
        ),
        (
            f"roundtrip={global_metrics['exact_roundtrip_rate']:.6f} "
            f"UNK={global_metrics['unknown_token_rate']:.6f} "
            f"byte-coverage={report['token_statistics']['full_byte_coverage']}"
        ),
        (
            f"fragmentation: CJK single-piece="
            f"{fragmentation['cjk_character']['single_piece_rate']:.4f}, "
            f"Latin-word single-piece="
            f"{fragmentation['latin_word']['single_piece_rate']:.4f}"
        ),
        f"correctness={report['assessment']['correctness']}",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.samples_per_file <= 0 or args.max_chars <= 0:
        raise ValueError("--samples-per-file and --max-chars must be positive")
    if args.speed_repeats <= 0:
        raise ValueError("--speed-repeats must be positive")

    sampling_summary = None
    if args.prepare:
        records, sampling_summary = sample_corpus(
            args.source_root,
            args.samples_per_file,
            args.max_chars,
            args.seed,
        )
        if not records:
            raise RuntimeError("Sampling produced no valid documents")
        write_eval_jsonl(args.eval_file, records, args.force_prepare)

    from transformers import AutoTokenizer

    tokenizer_path = str(args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=False,
        use_fast=True,
        local_files_only=args.tokenizer.exists(),
    )
    records = load_eval_jsonl(args.eval_file)
    report = evaluate(
        tokenizer,
        records,
        skip_speed=args.skip_speed,
        speed_repeats=args.speed_repeats,
        seed=args.seed,
    )
    report["eval_data"] = {
        "path": str(args.eval_file.resolve()),
        "sha256": sha256_file(args.eval_file),
        "documents": len(records),
        "characters": sum(len(record["text"]) for record in records),
        "languages": sorted({record["language"] for record in records}),
        "sources": sorted({record["source"] for record in records}),
    }
    report["sampling"] = sampling_summary

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(summarize_report(report, args.eval_file))
    print(f"report: {args.report}")
    if sampling_summary:
        print(
            f"sampled {sampling_summary['documents']} documents from "
            f"{sampling_summary['source_files']} source files"
        )


if __name__ == "__main__":
    main()
