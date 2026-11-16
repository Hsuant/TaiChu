"""Create an auditable, stratified sample from an SFT binary dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer

from dataset.storage import BinaryDatasetReader


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _stratified_source_sample(
    source: Path,
    *,
    seed: int,
    per_group: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    reservoirs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen: Counter[tuple[str, str]] = Counter()
    category_counts: Counter[str] = Counter()
    reasoning_counts: Counter[str] = Counter()
    legacy_tag_lines = 0
    total = 0
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            category = str(record.get("category") or "general")
            mode = str(record.get("reasoning_mode") or "no_think")
            key = (category, mode)
            bucket = reservoirs.setdefault(key, [])
            seen[key] += 1
            if len(bucket) < per_group:
                bucket.append(record)
            else:
                selected = rng.randrange(seen[key])
                if selected < per_group:
                    bucket[selected] = record
            category_counts[category] += 1
            reasoning_counts[mode] += 1
            if "<think>" in line or "</think>" in line:
                legacy_tag_lines += 1
            total += 1

    samples = []
    for key in sorted(reservoirs):
        for record in sorted(
            reservoirs[key],
            key=lambda item: str(item.get("sample_id", "")),
        ):
            samples.append(
                {
                    **record,
                    "_sample_category": key[0],
                    "_sample_reasoning_mode": key[1],
                }
            )
    summary = {
        "source_file": str(source),
        "source_bytes": source.stat().st_size,
        "source_records": total,
        "category_counts": dict(category_counts),
        "reasoning_mode_counts": dict(reasoning_counts),
        "legacy_tag_lines": legacy_tag_lines,
        "sample_count": len(samples),
        "sample_group_counts": {
            f"{category}/{mode}": count
            for (category, mode), count in sorted(seen.items())
        },
    }
    return samples, summary


def _binary_samples(
    root: Path,
    tokenizer_path: Path,
    *,
    seed: int,
    count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rng = random.Random(seed)
    rows = []
    with BinaryDatasetReader(root / "train") as reader:
        indices = sorted(
            rng.sample(range(len(reader)), k=min(count, len(reader)))
        )
        for index in indices:
            sample = reader[index]
            target_ids = [
                token_id
                for token_id, label in zip(
                    sample.input_ids,
                    sample.labels,
                    strict=True,
                )
                if label != -100
            ]
            rows.append(
                {
                    "index": index,
                    "sample_id": sample.sample_id,
                    "source_id": sample.source_id,
                    "metadata": dict(getattr(sample, "metadata", {})),
                    "input_tokens": len(sample.input_ids),
                    "supervised_tokens": len(target_ids),
                    "input_text": tokenizer.decode(
                        sample.input_ids,
                        skip_special_tokens=False,
                    ),
                    "supervised_text": tokenizer.decode(
                        target_ids,
                        skip_special_tokens=True,
                    ),
                }
            )
    return rows, {
        "split": "train",
        "sample_count": len(rows),
        "dataset_samples": len(reader),
    }


def _materialize_sample_preview(
    samples: list[dict[str, Any]],
    binary_samples: list[dict[str, Any]],
) -> str:
    lines = [
        "# SFT Dataset Sample Audit",
        "",
        "## Source Samples",
        "",
    ]
    for index, record in enumerate(samples, start=1):
        lines.extend(
            [
                f"### {index}. {record.get('sample_id', 'unknown')}",
                "",
                f"- category: `{record.get('category')}`",
                f"- reasoning_mode: `{record.get('reasoning_mode')}`",
                f"- source_group: `{record.get('source_group')}`",
                f"- source_id: `{record.get('source_id')}`",
                "",
            ]
        )
        for message in record.get("messages", []):
            lines.extend(
                [
                    f"**{message.get('role', 'unknown')}**",
                    "",
                    str(message.get("content", ""))[:2000],
                    "",
                ]
            )
    lines.extend(["## Binary Samples", ""])
    for row in binary_samples:
        lines.extend(
            [
                f"### index {row['index']}",
                "",
                f"- sample_id: `{row['sample_id']}`",
                f"- input_tokens: `{row['input_tokens']}`",
                f"- supervised_tokens: `{row['supervised_tokens']}`",
                f"- metadata: `{json.dumps(row['metadata'], ensure_ascii=False)}`",
                "",
                "**Decoded input**",
                "",
                "```text",
                row["input_text"][:4000],
                "```",
                "",
                "**Decoded supervised target**",
                "",
                "```text",
                row["supervised_text"][:4000],
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a prepared SFT dataset")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-source-group", type=int, default=1)
    parser.add_argument("--binary-samples", type=int, default=6)
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"sample output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        build_manifest = _read_json(root / "build_manifest.json")
        dataset_manifest = _read_json(root / "dataset_manifest.json")
        source_samples, source_summary = _stratified_source_sample(
            root / "source" / "train" / "sft.jsonl",
            seed=args.seed,
            per_group=args.per_source_group,
        )
        binary_samples, binary_summary = _binary_samples(
            root,
            Path(args.tokenizer).expanduser().resolve(),
            seed=args.seed + 1,
            count=args.binary_samples,
        )
        source_count = _write_jsonl(
            staging / "source_samples.jsonl",
            source_samples,
        )
        binary_count = _write_jsonl(
            staging / "binary_samples.jsonl",
            binary_samples,
        )
        summary = {
            "dataset_root": str(root),
            "seed": args.seed,
            "source_sampling": source_summary,
            "binary_sampling": binary_summary,
            "written": {
                "source_samples": source_count,
                "binary_samples": binary_count,
            },
            "build_manifest": build_manifest,
            "dataset_manifest": dataset_manifest,
        }
        _write_json(staging / "summary.json", summary)
        (staging / "report.md").write_text(
            _materialize_sample_preview(source_samples, binary_samples),
            encoding="utf-8",
        )
        os.replace(staging, output)
        print(json.dumps(summary["written"], ensure_ascii=False))
        print(f"output={output}")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
