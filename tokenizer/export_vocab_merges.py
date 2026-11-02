#!/usr/bin/env python3
"""Export GPT-2 style vocab.json and merges.txt from tokenizer.json.

The exported files describe the BPE model only. ``tokenizer.json`` remains the
canonical artifact because it also stores the normalizer, pre-tokenizer,
decoder, and added-token configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


MERGES_HEADER = "#version: 0.2"


def load_bpe_model(tokenizer_json: Path) -> Tuple[Dict[str, int], List[str]]:
    """Read and validate a Byte-Level BPE model from tokenizer.json."""

    with tokenizer_json.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    model = document.get("model")
    if not isinstance(model, dict):
        raise ValueError("tokenizer.json does not contain a model object")
    if model.get("type") != "BPE":
        raise ValueError(f"expected BPE model, found {model.get('type')!r}")

    raw_vocab = model.get("vocab")
    if not isinstance(raw_vocab, dict) or not raw_vocab:
        raise ValueError("tokenizer.json model.vocab must be a non-empty object")
    vocab: Dict[str, int] = {}
    for token, token_id in raw_vocab.items():
        if not isinstance(token, str):
            raise ValueError("all vocabulary tokens must be strings")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(f"token ID for {token!r} must be an integer")
        if token_id < 0:
            raise ValueError(f"token ID for {token!r} must be non-negative")
        vocab[token] = token_id

    expected_ids = set(range(len(vocab)))
    actual_ids = set(vocab.values())
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:10]
        duplicate_count = len(vocab) - len(actual_ids)
        raise ValueError(
            "vocabulary IDs must be contiguous from zero; "
            f"missing={missing}, duplicate_count={duplicate_count}"
        )

    raw_merges = model.get("merges")
    if not isinstance(raw_merges, list):
        raise ValueError("tokenizer.json model.merges must be a list")
    merges = [_normalize_merge(merge, index) for index, merge in enumerate(raw_merges)]
    return vocab, merges


def _normalize_merge(merge: Any, index: int) -> str:
    if isinstance(merge, str):
        parts = merge.split(" ")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"invalid merge string at index {index}: {merge!r}")
    elif isinstance(merge, list) and len(merge) == 2:
        parts = merge
    else:
        raise ValueError(f"invalid merge at index {index}: {merge!r}")

    normalized = []
    for part in parts:
        if not isinstance(part, str) or not part:
            raise ValueError(f"invalid merge token at index {index}: {part!r}")
        if any(character.isspace() for character in part):
            raise ValueError(
                f"merge token at index {index} contains whitespace: {part!r}"
            )
        normalized.append(part)
    return " ".join(normalized)


def _atomic_write(path: Path, lines: Iterable[str]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for line in lines:
                handle.write(line)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def export_vocab_merges(
    tokenizer_json: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> Tuple[Path, Path]:
    """Write vocab.json and merges.txt from one tokenizer.json file."""

    tokenizer_json = tokenizer_json.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not tokenizer_json.is_file():
        raise FileNotFoundError(f"tokenizer.json does not exist: {tokenizer_json}")
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"
    existing = [path for path in (vocab_path, merges_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"output files already exist: {[str(path) for path in existing]}; "
            "pass --overwrite to replace them"
        )

    vocab, merges = load_bpe_model(tokenizer_json)
    ordered_vocab = {
        token: token_id
        for token, token_id in sorted(vocab.items(), key=lambda item: item[1])
    }
    vocab_json = json.dumps(
        ordered_vocab,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    _atomic_write(vocab_path, (vocab_json, "\n"))
    _atomic_write(merges_path, (MERGES_HEADER + "\n", *[line + "\n" for line in merges]))
    return vocab_path, merges_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GPT-2 style vocab.json and merges.txt from tokenizer.json"
    )
    parser.add_argument(
        "--tokenizer_json",
        default="taichu_tokenizer/tokenizer.json",
        help="source tokenizer.json path",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="output directory; defaults to the tokenizer.json directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing vocab.json and merges.txt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer_json = Path(args.tokenizer_json)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else tokenizer_json.expanduser().resolve().parent
    )
    vocab_path, merges_path = export_vocab_merges(
        tokenizer_json=tokenizer_json,
        output_dir=output_dir,
        overwrite=args.overwrite,
    )
    vocab_size = len(json.loads(vocab_path.read_text(encoding="utf-8")))
    merge_lines = merges_path.read_text(encoding="utf-8").splitlines()
    if not merge_lines or merge_lines[0] != MERGES_HEADER:
        raise ValueError("generated merges.txt is missing the version header")
    merge_count = len(merge_lines) - 1
    print(f"vocab={vocab_path} entries={vocab_size}")
    print(f"merges={merges_path} pairs={merge_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
