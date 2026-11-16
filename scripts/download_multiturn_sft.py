"""Download the curated multi-turn SFT source subset from HF Mirror."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(
    os.getenv(
        "TAICHU_SFT_RAW_DIR",
        "./data/SFT/raw",
    )
)
ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

SMOLTALK_FILES = (
    "advice-seeking.parquet",
    "brainstorming.parquet",
    "coding.parquet",
    "creative-writing.parquet",
    "data-analysis.parquet",
    "document-qa.parquet",
    "editing.parquet",
    "everyday.parquet",
    "information-seeking.parquet",
    "math.parquet",
    "planning.parquet",
    "reasoning.parquet",
    "role-playing.parquet",
)

SOURCES = (
    (
        "opencsg/smoltalk-chinese",
        ROOT / "SmolTalk-Chinese",
        ("README.md", *SMOLTALK_FILES),
    ),
    (
        "shareAI/ShareGPT-Chinese-English-90k",
        ROOT / "ShareGPT-Chinese-English-90k",
        (
            "README.md",
            "sharegpt_jsonl/common_zh_70k.jsonl",
            "sharegpt_jsonl/common_en_70k.jsonl",
        ),
    ),
    (
        "HuggingFaceH4/ultrachat_200k",
        ROOT / "UltraChat-200k",
        (
            "README.md",
            "data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
            "data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
            "data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet",
        ),
    ),
    (
        "OpenMOSS-Team/moss-002-sft-data",
        ROOT / "MOSS-002-SFT",
        ("README.md", "zh_helpfulness.json", "zh_honesty.json"),
    ),
)


def main() -> None:
    for repo_id, directory, filenames in SOURCES:
        directory.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            print(f"[download] {repo_id}/{filename}")
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=directory,
                endpoint=ENDPOINT,
            )
            print(f"[ready] {downloaded}")


if __name__ == "__main__":
    main()
