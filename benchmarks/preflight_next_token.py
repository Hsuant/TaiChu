#!/usr/bin/env python3
"""End-to-end next-token, tensor and optimizer preflight for TaiChu training."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.batching import SFTBatchBuilder
from dataset.config import (
    BatchingConfig,
    SamplingConfig,
    StorageConfig,
    UnifiedDataConfig,
    UnifiedPackingConfig,
)
from dataset.offline.adapters import (
    RoleTagRenderer,
    SFTPreprocessor,
    TextTokenizer,
)
from dataset.offline.config import CleaningConfig, PreparationConfig
from dataset.offline.prepare import UnifiedDataPreparation
from dataset.packing import PretrainingPacker
from dataset.runtime import UnifiedDataLoaderFactory
from dataset.storage import BinaryDatasetReader
from model.model import TaiChuModel
from utils.config_loader import load_model_config


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    marker = "PASS" if passed else "FAIL"
    print(f"[{marker}] {name}: {detail}")


def _token_text(tokenizer: Tokenizer, token_id: int) -> str:
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
    )


def _shift_preview(
    tokenizer: Tokenizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    count: int = 10,
) -> list[dict[str, Any]]:
    valid = attention_mask.bool()
    ids = input_ids[valid][:count]
    targets = labels[valid][1 : count + 1]
    rows = []
    for position in range(min(len(ids) - 1, len(targets))):
        rows.append(
            {
                "position": position,
                "input_token_id": int(ids[position]),
                "input_token": _token_text(tokenizer, int(ids[position])),
                "target_token_id": int(targets[position]),
                "target_token": _token_text(tokenizer, int(targets[position])),
            }
        )
    return rows


def run(
    *,
    model_config_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    torch.manual_seed(20261102)
    tokenizer_path = ROOT / "taichu_tokenizer" / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    report_dir = report_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(
        tempfile.mkdtemp(prefix="next_token_preflight_", dir=report_dir)
    )

    raw_records = [
        {
            "sample_id": "preflight-000",
            "text": "next token shift must predict the following token",
        },
        {
            "sample_id": "preflight-001",
            "text": "tensor padding and causal masks must stay aligned",
        },
    ]
    train_jsonl = data_root / "train.jsonl"
    _write_jsonl(train_jsonl, raw_records)
    output_dir = data_root / "binary_dataset"

    preparation = PreparationConfig(
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        splits={"train": (train_jsonl,)},
        task_type="pretraining",
        dataset_id="next-token-preflight",
        version="1",
        max_seq_length=64,
        add_bos=True,
        add_eos=True,
        shard_size=1,
        cleaning=CleaningConfig(
            min_chars=1,
            max_chars=200000,
            deduplicate=True,
        ),
    )
    prepared_manifest = UnifiedDataPreparation(preparation).run()

    checks: list[dict[str, Any]] = []
    _record_check(
        checks,
        "offline_pipeline",
        prepared_manifest["splits"]["train"]["sample_count"] == 2,
        f"prepared samples={prepared_manifest['splits']['train']['sample_count']}",
    )

    with BinaryDatasetReader(output_dir / "train", checksums=True) as reader:
        stored_samples = [reader[index] for index in range(len(reader))]
    labels_match_inputs = all(
        sample.labels is None or sample.labels == sample.input_ids
        for sample in stored_samples
    )
    _record_check(
        checks,
        "storage_does_not_shift_labels",
        labels_match_inputs,
        "stored labels are position-aligned with input_ids",
    )

    data_config = UnifiedDataConfig(
        task_type="pretraining",
        dataset_id="next-token-preflight",
        storage=StorageConfig(
            train_path=output_dir / "train",
            num_workers=0,
        ),
        sampling=SamplingConfig(
            strategy="random",
            shuffle=False,
            seed=20261102,
        ),
        packing=UnifiedPackingConfig(
            enabled=False,
            strategy="none",
            max_seq_length=64,
        ),
        batching=BatchingConfig(
            strategy="fixed_length",
            micro_batch_size=2,
            pad_to_multiple_of=1,
        ),
        pad_token="<|pad|>",
        bos_token="<|taichu_start|>",
        eos_token="<|taichu_end|>",
    )
    pad_token_id = tokenizer.token_to_id("<|pad|>")
    if pad_token_id is None:
        raise ValueError("PAD token is missing from tokenizer")
    factory = UnifiedDataLoaderFactory(data_config, pad_token_id=pad_token_id)
    loader = factory.create(
        batch_size=2,
        pin_memory=False,
        persistent_workers=False,
    )
    try:
        batch = next(iter(loader))
    finally:
        loader.close()

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    _record_check(
        checks,
        "batch_label_alignment",
        torch.equal(
            labels[attention_mask.bool()],
            input_ids[attention_mask.bool()],
        )
        and torch.all(labels[~attention_mask.bool()] == -100),
        "valid labels equal input_ids; padded labels equal -100",
    )

    expected_targets = labels[:, 1:]
    shifted_targets = torch.roll(labels, shifts=-1, dims=1)[:, :-1]
    _record_check(
        checks,
        "next_token_shift_definition",
        torch.equal(expected_targets, shifted_targets),
        "logits[:, :-1] must predict labels[:, 1:]",
    )

    model_config = load_model_config(str(model_config_path))
    model = TaiChuModel(model_config)
    model.eval()
    with torch.no_grad():
        output = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    if output.loss is None:
        raise RuntimeError("model did not return training loss")
    manual_loss = F.cross_entropy(
        output.logits[:, :-1, :].reshape(-1, model_config.vocab_size),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    loss_diff = abs(float(output.loss.detach()) - float(manual_loss.detach()))
    _record_check(
        checks,
        "model_loss_matches_manual_shifted_cross_entropy",
        torch.allclose(output.loss, manual_loss, atol=1e-6, rtol=1e-6),
        f"model_loss={float(output.loss):.8f}, manual_loss={float(manual_loss):.8f}, "
        f"abs_diff={loss_diff:.3e}",
    )
    _record_check(
        checks,
        "logits_are_finite",
        bool(torch.isfinite(output.logits).all()),
        f"logits_shape={tuple(output.logits.shape)}",
    )

    padding_diffs = []
    for row in range(input_ids.size(0)):
        valid_length = int(attention_mask[row].sum())
        single_ids = input_ids[row : row + 1, :valid_length]
        with torch.no_grad():
            single_logits = model(single_ids).logits
        batch_logits = output.logits[row : row + 1, :valid_length]
        padding_diffs.append(
            float((single_logits - batch_logits).abs().max().detach())
        )
    max_padding_diff = max(padding_diffs, default=0.0)
    _record_check(
        checks,
        "padding_does_not_change_valid_logits",
        max_padding_diff <= 1e-5,
        f"max_abs_diff={max_padding_diff:.3e}",
    )

    first_length = int(attention_mask[0].sum())
    causal_prefix = input_ids[0 : 1, : first_length - 1]
    causal_extended = input_ids[0 : 1, :first_length]
    with torch.no_grad():
        prefix_logits = model(causal_prefix).logits
        extended_logits = model(causal_extended).logits[:, : causal_prefix.size(1)]
    causal_diff = float((prefix_logits - extended_logits).abs().max())
    _record_check(
        checks,
        "future_token_causality",
        causal_diff <= 1e-5,
        f"future_token_max_abs_diff={causal_diff:.3e}",
    )

    train_model = copy.deepcopy(model).train()
    train_output = train_model(
        input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    if train_output.loss is None:
        raise RuntimeError("training model did not return loss")
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    train_output.loss.backward()
    grad_norm = float(
        torch.sqrt(
            sum(
                parameter.grad.detach().pow(2).sum()
                for parameter in train_model.parameters()
                if parameter.grad is not None
            )
        )
    )
    grads_finite = all(
        torch.isfinite(parameter.grad).all()
        for parameter in train_model.parameters()
        if parameter.grad is not None
    )
    before = train_model.token_embedding.embedding.weight.detach().clone()
    optimizer.step()
    parameter_delta = float(
        (train_model.token_embedding.embedding.weight.detach() - before)
        .abs()
        .max()
    )
    _record_check(
        checks,
        "gradients_and_optimizer_step",
        grads_finite and grad_norm > 0 and parameter_delta > 0,
        f"grad_norm={grad_norm:.6f}, embedding_max_delta={parameter_delta:.3e}",
    )

    text_tokenizer = TextTokenizer(
        tokenizer,
        pad_token="<|pad|>",
        bos_token="<|taichu_start|>",
        eos_token="<|taichu_end|>",
    )
    sft_sample = SFTPreprocessor(
        text_tokenizer,
        RoleTagRenderer(),
        max_length=64,
        assistant_only_loss=True,
        add_eos=True,
        train_on_eos=True,
    ).prepare(
        {
            "sample_id": "sft-preflight",
            "messages": [
                {"role": "user", "content": "Say hello."},
                {"role": "assistant", "content": "Hello!"},
            ],
        }
    )
    masked_labels_correct = all(
        label == -100
        for label, mask in zip(
            sft_sample.labels,
            sft_sample.loss_mask,
            strict=True,
        )
        if not mask
    )
    trainable_tokens = sum(sft_sample.loss_mask)
    sft_batch = SFTBatchBuilder(pad_token_id)([sft_sample])
    sft_loss = model(
        sft_batch["input_ids"],
        attention_mask=sft_batch["attention_mask"],
        labels=sft_batch["labels"],
    ).loss
    _record_check(
        checks,
        "sft_assistant_only_loss_mask",
        masked_labels_correct and trainable_tokens > 0 and sft_loss is not None,
        f"trainable_tokens={trainable_tokens}, "
        f"sft_loss={float(sft_loss.detach()):.6f}",
    )

    packed_samples = stored_samples
    total_length = sum(len(sample.input_ids) for sample in packed_samples)
    packed = list(
        PretrainingPacker(
            total_length,
            pad_id=pad_token_id,
            pad_to_max=False,
            drop_last_remainder=False,
        ).pack(packed_samples)
    )
    packed_ids = [token for sequence in packed for token in sequence.input_ids]
    source_ids = [
        token for sample in packed_samples for token in sample.input_ids
    ]
    boundaries = [
        index
        for index in range(1, len(packed[0].document_ids))
        if packed[0].document_ids[index] != packed[0].document_ids[index - 1]
    ]
    _record_check(
        checks,
        "stream_packing_preserves_tokens_and_boundaries",
        packed_ids == source_ids and len(boundaries) == len(packed_samples) - 1,
        f"tokens={len(source_ids)}, document_boundaries={boundaries}",
    )

    preview = _shift_preview(
        tokenizer,
        input_ids,
        labels,
        attention_mask,
    )
    status = "PASS" if all(check["passed"] for check in checks) else "FAIL"
    report = {
        "status": status,
        "model_config": asdict(model_config),
        "generated_data_dir": str(data_root),
        "raw_records": raw_records,
        "stored_samples": [
            {
                "sample_id": sample.sample_id,
                "input_ids": sample.input_ids,
                "labels": sample.labels,
                "decoded": tokenizer.decode(
                    sample.input_ids,
                    skip_special_tokens=False,
                ),
            }
            for sample in stored_samples
        ],
        "batch": {
            "input_ids": input_ids.tolist(),
            "labels": labels.tolist(),
            "attention_mask": attention_mask.tolist(),
            "sample_ids": batch["sample_ids"],
        },
        "next_token_preview": preview,
        "comparisons": {
            "model_loss": float(output.loss.detach()),
            "manual_loss": float(manual_loss.detach()),
            "loss_abs_diff": loss_diff,
            "padding_max_abs_logit_diff": max_padding_diff,
            "future_token_causal_abs_diff": causal_diff,
            "gradient_norm": grad_norm,
            "embedding_max_delta_after_step": parameter_delta,
        },
        "stream_packing": {
            "packed_input_ids": packed[0].input_ids,
            "document_ids": packed[0].document_ids,
            "document_boundaries": boundaries,
            "warning": (
                "The current model attention does not isolate documents inside "
                "a packed sequence. Document IDs are preserved for a future "
                "block-diagonal mask."
            ),
            "next_token_at_boundary": (
                None
                if not boundaries
                else {
                    "last_token_before_boundary": packed[0].input_ids[
                        boundaries[0] - 1
                    ],
                    "first_token_after_boundary": packed[0].input_ids[
                        boundaries[0]
                    ],
                    "supervised_target": packed[0].labels[boundaries[0]],
                }
            ),
        },
        "checks": checks,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nNext-token preview:")
    for row in preview:
        print(
            f"  {row['input_token_id']:>6} {row['input_token']!r}"
            f" -> {row['target_token_id']:>6} {row['target_token']!r}"
        )
    print(
        f"\nStatus: {status}\n"
        f"Report: {report_path}\n"
        f"Generated data: {data_root}"
    )
    if status != "PASS":
        raise RuntimeError("next-token preflight failed; inspect the report")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        default="tests/test_train/model_sim.yaml",
    )
    parser.add_argument(
        "--report",
        default="reports/next_token_pipeline_report.json",
    )
    args = parser.parse_args()
    run(
        model_config_path=ROOT / args.model_config,
        report_path=ROOT / args.report,
    )


if __name__ == "__main__":
    main()
