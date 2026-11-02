"""Task-specific batch builders kept separate from storage and readers."""
from __future__ import annotations

from typing import Any

import torch

from ..packing import PackedSequence
from ..schema import (
    PreferenceSample,
    PretrainingSample,
    RewardSample,
    SFTSequenceSample,
    TrajectorySample,
)
from .alignment import aligned_segment_ids, segment_position_ids
from .padding import padded_float_tensor, padded_tensor


def _sequence_values(sample: Any) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    if isinstance(sample, PackedSequence):
        return (
            sample.input_ids,
            sample.labels,
            sample.attention_mask,
            sample.loss_mask,
            sample.segment_ids,
        )
    if isinstance(sample, PretrainingSample):
        labels = sample.labels or sample.input_ids
        mask = sample.attention_mask or [1] * len(sample.input_ids)
        return (
            sample.input_ids,
            labels,
            mask,
            [int(label != -100) for label in labels],
            [0] * len(sample.input_ids),
        )
    if isinstance(sample, SFTSequenceSample):
        mask = sample.attention_mask or [1] * len(sample.input_ids)
        segment_ids = sample.segment_ids or [0] * len(sample.input_ids)
        return sample.input_ids, sample.labels, mask, sample.loss_mask, segment_ids
    raise TypeError(f"unsupported sequence batch item {type(sample).__name__}")


class PretrainingBatchBuilder:
    def __init__(
        self,
        pad_token_id: int,
        *,
        pad_to_multiple_of: int = 1,
    ) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def __call__(self, samples: list[Any]) -> dict[str, Any]:
        rows = [_sequence_values(sample) for sample in samples]
        sample_ids = []
        for sample in samples:
            values = sample.sample_ids if isinstance(sample, PackedSequence) else [sample.sample_id]
            for value in values:
                if value not in sample_ids:
                    sample_ids.append(value)
        input_ids, _ = padded_tensor(
            [row[0] for row in rows],
            pad_id=self.pad_token_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        attention_mask, _ = padded_tensor(
            [row[2] for row in rows],
            pad_id=0,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        labels, _ = padded_tensor(
            [row[1] for row in rows],
            pad_id=-100,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        loss_mask, _ = padded_tensor(
            [row[3] for row in rows],
            pad_id=0,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        segment_ids, _ = padded_tensor(
            [row[4] for row in rows],
            pad_id=-1,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "segment_ids": aligned_segment_ids(attention_mask, segment_ids),
            "sample_ids": sample_ids,
        }


class SFTBatchBuilder(PretrainingBatchBuilder):
    def __init__(
        self,
        pad_token_id: int,
        *,
        padding_free: bool = False,
        pad_to_multiple_of: int = 1,
    ) -> None:
        super().__init__(pad_token_id, pad_to_multiple_of=pad_to_multiple_of)
        self.padding_free = bool(padding_free)

    def __call__(self, samples: list[Any]) -> dict[str, Any]:
        if self.padding_free:
            return self._padding_free(samples)
        batch = super().__call__(samples)
        batch["loss_mask"] = batch["loss_mask"].bool()
        return batch

    def _padding_free(self, samples: list[Any]) -> dict[str, Any]:
        rows = [_sequence_values(sample) for sample in samples]
        input_ids = []
        labels = []
        attention_mask = []
        loss_mask = []
        segment_ids = []
        position_ids = []
        sample_ids = []
        cu_seqlens = [0]
        previous_segment = None
        for sample, row in zip(samples, rows, strict=True):
            ids, targets, mask, losses, segments = row
            if len(set(segments)) > 1:
                segment_tensor = torch.tensor([segments], dtype=torch.long)
                local_positions = segment_position_ids(segment_tensor)[0].tolist()
            else:
                local_positions = list(range(len(ids)))
            for token, label, valid, loss, segment, position in zip(
                ids,
                targets,
                mask,
                losses,
                segments,
                local_positions,
                strict=True,
            ):
                if not valid:
                    continue
                if previous_segment is not None and segment != previous_segment:
                    cu_seqlens.append(len(input_ids))
                input_ids.append(token)
                labels.append(label)
                attention_mask.append(1)
                loss_mask.append(loss)
                segment_ids.append(segment)
                position_ids.append(position)
                previous_segment = segment
            if hasattr(sample, "sample_ids"):
                values = sample.sample_ids
            else:
                values = [sample.sample_id]
            for value in values:
                if value not in sample_ids:
                    sample_ids.append(value)
        cu_seqlens.append(len(input_ids))
        if not input_ids:
            raise ValueError("padding-free SFT batch contains no valid tokens")
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
            "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "sample_ids": sample_ids,
        }


class PreferenceBatchBuilder:
    def __init__(
        self,
        pad_token_id: int,
        *,
        pad_to_multiple_of: int = 1,
        max_length: int | None = None,
    ) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)
        self.max_length = max_length

    def _combined(
        self,
        sample: PreferenceSample,
    ) -> tuple[
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
    ]:
        prompt_mask = sample.prompt_mask or [1] * len(sample.prompt_input_ids)
        chosen_ids = sample.prompt_input_ids + sample.chosen_input_ids
        rejected_ids = sample.prompt_input_ids + sample.rejected_input_ids
        chosen_mask = prompt_mask + sample.chosen_mask
        rejected_mask = prompt_mask + sample.rejected_mask
        chosen_labels = [-100] * len(sample.prompt_input_ids) + [
            token if mask else -100
            for token, mask in zip(
                sample.chosen_input_ids,
                sample.chosen_mask,
                strict=True,
            )
        ]
        rejected_labels = [-100] * len(sample.prompt_input_ids) + [
            token if mask else -100
            for token, mask in zip(
                sample.rejected_input_ids,
                sample.rejected_mask,
                strict=True,
            )
        ]
        if self.max_length is not None:
            if len(chosen_ids) > self.max_length or len(rejected_ids) > self.max_length:
                raise ValueError("preference sequence exceeds configured max_length")
        return (
            chosen_ids,
            rejected_ids,
            chosen_labels,
            rejected_labels,
            chosen_mask,
            rejected_mask,
            sample.chosen_mask,
            sample.rejected_mask,
        )

    def __call__(self, samples: list[Any]) -> dict[str, Any]:
        if not all(isinstance(sample, PreferenceSample) for sample in samples):
            raise TypeError("preference batches require PreferenceSample values")
        pair_ids = [sample.pair_id for sample in samples]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("a preference batch cannot contain duplicate pair_id values")
        values = [self._combined(sample) for sample in samples]
        prompt_ids, _ = padded_tensor(
            [sample.prompt_input_ids for sample in samples],
            pad_id=self.pad_token_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        prompt_mask, _ = padded_tensor(
            [
                sample.prompt_mask
                if sample.prompt_mask is not None
                else [1] * len(sample.prompt_input_ids)
                for sample in samples
            ],
            pad_id=0,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        chosen_ids, _ = padded_tensor(
            [value[0] for value in values],
            pad_id=self.pad_token_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        rejected_ids, _ = padded_tensor(
            [value[1] for value in values],
            pad_id=self.pad_token_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        chosen_mask, _ = padded_tensor(
            [value[4] for value in values],
            pad_id=0,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        rejected_mask, _ = padded_tensor(
            [value[5] for value in values],
            pad_id=0,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        chosen_labels, _ = padded_tensor(
            [value[2] for value in values],
            pad_id=-100,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        rejected_labels, _ = padded_tensor(
            [value[3] for value in values],
            pad_id=-100,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        chosen_response_ids, _ = padded_tensor(
            [sample.chosen_input_ids for sample in samples],
            pad_id=self.pad_token_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        rejected_response_ids, _ = padded_tensor(
            [sample.rejected_input_ids for sample in samples],
            pad_id=self.pad_token_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        chosen_response_mask, _ = padded_tensor(
            [value[6] for value in values],
            pad_id=0,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        rejected_response_mask, _ = padded_tensor(
            [value[7] for value in values],
            pad_id=0,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        return {
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": prompt_mask,
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "chosen_labels": chosen_labels,
            "chosen_response_input_ids": chosen_response_ids,
            "chosen_response_attention_mask": chosen_response_mask,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
            "rejected_labels": rejected_labels,
            "rejected_response_input_ids": rejected_response_ids,
            "rejected_response_attention_mask": rejected_response_mask,
            "preference_label": torch.tensor(
                [sample.preference_label for sample in samples],
                dtype=torch.long,
            ),
            "pair_ids": pair_ids,
        }


class RewardBatchBuilder:
    def __init__(
        self,
        pad_token_id: int,
        *,
        pad_to_multiple_of: int = 1,
    ) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def __call__(self, samples: list[Any]) -> dict[str, Any]:
        if not all(isinstance(sample, RewardSample) for sample in samples):
            raise TypeError("reward batches require RewardSample values")
        input_rows = [
            sample.prompt_input_ids + sample.response_input_ids
            for sample in samples
        ]
        labels = [
            [-100] * len(sample.prompt_input_ids)
            + [
                token if mask else -100
                for token, mask in zip(
                    sample.response_input_ids,
                    sample.reward_mask or [1] * len(sample.response_input_ids),
                    strict=True,
                )
            ]
            for sample in samples
        ]
        input_ids, _ = padded_tensor(
            input_rows,
            pad_id=self.pad_token_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        attention_mask, _ = padded_tensor(
            [
                [1] * len(sample.prompt_input_ids)
                + (sample.reward_mask or [1] * len(sample.response_input_ids))
                for sample in samples
            ],
            pad_id=0,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        label_tensor, _ = padded_tensor(
            labels,
            pad_id=-100,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        ranking = [
            -1 if sample.ranking_label is None else sample.ranking_label
            for sample in samples
        ]
        return {
            "input_ids": input_ids,
            "labels": label_tensor,
            "attention_mask": attention_mask,
            "prompt_length": torch.tensor(
                [len(sample.prompt_input_ids) for sample in samples],
                dtype=torch.long,
            ),
            "reward_label": torch.tensor(
                [sample.reward_label for sample in samples],
                dtype=torch.float32,
            ),
            "ranking_label": torch.tensor(ranking, dtype=torch.long),
            "sample_ids": [sample.sample_id for sample in samples],
        }


class TrajectoryBatchBuilder:
    def __init__(
        self,
        pad_token_id: int,
        *,
        pad_to_multiple_of: int = 1,
    ) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def __call__(self, samples: list[Any]) -> dict[str, Any]:
        if not all(isinstance(sample, TrajectorySample) for sample in samples):
            raise TypeError("trajectory batches require TrajectorySample values")
        prompt_rows = [sample.prompt_input_ids for sample in samples]
        response_rows = [sample.response_input_ids for sample in samples]
        prompt_ids, prompt_mask = padded_tensor(
            prompt_rows,
            pad_id=self.pad_token_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        response_ids, _ = padded_tensor(
            response_rows,
            pad_id=self.pad_token_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        response_mask, _ = padded_tensor(
            [
                sample.response_mask
                if sample.response_mask is not None
                else [1] * len(sample.response_input_ids)
                for sample in samples
            ],
            pad_id=0,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        advantages = padded_float_tensor([sample.advantages for sample in samples])
        old_log_probs = padded_float_tensor([sample.old_log_probs for sample in samples])
        reference_log_probs = padded_float_tensor(
            [sample.reference_log_probs for sample in samples]
        )
        values = padded_float_tensor([sample.values for sample in samples])
        return {
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": prompt_mask,
            "response_input_ids": response_ids,
            "response_attention_mask": response_mask,
            "advantages": advantages,
            "old_log_probs": old_log_probs,
            "reference_log_probs": reference_log_probs,
            "values": values,
            "rewards": [list(sample.rewards) for sample in samples],
            "group_ids": [sample.group_id for sample in samples],
            "candidate_index": torch.tensor(
                [
                    -1 if sample.candidate_index is None else sample.candidate_index
                    for sample in samples
                ],
                dtype=torch.long,
            ),
            "stop_reason": [sample.stop_reason for sample in samples],
        }


__all__ = [
    "PreferenceBatchBuilder",
    "PretrainingBatchBuilder",
    "RewardBatchBuilder",
    "SFTBatchBuilder",
    "TrajectoryBatchBuilder",
]
