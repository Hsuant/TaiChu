"""Offline trajectory adapter; online rollouts use the same schema directly."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...exceptions import TaiChuDataFormatError
from ...schema import TrajectorySample
from .base import (
    BasePreprocessingAdapter,
    ConversationRenderer,
    TextTokenizer,
    dataset_identifier,
    sample_identifier,
    source_identifier,
)


class TrajectoryPreprocessor(BasePreprocessingAdapter):
    sample_type = "trajectory"

    def __init__(
        self,
        tokenizer: TextTokenizer,
        renderer: ConversationRenderer,
        *,
        add_eos: bool = True,
    ) -> None:
        if add_eos and tokenizer.eos_token_id is None:
            raise ValueError("EOS token could not be resolved")
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.add_eos = add_eos

    def validate(self, record: Mapping[str, Any]) -> None:
        super().validate(record)
        prompt = record.get("prompt_messages")
        if not isinstance(prompt, list) or not prompt:
            raise TaiChuDataFormatError("trajectory record requires prompt_messages")
        if not isinstance(record.get("response"), str):
            raise TaiChuDataFormatError("trajectory record requires response")

    def format(self, record: Mapping[str, Any]) -> tuple[str, str]:
        return (
            self.renderer.render_prompt(record["prompt_messages"]),
            record["response"],
        )

    def tokenize(
        self,
        record: Mapping[str, Any],
        formatted: tuple[str, str],
    ) -> TrajectorySample:
        prompt, response = formatted
        response_ids = self.tokenizer.encode(response)
        if self.add_eos and self.tokenizer.eos_token_id is not None:
            response_ids.append(self.tokenizer.eos_token_id)
        return TrajectorySample(
            sample_id=sample_identifier(record, "trajectory"),
            dataset_id=dataset_identifier(record, "trajectory"),
            source_id=source_identifier(record),
            prompt_input_ids=self.tokenizer.encode(prompt),
            response_input_ids=response_ids,
            rewards=list(record.get("rewards", [])),
            advantages=list(record.get("advantages", [])),
            old_log_probs=list(record.get("old_log_probs", [])),
            reference_log_probs=list(record.get("reference_log_probs", [])),
            values=list(record.get("values", [])),
            response_mask=[1] * len(response_ids),
            group_id=record.get("group_id"),
            candidate_index=record.get("candidate_index"),
            stop_reason=record.get("stop_reason"),
        )


__all__ = ["TrajectoryPreprocessor"]
