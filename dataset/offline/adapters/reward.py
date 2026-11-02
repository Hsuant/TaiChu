"""Prompt/response reward-model preprocessing."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...exceptions import TaiChuDataFormatError
from ...schema import RewardSample
from .base import (
    BasePreprocessingAdapter,
    ConversationRenderer,
    TextTokenizer,
    dataset_identifier,
    sample_identifier,
    source_identifier,
)


class RewardPreprocessor(BasePreprocessingAdapter):
    sample_type = "reward"

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
            raise TaiChuDataFormatError("reward record requires prompt_messages")
        response = record.get("response")
        if not isinstance(response, str) or not response:
            raise TaiChuDataFormatError("reward record requires non-empty response")
        if "reward_label" not in record:
            raise TaiChuDataFormatError("reward record requires reward_label")

    def format(self, record: Mapping[str, Any]) -> tuple[str, str]:
        return (
            self.renderer.render_prompt(record["prompt_messages"]),
            record["response"],
        )

    def tokenize(
        self,
        record: Mapping[str, Any],
        formatted: tuple[str, str],
    ) -> RewardSample:
        prompt, response = formatted
        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response)
        if self.add_eos and self.tokenizer.eos_token_id is not None:
            response_ids.append(self.tokenizer.eos_token_id)
        return RewardSample(
            sample_id=sample_identifier(record, "reward"),
            dataset_id=dataset_identifier(record, "reward"),
            source_id=source_identifier(record),
            prompt_input_ids=prompt_ids,
            response_input_ids=response_ids,
            reward_label=float(record["reward_label"]),
            reward_mask=[1] * len(response_ids),
            ranking_label=record.get("ranking_label"),
        )


__all__ = ["RewardPreprocessor"]
