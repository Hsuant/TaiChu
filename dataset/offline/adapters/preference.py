"""Prompt/chosen/rejected pair preprocessing."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...exceptions import TaiChuDataFormatError
from ...schema import PreferenceSample
from .base import (
    BasePreprocessingAdapter,
    ConversationRenderer,
    TextTokenizer,
    dataset_identifier,
    sample_identifier,
    source_identifier,
)


class PreferencePreprocessor(BasePreprocessingAdapter):
    sample_type = "preference"

    def __init__(
        self,
        tokenizer: TextTokenizer,
        renderer: ConversationRenderer,
        *,
        add_eos: bool = True,
        max_response_length: int | None = None,
    ) -> None:
        if add_eos and tokenizer.eos_token_id is None:
            raise ValueError("EOS token could not be resolved")
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.add_eos = add_eos
        self.max_response_length = max_response_length

    def validate(self, record: Mapping[str, Any]) -> None:
        super().validate(record)
        prompt = record.get("prompt_messages")
        if not isinstance(prompt, list) or not prompt:
            raise TaiChuDataFormatError("preference record requires prompt_messages")
        for message in prompt:
            if not isinstance(message, Mapping):
                raise TaiChuDataFormatError("prompt messages must be mappings")
            self.renderer.render(message)
        for name in ("chosen", "rejected"):
            value = record.get(name)
            if not isinstance(value, str) or not value:
                raise TaiChuDataFormatError(f"preference record requires non-empty {name}")

    def format(self, record: Mapping[str, Any]) -> tuple[str, str, str]:
        prompt = self.renderer.render_prompt(record["prompt_messages"])
        return prompt, record["chosen"], record["rejected"]

    def _response_ids(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text)
        if self.add_eos and self.tokenizer.eos_token_id is not None:
            ids.append(self.tokenizer.eos_token_id)
        if self.max_response_length is not None:
            ids = ids[: self.max_response_length]
        return ids

    def tokenize(
        self,
        record: Mapping[str, Any],
        formatted: tuple[str, str, str],
    ) -> PreferenceSample:
        prompt, chosen, rejected = formatted
        prompt_ids = self.tokenizer.encode(prompt)
        chosen_ids = self._response_ids(chosen)
        rejected_ids = self._response_ids(rejected)
        sample_id = sample_identifier(record, "preference")
        pair_id = record.get("pair_id", sample_id)
        if not isinstance(pair_id, str) or not pair_id:
            raise TaiChuDataFormatError("pair_id must be a non-empty string")
        label = record.get("preference_label", 1)
        if label not in {0, 1}:
            raise TaiChuDataFormatError("preference_label must be 0 or 1")
        return PreferenceSample(
            sample_id=sample_id,
            dataset_id=dataset_identifier(record, "preference"),
            source_id=source_identifier(record),
            pair_id=pair_id,
            prompt_input_ids=prompt_ids,
            chosen_input_ids=chosen_ids,
            rejected_input_ids=rejected_ids,
            prompt_mask=[1] * len(prompt_ids),
            chosen_mask=[1] * len(chosen_ids),
            rejected_mask=[1] * len(rejected_ids),
            preference_label=int(label),
        )


__all__ = ["PreferencePreprocessor"]
