"""Raw-text continuation engine for pretrained-model probes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch

from chat.config import SamplingConfig
from chat.sampling import sample_next_token
from chat.text import IncrementalTextDecoder
from chat.tokenizer import ChatTokenizer
from model.model import TaiChuModel


@dataclass(frozen=True, slots=True)
class ContinuationResult:
    prompt: str
    continuation: str
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    stop_reason: str

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.generated_tokens / self.elapsed_seconds

    @property
    def full_text(self) -> str:
        return self.prompt + self.continuation


class ContinuationEngine:
    """Generate a continuation without chat formatting or dialogue state."""

    def __init__(
        self,
        *,
        model: TaiChuModel,
        tokenizer: ChatTokenizer,
        device: torch.device,
        max_context_tokens: int,
        sampling: SamplingConfig,
    ) -> None:
        if sampling.max_new_tokens >= max_context_tokens:
            raise ValueError("max_new_tokens must be smaller than max_context_tokens")
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_context_tokens = max_context_tokens
        self._sampling = sampling
        torch.manual_seed(sampling.seed)
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(sampling.seed)

    def generate(
        self,
        prompt: str,
        *,
        stream: bool = True,
        on_text: Callable[[str], None] | None = None,
    ) -> ContinuationResult:
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        started = time.perf_counter()
        prompt_ids = self._tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        input_budget = self._max_context_tokens - self._sampling.max_new_tokens
        if len(prompt_ids) > input_budget:
            raise ValueError(
                f"prompt requires {len(prompt_ids)} tokens, but the context budget "
                f"allows {input_budget}; reduce the prompt or max_new_tokens"
            )

        decoder = IncrementalTextDecoder(
            self._tokenizer.decode,
            strip_leading=False,
        )
        generated, stop_reason = self._generate_token_ids(
            prompt_ids,
            decoder=decoder,
            on_text=on_text if stream else None,
        )
        tail = decoder.finish()
        if tail and stream and on_text is not None:
            on_text(tail)
        continuation = decoder.text
        if not continuation.strip():
            raise RuntimeError("model produced no visible continuation text")

        return ContinuationResult(
            prompt=prompt,
            continuation=continuation,
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(generated),
            elapsed_seconds=time.perf_counter() - started,
            stop_reason=stop_reason,
        )

    def _generate_token_ids(
        self,
        prompt_ids: list[int],
        *,
        decoder: IncrementalTextDecoder,
        on_text: Callable[[str], None] | None,
    ) -> tuple[list[int], str]:
        current = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=self._device,
        )
        past_key_values = None
        generated: list[int] = []
        stop_reason = "max_new_tokens"

        with torch.inference_mode(), torch.compiler.set_stance("force_eager"):
            for _ in range(self._sampling.max_new_tokens):
                output = self._model(
                    current,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                if output.logits is None:
                    raise RuntimeError("model did not return logits")
                next_token = sample_next_token(
                    output.logits[0, -1, :],
                    generated_token_ids=generated,
                    config=self._sampling,
                    generator=self._generator,
                )
                next_token_id = int(next_token.item())
                if next_token_id == self._tokenizer.eos_token_id:
                    stop_reason = "eos"
                    break
                generated.append(next_token_id)
                past_key_values = output.past_key_values
                current = next_token.view(1, 1)
                delta = decoder.push(next_token_id)
                if delta and on_text is not None:
                    on_text(delta)

        return generated, stop_reason


__all__ = ["ContinuationEngine", "ContinuationResult"]
