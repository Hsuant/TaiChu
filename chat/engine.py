"""Transport-independent conversational generation engine."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch

from chat.config import SamplingConfig
from chat.sampling import sample_next_token
from chat.session import Conversation
from chat.text import IncrementalTextDecoder
from chat.tokenizer import ChatTokenizer
from model.model import TaiChuModel


@dataclass(frozen=True, slots=True)
class ReplyResult:
    text: str
    generated_tokens: int
    prompt_tokens: int
    dropped_turns: int
    elapsed_seconds: float

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.generated_tokens / self.elapsed_seconds


class ChatEngine:
    """Coordinate one model, one conversation, and one decoding policy."""

    def __init__(
        self,
        *,
        model: TaiChuModel,
        tokenizer: ChatTokenizer,
        conversation: Conversation,
        device: torch.device,
        max_context_tokens: int,
        sampling: SamplingConfig,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._conversation = conversation
        self._device = device
        self._max_context_tokens = max_context_tokens
        self._sampling = sampling
        if sampling.max_new_tokens >= max_context_tokens:
            raise ValueError("max_new_tokens must be smaller than max_context_tokens")
        torch.manual_seed(sampling.seed)
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(sampling.seed)

    @property
    def conversation(self) -> Conversation:
        return self._conversation

    def respond(
        self,
        user_text: str,
        *,
        stream: bool = True,
        on_text: Callable[[str], None] | None = None,
    ) -> ReplyResult:
        """Generate one reply and commit the turn only after success."""

        started = time.perf_counter()
        self._conversation.add_user(user_text)
        try:
            input_budget = (
                self._max_context_tokens - self._sampling.max_new_tokens
            )
            prompt = self._conversation.prepare(input_budget)
            decoder = IncrementalTextDecoder(self._tokenizer.decode)
            generated_token_ids = self._generate_token_ids(
                prompt.token_ids,
                decoder=decoder,
                on_text=on_text if stream else None,
            )
            tail = decoder.finish()
            if tail and stream and on_text is not None:
                on_text(tail)
            text = decoder.text.strip()
            if not text:
                raise RuntimeError("model produced no visible response text")

            self._conversation.apply_prompt(prompt)
            self._conversation.add_assistant(text)
            elapsed = time.perf_counter() - started
            return ReplyResult(
                text=text,
                generated_tokens=len(generated_token_ids),
                prompt_tokens=len(prompt.token_ids),
                dropped_turns=prompt.dropped_turns,
                elapsed_seconds=elapsed,
            )
        except BaseException:
            self._conversation.remove_last_user()
            raise

    def _generate_token_ids(
        self,
        prompt_ids: tuple[int, ...],
        *,
        decoder: IncrementalTextDecoder,
        on_text: Callable[[str], None] | None,
    ) -> list[int]:
        current = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=self._device,
        )
        past_key_values = None
        generated: list[int] = []

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
                    break
                generated.append(next_token_id)
                past_key_values = output.past_key_values
                current = next_token.view(1, 1)
                delta = decoder.push(next_token_id)
                if delta and on_text is not None:
                    on_text(delta)

        return generated


__all__ = ["ChatEngine", "ReplyResult"]
