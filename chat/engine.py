"""Transport-independent conversational generation engine."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch

from chat.config import SamplingConfig
from chat.reasoning import ReasoningMode, parse_reasoning_tokens
from chat.sampling import sample_next_token
from chat.session import Conversation
from chat.text import IncrementalTextDecoder
from chat.tokenizer import ChatTokenizer
from model.model import TaiChuModel


@dataclass(frozen=True, slots=True)
class ReplyResult:
    text: str
    reasoning_text: str
    reasoning_mode: ReasoningMode
    thinking: bool
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

    _MAX_THINKING_TOKENS = 256
    _MIN_ANSWER_TOKENS = 16

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
        on_reasoning: Callable[[str], None] | None = None,
    ) -> ReplyResult:
        """Generate one reply and commit the turn only after success.

        ``on_reasoning`` receives reasoning-section deltas. If omitted,
        reasoning deltas fall back to ``on_text``.
        """

        started = time.perf_counter()
        self._conversation.add_user(user_text)
        try:
            reasoning_mode = self._conversation.reasoning_mode
            input_budget = (
                self._max_context_tokens - self._sampling.max_new_tokens
            )
            prompt = self._conversation.prepare(input_budget)
            decoder = IncrementalTextDecoder(self._tokenizer.decode)
            generated_token_ids, ended_in_reasoning = self._generate_token_ids(
                prompt.token_ids,
                decoder=decoder,
                on_text=on_text if stream else None,
                on_reasoning=on_reasoning if stream else None,
                reasoning_mode=reasoning_mode,
            )
            if stream:
                tail = decoder.finish()
                if tail:
                    if ended_in_reasoning:
                        if on_reasoning is not None:
                            on_reasoning(tail)
                        elif on_text is not None:
                            on_text(tail)
                    elif on_text is not None:
                        on_text(tail)
            parsed = parse_reasoning_tokens(
                generated_token_ids,
                mode=reasoning_mode,
                think_token_id=self._tokenizer.think_token_id,
                end_think_token_id=self._tokenizer.end_think_token_id,
                decode=self._tokenizer.decode,
            )
            text = parsed.answer_text
            if not text:
                if parsed.thinking and parsed.reasoning_text:
                    self._conversation.remove_last_user()
                    elapsed = time.perf_counter() - started
                    return ReplyResult(
                        text="",
                        reasoning_text=parsed.reasoning_text,
                        reasoning_mode=reasoning_mode,
                        thinking=True,
                        generated_tokens=len(generated_token_ids),
                        prompt_tokens=len(prompt.token_ids),
                        dropped_turns=prompt.dropped_turns,
                        elapsed_seconds=elapsed,
                    )
                raise RuntimeError("model produced no visible answer text")

            self._conversation.apply_prompt(prompt)
            self._conversation.add_assistant(
                text,
                reasoning_content=parsed.reasoning_text,
            )
            elapsed = time.perf_counter() - started
            return ReplyResult(
                text=text,
                reasoning_text=parsed.reasoning_text,
                reasoning_mode=reasoning_mode,
                thinking=parsed.thinking,
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
        on_reasoning: Callable[[str], None] | None,
        reasoning_mode: ReasoningMode,
    ) -> tuple[list[int], bool]:
        current = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=self._device,
        )
        past_key_values = None
        generated: list[int] = []
        thinking = reasoning_mode is ReasoningMode.ON
        thinking_closed = False
        answer_tokens = 0
        think_id = self._tokenizer.think_token_id
        end_think_id = self._tokenizer.end_think_token_id
        thinking_limit: int | None = None
        if reasoning_mode is not ReasoningMode.OFF:
            thinking_limit = min(
                self._sampling.max_new_tokens // 2,
                self._MAX_THINKING_TOKENS,
            )

        with torch.inference_mode(), torch.compiler.set_stance("force_eager"):
            for _ in range(self._sampling.max_new_tokens):
                output = self._model(
                    current,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                if output.logits is None:
                    raise RuntimeError("model did not return logits")
                suppress_token_ids = []
                if reasoning_mode is ReasoningMode.OFF:
                    suppress_token_ids.append(think_id)
                if (
                    thinking_closed
                    and answer_tokens < self._MIN_ANSWER_TOKENS
                ):
                    suppress_token_ids.append(self._tokenizer.eos_token_id)
                next_token = sample_next_token(
                    output.logits[0, -1, :],
                    generated_token_ids=generated,
                    config=self._sampling,
                    generator=self._generator,
                    suppress_token_ids=suppress_token_ids,
                )
                next_token_id = int(next_token.item())
                if next_token_id == self._tokenizer.eos_token_id:
                    if thinking and self._sampling.max_new_tokens > 1:
                        next_token_id = end_think_id
                        next_token = torch.tensor(
                            [end_think_id],
                            dtype=torch.long,
                            device=self._device,
                        )
                    else:
                        break

                if (
                    reasoning_mode is ReasoningMode.AUTO
                    and not generated
                    and next_token_id == think_id
                ):
                    thinking = True

                if (
                    thinking
                    and thinking_limit is not None
                    and self._sampling.max_new_tokens > 1
                    and len(generated) >= thinking_limit
                ):
                    next_token_id = end_think_id
                    next_token = torch.tensor(
                        [end_think_id],
                        dtype=torch.long,
                        device=self._device,
                    )

                generated.append(next_token_id)
                past_key_values = output.past_key_values
                current = next_token.view(1, 1)
                delta = decoder.push(next_token_id)
                if next_token_id in (think_id, end_think_id):
                    delta = ""
                if next_token_id == end_think_id and thinking:
                    thinking = False
                    thinking_closed = True
                    answer_tokens = 0
                elif thinking_closed and not thinking:
                    answer_tokens += 1

                if not delta:
                    continue
                callback = on_reasoning if thinking else on_text
                if callback is None and thinking:
                    callback = on_text
                if callback is not None:
                    callback(delta)

        return generated, thinking


__all__ = ["ChatEngine", "ReplyResult"]
