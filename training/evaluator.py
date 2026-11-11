"""Shared causal language-model evaluation and generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.amp.autocast_mode import autocast

from training.batch import move_batch_to_device
from training.checkpoint import unwrap_model
from training.validation import (
    compute_bits_per_byte,
    compute_empty_prefix_eos_metrics,
    compute_percentile_ce,
    compute_perplexity,
    compute_token_statistics_from_losses,
)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    loss: float
    perplexity: float | None
    bits_per_byte: float | None
    accuracy: float
    tokens: int
    target_bytes: int
    batches: int
    max_token_loss: float
    percentiles: dict[str, float | None]
    empty_prefix: dict[str, float | None]

    def metrics(self, prefix: str = "val/") -> dict[str, float]:
        values: dict[str, float] = {
            f"{prefix}loss": self.loss,
            f"{prefix}accuracy": self.accuracy,
            f"{prefix}tokens": self.tokens,
            f"{prefix}target_bytes": self.target_bytes,
            f"{prefix}batches": self.batches,
            f"{prefix}ce_max_exact": self.max_token_loss,
        }
        if self.perplexity is not None:
            values[f"{prefix}perplexity"] = self.perplexity
        if self.bits_per_byte is not None:
            values[f"{prefix}bpb"] = self.bits_per_byte
        for name, value in self.percentiles.items():
            if value is not None:
                values[f"{prefix}{name}"] = value
        for name, value in self.empty_prefix.items():
            if value is not None:
                values[f"{prefix}{name}"] = value
        return values


class CausalLMEvaluator:
    """Evaluate a causal language model on the unified validation loader."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer,
        device: torch.device,
        use_amp: bool,
        dtype: torch.dtype,
        bos_token_id: int | None,
        eos_token_id: int | None,
        logger=None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_amp = use_amp
        self.dtype = dtype
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.logger = logger
        self.rank = rank
        self.world_size = world_size

    @torch.no_grad()
    def evaluate(
        self,
        loader,
        *,
        compute_empty_prefix: bool,
        eval_log_interval: int = 10,
    ) -> EvaluationResult:
        was_training = self.model.training
        self.model.eval()
        try:
            return self._evaluate_impl(
                loader,
                compute_empty_prefix=compute_empty_prefix,
                eval_log_interval=eval_log_interval,
            )
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    def _evaluate_impl(
        self,
        loader,
        *,
        compute_empty_prefix: bool,
        eval_log_interval: int = 10,
    ) -> EvaluationResult:
        self.model.eval()
        total_nll = 0.0
        total_tokens = 0
        total_correct = 0
        total_bpb_nll = 0.0
        total_bpb_bytes = 0
        max_token_loss = 0.0
        local_losses: list[torch.Tensor] = []
        batches = 0
        try:
            total_batches = len(loader)
        except TypeError:
            total_batches = None
        accumulated_nll = 0.0
        accumulated_tokens = 0

        with torch.compiler.set_stance("force_eager"):
            for batch in loader:
                moved = move_batch_to_device(batch, self.device)
                with autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                    dtype=self.dtype,
                ):
                    output = self.model(
                        moved["input_ids"],
                        attention_mask=moved.get("attention_mask"),
                        segment_ids=moved.get("segment_ids"),
                        position_ids=moved.get("position_ids"),
                        labels=moved["labels"],
                        return_logits=False,
                        validate_inputs=False,
                        return_token_statistics=True,
                    )
                if (
                    output.loss is None
                    or not torch.isfinite(output.loss)
                    or output.token_losses is None
                    or output.correct_predictions is None
                ):
                    raise RuntimeError(
                        "model did not return finite validation statistics"
                    )
                stats = compute_token_statistics_from_losses(
                    output.token_losses,
                    moved["labels"],
                    correct_count=int(output.correct_predictions.item()),
                    tokenizer=self.tokenizer,
                )
                total_nll += stats.nll_nats
                total_tokens += stats.token_count
                total_correct += stats.correct_count
                total_bpb_nll += stats.bpb_nll_nats
                total_bpb_bytes += stats.bpb_bytes
                max_token_loss = max(max_token_loss, stats.max_token_loss)
                if stats.token_losses.numel():
                    local_losses.append(stats.token_losses)
                batches += 1
                accumulated_nll += stats.nll_nats
                accumulated_tokens += stats.token_count
                if (
                    self.logger is not None
                    and self.rank == 0
                    and eval_log_interval > 0
                    and batches % eval_log_interval == 0
                ):
                    mean = (
                        accumulated_nll / accumulated_tokens
                        if accumulated_tokens
                        else None
                    )
                    mean_text = "N/A" if mean is None else f"{mean:.4f}"
                    self.logger.info(
                        f"[Eval] batch {batches}/{total_batches or '?'} | "
                        f"loss={mean_text}"
                    )
                    accumulated_nll = 0.0
                    accumulated_tokens = 0

            if batches == 0:
                raise RuntimeError("validation data pipeline produced no batches")
            (
                total_nll,
                total_tokens,
                total_correct,
                total_bpb_nll,
                total_bpb_bytes,
                batches,
                max_token_loss,
            ) = self._reduce_totals(
                total_nll,
                total_tokens,
                total_correct,
                total_bpb_nll,
                total_bpb_bytes,
                batches,
                max_token_loss,
            )
            if total_tokens <= 0:
                raise RuntimeError(
                    "validation produced no supervised next-token positions"
                )
            loss = total_nll / total_tokens
            token_losses = (
                torch.cat(local_losses)
                if local_losses
                else torch.empty(0, dtype=torch.float32)
            )
            loss_distribution = self._gather_losses(token_losses)
            percentiles = (
                compute_percentile_ce(
                    loss_distribution,
                    (0.5, 0.9, 0.95),
                )
                if self.rank == 0
                else {}
            )
            empty_prefix = (
                compute_empty_prefix_eos_metrics(
                    unwrap_model(self.model),
                    bos_token_id=self.bos_token_id,
                    eos_token_id=self.eos_token_id,
                    device=self.device,
                )
                if compute_empty_prefix and self.rank == 0
                else {
                    "empty_prefix_eos_ce": None,
                    "empty_prefix_eos_ppl": None,
                }
            )
        if self.logger is not None and self.rank == 0:
            ppl = compute_perplexity(loss)
            ppl_text = "N/A" if ppl is None else f"{ppl:.2f}"
            p50 = percentiles.get("ce_p50")
            p90 = percentiles.get("ce_p90")
            self.logger.info(
                f"val/loss={loss:.4f} | val/ppl={ppl_text} | "
                f"val/acc={total_correct / total_tokens:.4f} | "
                f"val/tokens={total_tokens} | "
                f"val/ce_p50={p50 if p50 is not None else 'N/A'} | "
                f"val/ce_p90={p90 if p90 is not None else 'N/A'}"
            )

        return EvaluationResult(
            loss=loss,
            perplexity=compute_perplexity(loss),
            bits_per_byte=compute_bits_per_byte(
                total_bpb_nll,
                total_bpb_bytes,
            ),
            accuracy=total_correct / total_tokens,
            tokens=total_tokens,
            target_bytes=total_bpb_bytes,
            batches=batches,
            max_token_loss=max_token_loss,
            percentiles=percentiles,
            empty_prefix=empty_prefix,
        )

    def _reduce_totals(
        self,
        total_nll: float,
        total_tokens: int,
        total_correct: int,
        total_bpb_nll: float,
        total_bpb_bytes: int,
        batches: int,
        max_token_loss: float,
    ) -> tuple[float, int, int, float, int, int, float]:
        if self.world_size <= 1:
            return (
                total_nll,
                total_tokens,
                total_correct,
                total_bpb_nll,
                total_bpb_bytes,
                batches,
                max_token_loss,
            )
        totals = torch.tensor(
            [
                total_nll,
                float(total_tokens),
                float(total_correct),
                total_bpb_nll,
                float(total_bpb_bytes),
                float(batches),
            ],
            dtype=torch.float64,
            device=self.device,
        )
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        maximum = torch.tensor(
            max_token_loss,
            dtype=torch.float64,
            device=self.device,
        )
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
        values = totals.tolist()
        return (
            values[0],
            int(values[1]),
            int(values[2]),
            values[3],
            int(values[4]),
            int(values[5]),
            float(maximum.item()),
        )

    def _gather_losses(self, local: torch.Tensor) -> torch.Tensor:
        if self.world_size <= 1:
            return local
        gathered: list[Any] | None = (
            [None] * self.world_size if self.rank == 0 else None
        )
        torch.distributed.gather_object(
            local,
            object_gather_list=gathered,
            dst=0,
        )
        status: list[str | None] = [None]
        if self.rank == 0:
            try:
                return torch.cat(gathered)
            except Exception as exc:
                status[0] = f"{type(exc).__name__}: {exc}"
        torch.distributed.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(
                f"distributed percentile aggregation failed: {status[0]}"
            )
        return torch.empty(0)

    def generate(
        self,
        prompts: Sequence[str],
        *,
        num_tokens: int,
    ) -> list[dict[str, str]]:
        if not prompts or self.rank != 0:
            if self.world_size > 1:
                torch.distributed.barrier()
            return []
        model = unwrap_model(self.model)
        results: list[dict[str, str]] = []
        with torch.compiler.set_stance("force_eager"):
            for prompt in prompts:
                ids = self.tokenizer.encode(
                    prompt,
                    add_special_tokens=False,
                ).ids
                input_ids = torch.tensor(
                    [ids],
                    dtype=torch.long,
                    device=self.device,
                )
                generated = model.generate(
                    input_ids,
                    max_new_tokens=num_tokens,
                    temperature=0.8,
                    top_k=50,
                    eos_token_id=self.eos_token_id,
                )
                text = self.tokenizer.decode(generated[0].tolist())
                results.append({"prompt": prompt, "generated_text": text})
                if self.logger is not None:
                    self.logger.info(
                        f"[生成测试] 提示: {prompt}\n生成结果: {text}\n{'-' * 60}"
                    )
        if self.world_size > 1:
            torch.distributed.barrier()
        return results


__all__ = ["CausalLMEvaluator", "EvaluationResult"]
