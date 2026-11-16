"""Pretraining task entrypoint."""

from __future__ import annotations

import math

import torch

from model.config import load_model_config
from training.causal_lm import CausalLMTask, CausalLMTrainingPipeline
from training.cli import apply_cli_overrides, parse_pretraining_args
from training.config import load_pretraining_config, validate_pretraining_config
from training.optimization import StepBudget


class PretrainingTask(CausalLMTask):
    """Pretraining policy for the shared causal-LM runtime."""

    stage = "pretrain"

    def load_configs(self, args):
        model_config = load_model_config(args.model_config)
        config = load_pretraining_config(args.pretrain_config)
        model_config, config = apply_cli_overrides(
            model_config,
            config,
            args,
        )
        validate_pretraining_config(config, model_config)
        return model_config, config

    def build_step_budget(self, config, loaders, runtime) -> StepBudget:
        reader = loaders.train.dataset.reader
        actual_tokens = self._dataset_input_tokens(reader, runtime)
        sample_count = len(loaders.train.dataset)
        if actual_tokens <= 0 or sample_count <= 0:
            raise RuntimeError(
                "pretraining dataset contains no non-padding input tokens"
            )
        epoch_steps = (
            len(loaders.train)
            // config.training.gradient_accumulation_steps
        )
        if epoch_steps <= 0:
            raise RuntimeError(
                "pretraining loader contains no complete optimizer step"
            )
        target_tokens = config.training.target_tokens
        token_budget = target_tokens
        if target_tokens is None:
            target_tokens = actual_tokens
            step_limit = epoch_steps
        else:
            global_batch_size = (
                config.training.micro_batch_size
                * config.training.gradient_accumulation_steps
                * runtime.world_size
            )
            mean_tokens_per_sample = actual_tokens / sample_count
            estimated_steps = math.ceil(
                target_tokens
                / (global_batch_size * mean_tokens_per_sample)
            )
            step_limit = max(1, estimated_steps + 1)
        return StepBudget(
            target_tokens=target_tokens,
            micro_batch_size=config.training.micro_batch_size,
            gradient_accumulation_steps=(
                config.training.gradient_accumulation_steps
            ),
            world_size=runtime.world_size,
            sequence_length=config.data.packing.max_seq_length,
            step_limit=step_limit,
            token_budget=token_budget,
        )

    @staticmethod
    def _dataset_input_tokens(reader, runtime) -> int:
        metadata = reader.manifest.metadata
        selected = metadata.get("selection")
        if isinstance(selected, dict):
            value = selected.get("actual_total_input_tokens")
            if value is not None:
                return max(0, int(value))
        for key in ("actual_input_tokens", "input_tokens"):
            value = metadata.get(key)
            if value is not None:
                return max(0, int(value))

        total = 0
        if runtime.is_main:
            for record in reader.iter_metadata():
                sample_metadata = record.get("metadata", {})
                value = sample_metadata.get("input_token_count")
                if value is None:
                    raise ValueError(
                        "pretraining samples must record input_token_count"
                    )
                total += int(value)
        if not torch.distributed.is_initialized():
            return total
        values = [total if runtime.is_main else None]
        torch.distributed.broadcast_object_list(values, src=0)
        if values[0] is None:
            raise RuntimeError("failed to broadcast pretraining token count")
        return int(values[0])


def run_pretraining(args=None) -> None:
    args = parse_pretraining_args() if args is None else args
    CausalLMTrainingPipeline(PretrainingTask()).run(args)


def main() -> None:
    run_pretraining()


__all__ = ["PretrainingTask", "main", "run_pretraining"]


if __name__ == "__main__":
    main()
