"""Supervised fine-tuning entrypoint built on the shared causal-LM trainer."""

from __future__ import annotations

import math
from pathlib import Path

from dataset.offline.adapters import RoleTagRenderer
from model.config import load_model_config
from training.causal_lm import CausalLMTask, CausalLMTrainingPipeline
from training.cli import apply_cli_overrides, parse_sft_args
from training.config import load_sft_config, validate_sft_config
from training.model_artifacts import load_model_artifact
from training.optimization import StepBudget


SFT_SYSTEM_PROMPT = ""


def _initialization_path(args) -> Path | None:
    if args.resume is not None and args.init_from is not None:
        raise ValueError("--init_from and --resume are mutually exclusive")
    if args.resume is not None:
        return None
    if args.init_from is None:
        raise ValueError(
            "a new SFT run requires --init_from pointing to a pretrained "
            "model artifact"
        )
    path = Path(args.init_from).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"pretrained model artifact does not exist: {path}"
        )
    return path


class SFTPromptRenderer:
    """Render the exact SFT user prompt with the production system policy."""

    def __init__(self, *, system_prompt: str = SFT_SYSTEM_PROMPT) -> None:
        self._renderer = RoleTagRenderer()
        self._system_prompt = system_prompt.strip()

    def __call__(self, prompt: str) -> str:
        system = (
            self._renderer.render(
                {"role": "system", "content": self._system_prompt}
            )
            if self._system_prompt
            else None
        )
        user_turn = self._renderer.render({"role": "user", "content": prompt})
        system_prefix = (
            f"{system.prefix}{system.content}{system.suffix}"
            if system
            else ""
        )
        return (
            f"{system_prefix}{user_turn.prefix}{user_turn.content}"
            f"{user_turn.suffix}{self._renderer.start_token}assistant\n"
        )

class SFTTask(CausalLMTask):
    """Supervised fine-tuning policy for the shared causal-LM runtime."""

    stage = "sft"

    def validate_args(self, args) -> None:
        _initialization_path(args)

    def load_configs(self, args):
        model_config = load_model_config(args.model_config)
        config = load_sft_config(args.sft_config)
        model_config, config = apply_cli_overrides(
            model_config,
            config,
            args,
        )
        validate_sft_config(config, model_config)
        return model_config, config

    def build_step_budget(self, config, loaders, runtime) -> StepBudget:
        epochs = config.training.epochs
        if epochs is None:
            raise ValueError("SFT requires training.epochs")
        epoch_steps = (
            len(loaders.train)
            // config.training.gradient_accumulation_steps
        )
        if epoch_steps <= 0:
            raise RuntimeError(
                "SFT training loader contains no complete optimizer step"
            )
        max_steps = max(1, math.ceil(epoch_steps * epochs))
        tokens_per_step = (
            config.training.micro_batch_size
            * config.training.gradient_accumulation_steps
            * runtime.world_size
            * config.data.packing.max_seq_length
        )
        return StepBudget(
            target_tokens=max_steps * tokens_per_step,
            micro_batch_size=config.training.micro_batch_size,
            gradient_accumulation_steps=(
                config.training.gradient_accumulation_steps
            ),
            world_size=runtime.world_size,
            sequence_length=config.data.packing.max_seq_length,
            step_limit=max_steps,
        )

    def prepare_model(self, args, model) -> str | None:
        initialization_path = _initialization_path(args)
        if initialization_path is None:
            return None
        load_model_artifact(initialization_path, model)
        return str(initialization_path)

    def prompt_renderer_factory(self):
        return lambda _tokenizer: SFTPromptRenderer()


def run_sft(args=None) -> None:
    """Run SFT from a pretrained artifact or resume a full SFT checkpoint."""
    args = parse_sft_args() if args is None else args
    CausalLMTrainingPipeline(SFTTask()).run(args)


def main() -> None:
    run_sft()


__all__ = ["SFTTask", "main", "run_sft"]


if __name__ == "__main__":
    main()
