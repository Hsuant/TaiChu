"""Experiment directory resolution and creation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from training.checkpoint import resolve_experiment_dir_from_checkpoint


@dataclass(frozen=True, slots=True)
class ExperimentPaths:
    root: Path
    best_models: Path
    checkpoints: Path
    final_models: Path
    logs: Path

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        best_subdir: str = "best_models",
    ) -> "ExperimentPaths":
        root_path = Path(root)
        paths = cls(
            root=root_path,
            best_models=root_path / best_subdir,
            checkpoints=root_path / "checkpoints",
            final_models=root_path / "final_models",
            logs=root_path / "logs",
        )
        for directory in (
            paths.root,
            paths.best_models,
            paths.checkpoints,
            paths.final_models,
            paths.logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return paths


def resolve_experiment_root(
    *,
    output_dir: str,
    experiment_name: str,
    model_name: str,
    resume_path: str | None,
) -> Path:
    if resume_path is not None:
        return Path(resolve_experiment_dir_from_checkpoint(resume_path))
    name = experiment_name
    if not name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{model_name}_{timestamp}" if model_name else f"TaiChu_{timestamp}"
    root = Path(output_dir) / name
    candidate = root
    suffix = 1
    while candidate.exists():
        candidate = root.with_name(f"{root.name}_{suffix}")
        suffix += 1
    return candidate


__all__ = ["ExperimentPaths", "resolve_experiment_root"]
