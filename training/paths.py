"""Experiment directory resolution and creation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from training.checkpoint import resolve_experiment_dir_from_checkpoint


@dataclass(frozen=True, slots=True)
class ExperimentPaths:
    root: Path
    model_scale: str
    stage: str
    experiment_name: str
    config: Path
    evaluation: Path
    best_models: Path
    checkpoints: Path
    final_models: Path
    logs: Path
    metadata: Path

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        model_scale: str,
        stage: str,
        experiment_name: str,
    ) -> "ExperimentPaths":
        root_path = Path(root)
        paths = cls(
            root=root_path,
            model_scale=model_scale,
            stage=stage,
            experiment_name=experiment_name,
            config=root_path / "config",
            evaluation=root_path / "evaluation",
            best_models=root_path / "best_models",
            checkpoints=root_path / "checkpoints",
            final_models=root_path / "final_models",
            logs=root_path / "logs",
            metadata=root_path / "metadata.json",
        )
        for directory in (
            paths.root,
            paths.config,
            paths.evaluation,
            paths.best_models,
            paths.checkpoints,
            paths.final_models,
            paths.logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return paths


def _path_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return component.lower() or "unnamed"


def resolve_model_scale(model_name: str) -> str:
    """Extract a stable model-size component from names such as TaiChu-80M."""
    match = re.search(
        r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)([mbt])(?![A-Za-z0-9])",
        model_name,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return f"{match.group(1)}{match.group(2).lower()}"
    return _path_component(model_name)


def resolve_experiment_root(
    *,
    experiments_dir: str,
    stage: str,
    experiment_name: str,
    resume_path: str | None,
) -> Path:
    if resume_path is not None:
        return Path(resolve_experiment_dir_from_checkpoint(resume_path))
    name = experiment_name
    if not name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{stage}_{timestamp}"
    name_component = _path_component(name)
    base = Path(experiments_dir)
    root = base if base.name == name_component else base / name_component
    candidate = root
    suffix = 1
    while candidate.exists():
        candidate = root.with_name(f"{root.name}_{suffix}")
        suffix += 1
    return candidate


__all__ = [
    "ExperimentPaths",
    "resolve_experiment_root",
    "resolve_model_scale",
]
