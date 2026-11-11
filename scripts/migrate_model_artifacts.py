#!/usr/bin/env python3
"""Migrate legacy full best/final checkpoints to minimal model artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.model_artifacts import (  # noqa: E402
    read_model_artifact,
    save_serialized_model_artifact,
)


def _move(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)


def _is_minimal(path: Path) -> bool:
    try:
        read_model_artifact(path)
    except Exception:
        return False
    return True


def _legacy_metrics(checkpoint: dict, variant: str) -> dict[str, float]:
    metrics = {}
    raw_metrics = checkpoint.get("metrics")
    if isinstance(raw_metrics, dict):
        metrics.update(
            {
                str(name): value
                for name, value in raw_metrics.items()
                if isinstance(value, (int, float))
            }
        )
    best_metric = checkpoint.get("best_metric_value")
    if isinstance(best_metric, (int, float)):
        metrics.setdefault("val_loss", float(best_metric))
        metrics.setdefault("best_val_loss", float(best_metric))
    if variant == "best" and "val_loss" not in metrics:
        raise ValueError("legacy best checkpoint has no val_loss metric")
    return metrics


def _migrate_one(
    source: Path,
    *,
    variant: str,
    backup: Path,
    dry_run: bool,
) -> bool:
    if not source.is_file() or _is_minimal(source):
        return False
    action = f"{variant}: {source} -> {source} + backup {backup}"
    print(action)
    if dry_run:
        return True

    checkpoint = torch.load(
        source,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{source} is not a checkpoint dictionary")
    model_state_dict = checkpoint.get("model_state_dict")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_state_dict, dict) or not isinstance(
        model_config,
        dict,
    ):
        raise ValueError(f"{source} is not a recognized TaiChu checkpoint")
    if backup.exists():
        raise FileExistsError(f"migration backup already exists: {backup}")

    _move(source, backup)
    try:
        save_serialized_model_artifact(
            source,
            model_state_dict=model_state_dict,
            model_config=model_config,
            variant=variant,
            global_step=int(checkpoint.get("global_step", 0)),
            metrics=_legacy_metrics(checkpoint, variant),
            stop_reason="migrated" if variant == "final" else None,
        )
        read_model_artifact(source)
    except Exception:
        source.unlink(missing_ok=True)
        _move(backup, source)
        raise
    return True


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert legacy TaiChu best/final checkpoints"
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    experiment = Path(args.experiment).expanduser().resolve()
    if not experiment.is_dir():
        raise FileNotFoundError(f"experiment directory not found: {experiment}")
    checkpoints = experiment / "checkpoints"
    legacy = experiment / "legacy"

    best_changed = _migrate_one(
        experiment / "best_models" / "best_model.pt",
        variant="best",
        backup=legacy / "best_model-full.pt",
        dry_run=args.dry_run,
    )
    final_source = experiment / "final_models" / "final_model.pt"
    final_checkpoint = checkpoints / "checkpoint-final.pt"
    final_backup = (
        final_checkpoint
        if not final_checkpoint.exists()
        else legacy / "final_model-full.pt"
    )
    final_changed = _migrate_one(
        final_source,
        variant="final",
        backup=final_backup,
        dry_run=args.dry_run,
    )
    if not best_changed and not final_changed:
        print("No legacy model artifacts found; nothing to migrate.")


if __name__ == "__main__":
    main()
