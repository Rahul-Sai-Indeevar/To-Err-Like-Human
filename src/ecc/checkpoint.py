"""Checkpoint save/load with resume support for long Kaggle sessions.

A checkpoint stores: model state, optimizer state, scheduler state (when
available), the epoch just finished, the best metric value and the epoch it
was reached at. Loading permits epoch-boundary continuation. RNG streams and
AMP scaler state are not restored, so it does not reproduce an uninterrupted
training trajectory exactly.
"""
from __future__ import annotations

import copy
import json
import shutil
import warnings

import torch
from pathlib import Path
from typing import Any, Optional


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    extra: Optional[dict[str, Any]] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    run_config: Optional[dict[str, Any]] = None,
) -> None:
    ckpt: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
    }
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if run_config is not None:
        ckpt["run_config"] = copy.deepcopy(run_config)
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: Optional[str | torch.device] = None,
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def latest_checkpoint(output_dir: str | Path) -> Optional[Path]:
    """Return the path to checkpoint.pt if it exists, else None."""
    p = Path(output_dir) / "checkpoint.pt"
    return p if p.exists() else None


def select_resume_checkpoint(resume_from, output_dir, persistent_dir=None) -> Optional[Path]:
    """Explicit path, local checkpoint, then persistent checkpoint (without str(None))."""
    if resume_from:
        path = Path(resume_from)
        if not path.is_file():
            raise FileNotFoundError(f"Explicit resume checkpoint does not exist: {path}")
        return path
    local = latest_checkpoint(output_dir)
    if local is not None:
        return local
    return latest_checkpoint(persistent_dir) if persistent_dir is not None else None


def _semantic_config(config):
    """Runtime locations may change; only the epoch budget may grow semantically."""
    semantic = copy.deepcopy(config)
    for key in ("output_dir", "resume_from", "persistent_dir"):
        semantic.pop(key, None)
    semantic.get("optim", {}).pop("epochs", None)
    return semantic


def validate_run_config(previous, current, source):
    if _semantic_config(previous) != _semantic_config(current):
        raise ValueError(f"Run configuration changed relative to {source}; use a fresh output directory for a new experiment")
    old_epochs = previous.get("optim", {}).get("epochs")
    new_epochs = current.get("optim", {}).get("epochs")
    if old_epochs is not None and new_epochs is not None and new_epochs < old_epochs:
        raise ValueError(f"Epoch budget cannot shrink relative to {source}; use a fresh output directory")


def prepare_run_config(config, output_dir, persistent_dir=None):
    """Validate resume/output identity before replacing configuration or best model.

    New checkpoints embed the run configuration. Legacy checkpoints may use an
    adjacent config.json, otherwise continuation is explicitly marked unverified.
    An epoch-budget extension is allowed, but changing it can alter an SSL
    schedule; continuation is not a claim of uninterrupted-training equivalence.
    """
    output_dir = Path(output_dir)
    config_path = output_dir / "config.json"
    selected = select_resume_checkpoint(config.get("resume_from"), output_dir, persistent_dir)
    trained_artifacts = selected is not None or any(
        (output_dir / name).exists() for name in ("best.pt", "history.jsonl", "results.json")
    )
    if config_path.exists() and trained_artifacts:
        validate_run_config(json.loads(config_path.read_text(encoding="utf-8-sig")), config, config_path)
    provenance = {"resume_from": str(selected) if selected is not None else None,
                  "configuration_verification": "fresh_run", "upstream_unverified": False,
                  "exact_trajectory_resume": False}
    checkpoint = None
    source_best = None
    if selected is not None:
        checkpoint = torch.load(selected, map_location="cpu")
        saved_config = checkpoint.get("run_config")
        sidecar = selected.parent / "config.json"
        if saved_config is not None:
            validate_run_config(saved_config, config, selected)
            provenance["configuration_verification"] = "embedded_run_config"
        elif sidecar.is_file():
            validate_run_config(json.loads(sidecar.read_text(encoding="utf-8-sig")), config, sidecar)
            provenance["configuration_verification"] = "adjacent_config_json"
        else:
            provenance["configuration_verification"] = "unverified_legacy_checkpoint"
            provenance["upstream_unverified"] = True
            warnings.warn("Legacy checkpoint has no stored run configuration; configuration provenance cannot be verified", RuntimeWarning)
        provenance["upstream_unverified"] |= checkpoint.get("resume_provenance", {}).get("upstream_unverified", False)
        candidate_best = selected.parent / "best.pt"
        if candidate_best.is_file():
            source_best = candidate_best
        elif checkpoint.get("best_epoch", -1) >= 0:
            raise FileNotFoundError(f"Resume checkpoint records a validation-best model, but matching best.pt is missing: {candidate_best}")
    # All guards have passed; external resumes recover the chosen run's best,
    # even if this output directory contains a stale best from an earlier attempt.
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_best is not None and source_best.resolve() != (output_dir / "best.pt").resolve():
        shutil.copy2(source_best, output_dir / "best.pt")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf8")
    (output_dir / "resume_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf8")
    return selected, provenance
