"""Checkpoint save/load with resume support for long Kaggle sessions.

A checkpoint stores: model state, optimizer state, scheduler state (when
available), the epoch just finished, the best metric value and the epoch it
was reached at.  Loading restores everything so training continues exactly
where it left off.
"""
from __future__ import annotations

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
