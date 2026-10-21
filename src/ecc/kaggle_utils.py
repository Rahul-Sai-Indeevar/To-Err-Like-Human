"""Kaggle-specific utilities.

- Clone a GitHub repo using a GITHUB_TOKEN secret (Kaggle Settings -> Secrets).
- Persistent checkpoint directory: save ``checkpoint.pt`` to a mounted Kaggle
  Dataset so training can resume across 12-hour session limits.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def clone_with_token(repo_url: str, dest: str = ".") -> None:
    """Clone ``repo_url`` using the ``GITHUB_TOKEN`` environment variable.

    On Kaggle, add your GitHub personal access token as a secret named
    ``GITHUB_TOKEN`` (Settings -> Secrets -> Add).  It is then available as an
    environment variable in the notebook.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN not set. Add it in Kaggle Settings -> Secrets, "
            "then restart the session."
        )
    if repo_url.startswith("https://github.com/"):
        authed_url = repo_url.replace("https://github.com/", f"https://{token}@github.com/")
    else:
        authed_url = repo_url
    subprocess.run(["git", "clone", authed_url, dest], check=True)


def persistent_ckpt_dir(cfg: dict) -> Optional[Path]:
    """Return the persistent checkpoint directory from config, or None.

    Set ``persistent_dir`` in your config (or override with
    ``--set persistent_dir=/kaggle/input/your-checkpoint-dataset``) to keep
    checkpoints alive across Kaggle sessions.
    """
    pd = cfg.get("persistent_dir")
    return Path(pd) if pd else None


def save_to_persistent(ckpt_path: Path, persistent_dir: Optional[Path]) -> None:
    """Copy ``ckpt_path`` into ``persistent_dir`` if configured."""
    if persistent_dir is None:
        return
    persistent_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt_path, persistent_dir / ckpt_path.name)


def latest_checkpoint_in(path: Optional[str | Path]) -> Optional[Path]:
    """Return ``checkpoint.pt`` inside ``path`` if it exists, else None."""
    if path is None:
        return None
    p = Path(path) / "checkpoint.pt"
    return p if p.exists() else None
