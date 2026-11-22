"""Optional secondary storage for baseline checkpoints."""
from pathlib import Path
import shutil
from typing import Optional


def persistent_ckpt_dir(cfg: dict) -> Optional[Path]:
    """Return the persistent checkpoint directory from config, or None.

    Set ``persistent_dir`` in your config (or override with
    ``--set persistent_dir=/kaggle/input/your-checkpoint-dataset``) to keep
    checkpoints alive across Kaggle sessions.
    """
    pd = cfg.get("persistent_dir")
    return Path(pd) if pd else None


def save_to_persistent(ckpt_path: Path, persistent_dir: Optional[Path]) -> None:
    """Copy ``ckpt_path`` into ``persistent_dir`` if configured.

    Kaggle input mounts are read-only, so this silently skips if the
    directory is not writable. Checkpoints are still saved to the
    run's output_dir (which is writable). Use the notebook's auto-save
    cell to upload checkpoints to a Kaggle Dataset via the API.
    """
    if persistent_dir is None:
        return
    try:
        persistent_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt_path, persistent_dir / ckpt_path.name)
    except OSError:
        # Read-only filesystem (e.g. Kaggle input dataset mount) — skip silently.
        pass
