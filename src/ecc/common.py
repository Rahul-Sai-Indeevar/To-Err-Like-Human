from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .emotion import build_cost_matrix, build_rank_matrix, polarity_vector


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_classes(splits_dir) -> list[str]:
    with open(Path(splits_dir) / "classes.txt") as f:
        return [l.strip() for l in f if l.strip()]


def emotion_tensors(classes, C: int):
    W = build_cost_matrix(classes, C)
    return W, polarity_vector(classes), build_rank_matrix(W)


@torch.no_grad()
def predict_probs(model, loader, device, amp: bool = False):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)
        P.append(logits.float().softmax(1).cpu())
        Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def append_jsonl(path, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
