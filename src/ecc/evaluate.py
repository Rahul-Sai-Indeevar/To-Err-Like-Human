"""Evaluate a saved checkpoint on a split with ACC / ACC2 / ECC / EMC (argmax and MEC decoding)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import emotion_tensors, get_device, load_classes, predict_probs
from .config import load_config, parse_args
from .data import EmotionCSV, build_transforms
from .decode import mec_decode
from .metrics import confusion_matrix, evaluate_predictions
from .models import build_model


def main():
    ap = parse_args(__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    d, device = cfg["data"], get_device()
    classes = load_classes(d["splits_dir"])
    W, polarity, _ = emotion_tensors(classes, cfg["loss"]["polarity_C"])
    ds = EmotionCSV(Path(d["splits_dir"]) / f"{a.split}.csv", d["root"], classes,
                    build_transforms(False, d["image_size"], d["resize"]))
    dl = DataLoader(ds, batch_size=cfg["optim"]["batch_size"], num_workers=d["num_workers"])
    model = build_model(cfg["model"]["name"], len(classes), False).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    probs, labels = predict_probs(model, dl, device)
    print(json.dumps({"argmax": evaluate_predictions(labels, probs.argmax(1), W, polarity),
                      "mec": evaluate_predictions(labels, mec_decode(probs, W), W, polarity)}, indent=2))
    print("confusion matrix (rows=true):", classes)
    print(confusion_matrix(labels, probs.argmax(1), len(classes)))


if __name__ == "__main__":
    main()
