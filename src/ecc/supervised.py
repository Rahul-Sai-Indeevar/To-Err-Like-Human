"""Supervised training with L_c = L_CE + alpha * L_ListMLE (paper Tables 2-4).
Supports resume-from-checkpoint for long runs (e.g. Kaggle 12-hour sessions).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from .common import (append_jsonl, emotion_tensors, get_device, load_classes,
                     predict_probs, set_seed)
from .config import load_config, parse_args
from .data import EmotionCSV, build_transforms
from .decode import mec_decode
from .losses import CombinedLoss
from .metrics import confusion_matrix, evaluate_predictions
from .models import build_model


def lr_at(epoch: int, it: int, iters: int, o: dict) -> float:
    if epoch < o["warmup_epochs"]:
        return o["lr"] * (epoch * iters + it + 1) / (o["warmup_epochs"] * iters)
    return o["lr_after_decay"] if epoch >= o["decay_epoch"] else o["lr"]


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = get_device()
    d, o = cfg["data"], cfg["optim"]
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    classes = load_classes(d["splits_dir"])
    W, polarity, rank = emotion_tensors(classes, cfg["loss"]["polarity_C"])
    mk = lambda split, train: EmotionCSV(Path(d["splits_dir"]) / f"{split}.csv", d["root"], classes,
                                         build_transforms(train, d["image_size"], d["resize"]))
    loader = lambda ds, train: DataLoader(ds, batch_size=o["batch_size"], shuffle=train,
                                          num_workers=d["num_workers"], drop_last=train,
                                          pin_memory=device.type == "cuda")
    train_dl, val_dl, test_dl = (loader(mk("train", True), True), loader(mk("val", False), False),
                                 loader(mk("test", False), False))

    model = build_model(cfg["model"]["name"], len(classes), cfg["model"]["pretrained"]).to(device)
    crit = CombinedLoss(rank, cfg["loss"]["alpha"], cfg["loss"]["listmle_reduction"]).to(device)
    switch_epoch = cfg["loss"].get("switch_epoch", 0)
    opt = torch.optim.SGD(model.parameters(), lr=o["lr"], momentum=o["momentum"],
                          weight_decay=o["weight_decay"])
    use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- resume ----
    start_epoch = 0
    best, best_epoch = -1.0, -1
    sel = cfg["eval"]["select_by"]
    resume_from = cfg.get("resume_from")
    ckpt_path = resume_from or str(latest_checkpoint(out))
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = load_checkpoint(ckpt_path, model, opt, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        best = ckpt.get("best_metric", -1.0)
        best_epoch = ckpt.get("best_epoch", -1)
        print(f"Resumed from {ckpt_path} at epoch {start_epoch} (best {sel}={best:.4f})")

    for epoch in range(start_epoch, o["epochs"]):
        model.train()
        t0, run_loss = time.time(), 0.0
        crit.alpha = 0.0 if epoch < switch_epoch else cfg["loss"]["alpha"]
        for it, (x, y) in enumerate(train_dl):
            for g in opt.param_groups:
                g["lr"] = lr_at(epoch, it, len(train_dl), o)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
        probs, labels = predict_probs(model, val_dl, device, use_amp)
        m = evaluate_predictions(labels, probs.argmax(1), W, polarity)
        append_jsonl(out / "history.jsonl", {"epoch": epoch, "loss": run_loss / len(train_dl),
                                             "val": m, "sec": time.time() - t0})
        print(f"[{epoch+1}/{o['epochs']}] loss {run_loss/len(train_dl):.4f} val " +
              " ".join(f"{k} {v:.2f}" for k, v in m.items()), flush=True)
        if m[sel] > best:
            best, best_epoch = m[sel], epoch
            torch.save(model.state_dict(), out / "best.pt")
        # Save a resumable checkpoint every epoch.
        save_checkpoint(out / "checkpoint.pt", model, opt, epoch, best, best_epoch)

    # final test with the best-on-val checkpoint
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    probs, labels = predict_probs(model, test_dl, device, use_amp)
    res = {"best_epoch": best_epoch, "select_by": sel,
           "test_argmax": evaluate_predictions(labels, probs.argmax(1), W, polarity),
           "test_mec_decode": evaluate_predictions(labels, mec_decode(probs, W), W, polarity)}
    np.savetxt(out / "confusion_matrix.csv", confusion_matrix(labels, probs.argmax(1), len(classes)),
               fmt="%d", delimiter=",", header=",".join(classes), comments="")
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


def main():
    a = parse_args(__doc__).parse_args()
    cfg = load_config(a.config, a.set)
    if a.resume:
        cfg["resume_from"] = a.resume
    run(cfg)


if __name__ == "__main__":
    main()
