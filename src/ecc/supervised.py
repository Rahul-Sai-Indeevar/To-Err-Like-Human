"""Supervised training with L_c = L_CE + alpha * L_ListMLE (paper Tables 2-4)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

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
    opt = torch.optim.SGD(model.parameters(), lr=o["lr"], momentum=o["momentum"],
                          weight_decay=o["weight_decay"])
    use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, best_epoch, sel = -1.0, -1, cfg["eval"]["select_by"]
    for epoch in range(o["epochs"]):
        model.train()
        t0, run_loss = time.time(), 0.0
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
    run(load_config(a.config, a.set))


if __name__ == "__main__":
    main()
