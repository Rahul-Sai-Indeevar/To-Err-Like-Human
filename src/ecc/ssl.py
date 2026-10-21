"""FixMatch with an EMC-adaptive confidence threshold (paper Sec. 4.2.1).

Each epoch the pseudo-labels from weakly augmented unlabeled images are treated as ground truth and
the predictions on strongly augmented views as 'model output'; EMC of that pair
(computed from a confusion matrix accumulated during the epoch) sets the next epoch's threshold.

Supports resume-from-checkpoint for long runs (e.g. Kaggle 12-hour sessions).

NOT included: S^2-VER (paper Sec. 4.2.2). Use its official code and plug in L_c there.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from .checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from .common import (append_jsonl, emotion_tensors, get_device, load_classes,
                     predict_probs, set_seed)
from .config import load_config, parse_args
from .data import (EmotionCSV, TwoViewTransform, build_transforms,
                   class_balanced_indices)
from .decode import mec_decode
from .losses import CombinedLoss
from .metrics import emc as emc_fn
from .metrics import evaluate_predictions
from .models import build_model


def adaptive_threshold(tau: float, eps: float, emc: float, lo: float = 0.7, hi: float = 0.98,
                       rule: str = "ratio") -> float:
    """Threshold update from the epoch's EMC (paper Eq. 5).

    tau'_t = tau * eps / EMC_t, clipped to [lo, hi]. eps is a dataset constant
    (0.5 for FI, 0.4 for EmoSet), NOT Euler's number. Low EMC -> threshold moves
    up toward tau_max (fewer pseudo-labels); high EMC -> threshold moves down
    toward tau_min (more pseudo-labels). EMC == eps leaves tau unchanged.
    """
    if rule == "fixed":
        return tau
    if rule != "ratio":
        raise ValueError(rule)
    return float(min(hi, max(lo, tau * eps / max(emc, 1e-6))))


class EMA:
    def __init__(self, model, decay=0.999):
        self.model, self.decay = copy.deepcopy(model).eval(), decay
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for eb, b in zip(self.model.buffers(), model.buffers()):
            eb.copy_(b)

    def state_dict(self):
        return {"model": self.model.state_dict(), "decay": self.decay}

    def load_state_dict(self, ckpt, map_location=None):
        self.model.load_state_dict(ckpt["model"], strict=False)


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = get_device()
    d, o, s = cfg["data"], cfg["optim"], cfg["ssl"]
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    classes = load_classes(d["splits_dir"])
    N = len(classes)
    W, polarity, rank = emotion_tensors(classes, cfg["loss"]["polarity_C"])
    sp = Path(d["splits_dir"])
    targets = EmotionCSV(sp / "train.csv", d["root"], classes).targets
    lab_idx = class_balanced_indices(targets, s["n_labeled"] // N, cfg["seed"])
    unl_idx = np.setdiff1d(np.arange(len(targets)), lab_idx)
    print(f"labeled={len(lab_idx)} unlabeled={len(unl_idx)}")

    lab_ds = EmotionCSV(sp / "train.csv", d["root"], classes,
                        build_transforms(True, d["image_size"], d["resize"]), lab_idx)
    unl_ds = EmotionCSV(sp / "train.csv", d["root"], classes,
                        TwoViewTransform(d["image_size"], d["resize"]), unl_idx)
    ipe, bs, mu = s["iters_per_epoch"], o["batch_size"], s["mu"]
    mkdl = lambda ds, b: DataLoader(ds, batch_size=b, num_workers=d["num_workers"], drop_last=True,
                                    sampler=RandomSampler(ds, replacement=True, num_samples=ipe * b),
                                    pin_memory=device.type == "cuda")
    lab_dl, unl_dl = mkdl(lab_ds, bs), mkdl(unl_ds, bs * mu)
    ev = lambda split: DataLoader(EmotionCSV(sp / f"{split}.csv", d["root"], classes,
                                             build_transforms(False, d["image_size"], d["resize"])),
                                  batch_size=bs, num_workers=d["num_workers"])
    val_dl, test_dl = ev("val"), ev("test")

    model = build_model(cfg["model"]["name"], N, cfg["model"]["pretrained"]).to(device)
    ema = EMA(model, s["ema_decay"])
    sup = CombinedLoss(rank, cfg["loss"]["alpha"], cfg["loss"]["listmle_reduction"]).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=o["lr"], momentum=o["momentum"],
                          weight_decay=o["weight_decay"], nesterov=True)
    total = o["epochs"] * ipe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: math.cos(7 * math.pi * k / (16 * total)))

    tau, best, best_epoch, sel = s["tau"], -1.0, -1, cfg["eval"]["select_by"]
    start_epoch = 0

    # ---- resume ----
    resume_from = cfg.get("resume_from")
    ckpt_path = resume_from or str(latest_checkpoint(out))
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = load_checkpoint(ckpt_path, model, opt, sched, map_location=device)
        ema.load_state_dict(ckpt.get("ema", {}), map_location=device)
        start_epoch = ckpt["epoch"] + 1
        tau = ckpt.get("tau", s["tau"])
        best = ckpt.get("best_metric", -1.0)
        best_epoch = ckpt.get("best_epoch", -1)
        print(f"Resumed from {ckpt_path} at epoch {start_epoch} "
              f"(tau={tau:.3f}, best {sel}={best:.4f})")

    for epoch in range(start_epoch, o["epochs"]):
        model.train()
        S = torch.zeros(N * N, dtype=torch.long, device=device)  # weak-pseudo-label vs strong-pred
        stats = {"loss": 0.0, "mask": 0.0, "pl_acc": 0.0, "pl_n": 0}
        t0 = time.time()
        for (xl, yl), ((xw, xs), yu) in zip(lab_dl, unl_dl):
            xl, yl = xl.to(device), yl.to(device)
            xw, xs, yu = xw.to(device), xs.to(device), yu.to(device)
            logits = model(torch.cat([xl, xw, xs]))
            ll, lw, ls = logits.split([len(xl), len(xw), len(xs)])
            with torch.no_grad():
                conf, pseudo = lw.detach().softmax(1).max(1)
                mask = (conf >= tau).float()
                S += torch.bincount(pseudo * N + ls.argmax(1), minlength=N * N)
            loss_u = (F.cross_entropy(ls, pseudo, reduction="none") * mask).mean()
            loss = sup(ll, yl) + s["lambda_u"] * loss_u
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ema.update(model)
            stats["loss"] += loss.item()
            stats["mask"] += mask.mean().item()
            keep = mask.bool()
            stats["pl_acc"] += (pseudo[keep] == yu[keep]).sum().item()  # monitoring only
            stats["pl_n"] += keep.sum().item()

        emc_t = emc_fn(S.view(N, N).cpu().numpy(), W)
        tau_next = adaptive_threshold(s["tau"], s["eps"], emc_t, s["tau_min"], s["tau_max"], s["rule"])
        probs, labels = predict_probs(ema.model, val_dl, device)
        m = evaluate_predictions(labels, probs.argmax(1), W, polarity)
        rec = {"epoch": epoch, "loss": stats["loss"] / ipe, "tau": tau, "emc_weak_vs_strong": emc_t,
               "mask_rate": stats["mask"] / ipe, "pseudo_acc": stats["pl_acc"] / max(stats["pl_n"], 1),
               "val": m, "sec": time.time() - t0}
        append_jsonl(out / "history.jsonl", rec)
        print(f"[{epoch+1}/{o['epochs']}] tau {tau:.3f} emc {100*emc_t:.1f} mask {rec['mask_rate']:.2f} "
              f"pl_acc {rec['pseudo_acc']:.2f} val acc {m['acc']:.2f} ecc {m['ecc']:.2f}", flush=True)
        tau = tau_next
        if m[sel] > best:
            best, best_epoch = m[sel], epoch
            torch.save(ema.model.state_dict(), out / "best.pt")
        # Save a resumable checkpoint every epoch.
        save_checkpoint(out / "checkpoint.pt", model, opt, epoch, best, best_epoch,
                        extra={"ema": ema.state_dict(), "tau": tau_next},
                        scheduler=sched)

    ema.model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    probs, labels = predict_probs(ema.model, test_dl, device)
    res = {"best_epoch": best_epoch, "rule": s["rule"], "n_labeled": s["n_labeled"],
           "test_argmax": evaluate_predictions(labels, probs.argmax(1), W, polarity),
           "test_mec_decode": evaluate_predictions(labels, mec_decode(probs, W), W, polarity)}
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
