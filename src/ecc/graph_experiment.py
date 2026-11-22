"""Reproducible GSP comparison: frozen ImageNet features and full fine-tuning.

python -m ecc.graph_experiment --stage features
python -m ecc.graph_experiment --stage probe
python -m ecc.graph_experiment --stage finetune --epochs 6
"""

from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms as T
from PIL import Image

from .common import set_seed, predict_probs
from .data import EmotionCSV, build_transforms
from .emotion import build_cost_matrix, build_rank_matrix, polarity_vector
from .graph import build_loss, heat_filter
from .metrics import evaluate_predictions, confusion_matrix, graph_error_metrics

ROOT = Path("data/emoset_graph")
OUT = Path("outputs/graph_experiment")


class Uint8Normalize(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        return (x.float() / 255 - self.mean) / self.std


class CachedImages(torch.utils.data.Dataset):
    """Lossless uint8 cache of resized inputs; random crop/flip remain online.

    Avoids repeated JPEG decoding across 54 backbone-training epochs. Cache is
    identified by the exact manifest hash and stores no learned quantities.
    """

    def __init__(self, split, classes, train=False):
        ds = EmotionCSV(ROOT / f"{split}.csv", ROOT, classes)
        digest = hashlib.sha256((ROOT / "manifest.json").read_bytes()).hexdigest()[:16]
        path = ROOT / f"{split}_192_{digest}.npy"
        if not path.exists():
            temp = path.with_suffix(".part.npy")
            array = np.lib.format.open_memmap(
                temp, mode="w+", dtype=np.uint8, shape=(len(ds), 3, 192, 192)
            )
            resize = T.Resize((192, 192))
            for i, (rel, _) in enumerate(ds.items):
                with Image.open(ROOT / rel) as im:
                    array[i] = np.asarray(resize(im.convert("RGB"))).transpose(2, 0, 1)
            array.flush()
            del array
            temp.replace(path)
        self.array = np.load(path, mmap_mode="c")
        self.labels = ds.targets
        self.train = train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(self.array[i])
        a, b = torch.randint(33, (2,)).tolist() if self.train else (16, 16)
        x = x[:, a : a + 160, b : b + 160]
        if self.train and torch.rand(()) < 0.5:
            x = x.flip(-1)
        return x, int(self.labels[i])


def cached_loaders(classes, batch):
    return {
        s: DataLoader(
            CachedImages(s, classes, s == "train"),
            batch_size=batch,
            shuffle=s == "train",
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        for s in ("train", "val", "test")
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf8")


def loaders(classes, train=False, batch=64):
    return {
        s: DataLoader(
            EmotionCSV(
                ROOT / f"{s}.csv",
                ROOT,
                classes,
                build_transforms(train and s == "train", 160, 192),
            ),
            batch_size=batch,
            shuffle=train and s == "train",
            num_workers=0,
        )
        for s in ("train", "val", "test")
    }


def extract(classes, device):
    OUT.mkdir(parents=True, exist_ok=True)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    model.fc = torch.nn.Identity()
    features = {}
    with torch.no_grad():
        for split, loader in loaders(classes).items():
            xs = []
            ys = []
            for i, (x, y) in enumerate(loader):
                xs.append(model(x.to(device)).cpu())
                ys.append(y)
                if i % 20 == 0:
                    print("features", split, i, "/", len(loader), flush=True)
            features[split] = (torch.cat(xs), torch.cat(ys))
    features["metadata"] = {
        "classes": classes,
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "image_size": 160,
        "manifest_sha256": hashlib.sha256(
            (ROOT / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    torch.save(features, "data/emoset_graph/features.pt")


def evaluate(p, y, W, polarity):
    m = evaluate_predictions(y, p.argmax(1), W, polarity)
    m.update(graph_error_metrics(p, y, W))
    m["n_errors"] = int((y != p.argmax(1)).sum())
    return m


def fit(
    model, criterion, train, val, device, epochs, lr, seed, W, polarity, features=False
):
    set_seed(seed)
    criterion = criterion.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = -float("inf")
    state = None
    history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for x, y in train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(y)
            count += len(y)
        p, y = predict_probs(model, val, device)
        m = evaluate_predictions(y, p.argmax(1), W, polarity)
        history.append({"epoch": epoch + 1, "train_loss": total / count, "val": m})
        if m["ecc"] > best:
            best = m["ecc"]
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        if not features or epoch % 20 == 0:
            print("epoch", epoch + 1, "val", m, flush=True)
    model.load_state_dict(state)
    return history, best_epoch, time.perf_counter() - start


def candidates():
    return [
        dict(name="ce", kind="ce", alpha=0.0),
        dict(name="listmle", kind="listmle", alpha=0.2),
        dict(name="graph", kind="graph", alpha=1.0, gamma=1.0, beta=0.01),
        dict(name="graph_no_smooth", kind="graph", alpha=1.0, gamma=1.0, beta=0.0),
        dict(name="graph_flat", kind="graph", alpha=1.0, gamma=0.0, beta=0.0),
        dict(
            name="uniform_graph",
            kind="graph",
            alpha=1.0,
            gamma=1.0,
            beta=0.01,
            topology="uniform",
        ),
        dict(
            name="uniform_trace_matched",
            kind="graph",
            alpha=1.0,
            gamma=1.0,
            beta=0.01,
            topology="uniform_trace_matched",
        ),
    ]


def run(args):
    torch.set_num_threads(2 if args.stage == "probe" else 6)
    classes = (ROOT / "classes.txt").read_text().splitlines()
    W = build_cost_matrix(classes)
    rank = build_rank_matrix(W)
    polarity = polarity_vector(classes)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    print("device", device, flush=True)
    if args.stage == "features":
        return extract(classes, device)
    stage_out = OUT / args.stage
    stage_out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "manifest.json").read_text())
    protocol_path = stage_out / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        if (
            previous["args"]["epochs"] != args.epochs
            or previous["manifest"] != manifest
            or previous["device"] != str(device)
            or previous["torch"] != torch.__version__
        ):
            raise ValueError(
                "Existing stage uses a different protocol; use a fresh output directory"
            )
    write_json(
        stage_out / "protocol.json",
        {
            "args": vars(args),
            "device": str(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "classes": classes,
            "manifest": manifest,
            "selection": "maximum validation ECC; fixed candidates; test evaluated only after checkpoint selection",
            "image_size": 160,
            "resize": 192,
            "pretrained": "ResNet18_Weights.IMAGENET1K_V1",
            "optimizer": {
                "name": "AdamW",
                "lr": 0.003 if args.stage == "probe" else 0.0001,
                "weight_decay": 1e-4,
                "batch_size": 256 if args.stage == "probe" else args.batch_size,
                "gradient_clip_norm": 5.0,
            },
            "train_augmentation": "none (fixed features)"
            if args.stage == "probe"
            else "random crop 160 of resized 192; horizontal flip p=0.5",
            "checkpoint_model": "Linear(512,8)"
            if args.stage == "probe"
            else "Sequential(Uint8Normalize(), ResNet18(num_classes=8))",
        },
    )
    results = []
    methods = candidates() if args.stage == "probe" else candidates()[:3]
    cache = None
    if args.stage == "probe":
        cache = torch.load(ROOT / "features.pt", weights_only=False)
        digest = hashlib.sha256((ROOT / "manifest.json").read_bytes()).hexdigest()
        if cache["metadata"]["manifest_sha256"] != digest:
            raise ValueError("Stale feature cache")
    for seed in args.seeds:
        for cfg in methods:
            dest = stage_out / f"{cfg['name']}_seed{seed}"
            if (dest / "result.json").exists():
                saved = json.loads((dest / "result.json").read_text())
                if saved["config"] != cfg:
                    raise ValueError(f"Configuration changed for {dest}")
                results.append(saved)
                continue
            dest.mkdir(parents=True, exist_ok=True)
            set_seed(seed)
            if cache is not None:
                model = torch.nn.Linear(512, 8).to(device)
                ls = {
                    s: DataLoader(
                        TensorDataset(*cache[s]), batch_size=256, shuffle=s == "train"
                    )
                    for s in ("train", "val", "test")
                }
            else:
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                model.fc = torch.nn.Linear(model.fc.in_features, 8)
                model = torch.nn.Sequential(Uint8Normalize(), model).to(device)
                ls = cached_loaders(classes, args.batch_size)
            print(args.stage, cfg, "seed", seed, flush=True)
            history, epoch, seconds = fit(
                model,
                build_loss(cfg, W, rank),
                ls["train"],
                ls["val"],
                device,
                args.epochs,
                0.003 if cache is not None else 0.0001,
                seed,
                W,
                polarity,
                cache is not None,
            )
            vp, vy = predict_probs(model, ls["val"], device)
            heat_times = [0.0, 0.1, 0.3, 1.0]
            heat_val = [
                evaluate_predictions(vy, heat_filter(vp, W, t).argmax(1), W, polarity)[
                    "ecc"
                ]
                for t in heat_times
            ]
            chosen_time = heat_times[int(np.argmax(heat_val))]
            p, y = predict_probs(model, ls["test"], device)
            result = {
                "method": cfg["name"],
                "seed": seed,
                "config": cfg,
                "best_epoch": epoch,
                "seconds": seconds,
                "test": evaluate(p, y, W, polarity),
                "heat_filter": {
                    "times": heat_times,
                    "val_ecc": heat_val,
                    "chosen_time": chosen_time,
                    "test": evaluate(heat_filter(p, W, chosen_time), y, W, polarity),
                },
            }
            write_json(dest / "history.json", history)
            write_json(dest / "result.json", result)
            torch.save(model.state_dict(), dest / "best.pt")
            np.savez_compressed(dest / "predictions.npz", probabilities=p, labels=y)
            np.savetxt(
                dest / "confusion.csv",
                confusion_matrix(y, p.argmax(1), 8),
                fmt="%d",
                delimiter=",",
            )
            results.append(result)
            print("RESULT", result, flush=True)
    write_json(stage_out / "results.json", results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "probe", "finetune"], required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()
    if args.epochs is None:
        args.epochs = 60 if args.stage == "probe" else 6
    if args.epochs < 1:
        ap.error("epochs must be positive")
    run(args)


if __name__ == "__main__":
    main()
