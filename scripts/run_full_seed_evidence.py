"""Full-data fresh paired-seed CE/graph replication; one GPU task at a time.

This opt-in runner does not alter the paper training defaults. All geometry and
optimizer settings are locked before evaluation; checkpoints use validation ECC.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import Resize

from ecc.emotion import build_cost_matrix, polarity_vector
from ecc.graph import GraphSpectralLoss, graph_basis
from ecc.metrics import evaluate_predictions, graph_error_metrics


METHODS = ("ce", "graph")
SPLITS = ("train", "val", "test")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    """Hash an input cache with bounded memory, including the .npy header."""
    sha = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")
    temp.replace(path)


@contextmanager
def gpu_lock(path):
    """Advisory process lock: all launches of this runner share one lock file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another experiment holds the single-GPU lock") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class IsotropicMatchedLoss(nn.Module):
    """Match trace of BOTH graph quadratics, removing class geometry.

    Q=L^+ is replaced by q(I-J/K), q=tr(Q)/(K-1), and L by
    d(I-J/K), d=tr(L)/(K-1). Matching only tr(L) is insufficient.
    """
    def __init__(self, W, alpha=0.3, beta=0.01):
        super().__init__()
        _, _, values, _ = graph_basis(W)
        positive = values[values > 1e-10]
        self.q = float(np.mean(1 / positive))
        self.d = float(np.mean(positive))
        self.alpha, self.beta = alpha, beta

    def forward(self, logits, target):
        p = logits.softmax(1)
        y = F.one_hot(target, p.shape[1]).to(p.dtype)
        fidelity = self.q * (p - y).square().sum(1).mean()
        smooth = self.d * (p - 1 / p.shape[1]).square().sum(1).mean()
        return F.cross_entropy(logits, target) + self.alpha * (fidelity + self.beta * smooth)


def wrong_permutation(W, seed):
    """One independently preassigned wrong labeling per seed, never test-picked."""
    rng = np.random.default_rng(1717 + seed)
    n = len(W)
    while True:
        p = rng.permutation(n)
        if not np.array_equal(W[np.ix_(p, p)], W):
            return p


def loss_for(method, W, seed):
    if method not in METHODS:
        raise ValueError(f"Full-data replication permits only {METHODS}")
    if method == "ce":
        return nn.CrossEntropyLoss()
    if method == "isotropic":
        return IsotropicMatchedLoss(W)
    if method == "permuted":
        p = wrong_permutation(W, seed)
        W = W[np.ix_(p, p)]
    if method not in ("graph", "permuted"):
        raise ValueError(method)
    return GraphSpectralLoss(W, alpha=0.3, gamma=1, beta=0.01)


def augmentation_plan(n, epochs, seed):
    """Materialize independent CPU plans once; every method receives identical data."""
    rng = np.random.default_rng(400_000 + seed)
    plans = []
    sha = hashlib.sha256()
    for _ in range(epochs):
        plan = (rng.permutation(n).astype(np.int64),
                rng.integers(0, 33, size=(n, 2), dtype=np.int64),
                (rng.random(n) < 0.5))
        for x in plan:
            sha.update(x.tobytes())
        plans.append(plan)
    return plans, sha.hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for key, value in state.items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(root, classes, splits):
    datasets, metadata = {}, {}
    resize = Resize((192, 192))
    for split in splits:
        manifest = root / f"{split}.csv"
        rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")))
        if not rows:
            raise ValueError(f"Empty {manifest}")
        labels = np.array([classes.index(row["label"].lower()) for row in rows], dtype=np.int64)
        h = digest(manifest.read_bytes() + b"PIL_EXIF_RGB_torchvisionResize192_antialias_v1")
        cache = root / f"{split}_seed_evidence_{h[:16]}.npy"
        if not cache.exists():
            temp = cache.with_suffix(".part.npy")
            data = np.lib.format.open_memmap(temp, mode="w+", dtype=np.uint8,
                                             shape=(len(rows), 3, 192, 192))
            for i, row in enumerate(rows):
                image_path = (root / row["path"]).resolve()
                if not image_path.is_relative_to(root.resolve()):
                    raise ValueError("Dataset path escapes root")
                with Image.open(image_path) as im:
                    image = resize(ImageOps.exif_transpose(im).convert("RGB"))
                    data[i] = np.asarray(image).transpose(2, 0, 1)
                if i % 2000 == 0:
                    print("cache", split, i, "/", len(rows), flush=True)
            data.flush()
            del data
            temp.replace(cache)
        array = np.load(cache, mmap_mode="r")
        if array.shape != (len(rows), 3, 192, 192) or array.dtype != np.uint8:
            raise ValueError("Cache shape/dtype disagrees with manifest")
        datasets[split] = (array, labels)
        metadata[split] = {"n": len(rows), "csv_sha256": digest(manifest.read_bytes()),
                           "cache_key": h, "cache_sha256": file_digest(cache),
                           "cache_bytes": cache.stat().st_size,
                           "class_counts": np.bincount(labels, minlength=len(classes)).tolist()}
    return datasets, metadata


class Images:
    """Keep image arrays mapped on disk; allocate and transfer just one batch.

    In particular, never call np.array(images, copy=True) or transfer a complete
    split to CUDA. The full resized dataset exceeds available RAM/VRAM.
    """
    def __init__(self, datasets, device):
        self.device = device
        self.data, self.labels = {}, {}
        self.preloaded = False
        for name, (images, labels) in datasets.items():
            self.data[name] = images
            self.labels[name] = torch.from_numpy(labels.copy())
        self.mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)
        self.axis = torch.arange(160, device=device)

    def batch(self, split, index, crop=None, flip=None):
        source = self.data[split]
        ix = np.asarray(index, dtype=np.int64)
        batch = np.ascontiguousarray(source[ix])
        x = torch.from_numpy(batch).to(self.device)
        y = self.labels[split][torch.from_numpy(ix)].to(self.device)
        if crop is None:
            x = x[:, :, 16:176, 16:176]
        else:
            crop = torch.as_tensor(crop, device=self.device)
            flips = torch.as_tensor(flip, device=self.device)
            yy = crop[:, 0, None, None] + self.axis[None, :, None]
            xx = crop[:, 1, None, None] + torch.where(flips[:, None, None],
                    159 - self.axis[None, None, :], self.axis[None, None, :])
            offsets = (yy * 192 + xx).flatten(1)
            x = x.flatten(2).gather(2, offsets[:, None].expand(-1, 3, -1)).view(-1, 3, 160, 160)
        return (x.float() / 255 - self.mean) / self.std, y


@torch.no_grad()
def predict(model, images, split, batch_size, amp):
    model.eval()
    probabilities, labels = [], []
    n = len(images.labels[split])
    for start in range(0, n, batch_size):
        x, y = images.batch(split, np.arange(start, min(n, start + batch_size)))
        with torch.autocast(device_type=images.device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        probabilities.append(logits.float().softmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def evaluate(probabilities, labels, W, polarity):
    values = evaluate_predictions(labels, probabilities.argmax(1), W, polarity)
    values.update(graph_error_metrics(probabilities, labels, W))
    return values


def save_probabilities(path, p, y):
    temp = path.with_suffix(".part.npz")
    np.savez_compressed(temp, probabilities=p, labels=y)
    temp.replace(path)


def fit_one(args, method, seed, state, initial_hash, plans, plan_hash, images, W, polarity, protocol_hash):
    dest = args.out / f"{method}_seed{seed}"
    result_file = dest / "result.json"
    if result_file.exists():
        result = json.loads(result_file.read_text())
        if result["protocol_hash"] != protocol_hash or result["initial_state_sha256"] != initial_hash or result["augmentation_plan_sha256"] != plan_hash:
            raise ValueError(f"Completed result protocol mismatch: {dest}")
        print("SKIP", dest, flush=True)
        return result
    dest.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    device = images.device
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(W))
    model.load_state_dict(state)
    model.to(device)
    criterion = loss_for(method, W, seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    best_ecc, best_epoch, best_state, history = -float("inf"), None, None, []
    start_time = time.perf_counter()
    print("START", method, seed, flush=True)
    for epoch, (order, crop, flip) in enumerate(plans, 1):
        epoch_start = time.perf_counter()
        amp_scale_start = scaler.get_scale()
        skipped_steps = 0
        attempted_steps = 0
        model.train()
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            stop = min(len(order), start + args.batch_size)
            index = order[start:stop]
            x, y = images.batch("train", index, crop[start:stop], flip[start:stop])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = criterion(logits.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {method} seed{seed} epoch{epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            attempted_steps += 1
            # With this one optimizer and the standard GradScaler, a scale
            # decrease records an overflow-triggered skipped optimizer step.
            skipped_steps += int(amp and scaler.get_scale() < scale_before)
            total_loss += float(loss.detach()) * len(y)
        train_seconds = time.perf_counter() - epoch_start
        vp, vy = predict(model, images, "val", args.batch_size, amp)
        val = evaluate(vp, vy, W, polarity)
        history.append({"epoch": epoch, "train_loss": total_loss / len(order), "val": val})
        if val["ecc"] > best_ecc:
            best_ecc, best_epoch = val["ecc"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history[-1].update({"epoch_seconds": time.perf_counter() - epoch_start,
                            "train_seconds": train_seconds,
                            "optimizer_steps_attempted": attempted_steps,
                            "optimizer_steps_skipped_scale_decrease": skipped_steps,
                            "amp_scale_start": amp_scale_start,
                            "amp_scale_end": scaler.get_scale()})
        atomic_json(dest / "history.json", history)
        print("EPOCH", method, seed, epoch, "valECC", round(val["ecc"], 5),
              "seconds", round(history[-1]["epoch_seconds"], 1),
              "skipped_steps", skipped_steps,
              "elapsed", round(time.perf_counter() - start_time, 1), flush=True)
    model.load_state_dict(best_state)
    temp = dest / "best.part.pt"
    torch.save(best_state, temp)
    temp.replace(dest / "best.pt")
    vp, vy = predict(model, images, "val", args.batch_size, amp)
    save_probabilities(dest / "val_predictions.npz", vp, vy)
    result = {"method": method, "seed": seed, "protocol_hash": protocol_hash,
              "best_epoch": best_epoch, "best_validation_ecc": best_ecc,
              "initial_state_sha256": initial_hash, "augmentation_plan_sha256": plan_hash,
              "wrong_label_permutation": wrong_permutation(W, seed).tolist() if method == "permuted" else None,
              "val": evaluate(vp, vy, W, polarity), "seconds": time.perf_counter() - start_time}
    if not args.skip_test:
        p, y = predict(model, images, "test", args.batch_size, amp)
        save_probabilities(dest / "predictions.npz", p, y)
        result["test"] = evaluate(p, y, W, polarity)
    atomic_json(result_file, result)
    print("RESULT", json.dumps(result), flush=True)
    del model, optimizer, scaler, criterion, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run(args):
    source = Path(__file__).read_bytes()
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    classes = [c.strip() for c in (args.data_root / "classes.txt").read_text().splitlines() if c.strip()]
    W, polarity = build_cost_matrix(classes), polarity_vector(classes)
    datasets, metadata = load_dataset(args.data_root, classes, SPLITS[:2] if args.skip_test else SPLITS)
    if args.cache_only:
        print("Cache complete", json.dumps(metadata), flush=True)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    import ecc.graph, ecc.metrics, ecc.emotion
    sources = {"runner": digest(source)}
    module_sources = {}
    for module in (ecc.graph, ecc.metrics, ecc.emotion):
        module_sources[module.__name__] = Path(module.__file__).read_bytes()
        sources[module.__name__] = digest(module_sources[module.__name__])
    protocol = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "classes": classes, "datasets": metadata, "sources": sources,
                "data_manifest_sha256": digest((args.data_root / "manifest.json").read_bytes()) if (args.data_root / "manifest.json").exists() else None,
                "torch": torch.__version__, "torchvision": torchvision.__version__,
                "numpy": np.__version__, "python": platform.python_version(),
                "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "loss": {"alpha": .3, "gamma": 1, "beta": .01},
                "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-4, "clip_norm": 5},
                "selection": "highest validation ECC (earliest tied epoch); fixed maximum epochs; final test once",
                "transforms": "PIL EXIF transpose RGB; Resize192x192; random crop160/flip0.5 train; center160 val/test",
                "initialization": "ImageNet1K ResNet18 common pretrained backbone initialization plus seeded linear head; all parameters trained; identical state per paired methods",
                "memory_access": "disk-backed uint8 memmap; batch-sized contiguous CPU copy then transfer to GPU; no whole-split image copy or preload",
                "step_diagnostics": "overflow/skipped-step count inferred from standard single-optimizer GradScaler scale decreases; logging only",
                "seed_scope": "new initialization/augmentation seeds, same fixed dataset split; not independent data populations",
                "deterministic_algorithms": True}
    protocol_hash = digest(canonical(protocol))
    protocol["protocol_hash"] = protocol_hash
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing output uses different code, runtime, data or settings; choose fresh --out")
    atomic_json(protocol_path, protocol)
    (args.out / "runner_source.py").write_bytes(source)
    for name, contents in module_sources.items():
        (args.out / (name.split(".")[-1] + "_source.py")).write_bytes(contents)
    with gpu_lock(args.lock_file):
        images = Images(datasets, device)
        print("DEVICE", device, "GPUpreload", images.preloaded, flush=True)
        all_results = []
        for seed in args.seeds:
            seed_everything(seed)
            initial = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            initial.fc = nn.Linear(initial.fc.in_features, len(classes))
            state = {k: v.detach().cpu().clone() for k, v in initial.state_dict().items()}
            initial_hash = state_digest(state)
            del initial
            plans, plan_hash = augmentation_plan(len(datasets["train"][1]), args.epochs, seed)
            for method in args.methods:
                all_results.append(fit_one(args, method, seed, state, initial_hash, plans, plan_hash,
                                           images, W, polarity, protocol_hash))
                atomic_json(args.out / "results.json", all_results)
    print("COMPLETE", len(all_results), "runs", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data/kaggle_full_8epoch/data/emoset_full"))
    p.add_argument("--out", type=Path, default=Path("outputs/full_seed_replication"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(3, 9)))
    p.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--skip-test", action="store_true", help="Benchmark using validation only")
    p.add_argument("--lock-file", type=Path, default=Path("work/single_gpu.lock"))
    args = p.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 2 or len(args.seeds) != len(set(args.seeds)) or len(args.methods) != len(set(args.methods)):
        p.error("Positive epochs, batch>=2, unique seeds and methods required")
    if args.skip_test and len(args.methods) != 1:
        p.error("Benchmark --skip-test requires exactly one method")
    return args


def main():
    run(parse_args())


if __name__ == "__main__":
    main()


