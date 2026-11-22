"""Paired preprocessing, artifact writing and evaluation shared by the studies."""
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision.transforms import Resize

from .metrics import evaluate_predictions

SPLITS = ("train", "val", "test")

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


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


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


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
    """Decode the prepared splits without writing duplicate image caches."""
    root = root.resolve()
    datasets, metadata = {}, {}
    resize = Resize((192, 192))
    for split in splits:
        manifest = root / f"{split}.csv"
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"Empty manifest: {manifest}")
        labels = np.array([classes.index(row["label"].lower()) for row in rows], dtype=np.int64)
        pixels = np.empty((len(rows), 3, 192, 192), dtype=np.uint8)
        for i, row in enumerate(rows):
            image_path = (root / row["path"]).resolve()
            if not image_path.is_relative_to(root):
                raise ValueError(f"Dataset path escapes root: {image_path}")
            with Image.open(image_path) as source:
                image = resize(ImageOps.exif_transpose(source).convert("RGB"))
                pixels[i] = np.asarray(image).transpose(2, 0, 1)
        datasets[split] = (pixels, labels)
        metadata[split] = {
            "n": len(rows),
            "csv_sha256": file_sha(manifest),
            "pixels_sha256": digest(memoryview(pixels)),
            "class_counts": np.bincount(labels, minlength=len(classes)).tolist(),
        }
    return datasets, metadata


class Images:
    def __init__(self, datasets, device, preload):
        self.device = device
        self.data, self.labels = {}, {}
        total = sum(x.nbytes for x, _ in datasets.values())
        self.preloaded = device.type == "cuda" and preload and total < 2_000_000_000
        for name, (images, labels) in datasets.items():
            self.data[name] = torch.from_numpy(images)
            self.labels[name] = torch.from_numpy(labels.copy())
            if self.preloaded:
                self.data[name] = self.data[name].to(device)
                self.labels[name] = self.labels[name].to(device)
        self.mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)
        self.axis = torch.arange(160, device=device)

    def batch(self, split, index, crop=None, flip=None):
        source = self.data[split]
        ix = torch.as_tensor(index, device=source.device, dtype=torch.long)
        x = source[ix].to(self.device)
        y = self.labels[split][ix].to(self.device)
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
    return evaluate_predictions(labels, probabilities.argmax(1), W, polarity)


def save_probabilities(path, p, y):
    temp = path.with_suffix(".part.npz")
    np.savez_compressed(temp, probabilities=p, labels=y)
    temp.replace(path)
