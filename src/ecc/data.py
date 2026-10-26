"""CSV-backed dataset, transforms (paper Appendix C) and helpers for the semi-supervised setup."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(train: bool, image_size: int = 224, resize: int = 256):
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:  # resize 256 -> random crop 224 -> random h-flip
        return T.Compose([T.Resize((resize, resize)), T.RandomCrop(image_size),
                          T.RandomHorizontalFlip(), T.ToTensor(), norm])
    return T.Compose([T.Resize((resize, resize)), T.CenterCrop(image_size),
                      T.ToTensor(), norm])


class TwoViewTransform:
    """Weak + strong augmentation of the same image (FixMatch)."""

    def __init__(self, image_size: int = 224, resize: int = 256):
        norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.weak = T.Compose([T.Resize((resize, resize)), T.RandomCrop(image_size),
                               T.RandomHorizontalFlip(), T.ToTensor(), norm])
        self.strong = T.Compose([T.Resize((resize, resize)), T.RandomCrop(image_size),
                                 T.RandomHorizontalFlip(), T.RandAugment(num_ops=2, magnitude=10),
                                 T.ToTensor(), norm, T.RandomErasing(p=0.5)])

    def __call__(self, img):
        return self.weak(img), self.strong(img)


class EmotionCSV(Dataset):
    """Reads a CSV with columns `path,label` (path relative to `root`, label = class name)."""

    def __init__(self, csv_path, root, classes, transform=None, indices=None):
        self.root = Path(root)
        self.transform = transform
        c2i = {c: i for i, c in enumerate(classes)}
        with open(csv_path) as f:
            self.items = [(r["path"].replace("\\", "/"), c2i[r["label"]]) for r in csv.DictReader(f)]
        if indices is not None:
            self.items = [self.items[i] for i in indices]

    @property
    def targets(self) -> np.ndarray:
        return np.array([y for _, y in self.items])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = Image.open(self.root / path).convert("RGB")
        return (self.transform(img) if self.transform else img), y


def class_balanced_indices(targets: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Pick `per_class` labeled samples per class (e.g. FI: 80/800/1600 labels = 10/100/200 per class)."""
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(targets):
        cand = np.where(targets == c)[0]
        idx.append(rng.choice(cand, size=min(per_class, len(cand)), replace=False))
    return np.sort(np.concatenate(idx))
