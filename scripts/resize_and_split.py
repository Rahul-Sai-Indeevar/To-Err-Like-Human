"""Resize the FI dataset to 256x256 and create train/val/test splits for Kaggle.

The paper (Appendix C) resizes FI images to 256x256, then randomly crops 224x224
during training and center-crops 224x224 at test time. This script does the
one-time 256x256 resize so images are small and fast to decode on Kaggle.

Usage:
    python scripts/resize_and_split.py \
        --root /path/to/raw/FI          # folder with class subfolders (amusement/, anger/, ...)
        --out /path/to/kaggle_fi        # output directory

Output layout (upload this whole folder as a private Kaggle dataset):
    <out>/
        FI/
            amusement/  *.jpg (resized to 256x256)
            anger/
            ...
        FI_splits/
            classes.txt
            train.csv   (columns: path,label)
            val.csv
            test.csv

On Kaggle, mount this dataset and symlink:
    ln -s /kaggle/input/<your>/FI        data/FI
    ln -s /kaggle/input/<your>/FI_splits data/FI_splits
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET = 256


def resize_image(src: Path, dst: Path, size: int = TARGET) -> None:
    """Resize a single image to size x size, preserving RGB."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        img = img.resize((size, size), Image.BILINEAR)
        img.save(dst, quality=95)


def copy_and_resize(root: Path, out_root: Path) -> None:
    """Resize every image under root/<class>/ into out_root/FI/<class>/."""
    classes = sorted(p.name.lower() for p in root.iterdir() if p.is_dir())
    if not classes:
        print(f"No class folders found under {root}", file=sys.stderr)
        sys.exit(1)

    n = 0
    for cls in classes:
        src_dir = root / cls
        dst_dir = out_root / "FI" / cls
        for f in sorted(src_dir.rglob("*")):
            if f.suffix.lower() not in EXTS:
                continue
            resize_image(f, dst_dir / f.name)
            n += 1
            if n % 1000 == 0:
                print(f"  resized {n} images...", flush=True)
    print(f"Resized {n} images across {len(classes)} classes -> {out_root / 'FI'}")
    return classes


def make_splits(fi_root: Path, splits_out: Path, classes: list[str], val: float = 0.05,
                test: float = 0.15, seed: int = 0) -> None:
    """Create train/val/test CSVs + classes.txt using the repo's split logic."""
    # Reuse the repo's scanner + split logic inline to avoid a subprocess import.
    import random
    import csv

    EXT_LOW = {e.lower() for e in EXTS}

    rows = []
    for cls in classes:
        cls_dir = fi_root / "FI" / cls
        for f in sorted(cls_dir.rglob("*")):
            if f.suffix.lower() in EXT_LOW:
                rows.append((str(f.relative_to(fi_root / "FI")).replace("\\", "/"), cls))

    rng = random.Random(seed)

    # Stratified split
    by_class: dict[str, list[tuple[str, str]]] = {}
    for r in rows:
        by_class.setdefault(r[1], []).append(r)

    train_rows, val_rows, test_rows = [], [], []
    for cls_items in by_class.values():
        rng.shuffle(cls_items)
        n_test = int(round(len(cls_items) * test))
        n_val = int(round(len(cls_items) * val))
        test_rows += cls_items[:n_test]
        val_rows += cls_items[n_test:n_val + n_test]
        train_rows += cls_items[n_val + n_test:]

    splits_out.mkdir(parents=True, exist_ok=True)
    (splits_out / "classes.txt").write_text("\n".join(classes) + "\n")

    def write_csv(name: str, data: list[tuple[str, str]]) -> None:
        with open(splits_out / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            w.writerows(data)

    write_csv("train", train_rows)
    write_csv("val", val_rows)
    write_csv("test", test_rows)
    print(f"Splits: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} -> {splits_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="raw FI root with class subfolders")
    ap.add_argument("--out", required=True, help="output directory for Kaggle dataset")
    ap.add_argument("--val", type=float, default=0.05, help="validation fraction")
    ap.add_argument("--test", type=float, default=0.15, help="test fraction")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    classes = copy_and_resize(root, out)
    make_splits(out, out / "FI_splits", classes, val=a.val, test=a.test, seed=a.seed)
    print(f"\nDone. Upload the folder '{out}' as a private Kaggle dataset.")


if __name__ == "__main__":
    main()
