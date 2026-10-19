"""Build train/val/test CSV splits from a directory of class folders.

Two layouts are supported:
  root/<Class>/*.jpg                     -> stratified random split (use for FI)
  root/train/<Class>/*  root/test/<Class>/*  -> uses the given split, carves val out of train
Output: <out>/{train,val,test}.csv (columns path,label) and <out>/classes.txt
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _scan(folder: Path, root: Path):
    rows = []
    for cdir in sorted(p for p in folder.iterdir() if p.is_dir()):
        for f in sorted(cdir.rglob("*")):
            if f.suffix.lower() in EXT:
                rows.append((str(f.relative_to(root)), cdir.name.lower()))
    return rows


def _write(path: Path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(rows)


def _stratified(rows, frac, rng):
    by = {}
    for r in rows:
        by.setdefault(r[1], []).append(r)
    a, b = [], []
    for items in by.values():
        rng.shuffle(items)
        k = int(round(len(items) * frac))
        b += items[:k]
        a += items[k:]
    return a, b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root, out, rng = Path(a.root), Path(a.out), random.Random(a.seed)
    out.mkdir(parents=True, exist_ok=True)

    if (root / "train").is_dir() and (root / "test").is_dir():
        train_all, test = _scan(root / "train", root), _scan(root / "test", root)
        train, val = _stratified(train_all, a.val, rng)
    else:
        rows = _scan(root, root)
        rest, test = _stratified(rows, a.test, rng)
        train, val = _stratified(rest, a.val / (1 - a.test), rng)

    classes = sorted({r[1] for r in train + val + test})
    (out / "classes.txt").write_text("\n".join(classes) + "\n")
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        _write(out / f"{name}.csv", rows)
    print(f"classes={classes}\ntrain={len(train)} val={len(val)} test={len(test)} -> {out}")


if __name__ == "__main__":
    main()
