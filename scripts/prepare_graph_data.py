"""Extract a reproducible real EmoSet subset from pinned HF parquet shards.

Download shards first with scripts/download_emoset.py. This keeps the mirror's
train/val/test assignments. It is a shard subset, not an official benchmark split.
"""

import csv
import hashlib
import io
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps


def main():
    source = Path("data/emoset_source")
    root = Path("data/emoset_graph")
    root.mkdir(parents=True, exist_ok=True)
    classes = [
        "amusement",
        "anger",
        "awe",
        "contentment",
        "disgust",
        "excitement",
        "fear",
        "sadness",
    ]
    label_order = [
        "amusement",
        "awe",
        "contentment",
        "excitement",
        "anger",
        "disgust",
        "fear",
        "sadness",
    ]
    seen = set()
    summary = {}
    rng = np.random.default_rng(2026)
    for split, cap in [("train", 600), ("val", 100), ("test", 200)]:
        rows = []
        for shard in sorted(source.glob(f"{split}-*.parquet")):
            table = pq.read_table(shard)
            rows.extend(table.to_pylist())
        if not rows:
            raise RuntimeError(f"No {split} rows")
        counts = Counter()
        kept = []
        rejected = Counter()
        for index in rng.permutation(len(rows)):
            row = rows[index]
            name = label_order[row["label"]]
            if row.get("emotion") and row["emotion"].lower() != name:
                raise ValueError("Label metadata disagrees")
            if counts[name] >= cap:
                continue
            try:
                im = ImageOps.exif_transpose(
                    Image.open(io.BytesIO(row["image"]["bytes"]))
                ).convert("RGB")
                pixel_hash = hashlib.sha256(
                    str(im.size).encode() + im.tobytes()
                ).hexdigest()
                if pixel_hash in seen:
                    rejected["duplicates"] += 1
                    continue
                seen.add(pixel_hash)
                out = root / "images" / split / name / f"{pixel_hash}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                im.thumbnail((320, 320), Image.Resampling.LANCZOS)
                im.save(out, quality=95)
            except (OSError, ValueError):
                rejected["invalid"] += 1
                continue
            counts[name] += 1
            kept.append(
                {
                    "path": out.relative_to(root).as_posix(),
                    "label": name,
                    "source_id": row.get("image_id", ""),
                    "pixel_sha256": pixel_hash,
                }
            )
        if min(counts.get(c, 0) for c in classes) < 30:
            raise RuntimeError(f"Missing coverage: {counts}")
        with (root / f"{split}.csv").open("w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["path", "label", "source_id", "pixel_sha256"]
            )
            writer.writeheader()
            writer.writerows(kept)
        summary[split] = {
            "available_rows": len(rows),
            "selected": len(kept),
            "per_class": dict(counts),
            "excluded": dict(rejected),
        }
        print(split, summary[split], flush=True)
    (root / "classes.txt").write_text("\n".join(classes) + "\n")
    summary["source"] = json.loads((source / "provenance.json").read_text())
    summary["sampling_seed"] = 2026
    (root / "manifest.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
