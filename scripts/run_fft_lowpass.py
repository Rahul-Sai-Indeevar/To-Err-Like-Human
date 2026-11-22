"""Evaluate slide 10's clean + six low-pass conditions on frozen CE checkpoints."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import resnet18

from ecc.emotion import build_cost_matrix, polarity_vector
from ecc.experiment_data import atomic_json, file_sha, gpu_lock
from ecc.metrics import evaluate_predictions
from ecc.spatial_frequency import lowpass_rgb

CUTOFFS = (0.025, 0.05, 0.1, 0.2, 0.35, 0.5)


class Uint8Normalize(nn.Module):
    """Matches the archived CE checkpoint's sequential input module."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, pixels):
        return (pixels.float() / 255 - self.mean) / self.std


def load_model(path, classes, device):
    model = nn.Sequential(Uint8Normalize(), resnet18(weights=None, num_classes=len(classes)))
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def center_crops(root, rows):
    pixels = []
    for row in rows:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Image path escapes the dataset directory: {path}")
        with Image.open(path) as image:
            if image.size != (192, 192):
                raise ValueError(f"Expected a prepared 192×192 source image: {path}")
            pixels.append(np.asarray(image.convert("RGB"))[16:176, 16:176].transpose(2, 0, 1))
    return torch.from_numpy(np.stack(pixels))


@torch.inference_mode()
def evaluate(root, rows, classes, models, device, batch_size):
    conditions = [("clean", None)] + [(f"lowpass_{cutoff:g}", cutoff) for cutoff in CUTOFFS]
    predicted = {condition: {name: [] for name in models} for condition, _ in conditions}
    clipped = dict.fromkeys(predicted, 0)
    pixel_count = 0
    for start in range(0, len(rows), batch_size):
        pixels = center_crops(root, rows[start:start + batch_size]).to(device).float()
        pixel_count += pixels.numel()
        for condition, cutoff in conditions:
            filtered = pixels if cutoff is None else lowpass_rgb(pixels, cutoff)
            clipped[condition] += int(((filtered < 0) | (filtered > 255)).sum())
            filtered = filtered.clamp(0, 255)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                for name, model in models.items():
                    predicted[condition][name].append(model(filtered).float().argmax(1).cpu().numpy())
    labels = np.array([classes.index(row["label"]) for row in rows])
    cost = build_cost_matrix(classes)
    polarity = polarity_vector(classes)
    output = []
    for condition, cutoff in conditions:
        by_checkpoint = {
            name: evaluate_predictions(labels, np.concatenate(chunks), cost, polarity)
            for name, chunks in predicted[condition].items()
        }
        ecc = [metrics["ecc"] for metrics in by_checkpoint.values()]
        output.append({
            "condition": condition,
            "cutoff_cycles_per_pixel": cutoff,
            "ecc_mean": float(np.mean(ecc)),
            "ecc_std": float(np.std(ecc, ddof=1)) if len(ecc) > 1 else None,
            "clipped_fraction": clipped[condition] / pixel_count,
            "checkpoints": by_checkpoint,
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared directory with classes.txt and test.csv")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/fft_lowpass"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, help="Optional first N manifest rows for a smoke run")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch size and limit must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu or install a CUDA-compatible PyTorch.")
    torch.set_num_threads(3)
    root = args.dataset.resolve()
    classes = (root / "classes.txt").read_text(encoding="utf-8-sig").splitlines()
    build_cost_matrix(classes)
    with (root / "test.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))[:args.limit]
    if not rows:
        parser.error("The prepared test manifest is empty.")
    checkpoints = [path.resolve() for path in args.checkpoints]
    if len(set(checkpoints)) != len(checkpoints):
        parser.error("Pass each checkpoint only once.")
    metadata = {
        "images": len(rows),
        "test_manifest_sha256": file_sha(root / "test.csv"),
        "classes": classes,
        "checkpoints": {str(path): file_sha(path) for path in checkpoints},
        "butterworth_order": 2,
        "cutoffs_cycles_per_pixel": list(CUTOFFS),
        "batch_size": args.batch_size,
        "device": args.device,
        "torch": str(torch.__version__),
    }
    device = torch.device(args.device)
    with gpu_lock(Path("runs/.experiment.lock")):
        target = args.out / "results.json"
        if target.exists():
            parser.error(f"Results already exist at {target}. Use a fresh output directory.")
        models = {str(path): load_model(path, classes, device) for path in checkpoints}
        results = evaluate(root, rows, classes, models, device, args.batch_size)
        atomic_json(target, {"protocol": metadata, "results": results})
    print(f"Evaluated {len(rows)} images × {len(models)} checkpoints. Results: {target}")


if __name__ == "__main__":
    main()
