import csv
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch import nn
import yaml

from ecc.emotion import WHEEL_ORDER
from ecc.wavelets import wavelet_components
from scripts import run_wavelet_study as study


class TinyTwoBranchModel(nn.Module):
    def __init__(self, method, state, levels, head_seed):
        super().__init__()
        self.method = method
        self.classifier = nn.Linear(6, 8)
        self.classifier.load_state_dict(state)

    def forward(self, x):
        a, b = (x, x) if self.method == "rgb_dual" else wavelet_components(x)
        return self.classifier(torch.cat([a.mean((-2, -1)), b.square().mean((-2, -1))], 1))


def test_small_run_trains_then_replays_and_compares_both_methods(tmp_path, monkeypatch):
    dataset = tmp_path / "images"
    dataset.mkdir()
    (dataset / "classes.txt").write_text("\n".join(WHEEL_ORDER), encoding="utf8")
    rng = np.random.default_rng(42)
    rows = []
    for i, label in enumerate(WHEEL_ORDER):
        path = f"{i}.png"
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(dataset / path)
        rows.append({"path": path, "label": label})
    for split in ("train", "val", "test"):
        with (dataset / f"{split}.csv").open("w", encoding="utf8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "label"])
            writer.writeheader()
            writer.writerows(rows)
    state = nn.Linear(6, 8).state_dict()
    lock = study.shared.gpu_lock
    monkeypatch.setattr(study.shared, "gpu_lock", lambda unused: lock(tmp_path / "pipeline.lock"))
    monkeypatch.setattr(study, "make_initial", lambda seed, n: state)
    monkeypatch.setattr(study, "WaveletEmotionModel", TinyTwoBranchModel)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "datasets": {"fi": str(dataset)}, "methods": list(study.METHODS),
        "seeds": [15], "epochs": 1, "levels": 2, "batch_size": 4,
    }), encoding="utf8")
    out = tmp_path / "out"
    args = SimpleNamespace(config=config, out=out, stage="train", device="cpu")
    study.run(args)
    assert not list(out.glob("*/*/predictions.npz"))
    args.stage = "evaluate"
    study.run(args)
    summary = study.read_json(out / "summary.json")["fi"]
    assert summary["n_seed_pairs"] == 1
    assert len(list(out.glob("*/*/predictions.npz"))) == 2
    assert not list(dataset.glob("*.npy"))
    assert all(record["validation_replay_exact"] for record in study.read_json(out / "fi/results.json"))
