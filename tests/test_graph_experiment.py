import csv
import json
import numpy as np
import torch
from PIL import Image
from ecc.data import build_transforms
from ecc.graph_experiment import CachedImages, Uint8Normalize


def test_uint8_cache_preserves_evaluation_transform(tmp_path, monkeypatch):
    import ecc.graph_experiment as exp

    monkeypatch.setattr(exp, "ROOT", tmp_path)
    rng = np.random.default_rng(4)
    image = Image.fromarray(rng.integers(0, 256, (234, 310, 3), dtype=np.uint8))
    image.save(tmp_path / "sample.png")
    (tmp_path / "manifest.json").write_text(json.dumps({"fixture": 1}))
    with (tmp_path / "val.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerow(["sample.png", "amusement"])
    ds = CachedImages("val", ["amusement"])
    x, y = ds[0]
    actual = Uint8Normalize()(x.unsqueeze(0))[0]
    expected = build_transforms(False, 160, 192)(image)
    assert y == 0 and torch.allclose(actual, expected, atol=1e-6)
