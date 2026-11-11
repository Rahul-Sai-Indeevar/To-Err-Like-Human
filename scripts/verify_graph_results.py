"""Check saved real-data results and replay a trained checkpoint independently."""

import json
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from ecc.data import EmotionCSV
from ecc.emotion import build_cost_matrix, polarity_vector
from ecc.metrics import evaluate_predictions, graph_error_metrics
from ecc.graph_experiment import CachedImages, Uint8Normalize

root = Path("data/emoset_graph")
out = Path("outputs/graph_experiment")
classes = (root / "classes.txt").read_text().splitlines()
W = build_cost_matrix(classes)
pol = polarity_vector(classes)
truth = EmotionCSV(root / "test.csv", root, classes).targets
verified = []
for stage, expected in [("probe", 21), ("finetune", 9)]:
    paths = sorted((out / stage).glob("*/result.json"))
    assert len(paths) == expected, (stage, len(paths), expected)
    for path in paths:
        result = json.loads(path.read_text())
        history = json.loads((path.parent / "history.json").read_text())
        assert result["best_epoch"] == np.argmax([h["val"]["ecc"] for h in history]) + 1
        saved = np.load(path.parent / "predictions.npz")
        p, y = saved["probabilities"], saved["labels"]
        assert np.array_equal(y, truth)
        assert p.shape == (1600, 8) and np.isfinite(p).all() and np.all(p >= 0)
        assert np.allclose(p.sum(1), 1, atol=1e-6)
        metrics = evaluate_predictions(y, p.argmax(1), W, pol)
        metrics.update(graph_error_metrics(p, y, W))
        for key in ("acc", "acc2", "ecc", "emc", "gsee", "brier", "dirichlet"):
            assert np.isclose(metrics[key], result["test"][key], atol=1e-9), (path, key)
        recorded = np.loadtxt(path.parent / "confusion.csv", delimiter=",")
        assert recorded.sum() == len(y)
        assert int(recorded.trace()) == int((p.argmax(1) == y).sum())
        verified.append(str(path.relative_to(out)))

torch.set_num_threads(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.nn.Sequential(Uint8Normalize(), resnet18(weights=None, num_classes=8))
checkpoint = out / "finetune/graph_seed0/best.pt"
model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
model = model.to(device).eval()
x, y = next(iter(DataLoader(CachedImages("test", classes), batch_size=32)))
with torch.no_grad():
    replay = model(x.to(device)).softmax(1).cpu().numpy()
saved = np.load(checkpoint.parent / "predictions.npz")["probabilities"][:32]
diff = float(np.max(np.abs(replay - saved)))
assert np.allclose(replay, saved, atol=1e-5, rtol=1e-4), diff
report = {
    "verified_runs": len(verified),
    "runs": verified,
    "test_images_per_run": len(truth),
    "checkpoint_replay": str(checkpoint),
    "replay_device": str(device),
    "replayed_images": 32,
    "max_probability_difference": diff,
    "checks": [
        "label order",
        "finite normalized probabilities",
        "recomputed metrics",
        "validation-only epoch choice",
        "confusion totals",
        "checkpoint replay",
    ],
}
(out / "verification.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
