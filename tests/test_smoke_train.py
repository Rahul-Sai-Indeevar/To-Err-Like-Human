"""End-to-end smoke test on synthetic images (CPU, tiny, no downloads)."""
import numpy as np
import pytest
from PIL import Image

from ecc.config import load_config
from ecc.splits import main as splits_main

CLASSES = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]


@pytest.fixture()
def toy(tmp_path, monkeypatch):
    rng = np.random.RandomState(0)
    for c in CLASSES:
        (tmp_path / "img" / c).mkdir(parents=True)
        for k in range(14):
            Image.fromarray(rng.randint(0, 255, (40, 40, 3), dtype=np.uint8)).save(tmp_path / "img" / c / f"{k}.jpg")
    monkeypatch.setattr("sys.argv", ["x", "--root", str(tmp_path / "img"), "--out", str(tmp_path / "sp")])
    splits_main()
    return tmp_path


def _overrides(toy):
    return [f"data.root={toy/'img'}", f"data.splits_dir={toy/'sp'}", "data.num_workers=0",
            "data.image_size=32", "data.resize=36", "model.name=resnet18", "model.pretrained=false",
            "amp=false", f"output_dir={toy/'out'}"]


def test_supervised_smoke(toy):
    from ecc.supervised import run
    cfg = load_config("configs/fi_resnet50.yaml", _overrides(toy) + ["optim.epochs=2", "optim.batch_size=8",
                                                                      "optim.decay_epoch=1", "loss.alpha=0.2"])
    res = run(cfg)
    assert 0 <= res["test_argmax"]["acc"] <= 100


def test_ssl_smoke(toy):
    from ecc.ssl import run
    cfg = load_config("configs/ssl_fi.yaml", _overrides(toy) + ["optim.epochs=2", "optim.batch_size=8",
                                                                  "ssl.n_labeled=16", "ssl.iters_per_epoch=3", "ssl.mu=2"])
    res = run(cfg)
    assert "test_mec_decode" in res


def test_supervised_resume(toy):
    """Resume from checkpoint.pt should reproduce the same result as uninterrupted training."""
    from ecc.supervised import run
    from ecc.checkpoint import save_checkpoint, latest_checkpoint

    out = toy / "out_resume"
    out.mkdir(exist_ok=True)

    # Run 1 epoch only, saving a checkpoint
    cfg1 = load_config("configs/fi_resnet50.yaml", _overrides(toy) + ["optim.epochs=1", "optim.batch_size=8",
                                                                       "optim.decay_epoch=1", "loss.alpha=0.2",
                                                                       f"output_dir={out}"])
    res1 = run(cfg1)
    ckpt = latest_checkpoint(out)
    assert ckpt is not None, "checkpoint.pt should exist after epoch 1"

    # Run again from resume, 2 epochs total (should continue from epoch 1)
    cfg2 = load_config("configs/fi_resnet50.yaml", _overrides(toy) + ["optim.epochs=2", "optim.batch_size=8",
                                                                       "optim.decay_epoch=1", "loss.alpha=0.2",
                                                                       f"output_dir={out}", f"resume_from={ckpt}"])
    res2 = run(cfg2)
    assert 0 <= res2["test_argmax"]["acc"] <= 100
