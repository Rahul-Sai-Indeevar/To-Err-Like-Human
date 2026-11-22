"""Exercise the experiment lifecycle with real small CPU updates and saved arrays."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


def runner():
    import sys
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("wavelet_runner", scripts / "run_wavelet_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyModel(nn.Module):
    def __init__(self, method, state, levels=2, head_seed=0):
        super().__init__()
        self.classifier = nn.Linear(3, 8)
        self.classifier.load_state_dict({"weight": state["fc.weight"], "bias": state["fc.bias"]})
        self.levels = levels

    def forward(self, x):
        return self.classifier(x.mean((-2, -1)))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    r = runner()
    torch.set_num_threads(1)
    monkeypatch.setattr(r, "WaveletEmotionModel", TinyModel)
    from ecc.emotion import WHEEL_ORDER
    classes = list(WHEEL_ORDER)
    rng = np.random.default_rng(1)
    data = {s: (rng.integers(0, 256, (8, 3, 192, 192), dtype=np.uint8), np.arange(8)) for s in ("train", "val", "test")}
    state = {"fc.weight": torch.randn(8, 3) * .01, "fc.bias": torch.zeros(8)}
    protocol = {"protocol_hash": "fixture", "sources": {}, "dataset": "fixture", "datasets": {}, "classes": classes,
                "args": {"data_root": ".", "epochs": 2, "seeds": [31], "methods": ["rgb"], "batch_size": 4, "levels": 2}}
    images = r.shared.Images(data, torch.device("cpu"), False)
    plans, sha = r.shared.augmentation_plan(8, 2, 31)
    return r, data, state, protocol, images, plans, sha, tmp_path


def test_train_is_val_only_then_checkpoint_replays_and_test_is_reproducible(setup, monkeypatch):
    r, data, state, protocol, images, plans, sha, out = setup
    original = r.shared.predict

    def validation_only(model, imgs, split, batch_size, amp):
        assert split == "val", "training must not inspect test"
        return original(model, imgs, split, batch_size, amp)

    monkeypatch.setattr(r.shared, "predict", validation_only)
    result = r.fit_one(out, "rgb", 31, state, plans, sha, images, protocol)
    assert result["status"] == "trained" and "test" not in result
    assert result["optimizer_steps"] == 4 and result["amp_skipped_steps"] == 0
    assert not (out / "rgb_seed31/predictions.npz").exists()
    assert r.fit_one(out, "rgb", 31, state, plans, sha, images, protocol) == result
    r.shared.atomic_json(out / "protocol.json", protocol)
    monkeypatch.setattr(r, "make_initial", lambda seed, n: state)
    monkeypatch.setattr(r.shared, "load_dataset", lambda *a: (data, {}))
    r.evaluate_and_replay(out, torch.device("cpu"))
    assert json.loads((out / "rgb_seed31/result.json").read_text())["status"] == "evaluated"
    replay = json.loads((out / "checkpoint_replay.json").read_text())
    assert replay["arrays"][0]["bitwise_exact"]
    first = (out / "rgb_seed31/predictions.npz").read_bytes()
    r.evaluate_and_replay(out, torch.device("cpu"))
    assert (out / "rgb_seed31/predictions.npz").read_bytes() == first
    p = json.loads((out / "perturbations.json").read_text())
    assert {x["condition"] for x in p} == set(r.CONDITIONS)


def test_completed_checkpoint_and_protocol_tampering_are_rejected(setup):
    r, _, state, protocol, images, plans, sha, out = setup
    result = r.fit_one(out, "rgb", 31, state, plans, sha, images, protocol)
    dest = out / "rgb_seed31"
    with pytest.raises(ValueError, match="identity"):
        r.verify_completed(result, dict(protocol, protocol_hash="changed"), r.shared.state_digest(state), sha, dest)
    with (dest / "best.pt").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="Checkpoint changed"):
        r.verify_completed(result, protocol, r.shared.state_digest(state), sha, dest)


def test_noise_realization_is_shared_without_mutating_clean_images(setup):
    r, _, state, _, images, _, _, _ = setup
    model = TinyModel("rgb", state)
    clean_before = images.data["test"].clone()
    a, y = r.condition_predict(model, images, "test", 4, "awgn_003")
    b, yy = r.condition_predict(model, images, "test", 4, "awgn_003")
    assert np.array_equal(a, b) and np.array_equal(y, yy)
    assert torch.equal(images.data["test"], clean_before)
    with pytest.raises(ValueError):
        r.condition_predict(model, images, "test", 4, "undefined")


def test_global_test_gate_rejects_unexpected_run_even_when_file_count_matches(setup):
    r, _, _, protocol, _, _, _, out = setup
    r.shared.atomic_json(out / "protocol.json", protocol)
    wrong = out / "detail_seed31"
    wrong.mkdir()
    r.shared.atomic_json(wrong / "result.json", {})
    with pytest.raises(ValueError, match="Planned model set"):
        r.audit_training_complete(out)
