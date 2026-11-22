"""CPU-only validation of checkpoint-replay mismatch detection."""
import importlib.util
from pathlib import Path

import numpy as np


def checker():
    path = Path(__file__).with_name("verify_full_checkpoints.py")
    if not path.exists():
        path = Path(__file__).parents[1] / "scripts" / "verify_full_checkpoints.py"
    spec = importlib.util.spec_from_file_location("replay_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def values(module):
    metrics = {key: 1.0 for key in module.METRICS}
    metrics["error_energy"] = [0.0, 0.1]
    p = np.array([[.75, .25], [.2, .8]])
    y = np.array([0, 1])
    return p, y, metrics


def test_identical_replay_passes_all_cpu_checks():
    module = checker()
    p, y, metrics = values(module)
    result = module.compare_replay(p, y, p.copy(), y.copy(), metrics, metrics)
    assert result["passed"] is True
    assert result["max_probability_difference"] == 0
    assert result["argmax_agreement"] == 1


def test_probability_tolerance_and_label_order_mismatches_fail():
    module = checker()
    p, y, metrics = values(module)
    changed = p.copy()
    changed[0] += np.array([-.001, .001])
    result = module.compare_replay(p, y, changed, y[::-1], metrics, metrics)
    assert result["passed"] is False
    assert "probability_tolerance" in result["failures"]
    assert "label_identity_or_order" in result["failures"]
    assert result["argmax_agreement"] == 1  # Same predictions alone are insufficient.


def test_argmax_flip_and_saved_metric_mismatch_fail():
    module = checker()
    p, y, metrics = values(module)
    flipped = p[:, ::-1].copy()
    different = dict(metrics, ecc=2.0)
    result = module.compare_replay(p, y, flipped, y, metrics, different)
    assert result["passed"] is False
    assert "argmax_agreement" in result["failures"]
    assert "metric_ecc" in result["failures"]
