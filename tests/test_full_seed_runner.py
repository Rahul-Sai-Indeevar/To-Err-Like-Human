"""CPU checks for full-data streaming and unchanged paired transforms."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


def module(name, filename):
    path = Path(__file__).with_name(filename)
    if not path.exists():
        path = Path(__file__).parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_streaming_constructor_keeps_memmap_without_full_array_copy(tmp_path, monkeypatch):
    full = module("full_runner_memory", "run_full_seed_evidence.py")
    path = tmp_path / "images.npy"
    mapped = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(4, 3, 192, 192))
    labels = np.array([0, 1, 2, 3])
    def forbidden_array(*args, **kwargs):
        raise AssertionError("Whole-array np.array copy is forbidden")
    monkeypatch.setattr(full.np, "array", forbidden_array)
    images = full.Images({"train": (mapped, labels)}, torch.device("cpu"))
    assert images.data["train"] is mapped
    assert isinstance(images.data["train"], np.memmap)
    assert images.preloaded is False
    x, y = images.batch("train", [3, 1])
    assert x.shape == (2, 3, 160, 160)
    assert y.tolist() == [3, 1]


def test_streaming_batch_transform_exactly_matches_medium_runner(tmp_path):
    full = module("full_runner_transform", "run_full_seed_evidence.py")
    original = module("medium_runner_transform", "run_seed_evidence.py")
    path = tmp_path / "images.npy"
    data = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(5, 3, 192, 192))
    data[:] = np.random.default_rng(7).integers(0, 256, data.shape, dtype=np.uint8)
    labels = np.array([1, 0, 3, 2, 1])
    datasets = {"train": (data, labels)}
    streaming = full.Images(datasets, torch.device("cpu"))
    copied = original.Images(datasets, torch.device("cpu"), False)
    plans, sha = full.augmentation_plan(5, 2, 3)
    assert sha == original.augmentation_plan(5, 2, 3)[1]
    order, crop, flip = plans[0]
    for args in ((order,), (order, crop, flip)):
        actual, y_actual = streaming.batch("train", *args)
        expected, y_expected = copied.batch("train", *args)
        assert torch.equal(actual, expected)
        assert torch.equal(y_actual, y_expected)


def test_full_runner_cli_restricts_controls_and_locks_fresh_seed_defaults():
    full = module("full_runner_cli", "run_full_seed_evidence.py")
    args = full.parse_args([])
    assert args.methods == ["ce", "graph"]
    assert args.seeds == [3, 4, 5, 6, 7, 8]
    assert args.epochs == 8 and args.batch_size == 64
    assert str(args.lock_file).replace("\\", "/") == "work/single_gpu.lock"
    with pytest.raises(SystemExit):
        full.parse_args(["--methods", "permuted"])
    from ecc.emotion import WHEEL_ORDER, build_cost_matrix
    with pytest.raises(ValueError, match="only"):
        full.loss_for("isotropic", build_cost_matrix(WHEEL_ORDER), 3)


def test_bounded_cache_hash_matches_file_contents(tmp_path):
    full = module("full_runner_hash", "run_full_seed_evidence.py")
    path = tmp_path / "cache.npy"
    contents = b"cache-header-and-pixels" * 17
    path.write_bytes(contents)
    assert full.file_digest(path) == hashlib.sha256(contents).hexdigest()
