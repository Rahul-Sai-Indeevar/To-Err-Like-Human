"""CPU controls and paired input-plan checks for the optional seed study."""
import importlib.util
from pathlib import Path
import numpy as np
import torch


def runner_module():
    path = Path(__file__).with_name("run_seed_evidence.py")
    if not path.exists():
        path = Path(__file__).parents[1] / "scripts" / "run_seed_evidence.py"
    spec = importlib.util.spec_from_file_location("seed_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_isotropic_matches_both_quadratic_traces():
    from ecc.emotion import build_cost_matrix, WHEEL_ORDER
    from ecc.graph import graph_basis
    r = runner_module()
    W = build_cost_matrix(WHEEL_ORDER)
    _, L, vals, U = graph_basis(W)
    criterion = r.IsotropicMatchedLoss(W)
    inverse = np.zeros_like(vals)
    inverse[vals > 1e-10] = 1 / vals[vals > 1e-10]
    Q = (U * inverse) @ U.T
    assert np.isclose(criterion.q * 7, np.trace(Q))
    assert np.isclose(criterion.d * 7, np.trace(L))
    z = torch.randn(4, 8, requires_grad=True)
    y = torch.tensor([0, 1, 3, 5])
    criterion(z, y).backward()
    assert torch.isfinite(z.grad).all()


def test_permutation_preserves_spectrum_not_label_geometry():
    from ecc.emotion import build_cost_matrix, WHEEL_ORDER
    from ecc.graph import graph_basis
    r = runner_module()
    W = build_cost_matrix(WHEEL_ORDER)
    for seed in range(3, 15):
        p = r.wrong_permutation(W, seed)
        wrong = W[np.ix_(p, p)]
        assert not np.array_equal(wrong, W)
        assert np.allclose(graph_basis(wrong)[2], graph_basis(W)[2])


def test_plans_identical_and_batched_crops_match_reference():
    r = runner_module()
    (plan, sha) = r.augmentation_plan(6, 2, 3)
    assert r.augmentation_plan(6, 2, 3)[1] == sha
    assert r.augmentation_plan(6, 2, 4)[1] != sha
    images = np.random.default_rng(0).integers(0, 256, size=(6, 3, 192, 192), dtype=np.uint8)
    labels = np.arange(6)
    cache = r.Images({"train": (images, labels)}, torch.device("cpu"), False)
    order, crop, flip = plan[0]
    actual, yy = cache.batch("train", order, crop, flip)
    expected = []
    for i, (a, b), f in zip(order, crop, flip):
        x = torch.from_numpy(images[i, :, a:a+160, b:b+160].copy())
        if f:
            x = x.flip(-1)
        expected.append(x)
    expected = (torch.stack(expected).float() / 255 - cache.mean) / cache.std
    assert torch.equal(actual, expected)
    assert np.array_equal(yy.numpy(), labels[order])
