import numpy as np
import pytest
import torch
from ecc.emotion import WHEEL_ORDER, build_cost_matrix
from ecc.graph import graph_basis, spectral_weights, GraphSpectralLoss


def test_full_wheel_matches_author_evaluation_matrix():
    # ZhaoChenxi-nku/ECC train.py eval_training, pinned commit
    # 1c51f45cd1b17c12a009596109bb31c2f07bb84a, in WHEEL_ORDER.
    expected = np.array(
        [
            [1, 2, 3, 4, 8, 7, 6, 5],
            [2, 1, 2, 3, 7, 8, 7, 6],
            [3, 2, 1, 2, 6, 7, 8, 7],
            [4, 3, 2, 1, 5, 6, 7, 8],
            [8, 7, 6, 5, 1, 2, 3, 4],
            [7, 8, 7, 6, 2, 1, 2, 3],
            [6, 7, 8, 7, 3, 2, 1, 2],
            [5, 6, 7, 8, 4, 3, 2, 1],
        ]
    )
    assert np.array_equal(build_cost_matrix(WHEEL_ORDER), expected)


def test_transform_energy_and_dirichlet_identity():
    A, L, v, U = graph_basis(build_cost_matrix(WHEEL_ORDER))
    rng = np.random.default_rng(42)
    x = rng.normal(size=8)
    assert np.allclose(U.T @ U, np.eye(8))
    assert np.allclose(L.sum(1), 0)
    assert np.all(v >= 0) and np.count_nonzero(v == 0) == 1
    assert np.allclose(U @ np.diag(v) @ U.T, L)
    assert np.allclose(np.sum(x * x), np.sum((U.T @ x) ** 2))
    assert np.allclose(x @ L @ x, 0.5 * np.sum(A * (x[:, None] - x[None, :]) ** 2))
    assert np.allclose(x @ L @ x, np.sum(v * (U.T @ x) ** 2))


def test_permutation_invariance_and_gradients():
    torch.manual_seed(1)
    W = build_cost_matrix(WHEEL_ORDER)
    z = torch.randn(5, 8, requires_grad=True)
    y = torch.tensor([0, 2, 4, 6, 7])
    loss = GraphSpectralLoss(W)(z, y)
    loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    perm = np.array([3, 1, 6, 2, 0, 7, 5, 4])
    inv = np.argsort(perm)
    other = GraphSpectralLoss(W[np.ix_(perm, perm)])(
        z.detach()[:, perm], torch.tensor(inv[y])
    )
    assert torch.allclose(loss, other, atol=1e-6)


def test_dc_uniform_and_pseudoinverse():
    W = build_cost_matrix(WHEEL_ORDER)
    _, L, v, U = graph_basis(W)
    weights = spectral_weights(v)
    assert weights[0] == 0
    assert np.allclose(U @ np.diag(weights) @ U.T, np.linalg.pinv(L))
    # A zero-mass residual is also a balanced current injection on the graph.
    residual = np.eye(8)[0] - np.eye(8)[4]
    potential = np.linalg.pinv(L) @ residual
    assert np.allclose(L @ potential, residual)
    assert np.allclose(
        residual @ np.linalg.pinv(L) @ residual, potential @ L @ potential
    )
    uniform = np.ones(8) / 8
    assert abs(uniform @ L @ uniform) < 1e-12
    z = torch.randn(3, 8)
    y = torch.tensor([0, 1, 2])
    assert torch.equal(
        GraphSpectralLoss(W, alpha=0)(z, y), torch.nn.functional.cross_entropy(z, y)
    )


def test_invalid_graph():
    with pytest.raises(ValueError):
        graph_basis(np.zeros((8, 8)))


def test_uniform_control_preserves_graph_trace():
    W = build_cost_matrix(WHEEL_ORDER)
    A, L, v, _ = graph_basis(W)
    B, M, w, _ = graph_basis(W, "uniform_trace_matched")
    assert np.isclose(np.trace(L), np.trace(M))
    assert np.allclose(w[1:], w[1])


def test_heat_filter_mass_dc_and_energy():
    from ecc.graph import heat_filter

    W = build_cost_matrix(WHEEL_ORDER)
    _, L, _, _ = graph_basis(W)
    p = np.eye(8)
    assert np.allclose(heat_filter(p, W, 0), p)
    q = heat_filter(p, W, 1)
    assert np.allclose(q.sum(1), 1) and np.all(q >= 0)
    assert np.trace(q @ L @ q.T) < np.trace(p @ L @ p.T)
    assert np.allclose(heat_filter(np.ones((1, 8)) / 8, W, 3), np.ones((1, 8)) / 8)


def test_graph_training_step_and_metric():
    from ecc.metrics import graph_error_metrics

    W = build_cost_matrix(WHEEL_ORDER)
    torch.manual_seed(123)
    model = torch.nn.Linear(12, 8)
    x = torch.randn(32, 12)
    y = torch.arange(32) % 8
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = GraphSpectralLoss(W)
    before = criterion(model(x), y).item()
    for _ in range(15):
        opt.zero_grad()
        criterion(model(x), y).backward()
        opt.step()
    assert criterion(model(x), y).item() < before
    p = model(x).softmax(1).detach().numpy()
    m = graph_error_metrics(p, y.numpy(), W)
    assert abs(m["error_energy"][0]) < 1e-12
    assert np.isclose(sum(m["error_energy"]), m["brier"])
    _, _, v, _ = graph_basis(W)
    assert np.isclose(np.dot(m["error_energy"], spectral_weights(v)), m["gsee"])
