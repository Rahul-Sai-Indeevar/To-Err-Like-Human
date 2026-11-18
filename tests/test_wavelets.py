"""Mathematical filter-bank identities and genuine matched model controls."""
import pytest
import torch
from torch import nn
from torchvision.models import resnet18

from ecc.wavelets import (HaarBands, METHODS, WaveletEmotionModel,
                          haar_analysis2d, haar_synthesis2d, haar_decompose,
                          haar_reconstruct, reconstruct_bands, wavelet_components)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def initial_state():
    with torch.random.fork_rng():
        torch.manual_seed(13)
        base = resnet18(weights=None)
        base.fc = nn.Linear(512, 8)
        return {k: v.detach().clone() for k, v in base.state_dict().items()}


@pytest.mark.parametrize("levels", [1, 2, 3])
def test_perfect_reconstruction_parseval_and_orthogonal_components(levels):
    generator = torch.Generator().manual_seed(91)
    x = torch.randn(2, 3, 16, 24, dtype=torch.float64, generator=generator)
    pyramid = haar_decompose(x, levels)
    assert torch.allclose(haar_reconstruct(pyramid), x, atol=2e-15, rtol=2e-15)
    coefficients_energy = pyramid.approximation.square().sum()
    for d in pyramid.details:
        coefficients_energy += d.lh.square().sum() + d.hl.square().sum() + d.hh.square().sum()
    assert torch.allclose(coefficients_energy, x.square().sum(), atol=1e-11, rtol=1e-13)
    bands = reconstruct_bands(pyramid)
    assert len(bands) == 1 + 3 * levels
    assert all(component.shape == x.shape for component in bands.values())
    assert torch.allclose(sum(bands.values()), x, atol=2e-15, rtol=2e-15)
    assert torch.allclose(sum(b.square().sum() for b in bands.values()), x.square().sum(),
                          atol=1e-11, rtol=1e-13)
    components = list(bands.values())
    for i, a in enumerate(components):
        for b in components[i + 1:]:
            assert abs(float((a * b).sum())) < 2e-13


def test_explicit_axis_convention_and_coefficient_scaling():
    x = torch.tensor([[[[1., 2.], [3., 4.]]]])
    bands = haar_analysis2d(x)
    assert bands.ll.item() == 5.0
    assert bands.lh.item() == -1.0  # high along horizontal columns
    assert bands.hl.item() == -2.0  # high along vertical rows
    assert bands.hh.item() == 0.0
    assert torch.equal(haar_synthesis2d(bands), x)


def test_projection_is_block_mean_signed_unclipped_and_zero_dc():
    x = torch.tensor([[[[-4., 2., 8., -2.], [1., 3., -1., 5.],
                         [0., 2., -8., 7.], [-1., 4., 6., -3.]]]], dtype=torch.float64)
    low, detail = wavelet_components(x, levels=2)
    assert torch.equal(low, torch.full_like(x, x.mean()))
    assert torch.equal(low + detail, x)
    assert detail.min() < -1 and detail.max() > 1
    assert detail.mean() == 0
    assert torch.allclose((low * detail).sum(), torch.tensor(0., dtype=torch.float64))
    constant = torch.full((2, 3, 8, 12), -3.25, dtype=torch.float64)
    low, detail = wavelet_components(constant, 2)
    assert torch.equal(low, constant)
    assert torch.count_nonzero(detail) == 0


def test_normalized_decomposition_does_not_inject_detail_mean():
    x = torch.arange(3 * 8 * 8, dtype=torch.float64).reshape(1, 3, 8, 8) / 191
    mean = torch.tensor([.485, .456, .406], dtype=x.dtype).reshape(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], dtype=x.dtype).reshape(1, 3, 1, 1)
    raw_low, raw_detail = wavelet_components(x, 2)
    low, detail = wavelet_components((x - mean) / std, 2)
    assert torch.allclose(low, (raw_low - mean) / std, atol=2e-15, rtol=2e-15)
    assert torch.allclose(detail, raw_detail / std, atol=2e-15, rtol=2e-15)
    assert torch.allclose(detail.mean((-2, -1)), torch.zeros(1, 3, dtype=x.dtype), atol=1e-15)


def test_projection_rank_predicts_white_noise_energy_fraction():
    # Apply the projection to a complete orthonormal pixel basis: trace(P)=1
    # for one 4x4 block, hence iid noise retains exactly 1/16 expected energy.
    basis = torch.eye(16, dtype=torch.float64).reshape(16, 1, 4, 4)
    low, detail = wavelet_components(basis, 2)
    assert low.square().sum().item() == 1.0
    assert detail.square().sum().item() == 15.0


def test_decimation_is_shift_variant_but_covariant_to_full_block_shifts():
    x = torch.zeros(1, 1, 8, 12, dtype=torch.float64)
    x[0, 0, 1, 3] = 1
    low, _ = wavelet_components(x, 2)
    shifted_one, _ = wavelet_components(x.roll(1, -1), 2)
    shifted_four, _ = wavelet_components(x.roll(4, -1), 2)
    assert not torch.equal(shifted_one, low.roll(1, -1))
    assert torch.equal(shifted_four, low.roll(4, -1))


def test_gradients_and_noncontiguous_input():
    x = torch.randn(1, 1, 4, 8, dtype=torch.float64).transpose(-1, -2).requires_grad_()
    assert not x.is_contiguous()
    assert torch.autograd.gradcheck(lambda z: wavelet_components(z, 2), (x,))
    reconstructed = haar_reconstruct(haar_decompose(x, 2))
    reconstructed.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x), atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("x,levels", [
    (torch.ones(3, 4, 4), 1),
    (torch.ones(1, 1, 3, 4), 1),
    (torch.ones(1, 1, 4, 6), 2),
    (torch.ones(1, 1, 4, 4, dtype=torch.int64), 1),
    (torch.ones(0, 1, 4, 4), 1),
    (torch.ones(1, 1, 4, 4), 0),
    (torch.ones(1, 1, 4, 4), True),
    (torch.ones(1, 1, 4, 4), 1.5),
])
def test_invalid_analysis_inputs_fail(x, levels):
    with pytest.raises(ValueError):
        haar_decompose(x, levels)


def test_malformed_synthesis_fails():
    x = torch.ones(1, 1, 2, 2)
    with pytest.raises(ValueError):
        haar_synthesis2d(HaarBands(x, x[:, :, :, :1], x, x))


@pytest.mark.parametrize("method", METHODS)
def test_methods_forward_and_exact_input_routing(method, initial_state):
    model = WaveletEmotionModel(method, initial_state, levels=2, head_seed=29).eval()
    x = torch.randn(2, 3, 32, 32)
    low, detail = model.components(x)
    expected = {"rgb": (x,), "low": (low,), "detail": (detail,),
                "wavelet_fusion": (low, detail), "rgb_dual": (x, x), "low_dual": (low, low)}
    for a, b in zip(model.inputs(x), expected[method]):
        assert torch.equal(a, b)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 8)
    assert torch.isfinite(logits).all()


def test_dual_controls_match_parameters_and_head_but_break_gradient_symmetry(initial_state):
    single = WaveletEmotionModel("rgb", initial_state, head_seed=17).eval()
    dual = WaveletEmotionModel("rgb_dual", initial_state, head_seed=17).eval()
    fusion = WaveletEmotionModel("wavelet_fusion", initial_state, head_seed=17).eval()
    assert sum(p.numel() for p in dual.parameters()) == sum(p.numel() for p in fusion.parameters())
    assert torch.equal(dual.classifier.weight, fusion.classifier.weight)
    assert torch.equal(dual.classifier.bias, fusion.classifier.bias)
    left, right = dual.classifier.weight.chunk(2, dim=1)
    assert not torch.equal(left, right)
    assert torch.allclose(left + right, initial_state["fc.weight"], atol=4e-9, rtol=1e-6)
    assert dual.branches[0].conv1.weight.data_ptr() != dual.branches[1].conv1.weight.data_ptr()
    assert torch.equal(dual.branches[0].conv1.weight, dual.branches[1].conv1.weight)
    x = torch.randn(2, 3, 32, 32)
    assert torch.allclose(single(x), dual(x), atol=2e-5, rtol=2e-5)
    dual(x).square().mean().backward()
    a, b = dual.branches[0].conv1.weight.grad, dual.branches[1].conv1.weight.grad
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert not torch.allclose(a, b, atol=1e-8, rtol=1e-6)


def test_head_uses_local_rng_and_explicit_seed(initial_state):
    a = WaveletEmotionModel("rgb_dual", initial_state, head_seed=12)
    b = WaveletEmotionModel("rgb_dual", initial_state, head_seed=13)
    assert not torch.equal(a.classifier.weight, b.classifier.weight)
    assert torch.equal(a.branches[0].conv1.weight, b.branches[0].conv1.weight)
    with pytest.raises(ValueError):
        WaveletEmotionModel("unknown", initial_state)
