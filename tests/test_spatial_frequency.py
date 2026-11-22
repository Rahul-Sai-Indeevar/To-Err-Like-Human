import pytest
import torch
from ecc.spatial_frequency import butterworth_mask, transform_images


def test_butterworth_dc_complement_and_mean():
    x = torch.rand(3, 3, 32, 32) * 255
    low = transform_images(x, "lowpass", 0.1)
    high = transform_images(x, "highpass_dc", 0.1)
    mean = x.mean((-2, -1), keepdim=True)
    assert torch.allclose(low + high, x + mean, atol=1e-4)
    assert torch.allclose(low.mean((-2, -1)), mean[..., 0, 0], atol=2e-5)
    assert torch.allclose(high.mean((-2, -1)), mean[..., 0, 0], atol=2e-5)
    assert butterworth_mask((32, 32), 0.1)[0, 0] == 1


def test_phase_controls_preserve_unclipped_amplitude_and_parseval():
    torch.manual_seed(8)
    x = torch.rand(2, 3, 32, 32) * 255
    reference = torch.fft.fft2(x, norm="ortho").abs()
    for mode in ("phase_zero", "phase_random"):
        out = transform_images(x, mode)
        assert torch.allclose(
            torch.fft.fft2(out, norm="ortho").abs(), reference, atol=5e-4, rtol=1e-4
        )
        assert torch.allclose(
            out.square().sum((-2, -1)), x.square().sum((-2, -1)), rtol=1e-5
        )
    flat = transform_images(x, "amplitude_flat")
    assert torch.allclose(
        flat.square().sum((-2, -1)), x.square().sum((-2, -1)), rtol=1e-5
    )


def test_constant_signal_and_invalid_cutoff():
    x = torch.ones(1, 3, 16, 16) * 123
    for mode in ["clean", "lowpass", "highpass_dc"]:
        assert torch.allclose(transform_images(x, mode, 0.1), x)
    with pytest.raises(ValueError):
        butterworth_mask((16, 16), 0)
