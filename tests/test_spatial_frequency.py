import pytest
import torch

from ecc.spatial_frequency import butterworth_mask, lowpass_rgb


def test_mask_and_pixels_recover_butterworth_gain_at_cutoff():
    # One whole cycle in eight columns lies exactly at the chosen cutoff.
    f = 1 / 8
    pixels = 100 + 20 * torch.cos(2 * torch.pi * f * torch.arange(32))
    images = pixels.expand(2, 3, 32, 32)
    mask = butterworth_mask((32, 32), f)
    assert mask[0, 0] == 1
    assert mask[0, 4] == 0.5
    assert mask[0, 8] == pytest.approx(1 / 17)
    torch.testing.assert_close(lowpass_rgb(images, f), 100 + (images - 100) / 2)


def test_invalid_cutoff_and_constant_rgb():
    images = torch.full((4, 3, 8, 10), 42, dtype=torch.uint8)
    torch.testing.assert_close(lowpass_rgb(images, .10), images.float())
    with pytest.raises(ValueError):
        lowpass_rgb(images, 0)
