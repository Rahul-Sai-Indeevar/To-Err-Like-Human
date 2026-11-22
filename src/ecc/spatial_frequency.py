"""Radial Butterworth filtering of RGB pixel arrays."""
import torch


def butterworth_mask(shape, cutoff, order=2, *, device=None):
    """Return gains on the unshifted FFT grid, with cutoff in cycles per pixel."""
    if not 0 < cutoff or order < 1:
        raise ValueError("Use a positive cutoff and order")
    rows, columns = shape
    fy = torch.fft.fftfreq(rows, device=device)[:, None]
    fx = torch.fft.fftfreq(columns, device=device)[None, :]
    radius = torch.sqrt(fx.square() + fy.square())
    return 1 / (1 + (radius / cutoff).pow(2 * order))


def lowpass_rgb(images, cutoff):
    """Filter over the last two dimensions, independently per channel.

    Uses periodic boundaries and returns unclipped float pixels in the input scale.
    """
    pixels = images.float()
    spectrum = torch.fft.fft2(pixels, norm="ortho")
    gain = butterworth_mask(pixels.shape[-2:], cutoff, device=pixels.device)
    return torch.fft.ifft2(gain * spectrum, norm="ortho").real
