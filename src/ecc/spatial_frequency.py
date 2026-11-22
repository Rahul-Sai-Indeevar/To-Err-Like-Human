"""Spatial Fourier controls in cycles/pixel, separate from label-graph frequencies."""

from __future__ import annotations
import torch


def butterworth_mask(shape, cutoff, mode="lowpass", order=2, *, device=None):
    if cutoff <= 0 or order < 1 or mode not in ("lowpass", "highpass_dc"):
        raise ValueError("Use positive cutoff/order and lowpass or highpass_dc")
    h, w = shape
    fy = torch.fft.fftfreq(h, device=device)[:, None]
    fx = torch.fft.fftfreq(w, device=device)[None, :]
    radius = torch.sqrt(fx.square() + fy.square())
    low = 1 / (1 + (radius / cutoff).pow(2 * order))
    if mode == "lowpass":
        return low
    high = 1 - low
    high[0, 0] = 1  # Explicitly keep per-channel mean, avoiding trivial dark inputs.
    return high


def transform_images(images, mode, cutoff=None, order=2, generator=None):
    """Return unclipped float RGB in the input's 0..255 scale.

    FFT uses periodic boundaries and acts on space only, independently per RGB
    channel. Clipping is a separate measured intervention in the experiment.
    """
    x = images.float()
    if mode == "clean":
        return x.clone()
    spectrum = torch.fft.fft2(x, norm="ortho")
    if mode in ("lowpass", "highpass_dc"):
        mask = butterworth_mask(x.shape[-2:], cutoff, mode, order, device=x.device)
        changed = spectrum * mask
    elif mode == "phase_zero":
        changed = spectrum.abs().to(spectrum.dtype)
        changed[..., 0, 0] = spectrum[..., 0, 0]
    elif mode == "phase_random":
        noise = torch.randn(x.shape, device=x.device, generator=generator)
        phase = torch.angle(torch.fft.fft2(noise, norm="ortho"))
        changed = spectrum.abs() * torch.exp(1j * phase)
        changed[..., 0, 0] = spectrum[..., 0, 0]
    elif mode == "amplitude_flat":
        power = spectrum.abs().square()
        ac_power = (power.sum(dim=(-2, -1)) - power[..., 0, 0]).clamp_min(0)
        amplitude = (ac_power / (x.shape[-2] * x.shape[-1] - 1)).sqrt()
        changed = amplitude[..., None, None] * torch.exp(1j * torch.angle(spectrum))
        changed[..., 0, 0] = spectrum[..., 0, 0]
    else:
        raise ValueError(f"Unknown transform: {mode}")
    return torch.fft.ifft2(changed, norm="ortho").real
