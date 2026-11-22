"""Exact Haar filter banks and opt-in, capacity-controlled emotion classifiers.

The first band letter refers to filtering along the row (vertical) axis; the
second refers to columns (horizontal). Thus LH is low along rows and high
along columns. These names describe filter axes, not ambiguous edge labels.

Haar analysis is critically sampled and shift variant. Its *complete* bank is
orthonormal and perfectly reconstructing. A retained approximation alone is a
block-average projection, not an ideal low-pass or a shift-invariant filter.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torchvision.models import resnet18


METHODS = ("rgb", "low", "detail", "wavelet_fusion", "rgb_dual", "low_dual")
HEAD_ASYMMETRY_STD = 0.01


@dataclass(frozen=True)
class HaarDetails:
    lh: Tensor
    hl: Tensor
    hh: Tensor


@dataclass(frozen=True)
class HaarBands:
    ll: Tensor
    lh: Tensor
    hl: Tensor
    hh: Tensor


@dataclass(frozen=True)
class HaarPyramid:
    """Coarsest LL and detail tuples ordered finest to coarsest."""
    approximation: Tensor
    details: tuple[HaarDetails, ...]

    @property
    def levels(self) -> int:
        return len(self.details)


def _check_image(x: Tensor) -> None:
    if not isinstance(x, Tensor) or x.ndim != 4:
        raise ValueError("Expected a four-dimensional NCHW tensor")
    if not x.is_floating_point():
        raise ValueError("Haar transforms require a real floating-point tensor")
    if any(d < 1 for d in x.shape):
        raise ValueError("Every NCHW dimension must be positive")


def _check_levels(x: Tensor, levels: int) -> None:
    _check_image(x)
    if not isinstance(levels, int) or isinstance(levels, bool) or levels < 1:
        raise ValueError("levels must be a positive integer")
    if x.shape[-2] % (2 ** levels) or x.shape[-1] % (2 ** levels):
        raise ValueError("Both spatial dimensions must be divisible by 2**levels")


def haar_analysis2d(x: Tensor) -> HaarBands:
    """Apply separable [1,1]/sqrt(2), [1,-1]/sqrt(2) and decimate by two.

    For a 2x2 block [a,b;c,d], LL=(a+b+c+d)/2. LL is a
    coefficient, not the block mean; inverse reconstruction supplies the
    second factor of two. No padding, clipping or data-dependent gain occurs.
    """
    _check_levels(x, 1)
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return HaarBands((a + b + c + d) * 0.5,
                     (a - b + c - d) * 0.5,
                     (a + b - c - d) * 0.5,
                     (a - b - c + d) * 0.5)


def haar_synthesis2d(bands: HaarBands) -> Tensor:
    """Invert one level of the orthonormal bank, retaining signed details."""
    if not isinstance(bands, HaarBands):
        raise ValueError("Expected HaarBands")
    ll, lh, hl, hh = bands.ll, bands.lh, bands.hl, bands.hh
    _check_image(ll)
    for band in (lh, hl, hh):
        _check_image(band)
        if band.shape != ll.shape or band.dtype != ll.dtype or band.device != ll.device:
            raise ValueError("All Haar bands must have identical shape, dtype and device")
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    even = torch.stack((a, b), dim=-1).flatten(-2)
    odd = torch.stack((c, d), dim=-1).flatten(-2)
    return torch.stack((even, odd), dim=-2).flatten(-3, -2)


def haar_decompose(x: Tensor, levels: int = 2) -> HaarPyramid:
    """Repeatedly decompose LL; details retain finest-to-coarsest order."""
    _check_levels(x, levels)
    details = []
    approximation = x
    for _ in range(levels):
        bands = haar_analysis2d(approximation)
        approximation = bands.ll
        details.append(HaarDetails(bands.lh, bands.hl, bands.hh))
    return HaarPyramid(approximation, tuple(details))


def haar_reconstruct(pyramid: HaarPyramid) -> Tensor:
    """Reconstruct every level of a pyramid with its complete filter bank."""
    if not isinstance(pyramid, HaarPyramid) or not pyramid.details:
        raise ValueError("Expected a nonempty HaarPyramid")
    x = pyramid.approximation
    for detail in reversed(pyramid.details):
        if not isinstance(detail, HaarDetails):
            raise ValueError("Every detail level must be HaarDetails")
        x = haar_synthesis2d(HaarBands(x, detail.lh, detail.hl, detail.hh))
    return x


def reconstruct_bands(pyramid: HaarPyramid) -> dict[str, Tensor]:
    """Return same-size orthogonal components LL_J and every oriented detail.

    Keys are ``LL2``, ``L1_LH``, ``L1_HL``, ``L1_HH``, ``L2_LH``, etc. for
    a two-level pyramid. Each component is reconstructed with all other
    coefficients zero. Their sum reproduces the input; their squared norms
    sum to its squared norm, apart from floating-point roundoff.
    """
    if not isinstance(pyramid, HaarPyramid) or not pyramid.details:
        raise ValueError("Expected a nonempty HaarPyramid")
    zero_details = tuple(HaarDetails(torch.zeros_like(d.lh), torch.zeros_like(d.hl),
                                    torch.zeros_like(d.hh)) for d in pyramid.details)
    result = {f"LL{pyramid.levels}": haar_reconstruct(
        HaarPyramid(pyramid.approximation, zero_details))}
    for i, detail in enumerate(pyramid.details):
        for name in ("lh", "hl", "hh"):
            selected = list(zero_details)
            selected[i] = HaarDetails(*(getattr(detail, band) if band == name
                                       else torch.zeros_like(getattr(detail, band))
                                       for band in ("lh", "hl", "hh")))
            result[f"L{i + 1}_{name.upper()}"] = haar_reconstruct(HaarPyramid(
                torch.zeros_like(pyramid.approximation), tuple(selected)))
    return result


def wavelet_components(x: Tensor, levels: int = 2) -> tuple[Tensor, Tensor]:
    """Return P_J x and (I-P_J)x without clipping or magnitude conversion.

    P_J is the orthogonal projection onto blocks constant over 2**J pixels in
    each direction. For ImageNet-normalized z=(x-mu)/sigma, low=P_J z and
    detail=(I-P_J)z=(x-P_J x)/sigma. No artificial mean is injected into detail.
    """
    pyramid = haar_decompose(x, levels)
    approximation = pyramid.approximation
    for detail in reversed(pyramid.details):
        zeros = torch.zeros_like(detail.lh)
        approximation = haar_synthesis2d(HaarBands(approximation, zeros, zeros, zeros))
    return approximation, x - approximation


class WaveletEmotionModel(nn.Module):
    """Ablations of fixed signed Haar projections with matched dual controls.

    ``initial_state`` is a complete torchvision ResNet18 state with an already
    seeded K-class fc head. Every tower gets an independent copy of that same
    state. Dual classifiers concatenate two 512-dimensional features. Their
    weights are [W/2+delta, W/2-delta], delta~N(0,0.01**2), generated from a
    local CPU RNG seeded by ``head_seed``. Bias is the original head bias.

    The antisymmetric perturbation preserves the initial RGB-dual logits in
    real arithmetic while breaking tower-gradient symmetry. Without it,
    duplicate towers plus identical half-heads are a poor capacity control.
    Every dual method receives the same head for the same head_seed; nothing
    is tuned on validation or test outcomes. RGB-dual and low-dual genuinely
    have two independent trainable towers, with the same count as fusion.

    Inputs must already have the fixed RGB affine normalization. Transforms
    run in float32 (float64 preserved) under disabled autocast; each branch
    then follows its caller's autocast context. This opt-in class leaves the
    project's paper training defaults unchanged.
    """

    def __init__(self, method: str, initial_state: Mapping[str, Tensor],
                 levels: int = 2, head_seed: int = 0):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown wavelet method: {method}")
        if not isinstance(levels, int) or isinstance(levels, bool) or levels < 1:
            raise ValueError("levels must be a positive integer")
        if not isinstance(head_seed, int) or isinstance(head_seed, bool) or head_seed < 0:
            raise ValueError("head_seed must be a nonnegative integer")
        if "fc.weight" not in initial_state or "fc.bias" not in initial_state:
            raise ValueError("initial_state must include a complete ResNet18 fc head")
        weight, bias = initial_state["fc.weight"], initial_state["fc.bias"]
        if weight.ndim != 2 or weight.shape[1] != 512 or bias.shape != (weight.shape[0],):
            raise ValueError("Expected ResNet18 fc weight [classes,512] and bias [classes]")
        self.method, self.levels, self.head_seed = method, levels, head_seed
        self.dual = method in ("wavelet_fusion", "rgb_dual", "low_dual")
        self.branches = nn.ModuleList()
        for _ in range(2 if self.dual else 1):
            branch = resnet18(weights=None)
            branch.fc = nn.Linear(512, weight.shape[0])
            branch.load_state_dict(initial_state, strict=True)
            branch.fc = nn.Identity()
            self.branches.append(branch)
        self.classifier = nn.Linear(512 * len(self.branches), weight.shape[0])
        with torch.no_grad():
            source_weight = weight.detach().to(device="cpu", dtype=self.classifier.weight.dtype)
            if self.dual:
                generator = torch.Generator(device="cpu").manual_seed(head_seed)
                delta = torch.randn(source_weight.shape, generator=generator,
                                    dtype=source_weight.dtype) * HEAD_ASYMMETRY_STD
                self.classifier.weight.copy_(torch.cat((source_weight * 0.5 + delta,
                                                       source_weight * 0.5 - delta), dim=1))
            else:
                self.classifier.weight.copy_(source_weight)
            self.classifier.bias.copy_(bias.detach())

    def components(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return the same fixed low/detail preprocessing used in forward."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            return wavelet_components(x if x.dtype == torch.float64 else x.float(), self.levels)

    def inputs(self, x: Tensor) -> tuple[Tensor, ...]:
        """Expose the exact tensor given to each independent backbone."""
        if self.method == "rgb":
            return (x,)
        if self.method == "rgb_dual":
            return (x, x)
        low, detail = self.components(x)
        if self.method == "low":
            return (low,)
        if self.method == "detail":
            return (detail,)
        if self.method == "low_dual":
            return (low, low)
        return (low, detail)

    def forward(self, x: Tensor) -> Tensor:
        features = [branch(part) for branch, part in zip(self.branches, self.inputs(x))]
        return self.classifier(torch.cat(features, dim=1) if self.dual else features[0])
