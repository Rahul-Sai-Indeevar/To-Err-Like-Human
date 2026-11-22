"""Graph signal processing on the emotion-label domain (not image frequencies).

The proposed inverse-eigenvalue residual weights emphasize LOW-frequency error.
They are not the usual positive-order Sobolev norm. The DC residual vanishes for
two unit-mass distributions. Repeated eigenspaces are basis-invariant because
their weights coincide. No eigenvector signs are assumed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def graph_basis(W, topology="complete"):
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or not np.all(np.isfinite(W)):
        raise ValueError("W must be a finite square matrix")
    if not np.allclose(W, W.T) or np.any(W <= 0):
        raise ValueError("W must be positive and symmetric")
    A = 1.0 / W
    np.fill_diagonal(A, 0)
    if topology == "uniform":
        A = (A > 0).astype(float)
    elif topology == "uniform_trace_matched":
        # Remove emotional geometry while preserving total graph edge weight.
        scale = A.sum() / (len(A) * (len(A) - 1))
        A = (A > 0).astype(float) * scale
    elif topology != "complete":
        raise ValueError("Unknown graph topology")
    L = np.diag(A.sum(1)) - A
    values, U = np.linalg.eigh(L)
    values[np.abs(values) < 1e-10] = 0
    return A, L, values, U


def spectral_weights(values, gamma=1.0):
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    weights = np.zeros_like(values)
    positive = values > 1e-10
    weights[positive] = values[positive] ** (-gamma)
    return weights


def heat_filter(probabilities, W, time=0.0):
    """Mass-preserving graph diffusion exp(-t L); t is a graph scale, not seconds."""
    if time < 0:
        raise ValueError("Diffusion time must be nonnegative")
    _, _, values, U = graph_basis(W)
    kernel = (U * np.exp(-time * values)) @ U.T
    p = np.asarray(probabilities, dtype=np.float64) @ kernel
    # Roundoff only: exp(-tL) is nonnegative and doubly stochastic analytically.
    p = np.maximum(p, 0)
    return p / p.sum(axis=1, keepdims=True)


class GraphSpectralLoss(nn.Module):
    """CE + alpha * [sum lambda^-gamma |GFT(p-y)|^2 + beta p^T L p]."""

    def __init__(self, W, alpha=1.0, gamma=1.0, beta=0.01, topology="complete"):
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be nonnegative")
        _, L, values, U = graph_basis(W, topology)
        self.register_buffer("basis", torch.tensor(U, dtype=torch.float32))
        self.register_buffer("laplacian", torch.tensor(L, dtype=torch.float32))
        self.register_buffer(
            "weights",
            torch.tensor(spectral_weights(values, gamma), dtype=torch.float32),
        )
        self.alpha, self.beta = alpha, beta

    def components(self, logits, targets):
        p = logits.softmax(1)
        y = F.one_hot(targets, p.shape[1]).to(p.dtype)
        residual = (p - y) @ self.basis.to(p.dtype)
        spectral = (residual.square() * self.weights.to(p.dtype)).sum(1).mean()
        smoothness = ((p @ self.laplacian.to(p.dtype)) * p).sum(1).mean()
        return spectral, smoothness

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets)
        if self.alpha == 0:
            return ce
        spectral, smoothness = self.components(logits, targets)
        return ce + self.alpha * (spectral + self.beta * smoothness)


def build_loss(loss_cfg, W, rank):
    """Opt-in dispatch; existing configs retain exactly the ListMLE behavior."""
    from .losses import CombinedLoss

    kind = loss_cfg.get("kind", "listmle")
    if kind == "graph":
        return GraphSpectralLoss(
            W,
            alpha=loss_cfg["alpha"],
            gamma=loss_cfg.get("gamma", 1.0),
            beta=loss_cfg.get("beta", 0.01),
            topology=loss_cfg.get("topology", "complete"),
        )
    if kind not in ("ce", "listmle"):
        raise ValueError(f"Unknown loss {kind}")
    return CombinedLoss(
        rank,
        0.0 if kind == "ce" else loss_cfg["alpha"],
        loss_cfg.get("listmle_reduction", "sum"),
    )
