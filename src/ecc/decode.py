"""Test-time minimum-expected-cost decoding (idea from Conditional Risk Minimization).

Instead of argmax, predict the class j minimising  sum_i p_i * cost(i, j)
with cost(i, j) = 1 - 1/W_ij  (0 on the diagonal, grows with emotional distance).
This is an EXTENSION, not part of the paper.
"""
from __future__ import annotations

import numpy as np


def mec_decode(probs: np.ndarray, W: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    cost = 1.0 - 1.0 / W
    p = probs ** (1.0 / temperature)
    p = p / p.sum(1, keepdims=True)
    return (p @ cost).argmin(1)
