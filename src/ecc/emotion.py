"""Mikel's emotion wheel, emotional distance W (paper Eq. 1) and label ranks.

ASSUMPTION TO VERIFY: the circular order of the wheel below was derived so that the
excitement row reproduces the ranking given in the paper's Appendix B
(excitement, awe, contentment, amusement, fear, sadness, disgust, anger).
Compare with the order used in the official repo's metric.py and edit WHEEL_ORDER
if it differs -- it is the only place the wheel is defined.
"""
from __future__ import annotations

import numpy as np

WHEEL_ORDER = [
    "excitement", "awe", "contentment", "amusement",   # positive arc
    "anger", "disgust", "sadness", "fear",             # negative arc
]
POSITIVE = frozenset({"excitement", "awe", "contentment", "amusement"})
DEFAULT_C = 4  # polarity constant from the paper


def _check(names):
    bad = [n for n in names if n.lower() not in WHEEL_ORDER]
    if bad:
        raise ValueError(f"Unknown emotion class names {bad}; expected {WHEEL_ORDER}")


def wheel_distance(a: str, b: str) -> int:
    """Number of steps between two emotions on the (circular) wheel."""
    n = len(WHEEL_ORDER)
    d = abs(WHEEL_ORDER.index(a) - WHEEL_ORDER.index(b))
    return min(d, n - d)


def build_cost_matrix(class_names, C: int = DEFAULT_C) -> np.ndarray:
    """Symmetric emotional-distance matrix W (Eq. 1). W[i,i] == 1."""
    names = [n.lower() for n in class_names]
    _check(names)
    N = len(names)
    W = np.zeros((N, N), dtype=np.float64)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            same = (a in POSITIVE) == (b in POSITIVE)
            W[i, j] = (1 if same else C) + wheel_distance(a, b)
    return W


def polarity_vector(class_names) -> np.ndarray:
    """1 for positive emotions, 0 for negative (used for ACC2)."""
    return np.array([1 if n.lower() in POSITIVE else 0 for n in class_names])


def build_rank_matrix(W: np.ndarray) -> np.ndarray:
    """Row i = class indices sorted from closest to farthest from class i.

    Used as the target permutation for ListMLE. Ties are broken by class index
    (stable sort); the ground-truth class always comes first because W[i,i]=1.
    """
    return np.argsort(W, axis=1, kind="stable")
