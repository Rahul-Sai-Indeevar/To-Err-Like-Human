"""ACC, ACC2, ECC (Eq. 3) and EMC (Eq. 4), all computed from a confusion matrix.

Confusion matrix S: rows = true class, columns = predicted class.
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    S = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(S, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return S


def acc(S: np.ndarray) -> float:
    return float(np.trace(S) / S.sum())


def acc2(S: np.ndarray, polarity: np.ndarray) -> float:
    """Polarity accuracy: a prediction is right if it has the same polarity as the label."""
    same = polarity[:, None] == polarity[None, :]
    return float((S * same).sum() / S.sum())


def ecc(S: np.ndarray, W: np.ndarray) -> float:
    """Emotion Confusion Confidence: sum_ij S_ij / W_ij / N  (diagonal weight is 1)."""
    return float((S / W).sum() / S.sum())


def emc(S: np.ndarray, W: np.ndarray) -> float:
    """Emotional Misclassification Confidence over the misclassified samples only.

    EMC = sum_{i != j} S_ij / (W_ij - 1)  /  (N - N_c)
    Returns 1.0 when there are no mistakes (undefined in the paper; 1.0 is the
    'best possible' convention).
    """
    off = ~np.eye(len(S), dtype=bool)
    n_err = S[off].sum()
    if n_err == 0:
        return 1.0
    inv = np.zeros_like(W)
    inv[off] = 1.0 / (W[off] - 1.0)
    return float((S * inv).sum() / n_err)


def evaluate_predictions(y_true, y_pred, W: np.ndarray, polarity: np.ndarray) -> dict:
    """All four metrics in percent."""
    S = confusion_matrix(y_true, y_pred, len(W))
    return {
        "acc": 100 * acc(S),
        "acc2": 100 * acc2(S, polarity),
        "ecc": 100 * ecc(S, W),
        "emc": 100 * emc(S, W),
    }
