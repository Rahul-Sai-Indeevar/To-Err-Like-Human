import numpy as np
import pytest

from ecc.emotion import build_cost_matrix, build_rank_matrix, polarity_vector
from ecc.metrics import acc, acc2, confusion_matrix, ecc, emc, evaluate_predictions

CLASSES = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
W = build_cost_matrix(CLASSES)
POL = polarity_vector(CLASSES)
ix = {c: i for i, c in enumerate(CLASSES)}


def test_cost_matrix_properties():
    assert np.allclose(W, W.T)
    assert np.allclose(np.diag(W), 1)
    # any cross-polarity distance exceeds any same-polarity distance (the point of C=4)
    same = (POL[:, None] == POL[None, :]) & ~np.eye(8, dtype=bool)
    assert W[same].max() < W[POL[:, None] != POL[None, :]].min()


def test_rank_matches_paper_appendix_b():
    # Paper: excitement -> awe, contentment, amusement, fear, sadness, disgust, anger
    r = build_rank_matrix(W)[ix["excitement"]]
    assert [CLASSES[i] for i in r] == ["excitement", "awe", "contentment", "amusement",
                                       "fear", "sadness", "disgust", "anger"]


def test_perfect_classifier():
    y = np.arange(8).repeat(5)
    m = evaluate_predictions(y, y, W, POL)
    assert m == {"acc": 100, "acc2": 100, "ecc": 100, "emc": 100}


def test_ecc_between_acc_and_one_and_orders_mistakes():
    y = np.array([ix["excitement"]] * 10)
    near = np.array([ix["awe"]] * 10)       # same polarity, close
    far = np.array([ix["anger"]] * 10)      # opposite polarity
    S_near, S_far = confusion_matrix(y, near, 8), confusion_matrix(y, far, 8)
    assert acc(S_near) == acc(S_far) == 0
    assert ecc(S_near, W) > ecc(S_far, W) > 0
    assert emc(S_near, W) > emc(S_far, W)
    assert acc2(S_near, POL) == 1 and acc2(S_far, POL) == 0


def test_emc_range_and_max():
    y = np.array([ix["excitement"]] * 4)
    adj = np.array([ix["awe"]] * 4)          # W=2 -> weight 1/(W-1)=1 -> EMC=1
    assert emc(confusion_matrix(y, adj, 8), W) == pytest.approx(1.0)
    rng = np.random.RandomState(0)
    S = confusion_matrix(rng.randint(0, 8, 500), rng.randint(0, 8, 500), 8)
    assert 0 < emc(S, W) <= 1 and acc(S) <= ecc(S, W) <= 1
