import numpy as np
import torch
import torch.nn.functional as F

from ecc.emotion import build_cost_matrix, build_rank_matrix
from ecc.losses import CombinedLoss, listmle_loss

CLASSES = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
RANK = build_rank_matrix(build_cost_matrix(CLASSES))


def test_listmle_ge_cross_entropy_and_first_term_is_ce():
    torch.manual_seed(0)
    logits = torch.randn(16, 8)
    y = torch.randint(0, 8, (16,))
    perm = torch.as_tensor(RANK)[y]
    assert listmle_loss(logits, perm).item() >= F.cross_entropy(logits, y).item() - 1e-6


def test_listmle_prefers_correct_order():
    y = torch.tensor([5])                       # excitement
    perm = torch.as_tensor(RANK)[y]
    good = torch.zeros(1, 8)
    for pos, c in enumerate(perm[0].tolist()):
        good[0, c] = 8 - pos                    # scores follow the target ranking
    bad = good.flip(1)
    assert listmle_loss(good, perm) < listmle_loss(bad, perm)


def test_combined_loss_alpha_zero_is_ce_and_grads_flow():
    logits = torch.randn(4, 8, requires_grad=True)
    y = torch.randint(0, 8, (4,))
    assert torch.isclose(CombinedLoss(RANK, 0.0)(logits, y), F.cross_entropy(logits, y))
    CombinedLoss(RANK, 1.0)(logits, y).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
