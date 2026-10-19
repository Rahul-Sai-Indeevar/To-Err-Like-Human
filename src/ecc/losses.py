"""Cross-entropy and ListMLE (Xia et al. 2008) combined loss  L_c = L_CE + alpha * L_ListMLE."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def listmle_loss(logits: torch.Tensor, perm: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    """Plackett-Luce negative log-likelihood of the target ordering.

    logits: (B, N) scores. perm: (B, N) class indices, best first.
    The first term of the sum equals cross-entropy w.r.t. perm[:, 0].
    reduction: how to aggregate over the N list positions ("sum" = standard ListMLE,
    "mean" = scaled down by N). The paper does not say which it uses -- see README.
    """
    s = torch.gather(logits, 1, perm)
    tail_lse = torch.logcumsumexp(s.flip(1), dim=1).flip(1)  # log sum_{k>=j} exp(s_k)
    nll = tail_lse - s
    nll = nll.sum(1) if reduction == "sum" else nll.mean(1)
    return nll.mean()


class CombinedLoss(nn.Module):
    def __init__(self, rank_matrix, alpha: float = 0.0, reduction: str = "sum"):
        super().__init__()
        self.register_buffer("rank", torch.as_tensor(rank_matrix, dtype=torch.long))
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        loss = F.cross_entropy(logits, targets)
        if self.alpha > 0:
            loss = loss + self.alpha * listmle_loss(logits, self.rank[targets], self.reduction)
        return loss
