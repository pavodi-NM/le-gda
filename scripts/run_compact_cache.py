"""Seeded k-means++ used by the GPCache baseline (scripts/run_baselines_e4.py)
to build m cache keys per class from the K-shot support set.
"""

import torch
import torch.nn.functional as F


def kmeans(X, m, seed, iters=25):
    """Seeded k-means++ on rows of X [n, D]; returns [m, D] unit-norm centroids."""
    n = X.shape[0]
    if m >= n:
        return F.normalize(X, dim=-1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    # k-means++ init
    idx = [int(torch.randint(n, (1,), generator=g))]
    for _ in range(m - 1):
        d2 = torch.cdist(X, X[idx]).min(dim=1).values ** 2
        p = (d2 / d2.sum().clamp_min(1e-12)).cpu()
        idx.append(int(torch.multinomial(p, 1, generator=g)))
    C = X[idx].clone()
    for _ in range(iters):
        assign = torch.cdist(X, C).argmin(dim=1)
        for j in range(m):
            sel = assign == j
            if sel.any():
                C[j] = X[sel].mean(0)
            else:                                   # reseed empty cluster
                C[j] = X[torch.cdist(X, C).min(dim=1).values.argmax()]
    return F.normalize(C, dim=-1)
