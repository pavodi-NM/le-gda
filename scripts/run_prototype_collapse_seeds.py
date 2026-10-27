"""Shared helpers: the seeded few-shot draw, and one Tip-Adapter cache arm.

Every method draws its K-shot support set with `draw`, from the per-dataset
training pool written by extract_train_cache_pool.py, so all methods see the
same examples for a given (dataset, K, seed).
"""

import torch

from scripts.run_cache_retrieval_null import acc, search


def draw(pool_y, K, seed, device):
    """K indices per class, deterministic given seed."""
    g = torch.Generator().manual_seed(seed)
    idx = []
    for c in range(int(pool_y.max()) + 1):
        pos = (pool_y == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(pos), generator=g)[:K]
        idx.append(pos[perm])
    return torch.cat(idx).to(device)


def run_arm(keys, V, f_val, f_test, clip_val, clip_test, y_val, y_test,
            betas, alphas):
    """Test accuracy of a (keys, V) cache at its validation-selected beta, alpha."""
    b, a = search(betas, alphas, f_val @ keys, V, clip_val, y_val)
    return acc(clip_test + a * (((-1) * (b - b * (f_test @ keys))).exp() @ V),
               y_test)
