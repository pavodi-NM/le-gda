"""Cache-model construction utilities, shared by the baseline arms.

LE-GDA itself builds no cache. This module is here because the comparative
methods do, and they must build theirs the same way to stay comparable:

    kmeans / skmeans    m centroids per class (spherical variant for cosine)
    herd                m exemplars per class by herding
    build_class_cache   the per-class key/value tensors a cache model scores with
    acc_chunked         chunked accuracy, so a 40k x 1000 logit block fits in 8 GB

run_calibrated_ft.py (Tip-Adapter, Tip-Adapter-F, Proto-Adapter-F),
run_baselines_e4.py (GPCache) both import from here, so a
change in this file moves every baseline at once -- which is the point.

Running this module directly executes a standalone cache-size sweep that is not
part of the LE-GDA results; nothing in the README calls it.
"""

import argparse
import json
from statistics import mean, pstdev

import torch
import torch.nn.functional as F

import clip
from utils import clip_classifier
from datasets import build_dataset
from scripts.ds_meta import class_template
from scripts.run_cache_retrieval_null import ROOT, load_cfg, grid, acc, search
from scripts.run_prototype_collapse_seeds import draw, run_arm

DATASETS = ["caltech101", "dtd", "eurosat", "food101", "oxford_pets"]
BACKBONES = ["RN50", "RN101", "ViT-B/32", "ViT-B/16"]
SHOTS = [4, 16]
MS = [1, 2, 4, 8]
SEEDS = [0, 1, 2]


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


def skmeans(X, m, seed, iters=25):
    """Spherical k-means on rows of X [n, D]; returns [m, D] unit-norm centroids.

    Differs from `kmeans` in the metric, not the schedule: assignment is by
    maximum cosine similarity to UNIT-NORM centroids, and the update is the
    normalised mean of each cluster. `kmeans` assigns by Euclidean distance to
    un-normalised running means and only normalises at the very end, so the two
    disagree during the loop even though the features are unit-norm.

    This is the metric the cache actually retrieves under -- the affinity is
    exp(-beta(1 - f.k)) with unit-norm f and k -- so spherical k-means is the
    quantiser matched to the scoring function.

    At m=1 it is identical to `kmeans` by construction (both reduce to the
    normalised class mean); it is used here as an implementation check.
    """
    n = X.shape[0]
    if m >= n:
        return F.normalize(X, dim=-1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    # k-means++ init, cosine distance
    idx = [int(torch.randint(n, (1,), generator=g))]
    for _ in range(m - 1):
        d2 = (1 - X @ F.normalize(X[idx], dim=-1).t()).min(dim=1).values ** 2
        p = (d2 / d2.sum().clamp_min(1e-12)).cpu()
        idx.append(int(torch.multinomial(p, 1, generator=g)))
    C = F.normalize(X[idx].clone(), dim=-1)
    for _ in range(iters):
        sim = X @ C.t()
        assign = sim.argmax(dim=1)
        for j in range(m):
            sel = assign == j
            if sel.any():
                C[j] = F.normalize(X[sel].mean(0), dim=-1)
            else:                       # reseed on the least-covered point
                C[j] = X[(X @ C.t()).max(dim=1).values.argmin()]
    return C


def herd(X, m):
    """Classic herding selection on rows of X [n, D]; returns m selected rows."""
    mu = F.normalize(X.mean(0), dim=-1)
    picked, acc_sum = [], torch.zeros_like(mu)
    for _ in range(min(m, X.shape[0])):
        scores = (acc_sum + X) @ mu
        if picked:
            scores[torch.tensor(picked, dtype=torch.long, device=X.device)] = -1e9
        j = int(scores.argmax())
        picked.append(j)
        acc_sum = acc_sum + X[j]
    return X[picked]


def acc_chunked(keys, V, f, y, clip_logits, b, a, chunk=8192):
    """Test-side cache eval in chunks: the full affinity matrix on ImageNet
    (40k x 16k) plus its exp copy OOMs an 8 GB GPU."""
    correct = 0
    for i in range(0, f.shape[0], chunk):
        z = clip_logits[i:i + chunk] + a * (
            ((-1) * (b - b * (f[i:i + chunk] @ keys))).exp() @ V)
        correct += (z.argmax(1) == y[i:i + chunk]).sum().item()
    return correct / f.shape[0] * 100


def run_arm_chunked(keys, V, f_val, f_test, clip_val, clip_test,
                    y_val, y_test, betas, alphas):
    b, a = search(betas, alphas, f_val @ keys, V, clip_val, y_val)
    return acc_chunked(keys, V, f_test, y_test, clip_test, b, a)


def build_class_cache(f_tr, y_tr, n_cls, fn):
    """Apply fn per class; returns keys [D, m*C] and one-hot values [m*C, C]."""
    keys, labels = [], []
    for c in range(n_cls):
        Xc = f_tr[y_tr == c]
        Kc = fn(Xc)
        keys.append(Kc)
        labels += [c] * Kc.shape[0]
    keys = torch.cat(keys, 0)
    V = F.one_hot(torch.tensor(labels, device=keys.device), n_cls).float()
    return keys.t(), V


# This module is a library for the baseline runners: it has no entry point of
# its own. Import kmeans / skmeans / herd / build_class_cache / acc_chunked.
