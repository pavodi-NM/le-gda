"""Size-calibrated cache scoring: baseline arms at a matched grid.

A cache term is a SUM over a class's entries, so its magnitude scales with the
entry count and `alpha` means something different for every arm. That confound
is monotone in cache size, so comparing arms of different sizes on one grid is
not a fair comparison -- it is a measurement of the grid.

The fix is to calibrate the term rather than widen the grid:

    contribution to class c = alpha * (K/m_c) * sum_{j in c} exp(-beta(1-f.k_j))

At m_c = K the factor is 1, so the full-cache arm is bit-identical to stock
Tip-Adapter and the standard alpha/beta grid applies unchanged to every arm.
This is a reparameterisation, not a method change -- scaling V by c and alpha
by 1/c is the same function -- so it cannot flatter the smaller caches; it only
puts every arm on a common scale so one bounded grid is fair to all of them.

`build_calibrated` is what run_calibrated_ft.py imports to build the head-to-
head's cache arms. LE-GDA has no cache term and no beta.
"""

import argparse
import json
from statistics import mean, median, pstdev

import torch
import torch.nn.functional as F

import clip
from utils import clip_classifier
from scripts.ds_meta import class_template
from scripts.run_cache_retrieval_null import load_cfg, grid, acc, search
from scripts.run_prototype_collapse_seeds import draw
from scripts.run_compact_cache import kmeans, skmeans, herd, acc_chunked

DATASETS = ["caltech101", "dtd", "eurosat", "food101", "oxford_pets",
            "oxford_flowers", "ucf101", "stanford_cars", "fgvc", "sun397",
            "imagenet"]
BACKBONES = ["RN50", "RN101", "ViT-B/32", "ViT-B/16"]
SEEDS = [0, 1, 2]
K = 16
MS = [1, 2, 4, 8]


def build_calibrated(f_tr, y_tr, n_cls, fn, shots=K):
    """keys [D, sum m_c] and SIZE-CALIBRATED values [sum m_c, C].

    V[j, c] = shots / m_c, so every arm's cache term carries the same expected
    magnitude as the full `shots`-entry cache and alpha is comparable across
    arms. At m_c = shots this is the one-hot cache, i.e. Tip-Adapter unchanged.

    `shots` MUST be passed when sweeping K. It defaulted to the module constant
    16, which is correct only at 16 shots -- at K=4 it would inflate every
    cache term 4x, and a uniform inflation is invisible except through where
    alpha lands on its grid, i.e. exactly the alpha/cache-size confound.
    """
    keys, rows = [], []
    for c in range(n_cls):
        Xc = f_tr[y_tr == c]
        Kc = fn(Xc)
        m_c = Kc.shape[0]
        V = torch.zeros(m_c, n_cls, device=Kc.device)
        V[:, c] = shots / m_c
        keys.append(Kc)
        rows.append(V)
    return torch.cat(keys, 0).t(), torch.cat(rows, 0)


# Library only, no entry point. Import build_calibrated.
