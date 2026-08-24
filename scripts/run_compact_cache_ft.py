"""Fine-tuning for the trained baseline arms.

Supplies what run_calibrated_ft.py needs to train a cache model:

    train_keys   gradient descent on the cache keys, fixed LR grid (LRS)
    ft_arm       train + validation-select + evaluate one arm
    cache_acc    accuracy of a (keys, values) cache at given alpha/beta

Two of the head-to-head rows are trained here at a uniform 20 epochs:
Tip-Adapter-F (16C trainable keys) and Proto-Adapter-F / CC-F-1 (C trainable
keys, k-means initialised). LE-GDA trains nothing and does not use this path.
"""

import argparse
import json
from statistics import mean, median, pstdev

import torch
import torch.nn.functional as F

import clip
from utils import clip_classifier
from datasets import build_dataset
from scripts.ds_meta import class_template
from scripts.run_cache_retrieval_null import ROOT, load_cfg, grid, acc, search
from scripts.run_prototype_collapse_seeds import draw
from scripts.run_compact_cache import kmeans, build_class_cache

DATASETS = ["caltech101", "dtd", "eurosat", "food101", "oxford_pets"]
BACKBONES = ["RN50", "RN101", "ViT-B/32", "ViT-B/16"]
SEEDS = [0, 1, 2]
K = 16
M = 4
LRS = [1e-4, 1e-3, 1e-2]
EPOCHS = 200


def cache_acc(keys, V, f_val, y_val, f_te, y_te, clip_val, clip_te,
              betas, alphas):
    """Val-search alpha/beta, return test accuracy (as phase 1)."""
    from scripts.run_compact_cache import acc_chunked
    b, a = search(betas, alphas, f_val @ keys, V, clip_val, y_val)
    return acc_chunked(keys, V, f_te, y_te, clip_te, b, a)


def train_keys(keys0, V, f_tr, y_tr, clip_tr, a, b, lr, epochs=EPOCHS):
    """NOTE: the module default EPOCHS=200 is 10x Tip-Adapter's own
    `train_epoch` (20 in every config but EuroSAT's 100), and this loop has no
    LR schedule and no best-epoch selection. Every result in
    evidence/compact_cache_ft*.json, shot_sweep.json and ccf_capacity*.json was
    produced at 200 and is kept reproducible by leaving the default alone.
    The baseline runner passes epochs=20 explicitly."""
    keys = keys0.clone().requires_grad_(True)
    opt = torch.optim.Adam([keys], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logits = clip_tr + a * (((-1) * (b - b * (f_tr @ keys))).exp() @ V)
        F.cross_entropy(logits, y_tr).backward()
        opt.step()
    return keys.detach()


def ft_arm(keys0, V, f_tr, y_tr, clip_tr, f_val, y_val, f_te, y_te,
           clip_val, clip_te, betas, alphas):
    """Train at each LR with alpha/beta from the init cache's val search;
    select LR on val (with post-training alpha/beta re-search); report test."""
    b0, a0 = search(betas, alphas, f_val @ keys0, V, clip_val, y_val)
    best = (-1.0, None)
    for lr in LRS:
        keys = train_keys(keys0, V, f_tr, y_tr, clip_tr, a0, b0, lr)
        b, a = search(betas, alphas, f_val @ keys, V, clip_val, y_val)
        a_val = acc(clip_val + a * (((-1) * (b - b * (f_val @ keys))).exp()
                                    @ V), y_val)
        if a_val > best[0]:
            best = (a_val, (keys, b, a))
    keys, b, a = best[1]
    from scripts.run_compact_cache import acc_chunked
    return acc_chunked(keys, V, f_te, y_te, clip_te, b, a)


# Library only, no entry point: run_calibrated_ft.py drives the trained arms.
# Import train_keys / ft_arm / cache_acc / LRS.
