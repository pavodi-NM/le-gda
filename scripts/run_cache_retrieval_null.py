"""Shared helpers: dataset root, per-dataset config, Tip-Adapter's search grid,
top-1 accuracy, and the validation search over the cache's (beta, alpha).

The cache-model baselines score a test feature f as

    logits = 100 * f @ W_text.T  +  alpha * exp(-beta * (1 - f @ K)) @ V

with the keys K and one-hot values V built from the few-shot support set.
"""

import os

import yaml

# Dataset root. Override with the LEGDA_DATA_ROOT environment variable;
# see the README for the expected CoOp-style directory layout.
ROOT = os.environ.get("LEGDA_DATA_ROOT", "data/tip-adapter")


def load_cfg(dataset):
    with open(f"configs/{dataset}.yaml") as f:
        return yaml.safe_load(f)


def grid(cfg):
    """Tip-Adapter's linear (beta, alpha) grid from the dataset config."""
    ss, st = cfg["search_scale"], cfg["search_step"]
    betas = [i * (ss[0] - 0.1) / st[0] + 0.1 for i in range(st[0])]
    alphas = [i * (ss[1] - 0.1) / st[1] + 0.1 for i in range(st[1])]
    return betas, alphas


def acc(logits, y):
    return (logits.argmax(1) == y).float().mean().item() * 100


def search(betas, alphas, aff_val, V, clip_val, y_val):
    """Pick (beta, alpha) by validation accuracy, as utils.search_hp does.

    aff_val is the [n, NK] affinity between validation features and cache keys.
    """
    best = (-1.0, None, None)
    for b in betas:
        cache = ((-1) * (b - b * aff_val)).exp() @ V          # [n, C]
        for a in alphas:
            s = acc(clip_val + a * cache, y_val)
            if s > best[0]:
                best = (s, b, a)
    return best[1], best[2]
