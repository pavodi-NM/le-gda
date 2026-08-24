"""Does Tip-Adapter's cache retrieve, or does it just re-bias the class priors?

Tip-Adapter adds an image-dependent term to the CLIP logits:

    tip = 100 * f @ W_text.T  +  alpha * exp(-beta(1 - f @ K)) @ V

and the paper's story is nearest-neighbour retrieval: the test image queries the
few-shot cache. AMU-Tuning (CVPR 2024) reframes the whole family as "learning a
logit bias for zero-shot CLIP", but every bias in that framing still depends on
the query. Proto-Adapter collapses K to class prototypes, showing per-exemplar
structure is unnecessary -- also still query-dependent.

Nobody removes the query. That is the null here.

NULLS (each keeps the cache tensors, the alpha/beta grid, and the val search
exactly as the real method has them -- matched hyperparameter budget):

  bias-train   replace the query f by the L2-normalised mean of the cache keys,
               so the cache term is ONE constant vector added to every test
               image. Uses train data only, no leakage.
  bias-test    same, but the centroid is the mean test feature. Transductive,
               and an upper bound on what any image-independent bias can do.
  prototype    keys collapsed to per-class means (Proto-Adapter-like). Still
               query-dependent; included to separate "retrieval" from "which
               exemplars".
  shuffled     cache values permuted so keys point at wrong classes. Sanity
               floor: should destroy the gain.

Recovered fraction = (null - zeroshot) / (tip - zeroshot).

Pre-registered kill condition: the "gain is bias, not retrieval" hypothesis dies
if bias-train recovers < 50% of Tip-Adapter's gain averaged over the 15
(dataset, K) cells.

Cached tensors only -- no feature extraction.
"""

import argparse
import os
import json

import torch
import yaml

import clip
from utils import clip_classifier
from datasets import build_dataset

# Dataset root. Override with the LEGDA_DATA_ROOT environment variable;
# see the README for the expected CoOp-style directory layout.
ROOT = os.environ.get("LEGDA_DATA_ROOT", "data/tip-adapter")
DATASETS = ["caltech101", "dtd", "eurosat"]
SHOTS = [1, 2, 4, 8, 16]


def load_cfg(dataset):
    with open(f"configs/{dataset}.yaml") as f:
        return yaml.safe_load(f)


def grid(cfg):
    ss, st = cfg["search_scale"], cfg["search_step"]
    betas = [i * (ss[0] - 0.1) / st[0] + 0.1 for i in range(st[0])]
    alphas = [i * (ss[1] - 0.1) / st[1] + 0.1 for i in range(st[1])]
    return betas, alphas


def acc(logits, y):
    return (logits.argmax(1) == y).float().mean().item() * 100


def search(betas, alphas, aff_val, V, clip_val, y_val):
    """Pick (beta, alpha) by val accuracy, as utils.search_hp does.

    aff_val is [n, NK] for query-dependent variants, or [1, NK] for the
    image-independent nulls -- broadcasting makes the constant-bias case fall
    out of the same code path.
    """
    best = (-1.0, None, None)
    for b in betas:
        cache = ((-1) * (b - b * aff_val)).exp() @ V          # [n or 1, C]
        for a in alphas:
            s = acc(clip_val + a * cache, y_val)
            if s > best[0]:
                best = (s, b, a)
    return best[1], best[2]


def evaluate(name, aff_val, aff_test, V, clip_val, clip_test, y_val, y_test,
             betas, alphas):
    b, a = search(betas, alphas, aff_val, V, clip_val, y_val)
    cache = ((-1) * (b - b * aff_test)).exp() @ V
    return {"acc": acc(clip_test + a * cache, y_test), "beta": b, "alpha": a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--shots", nargs="+", type=int, default=SHOTS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="evidence/cache_retrieval_null.json")
    args = ap.parse_args()
    dev = args.device

    model, _ = clip.load("RN50", device=dev)
    model.eval()

    results = {}
    print(f"{'dataset':<12}{'K':>3}{'zs':>8}{'tip':>8}{'bias-tr':>9}{'bias-te':>9}"
          f"{'proto':>8}{'shuf':>8}{'rec%':>8}")
    print("-" * 72)

    for name in args.datasets:
        cfg = load_cfg(name)
        betas, alphas = grid(cfg)
        ds = build_dataset(name, ROOT, 1)

        f_val = torch.load(f"caches/{name}/val_f.pt", map_location=dev).float()
        y_val = torch.load(f"caches/{name}/val_l.pt", map_location=dev)
        f_test = torch.load(f"caches/{name}/test_f.pt", map_location=dev).float()
        y_test = torch.load(f"caches/{name}/test_l.pt", map_location=dev)

        W = clip_classifier(ds.classnames, ds.template, model, device=dev).float()
        clip_val, clip_test = 100.0 * f_val @ W, 100.0 * f_test @ W
        zs = acc(clip_test, y_test)
        results[name] = {"zeroshot": zs, "shots": {}}

        # Query-independent centroids. mu_train uses only cache keys.
        mu_test = f_test.mean(0, keepdim=True)
        mu_test = mu_test / mu_test.norm(dim=-1, keepdim=True)

        for K in args.shots:
            keys = torch.load(f"caches/{name}/keys_{K}shots.pt",
                              map_location=dev).float()          # [D, NK]
            V = torch.load(f"caches/{name}/values_{K}shots.pt",
                           map_location=dev).float()             # [NK, C]

            mu_train = keys.mean(1, keepdim=True).t()
            mu_train = mu_train / mu_train.norm(dim=-1, keepdim=True)

            # Prototype cache: average keys within class, renormalise.
            P = (keys @ V) / V.sum(0).clamp_min(1)               # [D, C]
            P = P / P.norm(dim=0, keepdim=True)
            Vp = torch.eye(V.shape[1], device=dev)

            g = torch.Generator(device="cpu").manual_seed(0)
            V_shuf = V[torch.randperm(V.shape[0], generator=g).to(dev)]

            variants = {
                "tip": (f_val @ keys, f_test @ keys, V),
                "bias-train": (mu_train @ keys, mu_train @ keys, V),
                "bias-test": (mu_test @ keys, mu_test @ keys, V),
                "prototype": (f_val @ P, f_test @ P, Vp),
                "shuffled": (f_val @ keys, f_test @ keys, V_shuf),
            }
            row = {k: evaluate(k, av, at, vv, clip_val, clip_test,
                               y_val, y_test, betas, alphas)
                   for k, (av, at, vv) in variants.items()}

            gain = row["tip"]["acc"] - zs
            rec = (row["bias-train"]["acc"] - zs) / gain * 100 if gain else float("nan")
            row["gain"] = gain
            row["recovered_pct"] = rec
            results[name]["shots"][K] = row

            print(f"{name:<12}{K:>3}{zs:>8.2f}{row['tip']['acc']:>8.2f}"
                  f"{row['bias-train']['acc']:>9.2f}{row['bias-test']['acc']:>9.2f}"
                  f"{row['prototype']['acc']:>8.2f}{row['shuffled']['acc']:>8.2f}"
                  f"{rec:>8.1f}")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)

    recs = [results[d]["shots"][K]["recovered_pct"]
            for d in args.datasets for K in args.shots]
    mean_rec = sum(recs) / len(recs)
    print("\n" + "=" * 72)
    print(f"mean recovered fraction (bias-train): {mean_rec:.1f}%  over {len(recs)} cells")
    print(f"KILL CONDITION (>= 50% to survive): "
          f"{'SURVIVES -- gain is largely a class bias' if mean_rec >= 50 else 'KILLED -- the cache genuinely retrieves'}")
    print("=" * 72)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
