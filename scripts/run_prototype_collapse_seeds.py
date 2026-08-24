"""Is prototype collapse of the Tip-Adapter cache lossless? Seed-controlled.

Proto-Adapter (Sensors 2024) replaces Tip-Adapter's per-exemplar cache with one
prototype per class and reports no loss. run_cache_retrieval_null.py, on the
repo's single cached draw, showed a gap that grows with K on DTD (+0.65/+3.07/
+4.49 at K=4/8/16) and Caltech101, but reversed on EuroSAT (-0.80 mean) where
K=2 swung -4.38. Single draws cannot tell that apart from few-shot draw noise.

Draws come from caches/<ds>/pool (see extract_train_cache_pool.py), which stores
the per-image augmented-mean features build_cache_model would compute, so a
subsample is an exact cache for that draw.

Both arms get the same draw, the same alpha/beta grid, and the same val search --
the comparison is paired within seed.

Pre-registered kill condition: "collapse is lossy and the loss grows with K on
texture/fine-grained data" survives iff, on DTD,
  (a) seed-mean (tip - proto) at K=16 is >= +2.0 AND >= 2x the seed std there,
  (b) seed-means increase monotonically across K = 4 -> 8 -> 16.
"""

import argparse
import json
from statistics import mean, pstdev

import torch

import clip
from utils import clip_classifier
from datasets import build_dataset
from scripts.run_cache_retrieval_null import ROOT, load_cfg, grid, acc, search

DATASETS = ["caltech101", "dtd", "eurosat"]
SHOTS = [1, 2, 4, 8, 16]
SEEDS = [0, 1, 2]


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
    b, a = search(betas, alphas, f_val @ keys, V, clip_val, y_val)
    return acc(clip_test + a * (((-1) * (b - b * (f_test @ keys))).exp() @ V),
               y_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--shots", nargs="+", type=int, default=SHOTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="evidence/prototype_collapse_seeds.json")
    args = ap.parse_args()
    dev = args.device

    model, _ = clip.load("RN50", device=dev)
    model.eval()

    results = {}
    print(f"{'dataset':<12}{'K':>3}  {'tip':>14}  {'proto':>14}  {'tip-proto':>16}")
    print("-" * 68)

    for name in args.datasets:
        cfg = load_cfg(name)
        betas, alphas = grid(cfg)
        ds = build_dataset(name, ROOT, 1)

        f_val = torch.load(f"caches/{name}/val_f.pt", map_location=dev).float()
        y_val = torch.load(f"caches/{name}/val_l.pt", map_location=dev)
        f_test = torch.load(f"caches/{name}/test_f.pt", map_location=dev).float()
        y_test = torch.load(f"caches/{name}/test_l.pt", map_location=dev)
        pool_f = torch.load(f"caches/{name}/pool/train_pool_f.pt",
                            map_location=dev).float()
        pool_y = torch.load(f"caches/{name}/pool/train_pool_l.pt",
                            map_location="cpu")

        W = clip_classifier(ds.classnames, ds.template, model, device=dev).float()
        clip_val, clip_test = 100.0 * f_val @ W, 100.0 * f_test @ W
        n_cls = int(pool_y.max()) + 1
        eye = torch.eye(n_cls, device=dev)
        results[name] = {"zeroshot": acc(clip_test, y_test), "shots": {}}

        for K in args.shots:
            tips, protos = [], []
            for s in args.seeds:
                idx = draw(pool_y, K, s, dev)
                keys = pool_f[idx].t()                      # [D, NK]
                V = eye[pool_y.to(dev)[idx]]                # [NK, C]

                P = (keys @ V) / V.sum(0).clamp_min(1)
                P = P / P.norm(dim=0, keepdim=True)

                tips.append(run_arm(keys, V, f_val, f_test, clip_val, clip_test,
                                    y_val, y_test, betas, alphas))
                protos.append(run_arm(P, eye, f_val, f_test, clip_val, clip_test,
                                      y_val, y_test, betas, alphas))

            d = [t - p for t, p in zip(tips, protos)]
            results[name]["shots"][K] = {
                "tip": tips, "proto": protos, "delta": d,
                "tip_mean": mean(tips), "proto_mean": mean(protos),
                "delta_mean": mean(d), "delta_std": pstdev(d),
                "tip_std": pstdev(tips),
            }
            print(f"{name:<12}{K:>3}  {mean(tips):>8.2f}+-{pstdev(tips):<5.2f}"
                  f"  {mean(protos):>8.2f}+-{pstdev(protos):<5.2f}"
                  f"  {mean(d):>+9.2f}+-{pstdev(d):<5.2f}")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)

    # ---- verdict -----------------------------------------------------------
    print("\n" + "=" * 68)
    r = results.get("dtd")
    if r:
        c16 = r["shots"][16]
        a = c16["delta_mean"] >= 2.0 and c16["delta_mean"] >= 2 * c16["delta_std"]
        seq = [r["shots"][k]["delta_mean"] for k in (4, 8, 16)]
        b = seq[0] < seq[1] < seq[2]
        print(f"DTD K=16 delta = {c16['delta_mean']:+.2f} +- {c16['delta_std']:.2f}"
              f"   (a) >=+2.0 and >=2x std: {'PASS' if a else 'FAIL'}")
        print(f"DTD delta at K=4,8,16 = {[round(v, 2) for v in seq]}"
              f"   (b) monotone increasing: {'PASS' if b else 'FAIL'}")
        print(f"VERDICT: {'SURVIVES' if a and b else 'KILLED'}")
    allv = [results[d]["shots"][k]["delta_mean"]
            for d in args.datasets for k in args.shots]
    print(f"\nmean tip-proto over all {len(allv)} cells: {mean(allv):+.2f}")
    print("=" * 68)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
