"""What does LE-GDA actually STORE at inference? A check on our own claim.

The naive accounting bills storage as |S| x (C*D + D^2) -- "C means plus one D x D
inverse covariance per selected space" -- and concludes that storage is the
method's worst property. That number is the reason storage appears as a
standing objection in the write-up, and as
limitation #2 of the write-up.

But gda_weights (run_baselines_e4.py:167) returns

    W = Sigma^-1 M   [D, C]        b = -0.5 * diag(M^T Sigma^-1 M)   [C]

and every evaluation path in this repo computes `features @ W + b`. Sigma^-1 is
a LOCAL inside gda_weights: it is consumed when W is formed and never returned,
stored, or referenced again. So the DEPLOYED head is C*D + C floats per space,
and the D^2 term is a fit-time working set, not storage.

Two further corrections the naive expression misses:

  1. greedy selects WITH REPLACEMENT, so |S| overcounts. ['R3','last','R3'] is
     two distinct heads, not three.
  2. D is not constant across spaces. On ViT-B/16 the CLS spaces are 768-d but
     `last` is the 512-d projected embedding; on RN50 the stages run
     256/512/1024/2048. the naive per-class figure uses a single D=768.

This script recomputes the deployed footprint from the SUBSETS THE METHOD
ACTUALLY SELECTED in the recorded F1 runs, so the answer is a distribution over
real cells rather than a worked example on dtd.

Accounting convention is taken from scripts/measure_cache_cost.py so the two
sides of the comparison use the same one: Tip-Adapter stores keys m*C*D plus
values (dense m*C*C float, or sparse m*C int32 labels), and the natural unit is
CLIP's own zero-shot classifier head, C*D_proj.

This script MEASURES; it does not edit any claim.
"""

import argparse
import glob
import json
import statistics as st
from collections import defaultdict

import torch

from scripts.ds_meta import class_template

K_TIP = 16                       # shots the cache baselines are quoted at


def dims_for(backbone, dataset="dtd"):
    """Feature width per space, read off the cached tensors themselves."""
    tag = backbone.replace("/", "-")
    out = {}
    for f in sorted(glob.glob(f"caches/{dataset}/collapse/{tag}_pool*_f.pt")):
        name = f.split("/")[-1][len(tag) + 1:-3]          # pool_L6_f -> L6
        s = name[len("pool_"):-2] if name != "pool_f" else "last"
        out[s if s else "last"] = tuple(torch.load(f, map_location="cpu").shape)[1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+",
                    default=["evidence/nonlinear_fusion_full_*.json"])
    ap.add_argument("--rule", default="F1")
    args = ap.parse_args()

    rows = []
    for pat in args.files:
        for f in sorted(glob.glob(pat)):
            rows += json.load(open(f))["rows"]
    if not rows:
        raise SystemExit("no rows")

    DIMS, NCLS = {}, {}
    per_cell = []
    for r in rows:
        bb, ds = r["backbone"], r["dataset"]
        if bb not in DIMS:
            DIMS[bb] = dims_for(bb)
        if ds not in NCLS:
            NCLS[ds] = len(class_template(ds)[0])
        d = DIMS[bb]
        C = NCLS[ds]
        sub = r["rules"][args.rule]["subset"]
        distinct = sorted(set(sub))
        # DEPLOYED: one linear head per DISTINCT selected space.
        deployed = sum(d[s] * C + C for s in distinct)
        # WHAT THE NAIVE FORMULA CLAIMS: |S| x (C*D + D^2), D taken per space.
        claimed = sum(d[s] * C + d[s] ** 2 for s in sub)
        proj = d["last"]
        per_cell.append({
            "backbone": bb, "dataset": ds, "K": r["K"], "seed": r["seed"],
            "n_sel": len(sub), "n_distinct": len(distinct), "C": C,
            "deployed": deployed, "claimed": claimed,
            "zs_head": C * proj,                      # CLIP's own classifier
            "tip_sparse": K_TIP * C * proj + K_TIP * C,
            "tip_dense": K_TIP * C * proj + K_TIP * C * C,
            "gda_clip": C * proj + C})

    k16 = [c for c in per_cell if c["K"] == 16]
    print("=" * 78)
    print(f"LE-GDA DEPLOYED STORAGE — {len(per_cell)} cells, rule {args.rule}")
    print("=" * 78)

    print("\nSELECTION — |S| overcounts because greedy picks with replacement")
    ns = [c["n_sel"] for c in k16]
    nd = [c["n_distinct"] for c in k16]
    print(f"  median |S| = {st.median(ns):.1f}   median |distinct S| = "
          f"{st.median(nd):.1f}   ({sum(1 for a,b in zip(ns,nd) if b<a)}"
          f"/{len(k16)} cells have a repeat)")

    print("\nPER CLASS, K=16 (floats)")
    print(f"  {'backbone':10s} {'claimed (naive)':>14s} {'deployed':>10s}"
          f" {'Tip (sparse)':>13s} {'GDA-CLIP':>9s} {'x Tip':>7s}")
    by_bb = defaultdict(list)
    for c in k16:
        by_bb[c["backbone"]].append(c)
    for bb, cs in sorted(by_bb.items()):
        cl = st.median([c["claimed"] / c["C"] for c in cs])
        dp = st.median([c["deployed"] / c["C"] for c in cs])
        tp = st.median([c["tip_sparse"] / c["C"] for c in cs])
        gd = st.median([c["gda_clip"] / c["C"] for c in cs])
        print(f"  {bb:10s} {cl:14.0f} {dp:10.0f} {tp:13.0f} {gd:9.0f}"
              f" {dp/tp:7.2f}")

    print("\n  'x Tip' < 1 means LE-GDA's deployed head is SMALLER than the")
    print("  Tip-Adapter cache it is currently reported as 6x larger than.")

    print("\nIN UNITS OF CLIP'S OWN ZERO-SHOT HEAD (measure_cache_cost's unit)")
    for bb, cs in sorted(by_bb.items()):
        u = st.median([c["deployed"] / c["zs_head"] for c in cs])
        uc = st.median([c["claimed"] / c["zs_head"] for c in cs])
        print(f"  {bb:10s} deployed {u:5.2f} x   |  as claimed {uc:8.1f} x")

    print("\nTHE NAIVE-FORMULA WORKED EXAMPLE, RECHECKED (dtd, ViT-B/16, K=16)")
    dd = [c for c in k16 if c["dataset"] == "dtd" and c["backbone"] == "ViT-B/16"]
    if dd:
        c = dd[0]
        print(f"  C={c['C']}  subset size {c['n_sel']} ({c['n_distinct']} distinct)")
        print(f"  the naive formula says ~53k floats per class; recomputed claim "
              f"{c['claimed']/c['C']:.0f}, deployed {c['deployed']/c['C']:.0f}")
        print(f"  Tip-Adapter (sparse values) {c['tip_sparse']/c['C']:.0f}"
              f" per class")

    print("\nWORST CASE FOR US — the biggest deployed head in the grid")
    w = max(k16, key=lambda c: c["deployed"] / c["C"])
    print(f"  {w['backbone']} / {w['dataset']}: {w['deployed']/w['C']:.0f}"
          f" per class vs Tip {w['tip_sparse']/w['C']:.0f}"
          f"  ({w['deployed']/w['tip_sparse']:.2f}x)")
    worse = [c for c in k16 if c["deployed"] > c["tip_sparse"]]
    print(f"  cells where LE-GDA is genuinely larger than Tip-Adapter: "
          f"{len(worse)}/{len(k16)}")


if __name__ == "__main__":
    main()
