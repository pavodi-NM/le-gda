"""W -- the CEILING on fusing LE-GDA's spaces better than a uniform mean.

Pre-registered before the code was written; the kill conditions are stated
in the paper.

THE QUESTION. The proposal is a "latent layer discriminant on top of LE-GDA":
some richer combiner over the per-space discriminant scores than LE-GDA's
uniform mean. Learned scalar weights, a latent map, a second-stage discriminant
-- every version lives in ONE family,

    logits(x) = a(x) + alpha * sum_l w_l ghat_l(x),      w on the simplex

and LE-GDA is the point where w is uniform over a val-chosen multiset. So rather
than build one combiner and measure it, MEASURE THE BEST THE FAMILY CAN DO with
the weights chosen on TEST. No validation-selected combiner can beat an oracle
that reads the test labels, so one run forecloses the family instead of one
variant at a time. Precedent: exemplar selection was killed this way (test-oracle
ceiling +1.60 against a pre-registered +2).

WHY `w_oracle_S >= le_gda` IS GUARANTEED, and why that is the point. The grid is
simplex compositions with denominator Q over the DISTINCT spaces LE-GDA
selected, and the oracle also searches alpha on test, so LE-GDA's own weight
vector must be a grid point for the difference to be a pure headroom. KC-W3
asserts it, so a violation is a bug and not a result.

    AMENDMENT, 2026-08-14, disclosed. The first run used Q=8 and KC-W3 FIRED:
    RN50/fgvc/seed 1 returned -0.030. Cause: greedy keeps the best-scoring
    PREFIX, so |S| is 1..4 and the multiplicities are 1/|S| -- including THIRDS
    (that cell selected ['R4','last','R4'] = 2/3, 1/3). Q=8 spans halves and
    quarters but not thirds, so the method's own weight vector was off-grid.
    The denominator must be a multiple of lcm(1,2,3,4) = 12. Fixed to Q=12,
    which is what the registered instrument condition required all along; the
    bar and the statistic are untouched. This is an implementation correction,
    not a moved goalpost, and the failing first run is kept in the log.

`w_oracle_all` is a max over THREE searches -- a coarser simplex over all
spaces, the fine simplex over the selected set, and greedy-on-test -- because a
single coarse grid over 7 spaces is only a LOWER bound on the all-spaces
ceiling, and under-stating a ceiling biases a kill test toward killing. Taking
the max over the union of what was searched is a genuine (if not tight) bound.

SCOPE: our own arms only. No baseline is run, re-run or touched. Standard test
splits. K=16 only. Development subset first (see the pre-registration); the full
grid runs only if the dev ceiling clears the bar.
"""

import argparse
import glob
import json
from itertools import combinations

import torch

import clip
from utils import clip_classifier
from scripts.ds_meta import class_template
from scripts.run_cache_retrieval_null import acc
from scripts.run_prototype_collapse_seeds import draw
from scripts.run_baselines_e4 import acc_batched, logspace
from scripts.run_layer_ensemble import gda_logits, greedy
from scripts.preflight import Preflight

DATASETS = ["stanford_cars", "fgvc", "oxford_flowers", "dtd"]
BACKBONES = ["ViT-B/16"]
SEEDS = [0, 1, 2]
N_LAM = 12
GREEDY_T = 4                  # identical to the method


def simplex(d, Q):
    """All weight vectors on the d-simplex with denominator Q, as tuples.

    Compositions of Q into d non-negative parts, normalised. Enumerated by
    stars-and-bars so the count is exactly C(Q+d-1, d-1) and no vector is
    produced twice.
    """
    out = []
    for cuts in combinations(range(Q + d - 1), d - 1):
        prev, parts = -1, []
        for c in list(cuts) + [Q + d - 1]:
            parts.append(c - prev - 1)
            prev = c
        out.append(tuple(p / Q for p in parts))
    return out


def best_over_weights(idxs, Q, cv_or_ct, base, y, ALPHAS):
    """max top-1 over (simplex weights on `idxs`) x (alpha), and the argmax.

    `cv_or_ct` is the list of per-space anchored logits on whichever split the
    caller is optimising on -- val for the method, TEST for a ceiling arm. The
    weighted sum is formed once per weight vector and handed to acc_batched, so
    the alpha axis stays chunked exactly as everywhere else in the project
    (imagenet at 40k x 1000 will not fit otherwise).
    """
    best = (-1.0, None, None)
    for w in simplex(len(idxs), Q):
        e = None
        for wi, i in zip(w, idxs):
            if wi == 0.0:
                continue
            t = cv_or_ct[i] * wi
            e = t if e is None else e + t
        if e is None:
            continue
        for al, sc in zip(ALPHAS, acc_batched(base, e, ALPHAS, y)):
            if sc > best[0]:
                best = (sc, w, al)
    return best


def stored_le_gda(bb, name, K, sd):
    """The recorded le_gda accuracy for this cell, for the KC-W3 check.

    Returns None if no testfull file covers the cell -- the check is then
    skipped for it rather than silently passing.
    """
    for f in sorted(glob.glob("evidence/layer_ensemble_*testfull*.json")):
        for c in json.load(open(f)).get("cells", []):
            if (c["backbone"], c["dataset"], c["K"], c["seed"]) == (bb, name, K, sd):
                return c["acc"]["le_gda"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--ks", nargs="+", type=int, default=[16],
                    help="K=16 only: K=1 has no within-class covariance and "
                         "K=4 is the method's weakest cell. Registered.")
    ap.add_argument("--layers", nargs="+", type=int, default=[4, 6, 8, 10, 11])
    ap.add_argument("--spaces", nargs="+", default=None)
    ap.add_argument("--q-sub", type=int, default=12,
                    help="simplex denominator over LE-GDA's distinct spaces. "
                         "MUST be a multiple of lcm(1,2,3,4)=12: greedy keeps "
                         "the best PREFIX, so its multiplicities are 1/|S| for "
                         "|S| in 1..4, thirds included. Q=8 misses thirds and "
                         "KC-W3 fired -- see the amendment in the docstring.")
    ap.add_argument("--q-all", type=int, default=6,
                    help="simplex denominator over ALL spaces (coarser: the "
                         "grid is C(Q+d-1,d-1) and d is 7 on ViTs).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--test-split", default="testfull")
    ap.add_argument("--out", default="evidence/weighting_ceiling.json")
    args = ap.parse_args()
    dev = args.device
    pf = Preflight("weighting_ceiling")

    ALPHAS = logspace(1e-3, 10 ** (3 + 60 / 29), 40)
    LAMS = logspace(1e-8, 1e2, N_LAM)
    SPACES = args.spaces or ([f"L{i}" for i in args.layers] + ["pre", "last"])
    assert "last" in SPACES, "the projected space anchors the nesting check"

    rows = []
    for bb in args.backbones:
        tag = bb.replace("/", "-")
        model, _ = clip.load(bb, device=dev)
        model.eval()
        for name in args.datasets:
            cn, tp = class_template(name)
            p = f"caches/{name}/collapse/{tag}"
            D = {}
            for sp in ("val", "test", "pool"):
                fsp = args.test_split if sp == "test" else sp
                for s in SPACES:
                    suf = {"pre": "_pre_f", "last": "_f"}.get(s, f"_{s}_f")
                    D[(sp, s)] = torch.load(f"{p}_{fsp}{suf}.pt",
                                            map_location=dev).float()
                D[(sp, "y")] = torch.load(f"{p}_{fsp}_l.pt", map_location=dev)
            pf.assert_fp32(**{f"val_{s}": D[("val", s)] for s in SPACES})
            W = clip_classifier(cn, tp, model, device=dev).float()
            a_val = 100.0 * D[("val", "last")] @ W
            a_te = 100.0 * D[("test", "last")] @ W
            y_val, y_te = D[("val", "y")], D[("test", "y")]
            n_cls = int(D[("pool", "y")].max()) + 1

            for K in args.ks:
                for sd in args.seeds:
                    idx = draw(D[("pool", "y")].cpu(), K, sd, dev)
                    ytr = D[("pool", "y")][idx]
                    cid = (bb, name, K, sd)

                    cv, ct = [], []
                    for s in SPACES:
                        v, t, _ = gda_logits(D[("pool", s)][idx], ytr, n_cls,
                                             D[("val", s)], D[("test", s)],
                                             LAMS, a_val, ALPHAS, y_val)
                        cv.append(v)
                        ct.append(t)          # kept on device: the oracle arms
                                              # sweep the TEST side many times

                    # ---- the method, reproduced exactly ----
                    S, al, vsc = greedy(cv, a_val, y_val, ALPHAS, GREEDY_T)
                    e_te = torch.stack([ct[i] for i in S]).mean(0)
                    a_le = acc(a_te + al * e_te, y_te)
                    sub = [SPACES[i] for i in S]
                    distinct = sorted(set(S))
                    pf.grid(cid, "le_gda", "alpha", al, ALPHAS)

                    # KC-W3, half 1: the recomputed method must match the record
                    rec = stored_le_gda(bb, name, K, sd)
                    if rec is not None and not pf.nests:
                        pf.nesting(f"le_gda reproduces the record ({name})",
                                   abs(a_le - rec), tol=1e-6)

                    # ---- CEILING 1: weights over LE-GDA's own distinct spaces
                    s_or, w_or, al_or = best_over_weights(
                        distinct, args.q_sub, ct, a_te, y_te, ALPHAS)
                    # ---- CEILING 2: weights over ALL spaces
                    s_grid, w_all, al_all = best_over_weights(
                        list(range(len(SPACES))), args.q_all, ct, a_te, y_te,
                        ALPHAS)
                    # ---- REFERENCE: the method's own procedure, test-scored
                    S_t, al_t, _ = greedy(ct, a_te, y_te, ALPHAS, GREEDY_T)
                    a_sel = acc(a_te + al_t * torch.stack(
                        [ct[i] for i in S_t]).mean(0), y_te)
                    # a coarse simplex over 7 spaces UNDER-states the all-spaces
                    # ceiling, and under-stating a ceiling biases a kill test
                    # toward killing. Take the max over everything searched.
                    s_all = max(s_grid, s_or, a_sel)

                    rows.append({
                        "backbone": bb, "dataset": name, "K": K, "seed": sd,
                        "test_split": args.test_split,
                        "le_gda": a_le, "le_gda_val": vsc, "subset": sub,
                        "n_distinct": len(distinct),
                        "w_oracle_S": s_or, "w_S": list(w_or),
                        "spaces_S": [SPACES[i] for i in distinct],
                        "w_oracle_all": s_all, "w_all": list(w_all),
                        "w_oracle_all_grid": s_grid,
                        "sel_oracle": a_sel,
                        "sel_oracle_subset": [SPACES[i] for i in S_t]})
                    r = rows[-1]
                    print(f"{bb:<9}{name:<15}s={sd} le_gda={a_le:6.2f} "
                          f"| ceil_S {r['w_oracle_S'] - a_le:+.3f} "
                          f"ceil_all {r['w_oracle_all'] - a_le:+.3f} "
                          f"sel_or {r['sel_oracle'] - a_le:+.3f} "
                          f"| S={sub}", flush=True)
                    json.dump({"rows": rows, "preflight": pf.to_dict()},
                              open(args.out, "w"), indent=2)
            del D
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    json.dump({"rows": rows, "preflight": pf.to_dict()},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}  ({len(rows)} rows)")
    pf.report()


if __name__ == "__main__":
    main()
