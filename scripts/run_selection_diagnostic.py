"""D -- is LE-GDA's remaining error SELECTION VARIANCE, and can a rule recover it?

Pre-registered before the code was written; the kill conditions are stated
in the paper.

THE DIAGNOSIS. Two independent measurements now point the same way. The Schur
kill found the joint model's mis-specification real and IRRELEVANT
("mis-specification was never the risk; estimation was"). The weighting ceiling
found every combiner that keeps the per-space scores fixed capped at +0.528
oracle, of which weighting is ~+0.04 -- while +0.384 is ALPHA ALONE, chosen on
validation. LE-GDA fits four things on a few hundred val items: lambda_l, the
anchor, the subset, and alpha. The hypothesis is that what is left is ESTIMATION
VARIANCE in that fitting, not model error.

Alpha enters linearly (a + alpha*ghat, so averaging logits over alpha is just
using the mean alpha), which makes this a SCALAR ESTIMATION problem.

WHAT IS MEASURED. Four alpha-selection rules on the method's own fused
validation curve -- none of which sees a test label -- against the test oracle;
and the per-space ridge chosen by validation against LEDOIT-WOLF computed in
closed form on the support set. Ledoit-Wolf shrinks toward m*I at intensity rho,
and a global scale is absorbed by alpha, so it is exactly our ridge family at
lambda = rho/(1-rho): a POINT IN THE EXISTING CLASS, not a new class. That is
the same reduction §2.6 performs for GDA-CLIP's Kubokawa-Srivastava
estimator.

THIS BUILDS NO NEW MODEL. It decides which method to build, or whether none.

SCOPE: our own arms only. Standard test splits. K=16. Same 24 dev cells as the
weighting ceiling, so KC-D3 can check the two runs agree about the oracle gap.
"""

import argparse
import glob
import json

import torch
import torch.nn.functional as F

import clip
from utils import clip_classifier
from scripts.ds_meta import class_template
from scripts.run_cache_retrieval_null import acc
from scripts.run_prototype_collapse_seeds import draw
from scripts.run_baselines_e4 import gda_weights, anchor_of, acc_batched, logspace
from scripts.run_layer_ensemble import greedy
from scripts.preflight import Preflight

DATASETS = ["stanford_cars", "fgvc", "oxford_flowers", "dtd"]
BACKBONES = ["ViT-B/16"]
SEEDS = [0, 1, 2]
N_LAM = 12
GREEDY_T = 4


def lw_intensity(R):
    """Ledoit-Wolf shrinkage intensity toward (tr S / D) I, from the support set.

    Sigma_LW = (1-rho) S + rho m I with m = tr(S)/D. Our ridge family is
    S + lambda m I, and Sigma_LW = (1-rho)[S + (rho/(1-rho)) m I], so the two
    coincide at lambda = rho/(1-rho) -- the leading (1-rho) is a global scale and
    is absorbed by alpha, exactly as §2.6 absorbs Kubokawa-Srivastava's
    D/(N-1).

    The b_bar^2 sum is computed in closed form rather than by looping over
    samples: sum_i ||x_i x_i^T - S||_F^2 = sum_i ||x_i||^4 - N ||S||_F^2, using
    sum_i x_i^T S x_i = N ||S||_F^2.
    """
    N, D = R.shape
    S = R.t() @ R / N
    m = S.diagonal().mean()
    s_fro2 = (S ** 2).sum()
    d2 = (s_fro2 - D * m ** 2) / D
    quartic = ((R ** 2).sum(1) ** 2).sum()
    b2 = ((quartic - N * s_fro2) / (N ** 2 * D)).clamp_min(0.0)
    b2 = torch.minimum(b2, d2.clamp_min(1e-30))
    rho = (b2 / d2.clamp_min(1e-30)).clamp(0.0, 1.0 - 1e-6)
    return float(rho), float(rho / (1.0 - rho))


def residuals(ftr, ytr, n_cls):
    """The within-class residuals gda_weights builds, for the LW estimate.

    Must match gda_weights EXACTLY -- it centres on the L2-NORMALISED class
    mean, not the raw one, so recomputing it any other way would estimate a
    different covariance than the one the method inverts.
    """
    res = []
    for c in range(n_cls):
        Xc = ftr[ytr == c]
        res.append(Xc - F.normalize(Xc.mean(0), dim=-1))
    return torch.cat(res, 0)


def anchored(ftr, ytr, n_cls, fv, ft, mode, reg):
    Wg, bg = gda_weights(ftr, ytr, n_cls, mode, reg)
    cv = fv @ Wg + bg
    a = anchor_of(cv)
    return cv / a, (ft @ Wg + bg) / a


def val_curves(a_val, e_val, ALPHAS, y_val):
    """(top-1 curve, cross-entropy curve) over the alpha grid, VALIDATION only."""
    accs = acc_batched(a_val, e_val, ALPHAS, y_val)
    nll = [float(F.cross_entropy(a_val + al * e_val, y_val)) for al in ALPHAS]
    return accs, nll


def rule_indices(v, nll, n_val):
    """Index chosen by each alpha rule. NO RULE SEES A TEST LABEL.

    R0  argmax val top-1                     -- what LE-GDA does today
    R1  plateau centre: median index among alphas within one binomial SE of the
        val max. The val curve is FLAT at large alpha (the write-up §5b.4
        measured 85.47 -> 85.37 -> constant), and the argmax of a flat noisy
        step function is close to a coin flip across the plateau.
    R2  3-point moving average over log alpha, then argmax
    R3  argmin val cross-entropy -- smooth, and uses the full predictive
        distribution instead of 0-1
    """
    n = len(v)
    best = max(v)
    r0 = v.index(best)
    p = best / 100.0
    se = 100.0 * (p * (1 - p) / max(1, n_val)) ** 0.5
    plateau = [i for i in range(n) if v[i] >= best - se]
    r1 = plateau[len(plateau) // 2]
    sm = [sum(v[max(0, i - 1):min(n, i + 2)]) / len(v[max(0, i - 1):min(n, i + 2)])
          for i in range(n)]
    r2 = sm.index(max(sm))
    r3 = nll.index(min(nll))
    return {"R0": r0, "R1": r1, "R2": r2, "R3": r3}, len(plateau)


def stored_le_gda(bb, name, K, sd):
    """The recorded le_gda for this cell, or None if no testfull file covers it."""
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
    ap.add_argument("--ks", nargs="+", type=int, default=[16])
    ap.add_argument("--layers", nargs="+", type=int, default=[4, 6, 8, 10, 11])
    ap.add_argument("--spaces", nargs="+", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--test-split", default="testfull")
    ap.add_argument("--out", default="evidence/selection_diagnostic.json")
    args = ap.parse_args()
    dev = args.device
    pf = Preflight("selection_diagnostic")

    ALPHAS = logspace(1e-3, 10 ** (3 + 60 / 29), 40)
    LAMS = logspace(1e-8, 1e2, N_LAM)
    SPACES = args.spaces or ([f"L{i}" for i in args.layers] + ["pre", "last"])
    assert "last" in SPACES

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
            n_val = int(y_val.shape[0])

            for K in args.ks:
                for sd in args.seeds:
                    idx = draw(D[("pool", "y")].cpu(), K, sd, dev)
                    ytr = D[("pool", "y")][idx]
                    cid = (bb, name, K, sd)

                    cv_v, ct_v, cv_l, ct_l, lams = [], [], [], [], []
                    for s in SPACES:
                        ftr = D[("pool", s)][idx]
                        # --- the method's own val-searched ridge ---
                        best = (-1.0, None, None, None)
                        for reg in LAMS:
                            cvx, ctx = anchored(ftr, ytr, n_cls, D[("val", s)],
                                                D[("test", s)], "ridge", reg)
                            sc = max(acc_batched(a_val, cvx, ALPHAS, y_val))
                            if sc > best[0]:
                                best = (sc, reg, cvx, ctx)
                        cv_v.append(best[2])
                        ct_v.append(best[3])
                        # --- Ledoit-Wolf, closed form, NO validation ---
                        rho, lam_lw = lw_intensity(residuals(ftr, ytr, n_cls))
                        cvx, ctx = anchored(ftr, ytr, n_cls, D[("val", s)],
                                            D[("test", s)], "lw", rho)
                        cv_l.append(cvx)
                        ct_l.append(ctx)
                        lams.append({"space": s, "lam_val": best[1],
                                     "rho_lw": rho, "lam_lw": lam_lw})
                        pf.grid(cid, f"gda_{s}", "reg", best[1], LAMS,
                                lo_boundary=True)

                    out = {"backbone": bb, "dataset": name, "K": K, "seed": sd,
                           "test_split": args.test_split, "n_val": n_val,
                           "lams": lams}

                    # ---- the method, on each lambda source ----
                    for tagl, cvs, cts in (("val", cv_v, ct_v),
                                           ("lw", cv_l, ct_l)):
                        S, al, _ = greedy(cvs, a_val, y_val, ALPHAS, GREEDY_T)
                        e_v = torch.stack([cvs[i] for i in S]).mean(0)
                        e_t = torch.stack([cts[i] for i in S]).mean(0)
                        v, nll = val_curves(a_val, e_v, ALPHAS, y_val)
                        t = acc_batched(a_te, e_t, ALPHAS, y_te)
                        idxs, width = rule_indices(v, nll, n_val)
                        out[f"lam_{tagl}"] = {
                            "subset": [SPACES[i] for i in S],
                            "val_curve": v, "test_curve": t, "nll_curve": nll,
                            "rule_idx": idxs,
                            "rule_test": {k: t[i] for k, i in idxs.items()},
                            "oracle_test": max(t),
                            "oracle_idx": t.index(max(t)),
                            "plateau_width": width}
                        if tagl == "val":
                            pf.grid(cid, "le_gda", "alpha", ALPHAS[idxs["R0"]],
                                    ALPHAS)

                    # NESTING: R0 on the val-lambda arm IS the method. This
                    # script rebuilds every piece of LE-GDA from scratch --
                    # per-space ridge search, anchoring, greedy, alpha argmax --
                    # so if it does not land on the recorded le_gda, one of
                    # those pieces has drifted and every rule delta below is
                    # measured against the wrong reference. Tolerance is one
                    # order below a single test item, per the weighting-ceiling
                    # fp32 count noise is ~1e-6, one item is >=2.5e-3.
                    rec = stored_le_gda(bb, name, K, sd)
                    if rec is not None and not pf.nests:
                        pf.nesting(
                            f"R0 on val-lambda reproduces recorded le_gda ({name})",
                            abs(out["lam_val"]["rule_test"]["R0"] - rec),
                            tol=1e-4)

                    rows.append(out)
                    a = out["lam_val"]
                    b = out["lam_lw"]
                    print(f"{bb:<9}{name:<15}s={sd} R0={a['rule_test']['R0']:6.2f} "
                          f"| R1{a['rule_test']['R1']-a['rule_test']['R0']:+.2f} "
                          f"R2{a['rule_test']['R2']-a['rule_test']['R0']:+.2f} "
                          f"R3{a['rule_test']['R3']-a['rule_test']['R0']:+.2f} "
                          f"orc{a['oracle_test']-a['rule_test']['R0']:+.2f} "
                          f"| LW{b['rule_test']['R0']-a['rule_test']['R0']:+.2f} "
                          f"| plateau {a['plateau_width']}/40", flush=True)
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
