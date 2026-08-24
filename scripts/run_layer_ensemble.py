"""Layer-Ensemble GDA. Pre-registered before the code was written; kill conditions are in the paper.

The head that wins (GDA), on the features that carry more (intermediate blocks).

NOTE ON THE NULL. GDA is affine-equivariant: for invertible T,
(Tf)^T (T Sigma T^T)^-1 (T mu) = f^T Sigma^-1 mu. So a "random projection of the
final layer" control would return IDENTICAL logits and controls for nothing --
the E10 failure mode.

A BOOTSTRAP null was tried first and was itself degenerate: resampling C*K
support points with replacement leaves many classes with zero samples, so GDA
cannot fit (measured 35.00 at K=4 and 1.05 -- chance -- at K=1). Beating a
broken arm is not a control. The null used here is `null_ridge`: len(SPACES)
GDA fits at the FINAL layer with different ridge strengths, then the identical
greedy selection and alpha search. Members are non-degenerate and
information-equivalent, so this isolates "several diverse estimators plus a
subset search" from "layers carry complementary information" at matched
validation budget.
"""

import argparse
import json

import torch
import torch.nn.functional as F

import clip
from utils import clip_classifier
from scripts.ds_meta import class_template
from scripts.run_cache_retrieval_null import acc
from scripts.run_prototype_collapse_seeds import draw
from scripts.run_baselines_e4 import gda_weights, anchor_of, acc_batched, logspace
from scripts.preflight import Preflight

DATASETS = ["stanford_cars", "fgvc", "oxford_flowers", "dtd"]
BACKBONES = ["ViT-B/16"]
SEEDS = [0, 1, 2]
KS = [16, 4, 1]
LAYERS = [4, 6, 8, 10, 11]
N_LAM = 12
GREEDY_T = 4                  # max ensemble members


def gda_logits(ftr, ytr, n_cls, fv, ft, LAMS, a_val, ALPHAS, y_val):
    """Ridge-searched GDA at one feature space -> anchored (val, test) logits."""
    best = (-1.0, None)
    for reg in LAMS:
        Wg, bg = gda_weights(ftr, ytr, n_cls, "ridge", reg)
        cv = fv @ Wg + bg
        v = max(acc_batched(a_val, cv / anchor_of(cv), ALPHAS, y_val))
        if v > best[0]:
            best = (v, reg)
    Wg, bg = gda_weights(ftr, ytr, n_cls, "ridge", best[1])
    cv = fv @ Wg + bg
    anc = anchor_of(cv)
    return cv / anc, (ft @ Wg + bg) / anc, best[1]


def greedy(cands, a_val, y_val, ALPHAS, T):
    """Forward selection with replacement on VALIDATION. Returns (subset, alpha).

    Identical procedure is given to the treatment and to the matched null, so
    the search budget is matched exactly.
    """
    S, best = [], (-1.0, None, None)
    for _ in range(T):
        step = (-1.0, None, None)
        for j in range(len(cands)):
            e = torch.stack([cands[i] for i in S + [j]]).mean(0)
            for al, s in zip(ALPHAS, acc_batched(a_val, e, ALPHAS, y_val)):
                if s > step[0]:
                    step = (s, j, al)
        S = S + [step[1]]
        if step[0] > best[0]:
            best = (step[0], list(S), step[2])
    return best[1], best[2], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--ks", nargs="+", type=int, default=KS)
    ap.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    ap.add_argument("--spaces", nargs="+", default=None,
                    help="explicit feature-space list, overriding --layers. "
                         "ResNets (Phase B) pass `R1 R2 R3 R4 last`: the stage "
                         "readout is GAP'd conv maps, named R not L so a stage "
                         "cannot be mistaken for a ViT block, and there is no "
                         "`pre` because AttentionPool2d never materialises one.")
    ap.add_argument("--device", default="cuda")
    # "test" is the CAPPED eval split (max(30, 3000/n_cls) per class) that
    # The write-up (§3) and §5c are measured on; "testfull" is the standard
    # uncapped split. The cap is invisible to the paired within-cell gates --
    # every arm sees the same items -- but it is fatal to the head-to-head
    # table, which is compared against published literature evaluated on the
    # full split. Same flag name as run_calibrated_ft.py so every arm of that
    # table can be moved with one token.
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--out", default="evidence/layer_ensemble.json")
    args = ap.parse_args()
    dev = args.device
    pf = Preflight("layer_ensemble")
    # ALPHA GRID, WIDENED FOR THE SCALE-UP. The 4 fine-grained datasets never
    # pinned alpha, so 1e-3..1e3 was enough for §3. eurosat does: it has
    # C = 42.8 (zs 43.5 -> gda_last 86.3), so the text logit is nearly worthless
    # and val wants alpha ~1.3e3, just OUTSIDE the old grid -- 100% hi-pinned on
    # gda_last, le_gda and null_ridge alike. A pinned alpha makes every arm a
    # BOUND, not a value (E13, and this project's alpha/cache-size confound).
    #
    # alpha -> infinity is the real domain limit (pure GDA, text ignored) and
    # val is FLAT there: measured 85.47 at 1.3e3 decaying to 85.37 by 4e4, then
    # constant to 1e9. So the grid only has to reach the plateau, not chase an
    # optimum that runs away. These 40 points EXTEND the old 30 at the identical
    # ratio 10^(6/29) -- the first 30 are unchanged, so nothing in §3 moves
    # for a grid reason -- and the last 10 put the asymptote inside the grid.
    ALPHAS = logspace(1e-3, 10 ** (3 + 60 / 29), 40)
    LAMS = logspace(1e-8, 1e2, N_LAM)
    SPACES = args.spaces or ([f"L{i}" for i in args.layers] + ["pre", "last"])
    assert "last" in SPACES, "the projected space anchors the nesting check"

    cells = []
    for bb in args.backbones:
        tag = bb.replace("/", "-")
        model, _ = clip.load(bb, device=dev)
        model.eval()
        for name in args.datasets:
            cn, tp = class_template(name)
            p = f"caches/{name}/collapse/{tag}"
            D = {}
            for sp in ("val", "test", "pool"):
                # the FILENAME token may differ from the split key; everything
                # downstream keeps addressing it as "test"
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
            zs = acc(a_te, y_te)
            n_cls = int(D[("pool", "y")].max()) + 1

            for K in args.ks:
                for sd in args.seeds:
                    idx = draw(D[("pool", "y")].cpu(), K, sd, dev)
                    ytr = D[("pool", "y")][idx]
                    accs, sel = {}, {}
                    cid = (bb, name, K, sd)

                    # ---- single-space GDA, every layer ----
                    cv, ct = [], []
                    for s in SPACES:
                        v, t, reg = gda_logits(D[("pool", s)][idx], ytr, n_cls,
                                               D[("val", s)], D[("test", s)],
                                               LAMS, a_val, ALPHAS, y_val)
                        cv.append(v)
                        # test-side logits sit on CPU: greedy() reads only the
                        # VAL side, and on imagenet 7 spaces x [40k, 1000] is
                        # 1.1 GB per arm, which does not fit next to the
                        # features on an 8 GB card. Same values, same device at
                        # the point of use -- the nesting check below still
                        # reproduces gda_last to max|d| = 0.
                        ct.append(t.cpu())
                        bb_ = (-1.0, None)
                        for al, sc in zip(ALPHAS,
                                          acc_batched(a_val, v, ALPHAS, y_val)):
                            if sc > bb_[0]:
                                bb_ = (sc, al)
                        accs[f"gda_{s}"] = acc(a_te + bb_[1] * t, y_te)
                        sel[f"gda_{s}"] = {"val": bb_[0], "alpha": bb_[1],
                                           "reg": reg}
                        pf.grid(cid, f"gda_{s}", "alpha", bb_[1], ALPHAS)
                        pf.grid(cid, f"gda_{s}", "reg", reg, LAMS,
                                lo_boundary=True)

                    # ---- THE METHOD: greedy layer ensemble ----
                    S, al, v = greedy(cv, a_val, y_val, ALPHAS, GREEDY_T)
                    accs["le_gda"] = acc(
                        a_te + al * torch.stack([ct[i] for i in S]).mean(0).to(dev),
                        y_te)
                    sel["le_gda"] = {"val": v, "alpha": al,
                                     "subset": [SPACES[i] for i in S]}
                    pf.grid(cid, "le_gda", "alpha", al, ALPHAS)

                    # ---- THE MATCHED NULL ----
                    # v1 was a BOOTSTRAP ensemble at the final layer and it was
                    # DEGENERATE: resampling C*K support points with replacement
                    # leaves many classes with zero samples, so GDA cannot fit.
                    # Measured 35.00 at K=4 and 1.05 at K=1 (chance). Beating a
                    # broken arm is not a control -- E10, committed by me.
                    # Replaced by a RIDGE ensemble: len(SPACES) GDA fits at the
                    # FINAL layer with different ridge strengths, then the
                    # identical greedy selection and alpha search. Every member
                    # is non-degenerate and information-equivalent (same
                    # features), so this isolates "several diverse estimators
                    # plus a subset search" from "layers carry complementary
                    # information", at exactly matched validation budget.
                    nv, nt = [], []
                    regs = logspace(1e-6, 1e2, len(SPACES))
                    for reg in regs:
                        Wg, bg = gda_weights(D[("pool", "last")][idx], ytr,
                                             n_cls, "ridge", reg)
                        c2 = D[("val", "last")] @ Wg + bg
                        an2 = anchor_of(c2)
                        nv.append(c2 / an2)
                        nt.append(((D[("test", "last")] @ Wg + bg) / an2).cpu())
                    Sb, alb, vb = greedy(nv, a_val, y_val, ALPHAS, GREEDY_T)
                    accs["null_ridge"] = acc(
                        a_te + alb * torch.stack([nt[i] for i in Sb]).mean(0).to(dev),
                        y_te)
                    sel["null_ridge"] = {"val": vb, "alpha": alb,
                                         "n_members": len(Sb)}
                    # activation: a null whose members are all identical would
                    # collapse to gda_last and control for nothing. Report the
                    # spread of member val-accuracies so that is visible.
                    mv = [max(acc_batched(a_val, x, ALPHAS, y_val)) for x in nv]
                    pf.activation(cid, "null_ridge", "member_spread",
                                  1.0 if (max(mv) - min(mv)) > 1e-6 else 0.0,
                                  0.0)

                    # ---- nesting: a 1-member ensemble at `last` == gda_last ----
                    if not pf.nests:
                        li = SPACES.index("last")
                        b1 = (-1.0, None)
                        for a_, s_ in zip(ALPHAS,
                                          acc_batched(a_val, cv[li], ALPHAS,
                                                      y_val)):
                            if s_ > b1[0]:
                                b1 = (s_, a_)
                        pf.nesting("1-member ensemble at `last` == gda_last",
                                   abs(acc(a_te + b1[1] * ct[li].to(dev), y_te)
                                       - accs["gda_last"]), tol=1e-9)

                    cells.append({"backbone": bb, "dataset": name, "K": K,
                                  "test_split": args.test_split,
                                  "seed": sd, "zs": zs,
                                  "C": accs["gda_last"] - zs,
                                  "acc": accs, "sel": sel})
                    other = [x for x in SPACES if x != "last"][:2]
                    print(f"{bb:<9}{name:<15}K={K:<3}s={sd} zs={zs:5.2f} "
                          f"last={accs['gda_last']:5.2f} "
                          + " ".join(f"{o}={accs[f'gda_{o}']:5.2f}"
                                     for o in other) + " "
                          f"LE={accs['le_gda']:5.2f} "
                          f"null={accs['null_ridge']:5.2f} | "
                          f"LE-last={accs['le_gda']-accs['gda_last']:+5.2f} "
                          f"LE-null={accs['le_gda']-accs['null_ridge']:+5.2f} "
                          f"{sel['le_gda']['subset']}", flush=True)
                    json.dump({"cells": cells, "preflight": pf.to_dict()},
                              open(args.out, "w"), indent=2)
            del D
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    json.dump({"cells": cells, "preflight": pf.to_dict()},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}  ({len(cells)} cells)")
    pf.report()


if __name__ == "__main__":
    main()
