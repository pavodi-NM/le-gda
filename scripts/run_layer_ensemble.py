"""Per-layer GDA and the greedy layer ensemble, on cached CLIP features.

For every (backbone, dataset, K, seed) cell this script fits a shared-covariance
Gaussian discriminant (GDA) with a validation-searched ridge on each candidate
feature space, and records:

  gda_<space>   GDA on that space alone, added to the zero-shot logits;
                gda_last (the final projected embedding) is GDA-CLIP
  le_gda        greedy forward selection (with replacement) of up to four
                spaces on the validation split, scores fused by a UNIFORM MEAN
  null_ridge    control: the same greedy search over GDA fits of the final
                embedding at different ridge strengths, so it has the same
                search budget but no extra layers

The paper's LE-GDA standardises each selected space's scores per query before
averaging; that fusion rule is computed by scripts/run_nonlinear_fusion.py,
which reads this script's output. Use this script for GDA-CLIP and the
per-space accuracies.

A random projection of the final layer would not work as a control: GDA is
affine-equivariant, (Tf)^T (T Sigma T^T)^-1 (T mu) = f^T Sigma^-1 mu, so it
returns identical logits.
"""

import argparse
import json

import torch

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
    """Forward selection with replacement on the validation split.

    At each of T steps, add the candidate (repeats allowed) and fusion weight
    alpha that maximise validation accuracy of zero-shot + alpha * mean(scores);
    return the best prefix. Returns (subset indices, alpha, val accuracy).
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
                         "ResNets use `R1 R2 R3 R4 last`: the globally pooled "
                         "output of each stage plus the final embedding (there "
                         "is no pre-projection feature on ResNets).")
    ap.add_argument("--device", default="cuda")
    # "testfull" is the standard test split. "test" addresses a capped subset
    # (max(30, 3000/n_cls) per class) on the ten CoOp datasets; on ImageNet,
    # whose features have a single test file, pass "test".
    ap.add_argument("--test-split", default="testfull")
    ap.add_argument("--out", default="evidence/layer_ensemble.json")
    args = ap.parse_args()
    dev = args.device
    pf = Preflight("layer_ensemble")
    # Fusion weight grid: 40 log-spaced points from 1e-3 to ~1.17e5 (ratio
    # 10^(6/29)). Datasets where zero-shot CLIP is weak (e.g. EuroSAT) select
    # alpha around 1e3; alpha -> infinity is pure GDA with the text logits
    # ignored, and validation accuracy is flat there, so the grid only needs to
    # reach that plateau.
    ALPHAS = logspace(1e-3, 10 ** (3 + 60 / 29), 40)
    LAMS = logspace(1e-8, 1e2, N_LAM)
    SPACES = args.spaces or ([f"L{i}" for i in args.layers] + ["pre", "last"])
    assert "last" in SPACES, "the final embedding is needed for GDA-CLIP"

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
                # the file name carries --test-split; it is addressed as "test"
                # below
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
                        # test-side logits are kept on the CPU: greedy() only
                        # reads the validation side, and on ImageNet 7 spaces x
                        # [40k, 1000] would not fit next to the features on an
                        # 8 GB GPU.
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

                    # ---- greedy layer ensemble, uniform mean ----
                    S, al, v = greedy(cv, a_val, y_val, ALPHAS, GREEDY_T)
                    accs["le_gda"] = acc(
                        a_te + al * torch.stack([ct[i] for i in S]).mean(0).to(dev),
                        y_te)
                    sel["le_gda"] = {"val": v, "alpha": al,
                                     "subset": [SPACES[i] for i in S]}
                    pf.grid(cid, "le_gda", "alpha", al, ALPHAS)

                    # ---- control: ridge ensemble at the final layer ----
                    # len(SPACES) GDA fits of the final embedding with different
                    # ridge strengths, then the identical greedy selection and
                    # alpha search. Same features for every member, so this
                    # separates "several estimators plus a subset search" from
                    # "intermediate layers add information".
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
                    # members that were all identical would collapse to
                    # gda_last; record whether their val accuracies differ.
                    mv = [max(acc_batched(a_val, x, ALPHAS, y_val)) for x in nv]
                    pf.activation(cid, "null_ridge", "member_spread",
                                  1.0 if (max(mv) - min(mv)) > 1e-6 else 0.0,
                                  0.0)

                    # ---- check: a 1-member ensemble at `last` == gda_last ----
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
