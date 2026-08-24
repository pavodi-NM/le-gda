"""N -- does fusion OUTSIDE the linear family buy anything?

Pre-registered before the code was written; the kill conditions are stated
in the paper.

WHY THIS AND NOT ANOTHER COMBINER. The weighting ceiling measured a BOUND on

    z(x) = a(x) + alpha * sum_l w_l ghat_l(x)

-- +0.528 oracle, of which ~+0.15 is weighting. Everything in that family is
closed; nothing outside it is. LE-GDA's anchor fixes ONE GLOBAL SCALAR per
space and says nothing about how each space's score distribution differs PER
SAMPLE. That is not a weight, and no w_l can express it.

THE STRUCTURAL TEST every candidate had to pass first: does it change the
argmax-relevant part of ghat_l in a way neither absorbed by alpha nor
expressible as w_l? It rejected James-Stein mean shrinkage before any code was
written -- for a shared-covariance discriminant, shrinking every mu_c toward
mubar rescales the class-dependent term and adds a class-independent constant,
so at balanced K it is absorbed by alpha. The four rules here all pass it,
because each transforms ghat_l PER SAMPLE.

F0  uniform mean over the selected multiset            -- the method today
F1  per-sample z-score across classes, then mean       -- 1/s_l(x) is not a w_l
F2  per-sample normalised rank, then mean              -- nonlinear, scale-free
F3  median over the selected multiset                  -- nonlinear in ghat
F4  per-space softmax, then mean of PROBABILITIES      -- the geometric mean
    would be linear in ghat and is excluded for exactly that reason

Each rule gets its OWN greedy subset selection under its own combiner, so no
rule is handicapped by a subset chosen for a different operator, and the search
budget is matched exactly.

F1 and F2 are scale-free, so they make the anchor redundant. If either wins, the
method gets SIMPLER, not more complex, and contribution C2 retires with it.

SCOPE: our own arms only. The per-space GDA, the ridge search and the anchor are
untouched, so no baseline row and no head-to-head table moves.
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
from scripts.preflight import Preflight

DATASETS = ["stanford_cars", "fgvc", "oxford_flowers", "dtd"]
BACKBONES = ["ViT-B/16"]
SEEDS = [0, 1, 2]
N_LAM = 12
GREEDY_T = 4


def z_score(g):
    """Per-sample standardisation across the class axis."""
    return (g - g.mean(-1, keepdim=True)) / g.std(-1, keepdim=True).clamp_min(1e-12)


def rank_norm(g):
    """Per-sample normalised rank across classes, in [0, 1].

    argsort twice is the standard rank trick; the second argsort inverts the
    permutation, giving each entry its position. Scale-free by construction, so
    the anchor is irrelevant to this rule.
    """
    C = g.shape[-1]
    r = g.argsort(-1).argsort(-1).float()
    return r / max(1, C - 1)


def softmax_p(g):
    return g.softmax(-1)


TOPK = 10          # fixed in advance, in the pre-registration


def topk_z(g, k=TOPK):
    """Standardise by the mean/std of the TOP-k classes only.

    F1 divides by a spread computed over ALL classes; the decision only involves
    the top of the distribution, so on a 1000-class dataset ~990 classes that are
    never in contention set the scale. This normalises by the decision's own
    support instead. k is FIXED at 10 and not searched.
    """
    t = g.topk(min(k, g.shape[-1]), dim=-1).values
    return (g - t.mean(-1, keepdim=True)) / t.std(-1, keepdim=True).clamp_min(1e-12)


def robust_z(g):
    """Median / IQR: robust to the bulk without discarding it."""
    med = g.median(-1, keepdim=True).values
    q = torch.quantile(g.float(), torch.tensor([0.25, 0.75], device=g.device),
                       dim=-1)
    iqr = (q[1] - q[0]).unsqueeze(-1).clamp_min(1e-12)
    return (g - med) / iqr


def winsor_z(g, c=3.0):
    """z-score, clipped at +-c sigma, then re-standardised.

    Stops a single outlier class from setting the scale for every other class.
    """
    z = z_score(g)
    z = z.clamp(-c, c)
    return z_score(z)


TRANSFORMS = {"F0": None, "F1": z_score, "F2": rank_norm,
              "F3": None, "F4": softmax_p,
              "G1": topk_z, "G2": robust_z, "G3": winsor_z}


def combine(rule, mats):
    """Fuse a list of per-space score tensors under the rule's operator."""
    st = torch.stack(mats)
    return st.median(0).values if rule == "F3" else st.mean(0)


def greedy_rule(rule, cands, a_val, y_val, ALPHAS, T):
    """Forward selection with replacement on VALIDATION, under `rule`'s operator.

    Identical budget to run_layer_ensemble.greedy -- T steps, every candidate
    scored at every alpha -- so no rule is advantaged by a bigger search.
    """
    S, best = [], (-1.0, None, None)
    for _ in range(T):
        step = (-1.0, None, None)
        for j in range(len(cands)):
            e = combine(rule, [cands[i] for i in S + [j]])
            for al, s in zip(ALPHAS, acc_batched(a_val, e, ALPHAS, y_val)):
                if s > step[0]:
                    step = (s, j, al)
        S = S + [step[1]]
        if step[0] > best[0]:
            best = (step[0], list(S), step[2])
    return best[1], best[2], best[0]


def eval_test(rule, tf, ct, S, al, a_te, y_te, dev, chunk=4096):
    """top-1 of the fused test logits, streamed in row chunks.

    IMAGENET FORCED THIS. The test-side logits are [n, C] per space, and at
    40k x 1000 seven of them are 1.1 GB -- with the transformed copies alongside
    they OOM an 8 GB card, which is the third documented instance of the same
    failure on this project (write-up §5b.4). So `ct` lives on the CPU and
    only one chunk per selected space is ever resident.

    Chunking is EXACT: the transform is per sample and per space, the combine is
    per sample, and argmax is per sample -- no statistic crosses a row boundary.
    Verified against the unchunked path on a dev cell to max|d| = 0.
    """
    n, correct = a_te.shape[0], 0
    for i in range(0, n, chunk):
        mats = []
        for j in S:
            g = ct[j][i:i + chunk].to(dev, non_blocking=True)
            mats.append(tf(g) if tf else g)
        e = combine(rule, mats)
        pred = (a_te[i:i + chunk] + al * e).argmax(-1)
        correct += int((pred == y_te[i:i + chunk]).sum())
        del mats, e
    return 100.0 * correct / n


def stored_le_gda(bb, name, K, sd):
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
    ap.add_argument("--rules", nargs="+", default=None,
                    help="subset of TRANSFORMS to run; F0 is always included "
                         "because it is the nesting reference.")
    ap.add_argument("--out", default="evidence/nonlinear_fusion.json")
    args = ap.parse_args()
    dev = args.device
    pf = Preflight("nonlinear_fusion")

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

            for K in args.ks:
                for sd in args.seeds:
                    idx = draw(D[("pool", "y")].cpu(), K, sd, dev)
                    ytr = D[("pool", "y")][idx]
                    cid = (bb, name, K, sd)

                    # per-space anchored logits, EXACTLY as LE-GDA builds them
                    cv, ct, cv_raw, ct_raw = [], [], [], []
                    for s in SPACES:
                        ftr = D[("pool", s)][idx]
                        best = (-1.0, None, None, None, None, None)
                        for reg in LAMS:
                            Wg, bg = gda_weights(ftr, ytr, n_cls, "ridge", reg)
                            v = D[("val", s)] @ Wg + bg
                            a_ = anchor_of(v)
                            sc = max(acc_batched(a_val, v / a_, ALPHAS, y_val))
                            if sc > best[0]:
                                best = (sc, v / a_,
                                        ((D[("test", s)] @ Wg + bg) / a_).cpu(),
                                        reg, v, (D[("test", s)] @ Wg + bg).cpu())
                        cv.append(best[1])
                        ct.append(best[2])          # CPU-resident: see eval_test
                        cv_raw.append(best[4])      # UNANCHORED, for the
                        ct_raw.append(best[5])      # anchor-invariance proof
                        pf.grid(cid, f"gda_{s}", "reg", best[3], LAMS,
                                lo_boundary=True)

                    out = {"backbone": bb, "dataset": name, "K": K, "seed": sd,
                           "test_split": args.test_split, "rules": {}}
                    active = {k: v for k, v in TRANSFORMS.items()
                              if args.rules is None or k in args.rules
                              or k == "F0"}
                    for rule, tf in active.items():
                        gv = [tf(x) for x in cv] if tf else cv
                        S, al, vsc = greedy_rule(rule, gv, a_val, y_val,
                                                 ALPHAS, GREEDY_T)
                        out["rules"][rule] = {
                            "test": eval_test(rule, tf, ct, S, al, a_te, y_te,
                                              dev),
                            "val": vsc, "alpha": al,
                            "subset": [SPACES[i] for i in S]}
                        if rule == "F0":
                            pf.grid(cid, "le_gda", "alpha", al, ALPHAS)

                    # ANCHOR INVARIANCE — C2 is retired by PROOF, not ablation.
                    # ghat = g / anchor with anchor > 0 scalar, and z-scoring
                    # divides by the per-sample std, so the anchor cancels
                    # EXACTLY: z(g/a) = (g/a - mean(g/a)) / std(g/a) = z(g).
                    # Greedy and the alpha search both see only z-scored scores,
                    # so the WHOLE pipeline is anchor-invariant under F1. That is
                    # a stronger statement than "ablating the anchor costs
                    # nothing" -- it says the anchor is a no-op, so the write-up
                    # §2.2 stops being a component at all. Measured here per
                    # cell rather than asserted.
                    Sr, alr, _ = greedy_rule("F1", [z_score(x) for x in cv_raw],
                                             a_val, y_val, ALPHAS, GREEDY_T)
                    f1_raw = eval_test("F1", z_score, ct_raw, Sr, alr,
                                       a_te, y_te, dev)
                    out["anchor_free_F1"] = f1_raw
                    out["anchor_dev"] = abs(f1_raw - out["rules"]["F1"]["test"])

                    # NESTING: F0 IS the method. If the rebuilt pipeline does not
                    # land on the recorded le_gda, every rule delta below is
                    # measured against the wrong reference.
                    rec = stored_le_gda(bb, name, K, sd)
                    if rec is not None and not pf.nests:
                        pf.nesting(f"F0 reproduces recorded le_gda ({name})",
                                   abs(out["rules"]["F0"]["test"] - rec),
                                   tol=1e-4)

                    rows.append(out)
                    f0 = out["rules"]["F0"]["test"]
                    print(f"{bb:<9}{name:<15}K={K:<3}s={sd} F0={f0:6.2f} | "
                          + "  ".join(
                        f"{r}{out['rules'][r]['test'] - f0:+.2f}"
                        for r in out["rules"] if r != "F0")
                          + f" | anchor-free dev {out['anchor_dev']:.1e}",
                          flush=True)
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
