"""LE-GDA: fusion of the per-layer GDA scores, and the fusion-rule comparison.

For every cell, each candidate space gets a ridge-searched GDA fit exactly as in
scripts/run_layer_ensemble.py. The per-space scores ghat_l(x) are then fused by
one of the rules below, added to the zero-shot logits a(x) with weight alpha,

    z(x) = a(x) + alpha * fuse_{l in S}( T(ghat_l(x)) ),

and each rule runs its OWN greedy subset selection on the validation split, so
no rule is evaluated on a subset chosen for another one.

    F1  per-query z-score across classes, then mean     <- LE-GDA (the paper)
    F0  uniform mean of the raw scores
    F2  per-query normalised rank, then mean
    F3  median over the selected spaces
    F4  per-space softmax, then mean of probabilities

The output JSON stores every rule under rows[*]["rules"][rule]; the tables and
figures read rules["F1"]. F1 is scale-free, so the per-space anchor used by F0
cancels; the script measures this per cell (anchor_free_F1, anchor_dev).
"""

import argparse
import glob
import json

import torch

import clip
from utils import clip_classifier
from scripts.ds_meta import class_template
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

    argsort twice gives each entry its position in the sorted order.
    """
    C = g.shape[-1]
    r = g.argsort(-1).argsort(-1).float()
    return r / max(1, C - 1)


def softmax_p(g):
    return g.softmax(-1)


TRANSFORMS = {"F0": None, "F1": z_score, "F2": rank_norm,
              "F3": None, "F4": softmax_p}


def combine(rule, mats):
    """Fuse a list of per-space score tensors under the rule's operator."""
    st = torch.stack(mats)
    return st.median(0).values if rule == "F3" else st.mean(0)


def greedy_rule(rule, cands, a_val, y_val, ALPHAS, T):
    """Forward selection with replacement on validation, under `rule`'s operator.

    Same procedure and budget as run_layer_ensemble.greedy: T steps, every
    candidate scored at every alpha.
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
    """Top-1 of the fused test logits, streamed in row chunks.

    On ImageNet the test-side logits are 40k x 1000 per space and do not fit on
    an 8 GB GPU together with their transformed copies, so `ct` is kept on the
    CPU and moved one chunk at a time. Chunking is exact: the transform, the
    fusion and the argmax are all per sample.
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
    """The uniform-mean le_gda recorded by run_layer_ensemble.py for this cell."""
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
                         "because it is checked against run_layer_ensemble.py.")
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

                    # per-space anchored logits, as in run_layer_ensemble.py
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
                        cv_raw.append(best[4])      # unanchored, for the
                        ct_raw.append(best[5])      # anchor-invariance check
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

                    # Anchor invariance of F1: ghat = g / anchor with a positive
                    # scalar anchor, and z(g/a) = z(g), so greedy and the alpha
                    # search see identical inputs with or without the anchor.
                    # Measured per cell: anchor_dev should be 0.
                    Sr, alr, _ = greedy_rule("F1", [z_score(x) for x in cv_raw],
                                             a_val, y_val, ALPHAS, GREEDY_T)
                    f1_raw = eval_test("F1", z_score, ct_raw, Sr, alr,
                                       a_te, y_te, dev)
                    out["anchor_free_F1"] = f1_raw
                    out["anchor_dev"] = abs(f1_raw - out["rules"]["F1"]["test"])

                    # Check: F0 must reproduce the uniform-mean le_gda recorded
                    # by run_layer_ensemble.py for the same cell.
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
