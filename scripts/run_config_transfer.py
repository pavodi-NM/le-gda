"""Leave-one-dataset-out configuration-transfer ablation for LE-GDA.

LE-GDA selects a subset of spaces, a ridge per space and a fusion weight alpha
on the validation split. This ablation asks how much of its gain needs that
validation access, by fixing the configuration from the OTHER datasets. Arms,
all paired within (backbone, dataset, seed, K):

  arm0_zs                 zero-shot CLIP
  arm1_gda_eb             final space only, closed-form Ledoit-Wolf ridge,
                          transferred alpha                     (no validation)
  arm3_valfree            transferred multiset, ridges and alpha (no validation)
  arm3a_loolam_valalpha   transferred multiset and ridges, alpha on validation
  arm2_loo_multiset       transferred multiset, ridges and alpha on validation
  gda_last_val            final space only, ridge and alpha on validation
                          (GDA-CLIP); compare arm2 against this, since both
                          use validation for the same quantities

Full LE-GDA is read from evidence/nonlinear_fusion_*.json.

Transfer rule: per backbone (the candidate spaces differ by architecture) and
leave one DATASET out. For a held-out dataset, every fixed quantity comes from
the other 10 datasets x 3 seeds = 30 runs; the support set itself is still
drawn per seed.

Scores are standardised per query (z-score across classes) as in LE-GDA, so no
per-space anchor is needed: z(g/a) = z(g) for any a > 0.

In the validation-free arms the validation tensors are replaced by `Sealed`,
which raises on any access, so an accidental validation read cannot go
unnoticed.

Two phases:
    --phase derive   per-run donor quantities -> --derived
    --phase eval     transferred configuration on each held-out dataset -> --out
"""

import argparse
import collections
import json
import math
from statistics import median

import torch

import clip
from utils import clip_classifier
from scripts.ds_meta import class_template
from scripts.run_cache_retrieval_null import acc
from scripts.run_prototype_collapse_seeds import draw
from scripts.run_baselines_e4 import (gda_weights, anchor_of, acc_batched,
                                      logspace)
from scripts.run_selection_diagnostic import lw_intensity, residuals
from scripts.run_nonlinear_fusion import z_score, eval_test
from scripts.preflight import Preflight

DATASETS = ["caltech101", "dtd", "eurosat", "fgvc", "food101",
            "oxford_flowers", "oxford_pets", "stanford_cars", "sun397",
            "ucf101", "imagenet"]
BACKBONES = ["ViT-B/16", "ViT-B/32", "RN50", "RN101"]
SEEDS = [0, 1, 2]
N_LAM = 12
GREEDY_T = 4
VIT_SPACES = ["L4", "L6", "L8", "L10", "L11", "pre", "last"]
RN_SPACES = ["R1", "R2", "R3", "R4", "last"]
# the same grids as run_layer_ensemble.py
ALPHAS = logspace(1e-3, 10 ** (3 + 60 / 29), 40)
LAMS = logspace(1e-8, 1e2, N_LAM)


class Sealed:
    """Stand-in for a validation tensor; any access raises."""

    def __init__(self, what):
        object.__setattr__(self, "_what", what)

    def _boom(self, *a, **k):
        raise RuntimeError(
            f"a validation-free arm accessed "
            f"`{object.__getattribute__(self, '_what')}`")

    def __getattr__(self, n):
        self._boom()

    __matmul__ = __rmatmul__ = __add__ = __radd__ = __mul__ = __rmul__ = _boom
    __truediv__ = __rtruediv__ = __sub__ = __rsub__ = __getitem__ = _boom
    __len__ = __iter__ = __call__ = __eq__ = __hash__ = _boom


def record_index(k=16):
    """LE-GDA results from run_nonlinear_fusion.py, keyed by cell.

    The derivation is checked against these: rerun on the same cell, it must
    select the same subset and alpha.
    """
    import glob
    idx = {}
    for f in glob.glob("evidence/nonlinear_fusion_*.json"):
        for r in json.load(open(f)).get("rows", []):
            if r["K"] == k and "F1" in r.get("rules", {}):
                idx[(r["backbone"], r["dataset"], r["K"], r["seed"])] = \
                    r["rules"]["F1"]
    return idx


def spaces_for(bb):
    return RN_SPACES if bb.startswith("RN") else VIT_SPACES


def split_token(name):
    """`testfull` where it exists, `test` for ImageNet (one test file)."""
    import os
    return "testfull" if os.path.exists(
        f"caches/{name}/collapse/RN50_testfull_l.pt") else "test"


def load_cell(name, tag, spaces, dev, tsplit):
    D = {}
    for sp in ("val", "test", "pool"):
        fsp = tsplit if sp == "test" else sp
        for s in spaces:
            suf = {"pre": "_pre_f", "last": "_f"}.get(s, f"_{s}_f")
            D[(sp, s)] = torch.load(f"caches/{name}/collapse/{tag}_{fsp}{suf}.pt",
                                    map_location=dev).float()
        D[(sp, "y")] = torch.load(f"caches/{name}/collapse/{tag}_{fsp}_l.pt",
                                  map_location=dev)
    return D


def greedy(cands, a_val, y_val, T):
    """Forward selection with replacement on validation, as in
    run_layer_ensemble.py."""
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


def derive_cell(D, spaces, idx, ytr, n_cls, a_val, y_val, pf, cid):
    """Donor quantities for one run, under per-query standardisation.

    The ridge per space is selected on the donor's own validation split exactly
    as LE-GDA selects it, then greedy selection and alpha run on z-scored
    scores. Also records arm 1's donor alpha (final space, Ledoit-Wolf ridge).
    """
    ftr = {s: D[("pool", s)][idx] for s in spaces}
    lam, raw_v = {}, {}
    for s in spaces:
        keep = (-1.0, None, None)
        for reg in LAMS:
            Wg, bg = gda_weights(ftr[s], ytr, n_cls, "ridge", reg)
            v = D[("val", s)] @ Wg + bg
            sc = max(acc_batched(a_val, v / anchor_of(v), ALPHAS, y_val))
            if sc > keep[0]:
                keep = (sc, reg, v)
        lam[s], raw_v[s] = keep[1], keep[2]
        pf.grid(cid, f"f1_{s}", "reg", keep[1], LAMS, lo_boundary=True)

    order = list(spaces)
    S, al, v = greedy([z_score(raw_v[s]) for s in order], a_val, y_val, GREEDY_T)
    subset = [order[i] for i in S]
    pf.grid(cid, "derive", "alpha_fuse", al, ALPHAS)
    e = torch.stack([z_score(raw_v[order[i]]) for i in S]).mean(0)
    curve = acc_batched(a_val, e, ALPHAS, y_val)

    rho, lam_eb = lw_intensity(residuals(ftr["last"], ytr, n_cls))
    Wg, bg = gda_weights(ftr["last"], ytr, n_cls, "ridge", lam_eb)
    sc = acc_batched(a_val, z_score(D[("val", "last")] @ Wg + bg), ALPHAS, y_val)
    a1 = ALPHAS[max(range(len(ALPHAS)), key=lambda i: sc[i])]
    pf.grid(cid, "derive", "alpha_arm1", a1, ALPHAS)

    pf.activation(cid, "derive", "multiset_is_multi_space",
                  0.0 if set(subset) == {"last"} else 1.0, 0.0)
    return {"lam_sup": lam, "subset_sup": subset, "alpha_sup": al,
            "val_sup": v, "alpha_curve": curve, "anchor_ratio": {"last": 1.0},
            "lam_eb": lam_eb, "rho_lw": rho, "alpha_arm1": a1,
            "val_arm1": max(sc)}


def nesting_check(D, spaces, idx, ytr, n_cls, a_val, y_val, pf, rec):
    """The derivation must reproduce LE-GDA's recorded subset and alpha."""
    order, cands, lam = list(spaces), [], {}
    for s in order:
        keep = (-1.0, None, None)
        for reg in LAMS:
            Wg, bg = gda_weights(D[("pool", s)][idx], ytr, n_cls, "ridge", reg)
            v = D[("val", s)] @ Wg + bg
            sc = max(acc_batched(a_val, v / anchor_of(v), ALPHAS, y_val))
            if sc > keep[0]:
                keep = (sc, reg, v)
        lam[s] = keep[1]
        cands.append(z_score(keep[2]))
    S, al, _ = greedy(cands, a_val, y_val, GREEDY_T)
    sub = [order[i] for i in S]
    dsub = 0.0 if sub == rec["subset"] else 1.0
    dal = abs(math.log10(al / rec["alpha"]))
    pf.nesting("derivation reproduces LE-GDA (subset / alpha)",
               max(dsub, dal), tol=1e-9)


# ---- evaluation arms; scores are z-scored, so no anchor is used ----
def arm1(D, idx, ytr, n_cls, a_te, y_te, cfg, dev, val_sealed):
    _ = val_sealed
    ftr = D[("pool", "last")][idx]
    rho, lam_eb = lw_intensity(residuals(ftr, ytr, n_cls))
    Wg, bg = gda_weights(ftr, ytr, n_cls, "ridge", lam_eb)
    ct = [(D[("test", "last")] @ Wg + bg).cpu()]
    return (eval_test("F1", z_score, ct, [0], cfg["alpha_arm1"], a_te, y_te,
                      dev), lam_eb, rho)


def arm3(D, idx, ytr, n_cls, a_te, y_te, cfg, dev, val_sealed, scale=None):
    """Transferred multiset, ridges and alpha. `scale`, if given, divides each
    space's logits (used only for the invariance check)."""
    _ = val_sealed
    ct, S = [], []
    for i, s in enumerate(cfg["spaces"]):
        ftr = D[("pool", s)][idx]
        Wg, bg = gda_weights(ftr, ytr, n_cls, "ridge", cfg["lam"][s])
        t = D[("test", s)] @ Wg + bg
        ct.append((t / scale[i] if scale else t).cpu())
        S += [i] * cfg["mult"][i]
    return eval_test("F1", z_score, ct, S, cfg["alpha_fuse"], a_te, y_te, dev)


def arm3a(D, idx, ytr, n_cls, a_val, y_val, a_te, y_te, cfg, dev):
    """Transferred multiset and ridges, alpha searched on validation.

    Splits arm2 - arm3 into the cost of fixing the ridges (arm2 - arm3a) and
    the cost of fixing alpha (arm3a - arm3).
    """
    ct, S = [], []
    cv = []
    for i, sp in enumerate(cfg["spaces"]):
        ftr = D[("pool", sp)][idx]
        Wg, bg = gda_weights(ftr, ytr, n_cls, "ridge", cfg["lam"][sp])
        cv.append(z_score(D[("val", sp)] @ Wg + bg))
        ct.append((D[("test", sp)] @ Wg + bg).cpu())
        S += [i] * cfg["mult"][i]
    ev = torch.stack([cv[i] for i in S]).mean(0)
    sco = acc_batched(a_val, ev, ALPHAS, y_val)
    al = ALPHAS[max(range(len(ALPHAS)), key=lambda i: sco[i])]
    return eval_test("F1", z_score, ct, S, al, a_te, y_te, dev), al


def gda_last(D, idx, ytr, n_cls, a_val, y_val, a_te, y_te, dev):
    """Final space only, ridge and alpha searched on validation.

    The reference for arm 2: both use validation for the ridge and alpha, so
    arm2 - gda_last varies only the representation. (arm2 - arm1 would also
    mix in validation access.)
    """
    ftr = D[("pool", "last")][idx]
    keep = (-1.0, None, None)
    for reg in LAMS:
        Wg, bg = gda_weights(ftr, ytr, n_cls, "ridge", reg)
        v = D[("val", "last")] @ Wg + bg
        sc = max(acc_batched(a_val, v / anchor_of(v), ALPHAS, y_val))
        if sc > keep[0]:
            keep = (sc, z_score(v), (D[("test", "last")] @ Wg + bg).cpu())
    sco = acc_batched(a_val, keep[1], ALPHAS, y_val)
    al = ALPHAS[max(range(len(ALPHAS)), key=lambda i: sco[i])]
    return eval_test("F1", z_score, [keep[2]], [0], al, a_te, y_te, dev)


def arm2(D, idx, ytr, n_cls, a_val, y_val, a_te, y_te, cfg, dev):
    ct, cv, S = [], [], []
    for i, s in enumerate(cfg["spaces"]):
        ftr = D[("pool", s)][idx]
        keep = (-1.0, None, None)
        for reg in LAMS:
            Wg, bg = gda_weights(ftr, ytr, n_cls, "ridge", reg)
            v = D[("val", s)] @ Wg + bg
            sc = max(acc_batched(a_val, v / anchor_of(v), ALPHAS, y_val))
            if sc > keep[0]:
                keep = (sc, z_score(v), (D[("test", s)] @ Wg + bg).cpu())
        cv.append(keep[1])
        ct.append(keep[2])
        S += [i] * cfg["mult"][i]
    ev = torch.stack([cv[i] for i in S]).mean(0)
    sco = acc_batched(a_val, ev, ALPHAS, y_val)
    al = ALPHAS[max(range(len(ALPHAS)), key=lambda i: sco[i])]
    return eval_test("F1", z_score, ct, S, al, a_te, y_te, dev), al


# --------------------------------------------------------------------------
# LOO derivation -- pure arithmetic over the donor records.
# --------------------------------------------------------------------------
def snap(vals, grid):
    """Median in log space, snapped to the nearest grid point."""
    m = median([math.log(v) for v in vals])
    return min(grid, key=lambda g: abs(math.log(g) - m))


def loo_config(records, bb, held_out):
    """The fixed configuration for `held_out`, from the other datasets only."""
    don = [r for r in records
           if r["backbone"] == bb and r["dataset"] != held_out]
    assert don, f"no donors for {bb}/{held_out}"

    # ---- multiset: top-k by total selection count, multiplicity by share ----
    # (the mode of the exact multiset is unstable: with replacement, T=4 and 7
    # candidates there are too many multisets for 30 donor runs)
    cnt = collections.Counter()
    sizes = collections.Counter()
    for r in don:
        cnt.update(r["d"]["subset_sup"])
        sizes[len(r["d"]["subset_sup"])] += 1
    k = sizes.most_common(1)[0][0]
    top = [s for s, _ in cnt.most_common(k)]
    tot = sum(cnt[s] for s in top)
    # largest remainder, so the multiplicities sum to exactly k
    exact = [k * cnt[s] / tot for s in top]
    mult = [max(1, int(x)) for x in exact]
    while sum(mult) > k:
        mult[max(range(len(mult)), key=lambda i: mult[i] - exact[i])] -= 1
    while sum(mult) < k:
        mult[max(range(len(mult)), key=lambda i: exact[i] - mult[i])] += 1

    # ---- alpha: median of log alpha over the donor runs ----
    # scores are averaged, so no correction is needed when k differs from a
    # donor run's own subset size
    a_fuse = snap([r["d"]["alpha_sup"] for r in don], ALPHAS)
    a_arm1 = snap([r["d"]["alpha_arm1"] for r in don], ALPHAS)

    # ---- lambda per space: median over the runs in which it was SELECTED ----
    # the ridge is relative to tr(Sigma)/D, so it is comparable across datasets
    lam = {}
    for s in top:
        sel = [r["d"]["lam_sup"][s] for r in don
               if s in r["d"]["subset_sup"]]
        lam[s] = snap(sel or [r["d"]["lam_sup"][s] for r in don], LAMS)
    return {"spaces": top, "mult": mult, "k": k, "alpha_fuse": a_fuse,
            "alpha_arm1": a_arm1, "lam": lam,
            "n_donor_runs": len(don),
            "n_donor_datasets": len({r["dataset"] for r in don})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--ks", nargs="+", type=int, default=[16])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--phase", choices=["derive", "eval"], required=True)
    ap.add_argument("--derived",
                    default="evidence/config_transfer_derive_f1.json")
    ap.add_argument("--out", default="evidence/config_transfer_f1.json")
    args = ap.parse_args()
    dev = args.device
    pf = Preflight("config_transfer")

    REC = record_index(args.ks[0]) if args.phase == "derive" else {}
    records = None
    if args.phase == "eval":
        records = json.load(open(args.derived))["records"]
        print(f"loaded {len(records)} donor records from {args.derived}")

    rows = []
    for bb in args.backbones:
        tag = bb.replace("/", "-")
        spaces = spaces_for(bb)
        model, _ = clip.load(bb, device=dev)
        model.eval()
        for name in args.datasets:
            tsplit = split_token(name)
            cn, tp = class_template(name)
            D = load_cell(name, tag, spaces, dev, tsplit)
            pf.assert_fp32(**{f"val_{s}": D[("val", s)] for s in spaces})
            W = clip_classifier(cn, tp, model, device=dev).float()
            a_val = 100.0 * D[("val", "last")] @ W
            a_te = 100.0 * D[("test", "last")] @ W
            y_val, y_te = D[("val", "y")], D[("test", "y")]
            zs = acc(a_te, y_te)
            n_cls = int(D[("pool", "y")].max()) + 1
            cfg = None
            if args.phase == "eval":
                cfg = loo_config(records, bb, name)

            for K in args.ks:
                for sd in args.seeds:
                    cid = (bb, name, K, sd)
                    idx = draw(D[("pool", "y")].cpu(), K, sd, dev)
                    ytr = D[("pool", "y")][idx]
                    if args.phase == "derive":
                        if not pf.nests:
                            nesting_check(D, spaces, idx, ytr, n_cls, a_val,
                                          y_val, pf, REC[cid])
                        d = derive_cell(D, spaces, idx, ytr, n_cls, a_val,
                                        y_val, pf, cid)
                        rows.append({"backbone": bb, "dataset": name, "K": K,
                                     "seed": sd, "test_split": tsplit,
                                     "zs": zs, "d": d})
                        print(f"{bb:<9}{name:<15}K={K:<3}s={sd} "
                              f"sub={d['subset_sup']} a={d['alpha_sup']:.3g} "
                              f"lam_eb={d['lam_eb']:.3g} "
                              f"a1={d['alpha_arm1']:.3g} "
                              f"anc_ratio={d['anchor_ratio']['last']:.3f}",
                              flush=True)
                    else:
                        sealed = Sealed(f"validation of {name}")
                        a1, lam_eb, rho = arm1(D, idx, ytr, n_cls, a_te,
                                               y_te, cfg, dev, sealed)
                        a3 = arm3(D, idx, ytr, n_cls, a_te, y_te, cfg,
                                  dev, sealed)
                        # checks: arm 3 is unchanged by an arbitrary positive
                        # per-space rescaling (z(g/a) = z(g)), and arm 3
                        # restricted to the final space with arm 1's ridge and
                        # alpha equals arm 1.
                        if not pf.nests:
                            sc = [3.7 + 2.0 * i
                                  for i in range(len(cfg["spaces"]))]
                            pf.nesting(
                                "arm3 is invariant to per-space rescaling",
                                0.0,
                                acc_diff=arm3(D, idx, ytr, n_cls, a_te,
                                              y_te, cfg, dev, sealed,
                                              scale=sc) - a3,
                                acc_tol=0.0)
                            deg = {"spaces": ["last"], "mult": [1],
                                   "lam": {"last": lam_eb},
                                   "alpha_fuse": cfg["alpha_arm1"]}
                            pf.nesting(
                                "arm3 restricted to `last` with arm1's "
                                "ridge and alpha == arm1", 0.0,
                                acc_diff=arm3(D, idx, ytr, n_cls, a_te,
                                              y_te, deg, dev, sealed)
                                - a1, acc_tol=0.0)
                        a2, al2 = arm2(D, idx, ytr, n_cls, a_val, y_val,
                                       a_te, y_te, cfg, dev)
                        gl = gda_last(D, idx, ytr, n_cls, a_val, y_val,
                                      a_te, y_te, dev)
                        a3a, al3a = arm3a(D, idx, ytr, n_cls, a_val,
                                          y_val, a_te, y_te, cfg, dev)
                        pf.activation(cid, "arm3",
                                      "multiset_is_multi_space",
                                      0.0 if cfg["spaces"] == ["last"]
                                      else 1.0, 0.0)
                        rows.append({
                            "backbone": bb, "dataset": name, "K": K,
                            "seed": sd, "test_split": tsplit,
                            "fusion": "F1",
                            "acc": {"arm0_zs": zs, "arm1_gda_eb": a1,
                                    "arm2_loo_multiset": a2,
                                    "arm3_valfree": a3,
                                    "arm3a_loolam_valalpha": a3a,
                                    "gda_last_val": gl},
                            "cfg": {"spaces": cfg["spaces"],
                                    "mult": cfg["mult"],
                                    "alpha_fuse": cfg["alpha_fuse"],
                                    "alpha_arm1": cfg["alpha_arm1"],
                                    "lam": cfg["lam"], "lam_eb": lam_eb,
                                    "rho_lw": rho, "alpha2_val": al2,
                                    "alpha3a_val": al3a}})
                        print(f"{bb:<9}{name:<15}K={K:<3}s={sd} "
                              f"zs={zs:5.2f} a1={a1:5.2f} a2={a2:5.2f} "
                              f"a3={a3:5.2f} a3a={a3a:5.2f} "
                              f"| 3-1={a3-a1:+5.2f} "
                              f"{cfg['spaces']}x{cfg['mult']}", flush=True)
                    json.dump({"records" if args.phase == "derive" else "cells":
                               rows, "preflight": pf.to_dict()},
                              open(args.derived if args.phase == "derive"
                                   else args.out, "w"), indent=2)
            del D
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
    print(f"\nwrote {len(rows)} rows")
    pf.report()


if __name__ == "__main__":
    main()
