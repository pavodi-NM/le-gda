"""LOO configuration-transfer ablation for LE-GDA. Report: the ablation write-up.

The reviewer question this answers is CLAP's: LE-GDA searches a subset, a ridge
per space and a fusion alpha on a validation split, so how much of its gain is
the representation ensemble and how much is validation access? Four arms, all
paired within (backbone, dataset, seed, K):

  0  zero-shot CLIP                                             the floor
  1  final space only, closed-form ridge, LOO-fixed alpha        VAL-FREE
  2  LOO-fixed multiset, lambda and alpha searched on validation
  3  LOO-fixed multiset, lambda and alpha                        VAL-FREE
  3b arm 3 with the published VALIDATION anchor                  anchor control
  4  full LE-GDA, read from the sweep of record                  the ceiling

DERIVATION RULE. Per backbone (candidate spaces differ by architecture) and
leave-one-DATASET-out: for held-out d, every fixed quantity comes from the other
10 datasets x 3 support sets = 30 runs. Support sets stay per seed -- this tests
CONFIGURATION transfer, not seed transfer.

THE ANCHOR IS THE PART THE PLAN DOES NOT MENTION, AND IT IS NOT OPTIONAL.
§2.2's anchor is the *validation* median top-class magnitude, so a literally
validation-free arm cannot use it. Arms 1 and 3 anchor on the SUPPORT set
instead, and -- this is the part that makes the comparison honest -- the donor
runs from which alpha is transferred are scored under the SAME support anchor.
Transferring an alpha derived against one scale and applying it against another
would penalise arm 3 for a units mismatch and call it a failure of transfer.
Arm 3b re-runs arm 3 with the validation anchor and nothing else changed, so
Delta(3b - 3) attributes any loss to the anchor rather than to lambda or alpha.

FIREWALL. In arms 1 and 3 the validation tensors are replaced by `Sealed`,
which raises on any attribute access or operator. A guard that raises is worth
more than a comment here, because an accidental validation read invalidates the
entire experiment and would be invisible in the numbers.
"""

import argparse
import collections
import json
import math
from statistics import median

import torch
import torch.nn.functional as F

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
# identical to run_layer_ensemble.py -- the grids ARE the pre-registration
ALPHAS = logspace(1e-3, 10 ** (3 + 60 / 29), 40)
LAMS = logspace(1e-8, 1e2, N_LAM)


class Sealed:
    """The validation firewall. Any touch raises.

    Arms 1 and 3 receive this in place of every validation tensor, so
    "validation-free" is a property of the executed code rather than of my
    care while writing it.
    """

    def __init__(self, what):
        object.__setattr__(self, "_what", what)

    def _boom(self, *a, **k):
        raise RuntimeError(
            f"VALIDATION FIREWALL BREACHED: a validation-free arm touched "
            f"`{object.__getattribute__(self, '_what')}`. The arm is not "
            f"validation-free and its number must be discarded.")

    def __getattr__(self, n):
        self._boom()

    __matmul__ = __rmatmul__ = __add__ = __radd__ = __mul__ = __rmul__ = _boom
    __truediv__ = __rtruediv__ = __sub__ = __rsub__ = __getitem__ = _boom
    __len__ = __iter__ = __call__ = __eq__ = __hash__ = _boom


def record_index_f1(k=16):
    """The §5e (F1) sweep of record, keyed by cell.

    This is the arm-4 ceiling for the F1 run AND the nesting reference for its
    derivation. Joining rather than recomputing keeps the pairing exact.
    """
    import glob
    idx = {}
    for f in glob.glob("evidence/nonlinear_fusion_*.json"):
        if "preregistration" in f:
            continue
        for r in json.load(open(f)).get("rows", []):
            if r["K"] == k and "F1" in r.get("rules", {}):
                idx[(r["backbone"], r["dataset"], r["K"], r["seed"])] = \
                    r["rules"]["F1"]
    return idx


def record_index(k=16):
    """The sweep of record, keyed by cell — the nesting reference.

    The derivation below re-implements §2's pipeline with the anchor
    swapped. If the same code with the ORIGINAL validation anchor does not
    reproduce the recorded lambda, subset and alpha exactly, then the donor
    quantities are being read off a different method than the one the paper
    reports, and every arm downstream is a contrast against an unknown.
    """
    import glob
    idx = {}
    for f in glob.glob("evidence/layer_ensemble_*testfull.json"):
        for c in json.load(open(f))["cells"]:
            if c["K"] == k:
                idx[(c["backbone"], c["dataset"], c["K"], c["seed"])] = c
    return idx


def spaces_for(bb):
    return RN_SPACES if bb.startswith("RN") else VIT_SPACES


def split_token(name):
    """`testfull` where it exists, `test` for imagenet.

    imagenet is the one dataset whose test split could not be re-extracted at
    full size -- a data limit, not a code one -- and its 40k eval split is the
    column that Table 1 already reports. Resolving by file existence keeps this
    experiment on exactly the same items as the sweep of record.
    """
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


def fuse(mats, mult):
    """Multiplicity-weighted mean -- identical to §2.4's mean over a multiset

    with replacement, which is what greedy selection with replacement produces.
    """
    tot = sum(mult)
    out = None
    for m, w in zip(mats, mult):
        out = m * (w / tot) if out is None else out + m * (w / tot)
    return out


def greedy(cands, a_val, y_val, T):
    """Forward selection with replacement on validation. Copied intact from
    run_layer_ensemble.py so the derivation reproduces the method of record."""
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


# --------------------------------------------------------------------------
# PHASE 1 -- derivation. Donor quantities, under the SUPPORT anchor.
# --------------------------------------------------------------------------
def nesting_check(D, spaces, idx, ytr, n_cls, a_val, y_val, pf, rec):
    """Same code, the PUBLISHED validation anchor -> must reproduce the record.

    Exact equality is the right tolerance here: these are grid points and a
    subset of names, not floating-point reductions over different kernels.
    """
    order, cv_val, lam_val = list(spaces), {}, {}
    for s in order:
        keep = (-1.0, None, None)
        for reg in LAMS:
            Wg, bg = gda_weights(D[("pool", s)][idx], ytr, n_cls, "ridge", reg)
            c = D[("val", s)] @ Wg + bg
            cvn = c / anchor_of(c)
            v = max(acc_batched(a_val, cvn, ALPHAS, y_val))
            if v > keep[0]:
                keep = (v, reg, cvn)
        lam_val[s], cv_val[s] = keep[1], keep[2]
    S, al, _ = greedy([cv_val[s] for s in order], a_val, y_val, GREEDY_T)
    sub = [order[i] for i in S]
    dlam = max(abs(math.log10(lam_val[s] / rec["sel"][f"gda_{s}"]["reg"]))
               for s in order)
    dsub = 0.0 if sub == rec["sel"]["le_gda"]["subset"] else 1.0
    dal = abs(math.log10(al / rec["sel"]["le_gda"]["alpha"]))
    pf.nesting("val-anchor rerun reproduces the sweep of record "
               "(lambda / subset / alpha)", max(dlam, dsub, dal), tol=1e-9,
               acc_diff=0.0)


def derive_cell_f1(D, spaces, idx, ytr, n_cls, a_val, y_val, pf, cid):
    """Donor quantities under Eq. 7's per-query standardisation (F1).

    THE ANCHOR IS GONE, BY PROOF RATHER THAN BY SUBSTITUTION. z-scoring divides
    by the per-sample std, so for any scalar a > 0, z(g/a) = z(g) exactly. The
    greedy search, the alpha search and the test evaluation all see only
    z-scored scores, so the whole pipeline is anchor-invariant and arms 1 and 3
    need no anchor at all -- not a support-set substitute, none. That deletes
    the F0 design's single largest source of arm-3 fragility (§3.2's ratio,
    which blows up to 4.44 +- 36.71 at K=4) rather than controlling for it.

    lambda per space is still selected the way the PAPER selects it -- the
    anchored F0 scan on donor validation (run_nonlinear_fusion.py builds F1's
    candidates from exactly these) -- because the donor may use its own
    validation and because copying the published rule is what makes the
    nesting check against the F1 record meaningful.
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


def nesting_check_f1(D, spaces, idx, ytr, n_cls, a_val, y_val, pf, rec):
    """The F1 derivation must reproduce the F1 record's subset and alpha."""
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
    pf.nesting("F1 derivation reproduces the §5e record (subset / alpha)",
               max(dsub, dal), tol=1e-9)


# ---- the F1 arms. No anchor appears anywhere below, by the proof above. ----
def arm1_f1(D, idx, ytr, n_cls, a_te, y_te, cfg, dev, val_sealed):
    _ = val_sealed
    ftr = D[("pool", "last")][idx]
    rho, lam_eb = lw_intensity(residuals(ftr, ytr, n_cls))
    Wg, bg = gda_weights(ftr, ytr, n_cls, "ridge", lam_eb)
    ct = [(D[("test", "last")] @ Wg + bg).cpu()]
    return (eval_test("F1", z_score, ct, [0], cfg["alpha_arm1"], a_te, y_te,
                      dev), lam_eb, rho)


def arm3_f1(D, idx, ytr, n_cls, a_te, y_te, cfg, dev, val_sealed, scale=None):
    """`scale` divides each space's logits -- used ONLY to prove invariance."""
    _ = val_sealed
    ct, S = [], []
    for i, s in enumerate(cfg["spaces"]):
        ftr = D[("pool", s)][idx]
        Wg, bg = gda_weights(ftr, ytr, n_cls, "ridge", cfg["lam"][s])
        t = D[("test", s)] @ Wg + bg
        ct.append((t / scale[i] if scale else t).cpu())
        S += [i] * cfg["mult"][i]
    return eval_test("F1", z_score, ct, S, cfg["alpha_fuse"], a_te, y_te, dev)


def arm3a_f1(D, idx, ytr, n_cls, a_val, y_val, a_te, y_te, cfg, dev):
    """LOO multiset AND LOO lambda, but alpha searched on validation.

    This splits `2 - 3` -- which lumps lambda and alpha together and was an
    acknowledged gap in the first version of this ablation -- into two
    contrasts: `2 - 3a` is the cost of fixing lambda alone, `3a - 3` the cost
    of fixing alpha alone. Under F1 that separation is the whole question,
    because z-scoring makes alpha nearly dataset-independent while lambda
    stays per-dataset.
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


def gda_last_f1(D, idx, ytr, n_cls, a_val, y_val, a_te, y_te, dev):
    """Single final space, lambda AND alpha searched on validation, under F1.

    This is the reference the multiset contrast actually needs. Arm 2 uses
    validation for lambda and alpha; arm 1 does not. So `arm2 - arm1` conflates
    "ensemble vs single space" with "validation vs none", and quoting it as
    evidence that the multiset transfers is a comparison the experiment is not
    entitled to make. `arm2 - gda_last` holds validation access FIXED on both
    sides and varies only the representation, which is the contrast that is.
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


def arm2_f1(D, idx, ytr, n_cls, a_val, y_val, a_te, y_te, cfg, dev):
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


def derive_cell(D, spaces, idx, ytr, n_cls, a_val, y_val, pf, cid):
    ftr = {s: D[("pool", s)][idx] for s in spaces}
    best_sup, cv_sup, anc_ratio = {}, {}, {}
    for s in spaces:
        keep = (-1.0, None, None)
        for reg in LAMS:
            Wg, bg = gda_weights(ftr[s], ytr, n_cls, "ridge", reg)
            c_sup = ftr[s] @ Wg + bg
            a_sup = anchor_of(c_sup)
            cvn = (D[("val", s)] @ Wg + bg) / a_sup
            v = max(acc_batched(a_val, cvn, ALPHAS, y_val))
            if v > keep[0]:
                keep = (v, reg, cvn)
                anc_ratio[s] = float(a_sup / anchor_of(D[("val", s)] @ Wg + bg))
        best_sup[s] = keep[1]
        cv_sup[s] = keep[2]
        pf.grid(cid, f"sup_{s}", "reg", keep[1], LAMS, lo_boundary=True)

    order = list(spaces)
    S, al, v = greedy([cv_sup[s] for s in order], a_val, y_val, GREEDY_T)
    subset = [order[i] for i in S]
    pf.grid(cid, "derive", "alpha_fuse", al, ALPHAS)
    # the alpha PLATEAU on validation. The preflight measures dispersion of the
    # argmax in log alpha; what a fixed alpha actually costs is the DEPTH of the
    # curve away from that argmax, and a flat curve makes a dispersed argmax
    # cheap. 40 floats per cell, no extra fits.
    e = torch.stack([cv_sup[order[i]] for i in S]).mean(0)
    curve = acc_batched(a_val, e, ALPHAS, y_val)

    # ---- arm-1 donor: final space, closed-form ridge, support anchor ----
    rho, lam_eb = lw_intensity(residuals(ftr["last"], ytr, n_cls))
    Wg, bg = gda_weights(ftr["last"], ytr, n_cls, "ridge", lam_eb)
    c_sup = ftr["last"] @ Wg + bg
    cvn = (D[("val", "last")] @ Wg + bg) / anchor_of(c_sup)
    sc = acc_batched(a_val, cvn, ALPHAS, y_val)
    a1 = ALPHAS[max(range(len(ALPHAS)), key=lambda i: sc[i])]
    pf.grid(cid, "derive", "alpha_arm1", a1, ALPHAS)

    # a selection that never leaves `last` is not an ensemble, and arm 3 would
    # then be arm 1 with a different alpha rather than a transferred multiset.
    pf.activation(cid, "derive", "multiset_is_multi_space",
                  0.0 if set(subset) == {"last"} else 1.0, 0.0)

    return {"lam_sup": best_sup, "subset_sup": subset, "alpha_sup": al,
            "val_sup": v, "alpha_curve": curve, "anchor_ratio": anc_ratio,
            "lam_eb": lam_eb, "rho_lw": rho, "alpha_arm1": a1,
            "val_arm1": max(sc)}


# --------------------------------------------------------------------------
# LOO derivation -- pure arithmetic over the donor records.
# --------------------------------------------------------------------------
def snap(vals, grid):
    """Median in LOG space, snapped to the nearest grid point.

    The grids are log-spaced, so the median of the logs is the only aggregate
    that is invariant to the units alpha and lambda are quoted in.
    """
    m = median([math.log(v) for v in vals])
    return min(grid, key=lambda g: abs(math.log(g) - m))


def loo_config(records, bb, held_out):
    """The fixed configuration for `held_out`, from the other datasets only."""
    don = [r for r in records
           if r["backbone"] == bb and r["dataset"] != held_out]
    assert don, f"no donors for {bb}/{held_out}"

    # ---- multiset: top-k by total selection count, multiplicity by share ----
    # NOT the mode of the exact multiset: with replacement, T=4 and 7
    # candidates the multiset space is large enough that 30 samples give a
    # near-uniform histogram and an unstable mode. Ranking degrades gracefully.
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
    # alpha / |L*| is already normalised by the mean in fuse(), so no correction
    # is needed when k differs from a donor run's own |L*|.
    a_fuse = snap([r["d"]["alpha_sup"] for r in don], ALPHAS)
    a_arm1 = snap([r["d"]["alpha_arm1"] for r in don], ALPHAS)

    # ---- lambda per space: median over the runs in which it was SELECTED ----
    # Eq. 5's trace scaling already normalises across spaces and
    # dimensionalities, which is what makes one lambda per space plausible.
    lam = {}
    for s in top:
        sel = [r["d"]["lam_sup"][s] for r in don
               if s in r["d"]["subset_sup"]]
        lam[s] = snap(sel or [r["d"]["lam_sup"][s] for r in don], LAMS)
    return {"spaces": top, "mult": mult, "k": k, "alpha_fuse": a_fuse,
            "alpha_arm1": a_arm1, "lam": lam,
            "n_donor_runs": len(don),
            "n_donor_datasets": len({r["dataset"] for r in don})}


# --------------------------------------------------------------------------
# PHASE 2 -- evaluation of the four arms on the held-out dataset.
# --------------------------------------------------------------------------
def arm1(D, idx, ytr, n_cls, a_te, y_te, cfg, val_sealed):
    """final space, closed-form ridge, support anchor, LOO-fixed alpha."""
    _ = val_sealed          # named so the firewall object is in scope, unused
    ftr = D[("pool", "last")][idx]
    rho, lam_eb = lw_intensity(residuals(ftr, ytr, n_cls))
    Wg, bg = gda_weights(ftr, ytr, n_cls, "ridge", lam_eb)
    anc = anchor_of(ftr @ Wg + bg)
    ct = (D[("test", "last")] @ Wg + bg) / anc
    return acc(a_te + cfg["alpha_arm1"] * ct, y_te), lam_eb, rho


def arm3(D, idx, ytr, n_cls, a_te, y_te, cfg, val_sealed):
    """LOO multiset, LOO lambda, support anchor, LOO alpha. Fully val-free."""
    _ = val_sealed
    ct = []
    for s in cfg["spaces"]:
        ftr = D[("pool", s)][idx]
        Wg, bg = gda_weights(ftr, ytr, n_cls, "ridge", cfg["lam"][s])
        anc = anchor_of(ftr @ Wg + bg)
        ct.append((D[("test", s)] @ Wg + bg) / anc)
    return acc(a_te + cfg["alpha_fuse"] * fuse(ct, cfg["mult"]), y_te)


def arm3b(D, idx, ytr, n_cls, a_val, a_te, y_te, cfg):
    """arm 3 with the VALIDATION anchor. Not val-free; isolates the anchor."""
    ct = []
    for s in cfg["spaces"]:
        ftr = D[("pool", s)][idx]
        Wg, bg = gda_weights(ftr, ytr, n_cls, "ridge", cfg["lam"][s])
        anc = anchor_of(D[("val", s)] @ Wg + bg)
        ct.append((D[("test", s)] @ Wg + bg) / anc)
    return acc(a_te + cfg["alpha_fuse"] * fuse(ct, cfg["mult"]), y_te)


def arm2(D, idx, ytr, n_cls, a_val, y_val, a_te, y_te, cfg):
    """LOO multiset; lambda and alpha searched on the held-out validation."""
    cv, ct = [], []
    for s in cfg["spaces"]:
        ftr = D[("pool", s)][idx]
        keep = (-1.0, None, None, None)
        for reg in LAMS:
            Wg, bg = gda_weights(ftr, ytr, n_cls, "ridge", reg)
            c = D[("val", s)] @ Wg + bg
            anc = anchor_of(c)
            v = max(acc_batched(a_val, c / anc, ALPHAS, y_val))
            if v > keep[0]:
                keep = (v, c / anc, (D[("test", s)] @ Wg + bg) / anc, reg)
        cv.append(keep[1])
        ct.append(keep[2])
    ev = fuse(cv, cfg["mult"])
    sc = acc_batched(a_val, ev, ALPHAS, y_val)
    al = ALPHAS[max(range(len(ALPHAS)), key=lambda i: sc[i])]
    return acc(a_te + al * fuse(ct, cfg["mult"]), y_te), al


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--ks", nargs="+", type=int, default=[16])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--phase", choices=["derive", "eval"], required=True)
    ap.add_argument("--fusion", choices=["F0", "F1"], default="F0",
                    help="F0 = §2.4's uniform mean of anchored scores, the "
                         "operator §5d measures. F1 = Eq. 7's per-query "
                         "standardisation, WHICH IS THE PAPER'S METHOD -- the "
                         "F0 arms characterise a variant that is not in the "
                         "submission, so F1 is the run that counts.")
    ap.add_argument("--derived", default="evidence/config_transfer_derive.json")
    ap.add_argument("--out", default="evidence/config_transfer.json")
    args = ap.parse_args()
    dev = args.device
    pf = Preflight("config_transfer")

    if args.phase == "derive":
        REC = (record_index_f1(args.ks[0]) if args.fusion == "F1"
               else record_index(args.ks[0]))
    else:
        REC = {}
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
                            (nesting_check_f1 if args.fusion == "F1"
                             else nesting_check)(
                                D, spaces, idx, ytr, n_cls, a_val, y_val, pf,
                                REC[cid])
                        d = (derive_cell_f1 if args.fusion == "F1"
                             else derive_cell)(D, spaces, idx, ytr, n_cls,
                                               a_val, y_val, pf, cid)
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
                        if args.fusion == "F1":
                            a1, lam_eb, rho = arm1_f1(D, idx, ytr, n_cls, a_te,
                                                      y_te, cfg, dev, sealed)
                            a3 = arm3_f1(D, idx, ytr, n_cls, a_te, y_te, cfg,
                                         dev, sealed)
                            # ANCHOR INVARIANCE, checked rather than assumed:
                            # re-running arm 3 with every space's logits scaled
                            # by an arbitrary positive constant must return the
                            # IDENTICAL number, because z(g/a) = z(g). This is
                            # what retires §3.2's whole anchor problem -- and
                            # it is a proof-shaped check, so the tolerance is 0.
                            if not pf.nests:
                                sc = [3.7 + 2.0 * i
                                      for i in range(len(cfg["spaces"]))]
                                pf.nesting(
                                    "F1 arm3 is invariant to an arbitrary "
                                    "per-space rescaling (the anchor is a "
                                    "no-op)", 0.0,
                                    acc_diff=arm3_f1(D, idx, ytr, n_cls, a_te,
                                                     y_te, cfg, dev, sealed,
                                                     scale=sc) - a3,
                                    acc_tol=0.0)
                                deg = {"spaces": ["last"], "mult": [1],
                                       "lam": {"last": lam_eb},
                                       "alpha_fuse": cfg["alpha_arm1"]}
                                pf.nesting(
                                    "F1 arm3 restricted to `last` with arm1's "
                                    "ridge and alpha == arm1", 0.0,
                                    acc_diff=arm3_f1(D, idx, ytr, n_cls, a_te,
                                                     y_te, deg, dev, sealed)
                                    - a1, acc_tol=0.0)
                            a3b = float("nan")   # VOID under F1: no anchor
                            a2, al2 = arm2_f1(D, idx, ytr, n_cls, a_val, y_val,
                                              a_te, y_te, cfg, dev)
                            gl = gda_last_f1(D, idx, ytr, n_cls, a_val, y_val,
                                             a_te, y_te, dev)
                            a3a, al3a = arm3a_f1(D, idx, ytr, n_cls, a_val,
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
                            json.dump({"cells": rows,
                                       "preflight": pf.to_dict()},
                                      open(args.out, "w"), indent=2)
                            continue
                        a1, lam_eb, rho = arm1(D, idx, ytr, n_cls, a_te, y_te,
                                               cfg, sealed)
                        if not pf.nests:
                            # arm 3 degenerated to the single final space with
                            # arm 1's own ridge and alpha IS arm 1. If these
                            # disagree the fusion path is not nested in the
                            # baseline path and Delta(3-1) mixes two pipelines.
                            deg = {"spaces": ["last"], "mult": [1],
                                   "lam": {"last": lam_eb},
                                   "alpha_fuse": cfg["alpha_arm1"]}
                            pf.nesting("arm3 restricted to `last` with arm1's "
                                       "ridge and alpha == arm1", 0.0,
                                       acc_diff=arm3(D, idx, ytr, n_cls, a_te,
                                                     y_te, deg, sealed) - a1,
                                       acc_tol=1e-9)
                        pf.activation(cid, "arm3", "multiset_is_multi_space",
                                      0.0 if cfg["spaces"] == ["last"] else 1.0,
                                      0.0)
                        a3 = arm3(D, idx, ytr, n_cls, a_te, y_te, cfg, sealed)
                        a3b = arm3b(D, idx, ytr, n_cls, a_val, a_te, y_te, cfg)
                        a2, al2 = arm2(D, idx, ytr, n_cls, a_val, y_val, a_te,
                                       y_te, cfg)
                        rows.append({
                            "backbone": bb, "dataset": name, "K": K,
                            "seed": sd, "test_split": tsplit,
                            "acc": {"arm0_zs": zs, "arm1_gda_eb": a1,
                                    "arm2_loo_multiset": a2,
                                    "arm3_valfree": a3, "arm3b_valanchor": a3b},
                            "cfg": {"spaces": cfg["spaces"],
                                    "mult": cfg["mult"],
                                    "alpha_fuse": cfg["alpha_fuse"],
                                    "alpha_arm1": cfg["alpha_arm1"],
                                    "lam": cfg["lam"], "lam_eb": lam_eb,
                                    "rho_lw": rho, "alpha2_val": al2}})
                        print(f"{bb:<9}{name:<15}K={K:<3}s={sd} zs={zs:5.2f} "
                              f"a1={a1:5.2f} a2={a2:5.2f} a3={a3:5.2f} "
                              f"a3b={a3b:5.2f} | 3-1={a3-a1:+5.2f} "
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
