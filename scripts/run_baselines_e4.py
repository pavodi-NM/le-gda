"""E4 — the mandatory published baselines. Plan §0.3 step 5(b), §8.

After §0.6's KC3 failure these are no longer diligence: they are the
experiment that decides whether the allocation work is a CONTRIBUTION or a
REPRODUCTION of kernel-space / shared-covariance methods under a budget.

    GPCache   Ding et al., IJCV, arXiv:2410.08895 (Oct 2024)
              logits_c = kappa_c(f, F) (K_c + sigma^2 I)^-1 Y
              Tip-Adapter's affinity with the Gram inverse inserted. Their
              headline: ImageNet 16-shot training-free 62.03 -> 64.17.
    GDA-CLIP  "A Hard-to-Beat Baseline", ICLR'24
              class means + a SINGLE SHARED covariance; Bayes collapses to the
              linear w_c = Sigma^-1 mu_c, ensembled with the zero-shot head.

*** THE STANDARD THIS SCRIPT IS HELD TO ***

The plan's standing rule is that an idea gets the same engineering effort aimed
at making it succeed as at making it fail. For a BASELINE that rule inverts:
every discretionary choice here must favour THEM, because a win over a crippled
baseline is not a result. Concretely, and each of these is deliberate:

  * They get a LARGER hyperparameter search than our own arms. Ours search
    50 beta x 20 alpha = 1000; GPCache searches 50 beta x 20 sigma^2 x 30 alpha
    = 30000 and GDA searches 20 lambda x 30 alpha per covariance estimator, over
    a closed-form solution, with the better of two estimators reported. Matched
    cardinality is the usual protocol control (prereg §7) and it is
    deliberately BROKEN in their favour here. The matched-cardinality number is
    also reported (`*_matched`, sigma^2 / lambda fixed to the val-median choice)
    so the cost of the advantage is visible, but the PRIMARY number for a
    baseline is its best.
  * Both regularizers are searched on validation over a wide log-spaced grid,
    and edge-pinning on sigma^2 and lambda is asserted like any other parameter.
    A pinned baseline is a crippled baseline and the run fails loudly.
  * GPCache at m=16 uses EXACTLY the keys our `cent16` uses (all 16 shots,
    l2-normalized), so `gpcache16 - cent16` isolates the Gram inverse and
    nothing else -- same draw, same features, same grids, same everything.
  * GDA gets both a ridge and a Ledoit-Wolf shrinkage estimator for the shared
    covariance, and we report whichever is better for it.
  * Budget-matched GPCache arms (m in {2,4,8}) use our k-means centroids -- the
    compression that is ours -- so the frontier comparison never charges them
    for a worse cache than the one we use ourselves.

STORAGE, stated honestly because it cuts both ways:
  GPCache at m keys stores m*D per class plus an m x m Gram (negligible), so it
  IS budget-comparable to our (m, 0) at the same m -- and NOT to (m-R, R).
  GDA-CLIP stores C means plus a SHARED D x D inverse covariance: D + D^2/C
  vectors-equivalent per class, e.g. ~22k per class on dtd/RN50. It is not a
  budget method at all. Report its accuracy fairly and its footprint honestly;
  do not use the footprint to wave away the accuracy.

Every contrast is paired with `evidence/allocation_stage1b.json` by
(backbone, dataset, seed) -- identical draws, identical cached features.
"""

import argparse
import json
import math

import torch
import torch.nn.functional as F

import clip
from utils import clip_classifier
from scripts.ds_meta import class_template
from scripts.run_cache_retrieval_null import load_cfg, grid, acc
from scripts.run_prototype_collapse_seeds import draw
from scripts.run_compact_cache import kmeans
from scripts.preflight import Preflight

DATASETS = ["stanford_cars", "fgvc", "oxford_flowers", "dtd"]
BACKBONES = ["RN50", "RN101", "ViT-B/32", "ViT-B/16"]
SEEDS = [0, 1, 2]
MS = [16, 8, 4, 2]
K = 16
N_BETA, N_ALPHA = 50, 20
N_SIG, N_LAM = 20, 20
# WIDENED 2026-08-13 after the pilot pre-flight pinned gpcache2/gpcache4 at the
# sigma^2 LOW edge and gda_lw at the alpha LOW edge. A pinned baseline is a
# crippled baseline, and E13 says fix the grid, not the verdict.
SIG_LO, SIG_HI = 1e-8, 1e2          # Gram diag is 1, so eigenvalues are O(m)
LAM_LO, LAM_HI = 1e-8, 1e2          # ridge relative to tr(Sigma)/D
# ANCHORED, SCALE-FREE alpha grid, shared by every baseline arm. A fixed numeric
# alpha grid is meaningless across arms whose cache terms differ in scale by
# orders of magnitude -- GDA's logit is O(1/sigma_within^2) while GPCache's is
# O(1/sigma^2) -- which is the alpha/cache-size confound
# (the alpha / cache-size confound) in a new place. Each arm's cache term is
# divided by the VALIDATION median of its own top-class magnitude (never the
# test median: that would leak), after which alpha is O(1) for every arm and one
# fixed log grid covers all of them without pinning. Same device as the evidence
# gate's grid anchoring (the pre-registration).
N_ALPHA_B = 30
ALPHA_LO, ALPHA_HI = 1e-3, 1e3


def logspace(lo, hi, n):
    r = math.log(hi / lo)
    return [lo * math.exp(r * i / (n - 1)) for i in range(n)]


def anchor_of(cv):
    """Validation median of the top-class cache magnitude. Scale reference."""
    return cv.max(-1).values.median().abs().clamp_min(1e-12)


def acc_batched(base, cache, alphas, y, chunk_elems=6e7):
    """top-1 for base + alpha*cache over every alpha at once -> list.

    base [n, C], cache [n, C]. Batching the alpha axis is a pure speedup; it
    changes no number, and is what makes the baselines' larger grid affordable
    rather than a reason to shrink it.

    The alpha axis is chunked to a fixed element budget for the same reason it
    is batched at all: on imagenet (n=40k, C=1000) one 30-alpha tensor is 1.2 GB
    and will not fit next to the features on an 8 GB card. argmax is taken per
    alpha, so chunking is exact -- no number changes, only the peak allocation.
    """
    A = torch.tensor(alphas, device=base.device)
    step = max(1, int(chunk_elems // max(1, base.shape[0] * base.shape[1])))
    out = []
    for i in range(0, len(A), step):
        L = base[None] + A[i:i + step][:, None, None] * cache[None]
        out += (L.argmax(-1) == y[None]).float().mean(-1).mul(100).tolist()
    return out


def gp_kernel(f, keys, beta, v=None, chunk=4096):
    """exp(-beta(1 - f@keys)) contracted with v -> [n, C]; v=None -> sum over m.

    The TEST side is evaluated only a few times per (m, seed) and never inside
    the beta loop, so materialising [n, C, m] up front (the E14 hoist, which is
    right for the val side) buys nothing there and costs 2.56 GB on imagenet at
    m=16 (40k x 1000 x 16) -- measured, it OOMs an 8 GB card. Chunking over rows
    is EXACT: output row i depends only on input row i.
    """
    out = []
    for i in range(0, f.shape[0], chunk):
        d = torch.einsum("nd,cmd->ncm", f[i:i + chunk], keys)
        A = torch.exp(-beta * (1.0 - d))
        out.append(A.sum(-1) if v is None
                   else torch.einsum("ncm,cm->nc", A, v))
    return torch.cat(out)


def build_keys(ftr, ytr, n_cls, m, seed):
    """[C, m, D] class-major keys, exactly m per class. m=16 == all 16 shots."""
    out = []
    for c in range(n_cls):
        Xc = ftr[ytr == c]
        out.append(F.normalize(Xc, dim=-1) if m >= Xc.shape[0]
                   else kmeans(Xc, m, seed))
    return torch.stack(out)


def gpcache_v(keys, beta, sig):
    """(K_c + sigma^2 I)^-1 1 per class -> v [C, m].

    K_c[i,j] = exp(-beta(1 - mu_i.mu_j)) is the SAME kernel as the query
    affinity, which is what makes this GP-Adapter's / GPCache's image branch
    rather than an arbitrary whitening.
    """
    C, m, _ = keys.shape
    G = torch.exp(-beta * (1.0 - torch.einsum("cid,cjd->cij", keys, keys)))
    A = G + sig * torch.eye(m, device=keys.device)[None]
    return torch.linalg.solve(A, torch.ones(C, m, 1, device=keys.device)).squeeze(-1)


def gda_weights(ftr, ytr, n_cls, mode, reg):
    """Shared-covariance Bayes classifier -> (W [D,C], b [C]).

    Bayes with a single shared Sigma collapses to the linear w_c = Sigma^-1 mu_c,
    b_c = -0.5 mu_c^T Sigma^-1 mu_c. This is GDA-CLIP exactly, and is also why
    it is a LINEAR head -- the point §1.2.2b makes against ourselves.
    """
    dev, D = ftr.device, ftr.shape[1]
    mus, res = [], []
    for c in range(n_cls):
        Xc = ftr[ytr == c]
        mu = F.normalize(Xc.mean(0), dim=-1)
        mus.append(mu)
        res.append(Xc - mu)
    M = torch.stack(mus)                                   # [C, D]
    R = torch.cat(res, 0)
    S = R.t() @ R / R.shape[0]
    if (R ** 2).sum().item() < 1e-10:
        # K=1: the residual from a single sample's own normalised mean is
        # EXACTLY zero, so Sigma is not merely ill-conditioned, it is the zero
        # matrix and there is nothing to estimate. The correct degeneration is
        # Sigma = I, i.e. proper nearest-class-mean -- which is what GDA-CLIP
        # reduces to at one shot. Inverting the degenerate matrix instead would
        # be a strawman (transfer prereg §5, plan §0.7's standard).
        S = torch.eye(D, device=dev)
    scale = S.diagonal().mean().clamp_min(1e-12)
    if mode == "ridge":
        S = S + reg * scale * torch.eye(D, device=dev)
    else:                                                  # Ledoit-Wolf-style
        S = (1 - reg) * S + reg * scale * torch.eye(D, device=dev)
    W = torch.linalg.solve(S, M.t())                       # [D, C]
    b = -0.5 * (M.t() * W).sum(0)                          # [C]
    return W, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--ks", nargs="+", type=int, default=[K],
                    help="shot counts to sweep (K enters the draw only here)")
    ap.add_argument("--ms", nargs="+", type=int, default=MS)
    ap.add_argument("--beta-mult", type=float, default=4.0)
    ap.add_argument("--alpha-mult", type=float, default=16.0)
    ap.add_argument("--device", default="cuda")
    # "test" is the CAPPED eval split (max(30, 3000/n_cls) per class);
    # "testfull" is the standard uncapped split. Same flag name as
    # run_calibrated_ft.py so every arm of the head-to-head table can be
    # moved with one token -- if they disagree,
    # make_le_gda_head_to_head.py's zero-shot alignment assert fires.
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--out", default="evidence/baselines_e4.json")
    args = ap.parse_args()
    dev = args.device
    pf = Preflight("baselines_e4")
    # sigma^2 = 0 IS a member of GPCache's model family -- the exact,
    # unregularized Gram inverse -- and the pilot showed the val optimum sits
    # there for m in {2,4}: k-means centroids are well separated, so the Gram is
    # near-identity and needs no ridge. Including 0 makes the low end of this
    # grid the parameter's DOMAIN boundary rather than an arbitrary cutoff, so
    # selecting it is a finding about the baseline, not a truncated search.
    # A tiny jitter keeps the solve numerically safe at m=16.
    SIGS = [1e-10] + logspace(SIG_LO, SIG_HI, N_SIG - 1)
    LAMS = logspace(LAM_LO, LAM_HI, N_LAM)
    ALPHAS = logspace(ALPHA_LO, ALPHA_HI, N_ALPHA_B)

    cells = []
    for bb in args.backbones:
        tag = bb.replace("/", "-")
        model, _ = clip.load(bb, device=dev)
        model.eval()
        for name in args.datasets:
            cfg = load_cfg(name)
            b_all, a_all = grid(cfg)
            btop, atop = b_all[-1] * args.beta_mult, a_all[-1] * args.alpha_mult
            betas = [0.1 + (btop - 0.1) * i / (N_BETA - 1) for i in range(N_BETA)]
            alphas = [0.1 + (atop - 0.1) * i / (N_ALPHA - 1)
                      for i in range(N_ALPHA)]
            cn, tp = class_template(name)
            p = f"caches/{name}/collapse/{tag}"
            f_val = torch.load(f"{p}_val_f.pt", map_location=dev).float()
            y_val = torch.load(f"{p}_val_l.pt", map_location=dev)
            f_te = torch.load(f"{p}_{args.test_split}_f.pt",
                              map_location=dev).float()
            y_te = torch.load(f"{p}_{args.test_split}_l.pt", map_location=dev)
            pool_f = torch.load(f"{p}_pool_f.pt", map_location=dev).float()
            pool_y = torch.load(f"{p}_pool_l.pt", map_location=dev)
            pf.assert_fp32(f_val=f_val, f_te=f_te, pool_f=pool_f)
            W = clip_classifier(cn, tp, model, device=dev).float()
            a_val, a_te = 100.0 * f_val @ W, 100.0 * f_te @ W
            zs = acc(a_te, y_te)
            n_cls = int(pool_y.max()) + 1

            for Ksh in args.ks:
                for sd in args.seeds:
                    idx = draw(pool_y.cpu(), Ksh, sd, dev)
                    ftr, ytr = pool_f[idx], pool_y[idx]
                    accs, sel = {}, {}
                    cid = (bb, name, sd)

                    # ---------------- GPCache, at every budget ----------------
                    for m in args.ms:
                        keys = build_keys(ftr, ytr, n_cls, m, sd)
                        # E14: hoist the f@keys product OUT of the beta loop. It is
                        # beta-free, and recomputing it 50x was ~50x the arm's whole
                        # cost. E14's own lesson is that this edit was once claimed
                        # and never applied, so it is asserted below, not assumed.
                        dv = torch.einsum("nd,cmd->ncm", f_val, keys)
                        best = (-1.0, None, None, None, None)
                        per_sig = {s: (-1.0, None, None, None) for s in SIGS}
                        for beta in betas:
                            Av = torch.exp(-beta * (1.0 - dv))
                            for sig in SIGS:
                                v = gpcache_v(keys, beta, sig)
                                cv = torch.einsum("ncm,cm->nc", Av, v)
                                anc = anchor_of(cv)
                                for al, s in zip(ALPHAS,
                                                 acc_batched(a_val, cv / anc,
                                                             ALPHAS, y_val)):
                                    if s > best[0]:
                                        best = (s, beta, al, sig, anc.item())
                                    if s > per_sig[sig][0]:
                                        per_sig[sig] = (s, beta, al, anc.item())
                            del Av
                        # primary: their best over the full (beta, alpha, sigma) grid
                        _, beta, al, sig, anc = best
                        ct = gp_kernel(f_te, keys, beta, gpcache_v(keys, beta, sig))
                        accs[f"gpcache{m}"] = acc(a_te + al * ct / anc, y_te)
                        # matched-cardinality control: sigma^2 fixed to the median
                        # of the per-sigma val optima, so the arm gets 50x20 like ours
                        smed = sorted(SIGS, key=lambda s: -per_sig[s][0])[
                            len(SIGS) // 2]
                        _, bm, am, anm = per_sig[smed]
                        cm_ = gp_kernel(f_te, keys, bm, gpcache_v(keys, bm, smed))
                        accs[f"gpcache{m}_matched"] = acc(a_te + am * cm_ / anm, y_te)
                        sel[f"gpcache{m}"] = {"val": best[0], "beta": beta,
                                              "alpha": al, "sigma": sig, "m": m,
                                              "anchor": anc}
                        pf.grid(cid, f"gpcache{m}", "beta", beta, betas)
                        pf.grid(cid, f"gpcache{m}", "alpha", al, ALPHAS)
                        pf.grid(cid, f"gpcache{m}", "sigma", sig, SIGS,
                                lo_boundary=True)
                        # the Gram inverse must actually be doing something: if
                        # sigma^2 is huge the arm collapses onto Tip-Adapter and
                        # "GPCache ties" would be an artefact, not a finding (E10)
                        Gd = torch.exp(-beta * (1.0 - torch.einsum(
                            "cid,cjd->cij", keys, keys)))
                        offdiag = (Gd - torch.diag_embed(
                            Gd.diagonal(dim1=1, dim2=2))).abs().mean().item()
                        pf.activation(cid, f"gpcache{m}", "gram",
                                      1.0 if offdiag / (offdiag + sig) > 0.01 else 0.0,
                                      1.0 if sig >= SIGS[-1] - 1e-12 else 0.0)

                        # NESTING (§0.5): as sigma^2 -> inf, (K+sigma^2 I)^-1 1 ->
                        # 1/sigma^2, so the GPCache logit -> (sum_j affinity)/sigma^2,
                        # i.e. EXACTLY Tip-Adapter's cache sum up to the scale that
                        # alpha absorbs. If this does not hold, their arm is not a
                        # superset of the baseline and every contrast is against an
                        # unknown quantity.
                        if not pf.nests or m == 16:
                            big = 1e8
                            got = gp_kernel(f_te, keys, 11.5,
                                            gpcache_v(keys, 11.5, big)) * big
                            want = gp_kernel(f_te, keys, 11.5)
                            pf.nesting(
                                f"gpcache m={m}, sigma^2->inf == Tip-Adapter sum",
                                (got - want).abs().max().item(), tol=1e-3,
                                acc_diff=acc(a_te + 3.7 * got, y_te)
                                - acc(a_te + 3.7 * want, y_te))
                        del dv, keys
                        torch.cuda.empty_cache()

                    # ---------------- GDA-CLIP ----------------
                    for mode in ("ridge", "lw"):
                        best = (-1.0, None, None, None)
                        for reg in (LAMS if mode == "ridge"
                                    else [l / (1 + l) for l in LAMS]):
                            Wg, bg = gda_weights(ftr, ytr, n_cls, mode, reg)
                            cv = f_val @ Wg + bg
                            anc = anchor_of(cv)
                            for al, s in zip(ALPHAS,
                                             acc_batched(a_val, cv / anc,
                                                         ALPHAS, y_val)):
                                if s > best[0]:
                                    best = (s, reg, al, anc.item())
                        _, reg, al, anc = best
                        Wg, bg = gda_weights(ftr, ytr, n_cls, mode, reg)
                        accs[f"gda_{mode}"] = acc(
                            a_te + al * (f_te @ Wg + bg) / anc, y_te)
                        sel[f"gda_{mode}"] = {"val": best[0], "reg": reg,
                                              "alpha": al, "anchor": anc}
                        pf.grid(cid, f"gda_{mode}", "alpha", al, ALPHAS)
                        pf.grid(cid, f"gda_{mode}", "reg", reg,
                                LAMS if mode == "ridge"
                                else [l / (1 + l) for l in LAMS],
                                lo_boundary=True)
                        del Wg, bg
                    # NESTING: at full shrinkage (Sigma = I) the Bayes rule reduces
                    # to the nearest-class-mean classifier, f.mu_c - 0.5, since the
                    # means are unit-norm. Checks gda_weights' algebra end to end.
                    if len([n for n in pf.nests if "gda" in n[0]]) == 0:
                        Wg, bg = gda_weights(ftr, ytr, n_cls, "lw", 1.0)
                        got = f_te @ Wg + bg
                        Mn = torch.stack([F.normalize(ftr[ytr == c].mean(0), dim=-1)
                                          for c in range(n_cls)])
                        sc = ftr.new_tensor(
                            [(ftr[ytr == c] - Mn[c]).pow(2).sum().item()
                             for c in range(n_cls)]).sum() / (ftr.shape[0])
                        want = (f_te @ Mn.t() - 0.5) / (sc / ftr.shape[1]).clamp_min(1e-12)
                        pf.nesting("gda Sigma=I == nearest-class-mean",
                                   (got - want).abs().max().item(), tol=1e-2,
                                   acc_diff=acc(a_te + 1.0 * got, y_te)
                                   - acc(a_te + 1.0 * want, y_te))
                        del Wg, bg

                    # report GDA at whichever estimator is better FOR IT
                    bmode = max(("ridge", "lw"), key=lambda t: accs[f"gda_{t}"])
                    accs["gda"] = accs[f"gda_{bmode}"]
                    sel["gda"] = dict(sel[f"gda_{bmode}"], mode=bmode)

                    cells.append({"backbone": bb, "dataset": name, "seed": sd,
                                  "test_split": args.test_split,
                                  "K": Ksh, "zs": zs, "acc": accs, "sel": sel})
                    print(f"{bb:<9}{name:<16}s={sd} zs={zs:5.2f} "
                          f"gp16={accs['gpcache16']:5.2f} gp8={accs['gpcache8']:5.2f} "
                          f"gp4={accs['gpcache4']:5.2f} gp2={accs['gpcache2']:5.2f} "
                          f"gda={accs['gda']:5.2f}({bmode})", flush=True)
                    json.dump({"cells": cells, "preflight": pf.to_dict()},
                              open(args.out, "w"), indent=2)
        del model
        torch.cuda.empty_cache()

    json.dump({"cells": cells, "preflight": pf.to_dict()},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}  ({len(cells)} cells)")
    pf.report()


if __name__ == "__main__":
    main()
