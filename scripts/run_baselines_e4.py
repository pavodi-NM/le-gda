"""GPCache baseline, and the shared GDA building blocks used by LE-GDA.

    GPCache   Ding et al., IJCV, arXiv:2410.08895 (2024)
              logits_c = kappa_c(f, F) (K_c + sigma^2 I)^-1 Y
              Tip-Adapter's affinity with the Gram inverse inserted.
    GDA-CLIP  Wang et al., "A Hard-to-Beat Baseline for Training-free CLIP-based
              Adaptation", ICLR 2024: class means and one shared covariance,
              which gives the linear classifier w_c = Sigma^-1 mu_c.

The functions gda_weights, anchor_of, acc_batched and logspace are imported by
the LE-GDA scripts. The GDA-CLIP numbers in the paper come from
scripts/run_layer_ensemble.py (`gda_last`); the gda_* arms recorded here are a
cross-check.

Baseline hyperparameters are searched on the validation split over grids at
least as large as LE-GDA's: GPCache over 50 beta x 20 sigma^2 x 30 alpha. A
`*_matched` variant with sigma^2 fixed to the median of the per-sigma optima is
also recorded. GPCache at m=16 uses all 16 shots as keys; m < K uses k-means
centroids per class.
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
SIG_LO, SIG_HI = 1e-8, 1e2          # Gram diag is 1, so eigenvalues are O(m)
LAM_LO, LAM_HI = 1e-8, 1e2          # ridge relative to tr(Sigma)/D
# Anchored, scale-free alpha grid shared by every arm. The added score terms
# differ in scale by orders of magnitude between methods (GDA's logit is
# O(1/sigma_within^2), GPCache's O(1/sigma^2)), so each arm's term is divided by
# the validation median of its own top-class score (anchor_of) before the alpha
# search; one fixed log grid then fits every arm.
N_ALPHA_B = 30
ALPHA_LO, ALPHA_HI = 1e-3, 1e3


def logspace(lo, hi, n):
    r = math.log(hi / lo)
    return [lo * math.exp(r * i / (n - 1)) for i in range(n)]


def anchor_of(cv):
    """Validation median of the top-class score magnitude; the scale reference."""
    return cv.max(-1).values.median().abs().clamp_min(1e-12)


def acc_batched(base, cache, alphas, y, chunk_elems=6e7):
    """top-1 for base + alpha*cache over every alpha at once -> list.

    base [n, C], cache [n, C]. The alpha axis is batched for speed and chunked
    to a fixed element budget so that ImageNet (n=40k, C=1000) fits on an 8 GB
    GPU; argmax is taken per alpha, so the result is exact.
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

    Used on the test side, chunked over rows so that ImageNet at m=16
    (40k x 1000 x 16) fits on an 8 GB GPU. Output row i depends only on input
    row i, so chunking is exact.
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

    K_c[i,j] = exp(-beta(1 - mu_i.mu_j)) is the same kernel as the query
    affinity, as in GPCache's image branch.
    """
    C, m, _ = keys.shape
    G = torch.exp(-beta * (1.0 - torch.einsum("cid,cjd->cij", keys, keys)))
    A = G + sig * torch.eye(m, device=keys.device)[None]
    return torch.linalg.solve(A, torch.ones(C, m, 1, device=keys.device)).squeeze(-1)


def gda_weights(ftr, ytr, n_cls, mode, reg):
    """Shared-covariance Bayes classifier -> (W [D,C], b [C]).

    Bayes with a single shared Sigma gives the linear classifier
    w_c = Sigma^-1 mu_c, b_c = -0.5 mu_c^T Sigma^-1 mu_c (GDA-CLIP). Class means
    are L2-normalised. mode="ridge" adds reg * tr(Sigma)/D * I; otherwise reg is
    a Ledoit-Wolf-style shrinkage weight in [0, 1].
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
        # K=1 on unit-norm features: every residual is zero, so Sigma is the
        # zero matrix. Fall back to Sigma = I, i.e. nearest class mean.
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
    # "testfull" is the standard test split; on ImageNet pass "test" (see
    # run_layer_ensemble.py).
    ap.add_argument("--test-split", default="testfull")
    ap.add_argument("--out", default="evidence/baselines_e4.json")
    args = ap.parse_args()
    dev = args.device
    pf = Preflight("baselines_e4")
    # sigma^2 = 0 (the exact Gram inverse) is part of GPCache's model family and
    # is included, with a tiny jitter to keep the solve stable at m=16.
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
                        # f @ keys does not depend on beta: compute it once.
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
                        # best over the full (beta, alpha, sigma^2) grid
                        _, beta, al, sig, anc = best
                        ct = gp_kernel(f_te, keys, beta, gpcache_v(keys, beta, sig))
                        accs[f"gpcache{m}"] = acc(a_te + al * ct / anc, y_te)
                        # sigma^2 fixed to the median of the per-sigma val optima
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
                        # check the Gram inverse is active: with a very large
                        # sigma^2 the arm collapses onto Tip-Adapter
                        Gd = torch.exp(-beta * (1.0 - torch.einsum(
                            "cid,cjd->cij", keys, keys)))
                        offdiag = (Gd - torch.diag_embed(
                            Gd.diagonal(dim1=1, dim2=2))).abs().mean().item()
                        pf.activation(cid, f"gpcache{m}", "gram",
                                      1.0 if offdiag / (offdiag + sig) > 0.01 else 0.0,
                                      1.0 if sig >= SIGS[-1] - 1e-12 else 0.0)

                        # check: as sigma^2 -> inf, (K + sigma^2 I)^-1 1 -> 1/sigma^2,
                        # so the GPCache logit becomes Tip-Adapter's cache sum up to
                        # a scale that alpha absorbs.
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
                    # check: at full shrinkage (Sigma = I) the rule reduces to the
                    # nearest-class-mean classifier f.mu_c - 0.5 (unit-norm means).
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

                    # report GDA at whichever estimator scores higher
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
