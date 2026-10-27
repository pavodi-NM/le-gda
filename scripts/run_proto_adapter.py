"""Proto-Adapter and Proto-Adapter-F: runner.

Method: scripts/proto_adapter.py (implemented from the paper; its docstring
lists every implementation choice). Protocol: scripts/protocol_adapter.py.

  --protocol ours     P1-P6: our seeded K-shot draws from the 32-per-class pool,
                      our validation split, the standard `testfull` test split,
                      seeds 0 1 2.
  --protocol theirs   The paper's own setting, as a check against its published
                      numbers: Tip-Adapter's fixed draw
                      (`random.seed(1); build_dataset(ds, ROOT, 16)`), the
                      dataset's own val and test splits, one draw.

Under --protocol ours the training-free variant reads the support features
from the cached pool (`caches/<ds>/collapse/<tag>_pool_f.pt`, already the
10-pass augmented mean). The -F variant fine-tunes for 20 epochs with live
augmentation through the frozen encoder, as the paper does. The five margins
share one augmentation stream; `--prove-margin-sharing` checks that this is
identical to training them separately.
"""

import argparse
import math
import os
import random
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm as _tqdm


def tqdm(*a, **k):
    """Progress bars only when stderr is a terminal."""
    k.setdefault("disable", not sys.stderr.isatty())
    return _tqdm(*a, **k)

import clip
from datasets import build_dataset
from scripts.extract_backbone_all import ROOT, ImgList
from scripts.extract_train_cache_pool import TRAIN_TFM
from scripts.preflight import Preflight
from scripts.protocol_adapter import (assert_pool_matches, report,
                                      support_items, text_head)
from scripts.run_baselines_e4 import logspace
from scripts.run_prototype_collapse_seeds import draw
from scripts import proto_adapter as PA

# U6 / P4: the paper gives no alpha grid; `p_proto` is a cosine in [-1, 1]
# against a zero-shot logit scaled by 100 (U9), so useful alphas are O(10-1000).
ALPHAS = logspace(1e-3, 10 ** (3 + 60 / 29), 40)

DATASETS = ["imagenet", "caltech101", "oxford_pets", "stanford_cars",
            "oxford_flowers", "food101", "fgvc", "sun397", "dtd", "eurosat",
            "ucf101"]
BACKBONES = ["RN50", "RN101", "ViT-B/32", "ViT-B/16"]


# ------------------------------------------------------------- features ----
def cached(ds, tag, split, dev):
    """Cached features. On ImageNet the single test file is named `test`.

    On the other datasets `test` is a capped subset and `testfull` the
    standard split; ImageNet has one test split (40 per class of the official
    validation set), stored as `test`.
    """
    if ds == "imagenet" and split == "testfull":
        split = "test"
    p = f"caches/{ds}/collapse/{tag}_{split}"
    return (torch.load(p + "_f.pt", map_location=dev).float(),
            torch.load(p + "_l.pt", map_location=dev))


def support_cache_path(ds, tag, k, epochs):
    return (f"caches/{ds}/proto_adapter/{tag}_theirs_K{k}_a{epochs}.pt")


@torch.no_grad()
def encode_support(items, model, dev, epochs, bs, workers):
    """U2/U3: `normalize(mean_a encode(aug_a(img)))`, per image.

    Same form as `utils.build_cache_model` and
    `extract_train_cache_pool.encode_pool`: the augmentation mean is taken on
    raw features, then each image's feature is L2-normalized once.
    """
    # one loader with persistent workers, iterated `epochs` times
    loader = DataLoader(ImgList(items, TRAIN_TFM), batch_size=bs, shuffle=False,
                        num_workers=workers, pin_memory=True,
                        persistent_workers=workers > 0)
    total, ys = None, None
    for e in range(epochs):
        fs, yy = [], []
        for x, y in tqdm(loader, desc=f"aug {e+1}/{epochs}", leave=False):
            fs.append(model.encode_image(x.to(dev)).float())
            if e == 0:
                yy.append(y)
        f = torch.cat(fs)
        total = f if total is None else total + f
        if e == 0:
            ys = torch.cat(yy)
    f = total / epochs
    return f / f.norm(dim=-1, keepdim=True), ys.to(dev)


# ------------------------------------------------------------- the arms ----
def run_free(sup_f, sup_y, n_cls, val, test, alphas, pf, cell,
             channel_wise=True, class_wise=True):
    """Proto-Adapter. S1-S4: no training, no beta, one matrix product."""
    (f_val, y_val, zs_val), (f_te, y_te, zs_te) = val, test
    # Eq.(6) uses F_proto exactly as constructed. No extra normalization here:
    # it would change nothing for the method but would undo the class-wise
    # setting of the Table 3 ablation (--norm-ablation).
    P = PA.build_prototypes(sup_f, sup_y, n_cls, channel_wise, class_wise)
    cos_te = f_te @ P.t()
    # check: at alpha = 0, Eq.(7) is zero-shot CLIP
    pf.nesting("alpha=0 == zero-shot CLIP", 0.0,
               acc_diff=PA.accuracy(cos_te, zs_te, y_te, 0.0)
               - PA.accuracy(torch.zeros_like(zs_te), zs_te, y_te, 0.0))
    a, _ = PA.search_alpha(f_val @ P.t(), zs_val, y_val, alphas)
    pf.grid(cell, "PROTO", "alpha", a, alphas)
    return PA.accuracy(cos_te, zs_te, y_te, a), a


def train_heads(P, items, model, dev, heads, epochs, bs, workers, seed,
                log=None, fused=None):
    """S5-S8 + U8: ONE augmentation stream, one adapter per (margin, scale).

    `heads` is a list of `(m, s)` pairs. Returns `{(m, s): ProtoAdapter}`.

    The augmented batch through the frozen encoder does not depend on m or s,
    and the heads have independent parameters and optimizers, so training them
    side by side on the same batches equals training each alone on that
    stream. The stream uses a local `torch.Generator`, so
    `--prove-margin-sharing` can replay it exactly.
    """
    heads = [(float(m), float(sc)) for m, sc in heads]
    ads = {h: PA.ProtoAdapter(P).to(dev) for h in heads}
    # eps=1e-4 rather than torch's 1e-8, as in Tip-Adapter-F's AdamW call: the
    # gradient of a cosine logit is small and fp32 Adam's second moment
    # approaches the default eps.
    opts = {h: torch.optim.Adam(a.parameters(), lr=PA.TRAIN["lr"], eps=1e-4)
            for h, a in ads.items()}
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ImgList(items, TRAIN_TFM), batch_size=bs, shuffle=True,
                        num_workers=workers, pin_memory=True, generator=g,
                        drop_last=False, persistent_workers=workers > 0)
    total = epochs * len(loader)
    step = 0
    for ep in range(epochs):
        losses = defaultdict(float)
        for x, y in tqdm(loader, desc=f"ep {ep+1}/{epochs}", leave=False):
            with torch.no_grad():
                f = model.encode_image(x.to(dev)).float()
                f = f / f.norm(dim=-1, keepdim=True)
            y = y.to(dev)
            lr = PA.cosine_lr(step, total)
            for (m, sc), ad in ads.items():
                for gp in opts[(m, sc)].param_groups:
                    gp["lr"] = lr
                # U4: cross-entropy on the adapter's margin logits alone
                if fused is None:
                    logits = PA.arcface_logits(ad(f), y, m, sc)
                else:
                    # U4 alternative (--loss-on fused): loss on the fused
                    # logits `100 * f @ W_text + alpha * phi`, as Tip-Adapter-F
                    # does. The 100 sets the temperature, so the margin is
                    # applied at s = 1. The training alpha is the training-free
                    # variant's validation-selected alpha.
                    W_t, a_t = fused
                    phi = PA.arcface_logits(ad(f), y, m, 1.0)
                    logits = PA.zeroshot_logits(f, W_t) + a_t * phi
                loss = F.cross_entropy(logits, y)
                opts[(m, sc)].zero_grad(set_to_none=True)
                loss.backward()
                opts[(m, sc)].step()
                losses[(m, sc)] += loss.item()
            step += 1
        if log is not None and (ep + 1) % 5 == 0:
            log(f"    ep {ep+1:2d} lr {lr:.2e} " + " ".join(
                f"m{m}s{sc:g}:{losses[(m, sc)]/len(loader):.3f}"
                for m, sc in heads))
    for a in ads.values():
        a.eval()
    return ads


def select_ft(ads, val, test, alphas, pf, cell, P, sup_f, sup_y):
    """S9/U8: (m, alpha) by validation top-1; test is evaluated afterwards.

    `ads` is keyed by `(m, scale)`. A normal run passes one scale (U1), so the
    selection is over margins; several scales are only passed when selecting
    the scale itself.
    """
    (f_val, y_val, zs_val), (f_te, y_te, zs_te) = val, test
    best, per_head = None, {}
    for h, ad in ads.items():
        with torch.no_grad():
            cv = ad(f_val)
        a, acc_v = PA.search_alpha(cv, zs_val, y_val, alphas)
        with torch.no_grad():
            per_head[h] = (PA.accuracy(ad(f_te), zs_te, y_te, a), a, acc_v)
        if best is None or acc_v > best[0]:
            best = (acc_v, h, a, ad)
    acc_v, (m, sc), a, ad = best
    pf.grid(cell, "PROTO-F", "alpha", a, alphas)
    # m = 0 is the lower end of the margin's domain, not a grid cutoff
    pf.grid(cell, "PROTO-F", "margin", m, sorted({h[0] for h in ads}),
            lo_boundary=True)
    # activation: did the adapter move from its initialization, and on what
    # fraction of the support set does U12's guard bind?
    with torch.no_grad():
        rot = 1.0 - F.cosine_similarity(F.normalize(ad.weight, dim=1),
                                        F.normalize(P, dim=1), dim=1)
        frac_active = (rot > 1e-6).float().mean().item()
        pos = ad(sup_f).gather(1, sup_y[:, None]).squeeze(1)
        at_bound = 0.0 if m == 0 else (
            pos <= math.cos(math.pi - m)).float().mean().item()
    pf.activation(cell, "PROTO-F", "adapter", frac_active, at_bound)
    return per_head[(m, sc)][0], m, sc, a, acc_v, per_head


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--backbones", nargs="+", default=["RN50"])
    ap.add_argument("--protocol", choices=["ours", "theirs"], default="ours")
    ap.add_argument("--arms", nargs="+", default=["free", "ft"])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--scale", type=float, nargs="+", default=None,
                    help="U1, the ArcFace scale; required for the -F variant. "
                         "The reported runs use 64, selected on validation. "
                         "Several values train one head per scale in one pass "
                         "(for selecting the scale, with --report-heads).")
    ap.add_argument("--loss-on", choices=["adapter", "fused"],
                    default="adapter",
                    help="U4. 'adapter' (default) is the paper's reading; "
                         "'fused' is Tip-Adapter-F's convention and takes its "
                         "training alpha from the training-free variant.")
    ap.add_argument("--report-heads", action="store_true",
                    help="also record every (margin, scale) head's test and "
                         "val accuracy (Table 4 reproduction, scale selection)")
    ap.add_argument("--margins", nargs="+", type=float, default=list(PA.MARGINS))
    ap.add_argument("--norm-ablation", action="store_true",
                    help="reproduce Table 3 instead of running the method")
    ap.add_argument("--prove-margin-sharing", action="store_true")
    ap.add_argument("--test-split", default="testfull")
    ap.add_argument("--augment-epoch", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=PA.TRAIN["batch_size"])
    ap.add_argument("--epochs", type=int, default=PA.TRAIN["epochs"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = args.device
    if "ft" in args.arms and args.scale is None and not args.norm_ablation:
        raise SystemExit("--scale is required for the -F variant (the paper "
                         "gives none; the reported runs use 64, see "
                         "scripts/proto_adapter.py U1)")

    pf = Preflight("proto_adapter")
    cells, t0 = [], time.time()
    for bb in args.backbones:
        tag = bb.replace("/", "-")
        model = None
        for ds in args.datasets:
            cell = f"{ds}/{tag}"
            f_val, y_val = cached(ds, tag, "val", dev)
            f_te, y_te = cached(ds, tag, args.test_split, dev)
            f_pool, y_pool = cached(ds, tag, "pool", dev)
            pf.assert_fp32(f_val=f_val, f_te=f_te, f_pool=f_pool)
            n_cls = int(y_te.max()) + 1
            if model is None:
                model, _ = clip.load(bb, device=dev)
                model.eval()
            W = text_head(ds, model, dev).float()
            val = (f_val, y_val, PA.zeroshot_logits(f_val, W))
            test = (f_te, y_te, PA.zeroshot_logits(f_te, W))
            zs = PA.accuracy(torch.zeros_like(test[2]), test[2], y_te, 0.0)
            print(f"\n=== {cell}  n_cls={n_cls} val={len(y_val)} "
                  f"test[{args.test_split}]={len(y_te)}  zs={zs:.2f} ===")

            arms = defaultdict(list)
            seeds = args.seeds if args.protocol == "ours" else [1]
            for seed in seeds:
                if args.protocol == "ours":
                    assert_pool_matches(ds, tag)
                    idx = draw(y_pool.cpu(), args.k, seed, dev)
                    sup_f, sup_y = f_pool[idx], y_pool[idx]
                    items, lab, _ = support_items(ds, args.k, seed)
                    assert torch.equal(torch.tensor(lab, device=dev), sup_y), \
                        f"{cell}: support item labels != cached pool rows"
                else:
                    # the paper's setting: Tip-Adapter's fixed draw
                    random.seed(1)
                    torch.manual_seed(1)
                    items = build_dataset(ds, ROOT, args.k).train_x
                    # the draw is fixed by seed 1, so its augmented-mean
                    # encoding is cached and reused (file name includes the
                    # protocol)
                    cp = support_cache_path(ds, tag, args.k, args.augment_epoch)
                    if os.path.exists(cp):
                        d = torch.load(cp, map_location=dev)
                        assert d["impaths"] == [x.impath for x in items], (
                            f"{cp}: cached support set is a DIFFERENT draw")
                        sup_f, sup_y = d["f"].float(), d["y"]
                    else:
                        sup_f, sup_y = encode_support(
                            items, model, dev, args.augment_epoch,
                            args.batch_size, args.workers)
                        os.makedirs(os.path.dirname(cp), exist_ok=True)
                        torch.save({"f": sup_f.cpu(), "y": sup_y.cpu(),
                                    "impaths": [x.impath for x in items]}, cp)

                if args.norm_ablation:
                    for cw in (False, True):
                        for lw in (False, True):
                            a, _ = run_free(sup_f, sup_y, n_cls, val, test,
                                            ALPHAS, pf, cell, cw, lw)
                            arms[f"PROTO[ch={int(cw)},cls={int(lw)}]"].append(a)
                    continue

                if "free" in args.arms:
                    a, alpha = run_free(sup_f, sup_y, n_cls, val, test,
                                        ALPHAS, pf, cell)
                    arms["PROTO"].append(a)
                    print(f"  seed {seed}: PROTO   {a:.2f} (alpha={alpha:.4g})")
                if "ft" in args.arms:
                    P = PA.build_prototypes(sup_f, sup_y, n_cls)
                    # check: an untrained ProtoAdapter equals the training-free
                    # variant (forward()'s class-wise normalization is a no-op
                    # on P)
                    with torch.no_grad():
                        d0 = (PA.ProtoAdapter(P).to(dev)(f_val)
                              - f_val @ F.normalize(P, dim=1).t()).abs().max()
                    pf.nesting("PROTO-F at 0 steps == PROTO", d0.item(), tol=1e-6)
                    heads = [(m, sc) for sc in args.scale
                             for m in args.margins]
                    fused = None
                    if args.loss_on == "fused":
                        if not (args.arms and "free" in args.arms):
                            raise SystemExit("--loss-on fused needs the free "
                                             "arm in the same run: U4b takes "
                                             "its training alpha from it")
                        fused = (W, alpha)
                    ads = train_heads(P, items, model, dev, heads,
                                      args.epochs, args.batch_size,
                                      args.workers, seed, log=print,
                                      fused=fused)
                    a, m, sc, alpha, av, per = select_ft(
                        ads, val, test, ALPHAS, pf, cell, P, sup_f, sup_y)
                    arms["PROTO-F"].append(a)
                    print(f"  seed {seed}: PROTO-F {a:.2f} "
                          f"(m={m} s={sc:g} alpha={alpha:.4g} val={av:.2f})")
                    if args.report_heads:
                        # per-head results (not the method's number), under
                        # separate keys
                        for (hm, hs), (ta, ha, hv) in sorted(per.items()):
                            arms[f"PROTO-F[m={hm:g},s={hs:g}]"].append(ta)
                            # validation accuracy, used to select the scale
                            arms[f"VAL[m={hm:g},s={hs:g}]"].append(hv)
                            print(f"      head m={hm:g} s={hs:g}: test {ta:.2f} "
                                  f"(alpha={ha:.4g} val={hv:.2f})")
                    if args.prove_margin_sharing:
                        prove_sharing(P, items, model, dev, heads,
                                      args.epochs, args.batch_size,
                                      args.workers, seed, ads)

            for k, v in arms.items():
                print(f"  {cell} {k:28s} {[round(x,2) for x in v]} "
                      f"mean {sum(v)/len(v):.2f}")
            cells.append({"backbone": bb, "dataset": ds, "K": args.k,
                          "test_split": args.test_split,
                          "protocol": args.protocol, "zs": zs,
                          "scale": args.scale, "epochs": args.epochs,
                          "loss_on": args.loss_on,
                          "impl": "ours-from-paper", "arms": dict(arms)})
    ok = pf.report()
    if args.out:
        report(cells, args.out,
               meta={"protocol": args.protocol, "seeds": args.seeds,
                     "scale": args.scale, "margins": args.margins,
                     "loss_on": args.loss_on,
                     "alphas": [ALPHAS[0], ALPHAS[-1], len(ALPHAS)],
                     "preflight_ok": ok,
                     "wall_s": round(time.time() - t0, 1)})
        print(f"\nwrote {args.out}  ({time.time()-t0:.0f}s)  preflight_ok={ok}")


def prove_sharing(P, items, model, dev, heads, epochs, bs, workers, seed,
                  shared):
    """Check that training each head separately equals the shared stream."""
    print("  [prove] re-running each head separately...")
    for h in heads:
        solo = train_heads(P, items, model, dev, [h], epochs, bs, workers,
                           seed)[(float(h[0]), float(h[1]))]
        d = (solo.weight - shared[(float(h[0]), float(h[1]))]
             .weight).abs().max().item()
        print(f"  [prove] m={h[0]} s={h[1]}: max|dW| = {d:.3e}  "
              f"{'IDENTICAL' if d == 0.0 else 'DIFFERS'}")
        assert d == 0.0, f"head sharing is NOT exact at {h} (max|dW|={d})"


if __name__ == "__main__":
    main()
