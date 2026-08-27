"""Proto-Adapter and Proto-Adapter-F: calibration, then our protocol.

Method:   `scripts/proto_adapter.py`            (implemented from the paper)
Reading:  the design note in `scripts/proto_adapter.py` (S1-S13 stated,
          U1-U12 chosen)
Protocol: `scripts/protocol_adapter.py`          (P1-P6)
Protocol: P1-P6 in `scripts/protocol_adapter.py`

TWO PROTOCOLS, and the distinction is the whole point of sec.1.11.1.

  --protocol theirs   THE BAR. Upstream's own hardcoded draw
                      (`random.seed(1); build_dataset(ds, ROOT, 16)`), upstream's
                      `dataset.val` and `dataset.test`, one draw, RN50. There is
                      no "Run A = their code" for this method, so reproducing
                      their PUBLISHED numbers is the only external check that
                      exists. If this does not land near Table 2, we stop and
                      say so rather than shipping a number.

  --protocol ours     P1-P6: our seeded K=16 draws from the 32/class pool, our
                      val, the uncapped `testfull` split, seeds [0,1,2], no
                      test-set selection anywhere.

WHERE THE COMPUTE GOES, and why this is not a 5x bill. Everything except the
support set is cached: `caches/<ds>/collapse/<tag>_{val,testfull}_f.pt` are the
same images under the same `preprocess` for BOTH protocols, because
`eval_items(ds,'testfull')` is `sorted(ds.test)` -- the same SET, and top-1 does
not care about order. So the only live encoding is the support images:

  training-free   10 augmentation passes over K*n_cls images (sec.U3), and under
                  --protocol ours not even that -- the cached `_pool_f.pt` rows
                  ARE those features, which is the whole reason that pool exists.
  -F              20 epochs x K*n_cls images with live augmentation through the
                  frozen encoder. This is the expensive arm and it is expensive
                  on purpose: sec.1.4 measured that doing it on frozen features
                  instead costs -4.60 on one cell and sec.1.8 measured +2.30
                  grid-wide, which is exactly the defect this run exists to not
                  repeat.

U8's five margins share ONE augmentation stream. Five adapters are stepped on
the identical batches, which is step-for-step identical to five independent runs
on that stream because the heads do not interact. `--prove-margin-sharing`
re-runs a cell the slow way and asserts bit-equality.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm as _tqdm


def tqdm(*a, **k):
    """Silent unless a human is watching. A 4000-line progress bar in a
    nohup log is not a record of anything and buries the one line per seed
    that is."""
    k.setdefault("disable", not sys.stderr.isatty())
    return _tqdm(*a, **k)

import clip
from datasets import build_dataset
from scripts.extract_backbone_all import ROOT, ImgList
from scripts.extract_train_cache_pool import TRAIN_TFM
from scripts.preflight import Preflight
from scripts.protocol_adapter import (assert_pool_matches, eval_items, report,
                                      support_items, text_head)
from scripts.run_baselines_e4 import logspace
from scripts.run_prototype_collapse_seeds import draw
from scripts import proto_adapter as PA

# U6 / P4. Ours, widened, because the method has no grid of its own and
# Tip-Adapter's (ceiling 13, or 3 on imagenet) cannot reach the alpha this
# method needs -- `p_proto` is a cosine in [-1,1] against a zero-shot logit
# already multiplied by 100 (U9), so the useful alpha is O(10-1000).
ALPHAS = logspace(1e-3, 10 ** (3 + 60 / 29), 40)

DATASETS = ["imagenet", "caltech101", "oxford_pets", "stanford_cars",
            "oxford_flowers", "food101", "fgvc", "sun397", "dtd", "eurosat",
            "ucf101"]
BACKBONES = ["RN50", "RN101", "ViT-B/32", "ViT-B/16"]


def install_imagenet():
    """Register the sec.1.9 ImageNet loader without editing datasets/__init__.py.

    `patches/imagenet_subset_for_tip_adapter.py` is a DATA-ACCESS adapter and
    nothing else (its own docstring says so, and sec.1.9 verified its label
    order against all four cached backbones). It lives in patches/ rather than
    datasets/ so that the upstream tree stays pristine; registering it at
    runtime keeps it that way and keeps the change reversible.
    """
    import datasets as D
    if "imagenet" in D.dataset_list:
        return
    import importlib.util
    import sys
    src = "patches/imagenet_subset_for_tip_adapter.py"
    spec = importlib.util.spec_from_file_location("_in_subset", src)
    mod = importlib.util.module_from_spec(spec)
    # the file does `from .imagenet_classnames import imagenet_classes`; this
    # repo keeps that list in datasets/imagenet.py instead. Same 1000 names in
    # the same order -- assert_pool_matches would fail loudly otherwise.
    sys.modules["_in_subset"] = mod
    mod.__package__ = "datasets"
    import datasets.imagenet as _in
    sys.modules["datasets.imagenet_classnames"] = _in
    spec.loader.exec_module(mod)
    D.dataset_list["imagenet"] = mod.ImageNetSubset


# ------------------------------------------------------------- features ----
def cached(ds, tag, split, dev):
    """Cached features. `testfull` is spelled `test` on imagenet, and must be.

    Everywhere else `test` is the EVAL_TARGET-capped split and `testfull` the
    standard uncapped one, so the two names address different files and mixing
    them is P7's exact failure. On imagenet there is only one file because
    there is only one split: 40/class of the official val set is not a cap we
    chose, it is all the test data on disk (sec.1.5, and the standing
    capped-test-splits record). sec.1.9 already runs it under the name
    `testfull` for the same reason -- 40000 items either way.
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

    Byte-identical in form to `utils.build_cache_model` and to
    `extract_train_cache_pool.encode_pool`: the augmentation mean is taken on
    RAW features and the L2 normalization is applied once, per image, at the
    end. That per-image unit vector is what the paper calls `CLIP_vis(I)`
    (sec.2.1) and what Eq.(5) then averages over the K images of a class.
    """
    # ONE loader, iterated `epochs` times with persistent workers. Building it
    # inside the loop respawns `workers` processes on every augmentation pass,
    # and that spawn cost does not scale with the dataset -- it was ~12 minutes
    # per cell whether the cell had 160 images or 16 000, which is most of what
    # the first calibration attempt spent its time on.
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
    # Eq.(6) is `p_proto = F_proto^T f` with F_proto EXACTLY as constructed --
    # no further normalization. Re-normalizing here would be a no-op for the
    # method (S2 leaves the rows on the unit sphere anyway) but it silently
    # DELETES the class-wise arm of the Table 3 ablation, which is the one the
    # paper says carries most of the gain. Measured on dtd/RN50: with the extra
    # normalize, cls=0 and cls=1 both returned 65.19, i.e. the ablation
    # reported no effect because the code had undone the thing being ablated.
    P = PA.build_prototypes(sup_f, sup_y, n_cls, channel_wise, class_wise)
    cos_te = f_te @ P.t()
    # NESTING (plan sec.4.6 step 4). alpha -> 0 removes the adapter from Eq.(7)
    # entirely, so the arm MUST collapse onto zero-shot CLIP. If it does not,
    # the fusion is wired wrong and every alpha above it is measuring something
    # other than what the equation says.
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

    Why the cross product and not a loop over runs: the expensive object is the
    augmented batch pushed through the frozen encoder, and it depends on
    NEITHER m nor s. The heads are independent `nn.Linear`s with independent
    optimizers and they never read each other, so stepping them side by side on
    the same batches is step-for-step identical to training each alone on that
    stream. That is what makes U1's scale calibration one ImageNet pass instead
    of five, and U8's margin sweep free.

    The stream is driven by a local `torch.Generator`, not the global RNG, so
    nothing else in the process can perturb it and `--prove-margin-sharing` can
    replay it exactly.
    """
    heads = [(float(m), float(sc)) for m, sc in heads]
    ads = {h: PA.ProtoAdapter(P).to(dev) for h in heads}
    # eps=1e-4 rather than torch's 1e-8: the gradient of a cosine logit is
    # O(1/D) and fp32 Adam's second moment underflows toward the default eps,
    # which silently rescales the step. Tip-Adapter-F's own AdamW call uses the
    # same value for the same reason.
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
                # U4: cross-entropy on the ADAPTER's margin-modified logits
                # alone. Neither alpha nor the text head appears in sec.2.2.2.
                if fused is None:
                    logits = PA.arcface_logits(ad(f), y, m, sc)
                else:
                    # U4, THE ALTERNATIVE READING. sec.2.2.2 says "apply the
                    # cross-entropy loss to the obtained logits" and never
                    # mentions alpha or the text head, which is why the default
                    # trains the adapter alone. Tip-Adapter-F instead trains on
                    # the FUSED logits, so its cache learns to CORRECT the text
                    # head rather than to classify beside it.
                    #
                    # NOTE THE SCALE DISAPPEARS. The fused form mirrors Eq.(7)
                    # exactly -- `100 * f @ W_text + alpha * phi` -- and the
                    # 100 already sets the softmax temperature, so ArcFace's
                    # `s` has no role at all here and the margin is applied at
                    # s = 1. That is itself an argument FOR this reading: it
                    # explains why the paper never states a scale, because
                    # under Tip-Adapter's convention there is no scale to
                    # state.
                    #
                    # U4b, forced by U4: the fused reading needs a TRAINING
                    # alpha, and the paper gives none (alpha is searched
                    # afterwards). We take the training-free arm's val-selected
                    # alpha -- the only value available without touching test.
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
    """S9/U8: (m, alpha) by VALIDATION top-1. P3 -- test is touched once, after.

    `ads` is keyed by `(m, scale)`. The scale is U1 and is NOT a per-cell knob:
    a run passes ONE scale, so this argmax is over margins only. The keys carry
    it anyway so the scale-calibration mode can reuse the same selector.
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
    # m = 0 is the DOMAIN boundary of the additive angular margin, not an
    # arbitrary cutoff: a negative margin is not a member of the method's
    # family, so there is nowhere to widen the grid to. Selecting it is
    # informative -- it says the margin bought nothing on this cell -- and
    # preflight is told so rather than failing the run for it (E13's carve-out).
    pf.grid(cell, "PROTO-F", "margin", m, sorted({h[0] for h in ads}),
            lo_boundary=True)
    # E10/E11. Two ways this arm can silently do nothing and still produce a
    # plausible number: the adapter never leaves its initialization (then
    # PROTO-F is PROTO wearing a hat), or U12's guard binds on so much of the
    # support set that the margin is no longer a margin.
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
                    help="U1, the ArcFace scale. Required for the -F arm; "
                         "calibrated ONCE against Table 4 and then frozen. "
                         "Several values sweep it in one pass -- calibration "
                         "only, never in a reported run.")
    ap.add_argument("--loss-on", choices=["adapter", "fused"],
                    default="adapter",
                    help="U4. 'adapter' is the paper's literal reading and the "
                         "default; 'fused' is Tip-Adapter-F's convention and "
                         "takes its training alpha from the training-free arm "
                         "(U4b). Never change this silently -- it is a "
                         "different reading of sec.2.2.2, not a tuning knob.")
    ap.add_argument("--report-heads", action="store_true",
                    help="also record every (margin, scale) head's test "
                         "accuracy, for the Table 3/4 reproductions")
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
    install_imagenet()
    dev = args.device
    if "ft" in args.arms and args.scale is None and not args.norm_ablation:
        raise SystemExit("--scale is U1 and has no paper default; pass it "
                         "explicitly (see the design note in scripts/proto_adapter.py)")

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
                    # THE BAR: upstream's own hardcoded draw, verbatim.
                    random.seed(1)
                    torch.manual_seed(1)
                    items = build_dataset(ds, ROOT, args.k).train_x
                    # Their draw is a fixed function of seed 1, so the
                    # augmented-mean encode of it is reusable across the
                    # Table 2 / 3 / 4 reproductions. Cached under a name that
                    # spells out the protocol -- an 'ours' draw must NEVER land
                    # in the same file, which is why the protocol is in it.
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
                    # NESTING. An UNTRAINED ProtoAdapter is exactly the
                    # training-free arm: forward() re-normalizes class-wise,
                    # which is idempotent on an already class-normalized P. So
                    # PROTO-F at zero steps must equal PROTO to the bit. This is
                    # what makes "fine-tuning gained X" a statement about
                    # training rather than about two different constructions.
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
                        # NOT the method's number. Table 4 reports a test
                        # accuracy PER MARGIN, so reproducing it needs the
                        # whole grid exposed. Kept in separate arm keys so it
                        # can never be mistaken for the val-selected result.
                        for (hm, hs), (ta, ha, hv) in sorted(per.items()):
                            arms[f"PROTO-F[m={hm:g},s={hs:g}]"].append(ta)
                            # U1 is chosen on VAL, so the val accuracy of every
                            # head has to survive into the evidence file --
                            # otherwise picking the scale later would have
                            # nothing to pick on but test.
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
               meta={"protocol": "P1-P6, scripts/protocol_adapter.py",
                     "design_note": "docstring of scripts/proto_adapter.py",
                     "protocol": args.protocol, "seeds": args.seeds,
                     "scale": args.scale, "margins": args.margins,
                     "loss_on": args.loss_on,
                     "alphas": [ALPHAS[0], ALPHAS[-1], len(ALPHAS)],
                     "preflight_ok": ok,
                     "wall_s": round(time.time() - t0, 1)})
        print(f"\nwrote {args.out}  ({time.time()-t0:.0f}s)  preflight_ok={ok}")


def prove_sharing(P, items, model, dev, heads, epochs, bs, workers, seed,
                  shared):
    """U8's cost note, proven: N separate trainings == the shared stream.

    Same standard as sec.1.9(b)'s search_hp hoist -- exact, not argued.
    """
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
