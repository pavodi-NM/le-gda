"""Intermediate-block CLS features from CLIP ViTs, for layer-ensemble GDA.

Motivated by everything this session measured, in order:

  * plan §0.7 / KC-P3: a discriminative linear probe and a generative
    Gaussian discriminant TIE (n.s.) and both beat every cache arm by ~5 points.
    Two unrelated estimators hitting the same number is a FUNCTION-CLASS
    ceiling, not an estimation shortfall -- so improving the head is done.
  * The only intervention that moved that ceiling all session was changing the
    FEATURES: pre-projection features lifted GDA by +0.221 (18/24, p=0.023) and
    by +1.55 on fgvc, the most fine-grained dataset.
  * SimCLR (Chen et al. 2020) is the mechanism: the representation before a
    contrastively-trained head is much better downstream, because the objective
    makes the last layers invariant to detail that is useful for classification.
    CLIP's final projection is a single LINEAR map, which is why our
    pre-projection gain was +0.2 and not SimCLR's +10. Intermediate transformer
    blocks strip far more of that specialisation, and are untested here.

So this extracts the CLS token after selected residual blocks. CLIP's
VisionTransformer runs the transformer in LND layout:

    x = x.permute(1, 0, 2)      # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)      # LND -> NLD
    x = self.ln_post(x[:, 0, :])

so hooking `visual.transformer.resblocks[i]` gives [L, N, D] and the CLS token
is `out[0]`. `ln_post` is applied only at the end; each layer is L2-normalised
here instead, which is what every retrieval and covariance estimator in this
project assumes.

The final block's CLS, L2-normalised, is NOT identical to the cached
pre-projection feature (which has `ln_post` applied). Both are kept: `_pre_f`
from extract_preproj.py is the ln_post'd penultimate, and `_L{i}_f` here is the
raw block output. The last-layer file therefore doubles as a consistency check
rather than a duplicate.

`--with-pre` writes that `_pre_f` file from this same forward pass instead of
paying for a second one in extract_preproj.py. `pre` is the most-selected space
at K=16, so the phase-A sweep needs it on all 11 datasets, and the two scripts
hook different modules of the same forward. The nesting check is the one
extract_preproj.py uses and is not weakened: the projected output is
recomputed here and must reproduce the cached `_f.pt` bit-for-bit on the
deterministic splits.

imagenet has no CoOp-style builder entry, so `build_dataset` (and therefore
`splits_for`) raises KeyError on it. Its splits come from
extract_imagenet_subset.build_items instead -- the SAME function that built the
cached projected features, which is what makes the label assert below meaningful
rather than a comparison of two unrelated item orders.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import clip
from scripts.extract_backbone_all import (
    ImgList, splits_for, TRAIN_TFM, add_test_split_args, resolve_test_tag)

BACKBONES = ["ViT-B/16"]
DATASETS = ["stanford_cars", "fgvc", "oxford_flowers", "dtd"]
LAYERS = [4, 6, 8, 10, 11]      # 0-indexed blocks of the 12-block ViT-B


def splits_of(name, full_test=False):
    """{split: items}, routing imagenet to the builder its cache was made with.

    IMAGENET CANNOT HONOUR `full_test`, and the reason is data, not code.
    extract_imagenet_subset.build_items reads data/tip-adapter/imagenet_subset,
    which holds the official VAL set at 50/class and 32/class of train --
    nothing else. Our "test" is not a capped standard test set: it is 40/class
    carved out of that val set, with the other 10/class held out as our
    validation split. Reaching the standard 50k evaluation would mean spending
    the whole val set on test, leaving LE-GDA with nothing to search lambda,
    S and alpha on -- the method has no training loop, so validation is not a
    convenience, it is the entire fitting procedure. So imagenet stays at 40k
    and that is a stated scope limit, not an oversight.
    """
    if name == "imagenet":
        assert not full_test, (
            "imagenet has no full-size test split available: only the official "
            "val set (50/class) is on disk, and 10/class of it is our "
            "validation split. See the docstring.")
        from scripts.extract_imagenet_subset import build_items
        return build_items()
    return splits_for(name, full_test=full_test)[0]


@torch.no_grad()
def encode_layers(items, model, tfm, dev, bs, workers, layers, epochs=1,
                  with_pre=False):
    """{key: [N, D] L2-normalised feature}, averaged over augmentations.

    Keys are the block indices in `layers`, plus "pre" (the `ln_post` output)
    and "post" (the projected embedding) when `with_pre` is set. "post" is not
    a feature anyone wants -- it is already cached -- but recomputing it is what
    lets the caller check bit-for-bit that these hooks saw the items the cached
    labels belong to.

    Batches are accumulated on CPU: imagenet's 40k test split at 7 keys does not
    fit alongside the model on an 8 GB card, and the values are identical.
    """
    grabbed = {}

    def mk(i):
        def hook(_m, _i, o):
            # o is [L, N, D] in CLIP's LND transformer layout; CLS is index 0
            grabbed[i] = o[0].float()
        return hook

    hs = [model.visual.transformer.resblocks[i].register_forward_hook(mk(i))
          for i in layers]
    if with_pre:
        hs.append(model.visual.ln_post.register_forward_hook(
            lambda _m, _i, o: grabbed.__setitem__("pre", o.float())))
    keys = list(layers) + (["pre", "post"] if with_pre else [])
    loader = DataLoader(ImgList(items, tfm), batch_size=bs, shuffle=False,
                        num_workers=workers, pin_memory=True)
    tot = {k: None for k in keys}
    labels = None
    for e in range(epochs):
        acc = {k: [] for k in keys}
        ys = []
        for x, y in tqdm(loader, leave=False):
            post = model.encode_image(x.to(dev)).float()
            if with_pre:
                grabbed["post"] = post
            for k in keys:
                acc[k].append(grabbed[k].cpu())
            if e == 0:
                ys.append(y)
        for k in keys:
            f = torch.cat(acc[k])
            tot[k] = f if tot[k] is None else tot[k] + f
        if e == 0:
            labels = torch.cat(ys)
    for h in hs:
        h.remove()
    out = {}
    for k in keys:
        f = tot[k] / epochs
        out[k] = f / f.norm(dim=-1, keepdim=True)
    return out, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    ap.add_argument("--with-pre", action="store_true",
                    help="also write the ln_post pre-projection feature, "
                         "instead of a second full pass in extract_preproj.py")
    ap.add_argument("--augment-epoch", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    add_test_split_args(ap)
    args = ap.parse_args()
    ttag = resolve_test_tag(args)
    fname = {"test": ttag}          # split key -> filename token

    for bb in args.backbones:
        assert bb.startswith("ViT"), f"{bb}: ResNet blocks are not CLS-token"
        model, preprocess = clip.load(bb, device=args.device)
        model.eval()
        tag = bb.replace("/", "-")
        nblk = len(model.visual.transformer.resblocks)
        print(f"{bb}: {nblk} blocks, extracting {args.layers}")
        for name in args.datasets:
            out = os.path.join("caches", name, "collapse")

            def done(osplit):
                ok = all(os.path.exists(f"{out}/{tag}_{osplit}_L{i}_f.pt")
                         for i in args.layers)
                if args.with_pre:
                    ok &= os.path.exists(f"{out}/{tag}_{osplit}_pre_f.pt")
                return ok

            if all(done(fname.get(s, s)) for s in ("val", "test", "pool")):
                print(f"{bb:<10}{name:<16}cached", flush=True)
                continue
            src = splits_of(name, full_test=args.full_test)
            for split, items in src.items():
                osplit = fname.get(split, split)
                if done(osplit):
                    continue
                # the projected split is what KC-B0a/B0b check against, so a
                # missing reference must STOP rather than silently skip both
                # gates -- which is exactly what a new --test-tag namespace
                # would do on its first run
                if split != "pool":
                    assert os.path.exists(f"{out}/{tag}_{osplit}_f.pt"), (
                        f"{name}/{osplit}: no projected reference to check "
                        f"alignment against. Run extract_backbone_all.py with "
                        f"the same --full-test/--test-tag first.")
                tfm = TRAIN_TFM if split == "pool" else preprocess
                ep = args.augment_epoch if split == "pool" else 1
                feats, lab = encode_layers(items, model, tfm, args.device,
                                           args.batch_size, args.workers,
                                           args.layers, ep, args.with_pre)
                # label alignment against the already-cached projected split --
                # the same nesting discipline extract_preproj.py uses
                ref = f"{out}/{tag}_{osplit}_l.pt"
                if os.path.exists(ref):
                    dl = (torch.load(ref, map_location="cpu") - lab).abs().max()
                    assert dl.item() == 0, (
                        f"{name}/{osplit}: labels do not match the cached split "
                        f"-- item alignment is broken, stop")
                post = feats.pop("post", None)
                if post is not None:
                    # the deterministic splits must reproduce the cached
                    # projected features exactly, or the hooks above read items
                    # that the cached labels do not describe. `pool` averages a
                    # RANDOM train transform and is not expected to reproduce.
                    if split != "pool" and os.path.exists(f"{out}/{tag}_{osplit}_f.pt"):
                        old = torch.load(f"{out}/{tag}_{osplit}_f.pt",
                                         map_location="cpu").float()
                        # PER-ITEM COSINE, not max|d|. The check exists to catch
                        # item MISALIGNMENT, and a wrong image scores ~0 here
                        # while fp16 kernel choice scores ~1 -- max|d| cannot
                        # tell those apart. Measured on sun397/test: labels
                        # identical, min cosine 0.999995, and the only 6 items
                        # of 11910 deviating above 1e-4 are indices 11904-11909,
                        # i.e. exactly the final partial batch (11910 = 93*128
                        # + 6). A 6-image batch picks a different GEMM kernel
                        # than a 128-image one, so max|d| hit 1.3e-3 on images
                        # that are provably the right images. Cosine is the
                        # STRICTER test against the failure that matters.
                        cos = torch.nn.functional.cosine_similarity(
                            old, post, dim=-1)
                        c, d = cos.min().item(), (old - post).abs().max().item()
                        print(f"  nesting {name}/{osplit}: min cos={c:.6f} "
                              f"max|d_feat|={d:.2e} "
                              f"{'OK' if c > 0.999 else 'MISALIGNED'}")
                        assert c > 0.999, (
                            f"{name}/{osplit}: re-extraction does not reproduce "
                            f"the cached projected features (min cosine {c:.4f})"
                            f" -- item alignment is broken, stop")
                    pre = feats.pop("pre")
                    # never overwrite: `pool` averages a random transform, so a
                    # rewrite would silently change features that published
                    # numbers were fitted on (§3 of the write-up).
                    if not os.path.exists(f"{out}/{tag}_{osplit}_pre_f.pt"):
                        torch.save(pre, f"{out}/{tag}_{osplit}_pre_f.pt")
                        torch.save(lab, f"{out}/{tag}_{osplit}_pre_l.pt")
                for i, f in feats.items():
                    torch.save(f, f"{out}/{tag}_{osplit}_L{i}_f.pt")
                print(f"{bb:<10}{name:<16}{osplit:<9}"
                      + " ".join(f"L{i}={tuple(f.shape)}"
                                 for i, f in feats.items()), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
