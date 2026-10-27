"""Intermediate-block CLS features from CLIP ViTs, for LE-GDA.

Extracts the CLS token after selected residual blocks. CLIP's
VisionTransformer runs the transformer in LND layout:

    x = x.permute(1, 0, 2)      # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)      # LND -> NLD
    x = self.ln_post(x[:, 0, :])

so hooking `visual.transformer.resblocks[i]` gives [L, N, D] and the CLS token
is `out[0]`. Each block feature is L2-normalised (no `ln_post`), saved as
`<tag>_<split>_L{i}_f.pt`.

`--with-pre` also writes the pre-projection feature (`ln_post` of the final
CLS token, L2-normalised) as `<tag>_<split>_pre_f.pt` from the same forward
pass, and recomputes the projected output to check it against the cached
`_f.pt` (min per-item cosine > 0.999 on the val/test splits) together with
the labels.

ImageNet items come from extract_imagenet_subset.build_items, the function
that built its cached projected features.
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

    ImageNet has no `full_test` variant: data/tip-adapter/imagenet_subset holds
    the official validation set (50 per class) and 32 per class of train. Its
    test split is 40 per class of that validation set; the other 10 per class
    are the validation split LE-GDA selects its hyperparameters on.
    """
    if name == "imagenet":
        assert not full_test, (
            "imagenet has a single test split (40/class of the official val "
            "set); run it without --full-test")
        from scripts.extract_imagenet_subset import build_items
        return build_items()
    return splits_for(name, full_test=full_test)[0]


@torch.no_grad()
def encode_layers(items, model, tfm, dev, bs, workers, layers, epochs=1,
                  with_pre=False):
    """{key: [N, D] L2-normalised feature}, averaged over augmentations.

    Keys are the block indices in `layers`, plus "pre" (the `ln_post` output)
    and "post" (the projected embedding) when `with_pre` is set; "post" is only
    used to check that these items match the cached projected features.

    Batches are accumulated on the CPU so ImageNet fits on an 8 GB GPU.
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
                    help="also write the ln_post pre-projection feature "
                         "(`pre`) from the same forward pass")
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
                # the checks below need this split's projected features
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
                # check: labels match the cached projected split
                ref = f"{out}/{tag}_{osplit}_l.pt"
                if os.path.exists(ref):
                    dl = (torch.load(ref, map_location="cpu") - lab).abs().max()
                    assert dl.item() == 0, (
                        f"{name}/{osplit}: labels do not match the cached split "
                        f"-- item alignment is broken")
                post = feats.pop("post", None)
                if post is not None:
                    # check: val/test reproduce the cached projected features
                    # (`pool` averages random augmentations, so it is skipped)
                    if split != "pool" and os.path.exists(f"{out}/{tag}_{osplit}_f.pt"):
                        old = torch.load(f"{out}/{tag}_{osplit}_f.pt",
                                         map_location="cpu").float()
                        # per-item cosine rather than max|d|: a wrong image
                        # scores ~0, while fp16 kernel differences (e.g. a
                        # smaller final batch) can move max|d| to ~1e-3 on the
                        # right images
                        cos = torch.nn.functional.cosine_similarity(
                            old, post, dim=-1)
                        c, d = cos.min().item(), (old - post).abs().max().item()
                        print(f"  projected {name}/{osplit}: min cos={c:.6f} "
                              f"max|d_feat|={d:.2e} "
                              f"{'OK' if c > 0.999 else 'MISALIGNED'}")
                        assert c > 0.999, (
                            f"{name}/{osplit}: re-extraction does not reproduce "
                            f"the cached projected features (min cosine {c:.4f})"
                            f" -- item alignment is broken")
                    pre = feats.pop("pre")
                    # never overwrite: `pool` averages random augmentations,
                    # so a rewrite would change existing features
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
