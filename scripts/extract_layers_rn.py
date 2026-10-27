"""Intermediate-stage features from CLIP ResNets, for LE-GDA.

ResNets have no CLS token, and `AttentionPool2d` applies `c_proj` inside
`F.multi_head_attention_forward`, so no pre-projection tensor exists to hook.
The readout used here for each stage visual.layer{1..4} is:

    global-average-pool the spatial map, then L2-normalise.

This is the convolutional analogue of a CLS readout; note that the ViT and
ResNet families are therefore read out differently.

Checks run during extraction:

  labels      re-extracted labels match the cached split exactly
  projected   re-extracting the projected feature reproduces the cached
              `_f.pt` with min per-item cosine > 0.999 (val/test splits)
  readout     on the pool split, the class-mean Gram matrices of GAP(layer4)
              and of the projected feature agree (Spearman rho >= --rho-bar,
              default 0.30); layer4 feeds attnpool, so a low value would mean
              the pooling is wrong

Files are `<tag>_<split>_R<i>_f.pt` (R for stage, L for ViT block). GAP widths
on RN50/RN101 are 256/512/1024/2048 for layer1..4.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import clip
from scripts.extract_backbone_all import (
    ImgList, TRAIN_TFM, add_test_split_args, resolve_test_tag)
from scripts.extract_layers import splits_of

BACKBONES = ["RN50", "RN101"]
DATASETS = ["stanford_cars", "fgvc", "oxford_flowers", "dtd"]
STAGES = [1, 2, 3, 4]


@torch.no_grad()
def encode_stages(items, model, tfm, dev, bs, workers, stages, epochs=1):
    """{stage: [N, C_stage] GAP'd, L2-normalised} + the projected feature.

    Accumulated on CPU for the same reason as the ViT path: imagenet's 40k test
    split at these widths does not fit next to the model on an 8 GB card.
    """
    grabbed = {}

    def mk(i):
        def hook(_m, _i, o):
            # o is [N, C, H, W]; global-average-pool the spatial map
            grabbed[i] = o.float().mean(dim=(2, 3))
        return hook

    hs = [getattr(model.visual, f"layer{i}").register_forward_hook(mk(i))
          for i in stages]
    keys = list(stages) + ["post"]
    loader = DataLoader(ImgList(items, tfm), batch_size=bs, shuffle=False,
                        num_workers=workers, pin_memory=True)
    tot = {k: None for k in keys}
    labels = None
    for e in range(epochs):
        acc = {k: [] for k in keys}
        ys = []
        for x, y in tqdm(loader, leave=False):
            grabbed["post"] = model.encode_image(x.to(dev)).float()
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


def gram_agreement(a, b, y, n_cls):
    """Spearman rho between the two spaces' class-mean Gram matrices.

    Both spaces describe the same classes, so their class-mean similarity
    structure must agree far above chance if the readout is meaningful. This is
    invariant to dimension, which a direct feature correlation is not.
    """
    from scipy.stats import spearmanr
    ga, gb = [], []
    for f in (a, b):
        M = torch.stack([f[y == c].mean(0) for c in range(n_cls)])
        M = M / M.norm(dim=-1, keepdim=True)
        G = (M @ M.t())
        iu = torch.triu_indices(n_cls, n_cls, offset=1)
        (ga if f is a else gb).append(G[iu[0], iu[1]])
    return float(spearmanr(ga[0].numpy(), gb[0].numpy()).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--stages", nargs="+", type=int, default=STAGES)
    ap.add_argument("--augment-epoch", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rho-bar", type=float, default=0.30,
                    help="minimum Gram agreement of GAP(layer4) with the "
                         "projected feature (readout check)")
    add_test_split_args(ap)
    args = ap.parse_args()
    ttag = resolve_test_tag(args)
    fname = {"test": ttag}          # split key -> filename token

    for bb in args.backbones:
        assert bb.startswith("RN"), f"{bb}: this is the ResNet extractor"
        model, preprocess = clip.load(bb, device=args.device)
        model.eval()
        tag = bb.replace("/", "-")
        widths = {i: getattr(model.visual, f"layer{i}")[-1].bn3.num_features
                  for i in args.stages}
        print(f"{bb}: GAP widths {widths}", flush=True)
        for name in args.datasets:
            out = os.path.join("caches", name, "collapse")
            os.makedirs(out, exist_ok=True)
            if all(os.path.exists(f"{out}/{tag}_{fname.get(s, s)}_R{i}_f.pt")
                   for s in ("val", "test", "pool") for i in args.stages):
                print(f"{bb:<8}{name:<16}cached", flush=True)
                continue
            src = splits_of(name, full_test=args.full_test)
            for split, items in src.items():
                osplit = fname.get(split, split)
                if all(os.path.exists(f"{out}/{tag}_{osplit}_R{i}_f.pt")
                       for i in args.stages):
                    continue
                # the label and projected checks need the projected features
                # of this split, so require them
                if split != "pool":
                    assert os.path.exists(f"{out}/{tag}_{osplit}_f.pt"), (
                        f"{name}/{osplit}: no projected reference to check "
                        f"alignment against. Run extract_backbone_all.py with "
                        f"the same --full-test/--test-tag first.")
                tfm = TRAIN_TFM if split == "pool" else preprocess
                ep = args.augment_epoch if split == "pool" else 1
                feats, lab = encode_stages(items, model, tfm, args.device,
                                           args.batch_size, args.workers,
                                           args.stages, ep)
                post = feats.pop("post")

                # ---- check: label alignment ----
                ref = f"{out}/{tag}_{osplit}_l.pt"
                if os.path.exists(ref):
                    dl = (torch.load(ref, map_location="cpu") - lab).abs().max()
                    assert dl.item() == 0, (
                        f"{name}/{osplit}: labels do not match the cached split "
                        f"-- item alignment is broken")

                # ---- check: the projected feature reproduces ----
                fp = f"{out}/{tag}_{osplit}_f.pt"
                if split != "pool" and os.path.exists(fp):
                    old = torch.load(fp, map_location="cpu").float()
                    c = torch.nn.functional.cosine_similarity(
                        old, post, dim=-1).min().item()
                    print(f"  projected {name}/{osplit}: min cos={c:.6f} "
                          f"{'OK' if c > 0.999 else 'MISALIGNED'}")
                    assert c > 0.999, (
                        f"{name}/{osplit}: re-extraction does not reproduce the "
                        f"cached projected features (min cosine {c:.4f}) -- "
                        f"item alignment is broken")

                # ---- check: GAP(layer4) shares class structure ----
                if split == "pool" and 4 in feats:
                    n_cls = int(lab.max()) + 1
                    rho = gram_agreement(feats[4], post, lab, n_cls)
                    ok = rho >= args.rho_bar
                    print(f"  readout {name}: GAP(layer4) vs projected, "
                          f"class-Gram rho={rho:+.3f} "
                          f"(bar {args.rho_bar}) {'PASS' if ok else 'FAIL'}")
                    assert ok, (
                        f"{name}: GAP(layer4) shares almost no class structure "
                        f"with the projected feature (rho={rho:.3f}); the "
                        f"readout is likely wrong")

                for i, f in feats.items():
                    torch.save(f, f"{out}/{tag}_{osplit}_R{i}_f.pt")
                print(f"{bb:<8}{name:<16}{osplit:<9}"
                      + " ".join(f"R{i}={tuple(f.shape)}"
                                 for i, f in sorted(feats.items())),
                      flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
