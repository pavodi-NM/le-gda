"""Features for the ImageNet subset.

Reads data/tip-adapter/imagenet_subset/{val,train}/<label>/*.jpg written by
fetch_imagenet_subset.py. Splits the official val 50/class into val 10/class
+ test 40/class (deterministic per-class shuffle, seed = class index); the
pool is the 32/class train subset with 10-pass augmented-mean features.
Writes caches/imagenet/collapse/<tag>_{val,test,pool}_{f,l}.pt.
"""

import argparse
import os
import random
from collections import namedtuple

import torch

import clip
from scripts.extract_backbone_all import encode
from scripts.extract_train_cache_pool import TRAIN_TFM

SUB = os.path.join(os.environ.get("LEGDA_DATA_ROOT", "data/tip-adapter"),
                   "imagenet_subset")
BACKBONES = ["RN50", "RN101", "ViT-B/32", "ViT-B/16"]
Item = namedtuple("Item", "impath label")


def build_items():
    val, test, pool = [], [], []
    for c in range(1000):
        d = os.path.join(SUB, "val", str(c))
        paths = sorted(os.listdir(d))
        assert len(paths) == 50, (c, len(paths))
        idx = list(range(50))
        random.Random(c).shuffle(idx)
        val += [Item(os.path.join(d, paths[i]), c) for i in idx[:10]]
        test += [Item(os.path.join(d, paths[i]), c) for i in idx[10:]]
        dt = os.path.join(SUB, "train", str(c))
        pool += [Item(os.path.join(dt, p), c) for p in sorted(os.listdir(dt))]
    for split in (val, test, pool):
        split.sort(key=lambda it: it.impath)
    return {"val": val, "test": test, "pool": pool}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--augment-epoch", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src = build_items()
    out_dir = "caches/imagenet/collapse"
    os.makedirs(out_dir, exist_ok=True)
    for bb in args.backbones:
        model, preprocess = clip.load(bb, device=args.device)
        model.eval()
        tag = bb.replace("/", "-")
        for split, items in src.items():
            fp = os.path.join(out_dir, f"{tag}_{split}_f.pt")
            if os.path.exists(fp):
                print(f"{bb:<10}{split:<6}cached", flush=True)
                continue
            tfm = TRAIN_TFM if split == "pool" else preprocess
            ep = args.augment_epoch if split == "pool" else 1
            f, y = encode(items, model, tfm, args.device,
                          args.batch_size, args.workers, ep)
            torch.save(f, fp)
            torch.save(y, os.path.join(out_dir, f"{tag}_{split}_l.pt"))
            print(f"{bb:<10}{split:<6}{tuple(f.shape)}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
