"""Reusable few-shot cache pool: per-image augmented-mean train features.

utils.build_cache_model computes, for each training image independently,

    key_i = normalize( mean_a encode(aug_a(img_i)) )

over cfg['augment_epoch'] augmentation passes, then stacks the K-shot subset.
Because the averaging and the normalisation are both per-image, encoding the
whole train pool once and subsampling reproduces the cache for ANY few-shot draw
exactly -- which is what makes multi-seed experiments cheap instead of requiring
a re-encode per seed.

Per-class cap keeps EuroSAT (1000-1500 imgs/class) affordable; any cap well above
the largest K still leaves the draws properly random.

Writes caches/<dataset>/pool/train_pool_f.pt and train_pool_l.pt.
"""

import argparse
import os
import random
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import clip
from datasets import build_dataset
from datasets.oxford_pets import OxfordPets

# Dataset root. Override with the LEGDA_DATA_ROOT environment variable;
# see the README for the expected CoOp-style directory layout.
ROOT = os.environ.get("LEGDA_DATA_ROOT", "data/tip-adapter")
DATASETS = ["caltech101", "dtd", "eurosat"]

# Identical to main.py's train_tranform.
TRAIN_TFM = transforms.Compose([
    transforms.RandomResizedCrop(
        size=224, scale=(0.5, 1),
        interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])


class AugList(Dataset):
    def __init__(self, items, tfm):
        self.items, self.tfm = list(items), tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d = self.items[i]
        return self.tfm(Image.open(d.impath).convert("RGB")), d.label


def capped_train(dataset, cap, seed=0):
    ds = build_dataset(dataset, ROOT, 1)
    train, _, _ = OxfordPets.read_split(ds.split_path, ds.image_dir)
    by_cls = defaultdict(list)
    for d in train:
        by_cls[d.label].append(d)
    rng = random.Random(seed)
    out = []
    for c in sorted(by_cls):
        pool = by_cls[c]
        out.extend(pool if len(pool) <= cap else rng.sample(pool, cap))
    out.sort(key=lambda d: d.impath)
    return out


@torch.no_grad()
def encode_pool(items, model, dev, epochs, batch_size, workers):
    """Mean over augmentation passes, per image, then normalise."""
    loader = DataLoader(AugList(items, TRAIN_TFM), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=True)
    total, labels = None, None
    for e in range(epochs):
        feats, ys = [], []
        for x, y in tqdm(loader, desc=f"aug {e + 1}/{epochs}", leave=False):
            feats.append(model.encode_image(x.to(dev)).float())
            if e == 0:
                ys.append(y)
        f = torch.cat(feats)
        total = f if total is None else total + f
        if e == 0:
            labels = torch.cat(ys)
    f = total / epochs
    return (f / f.norm(dim=-1, keepdim=True)).cpu(), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--cap", type=int, default=64, help="max images per class")
    ap.add_argument("--augment-epoch", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    model, _ = clip.load("RN50", device=args.device)
    model.eval()

    for name in args.datasets:
        out_dir = os.path.join("caches", name, "pool")
        os.makedirs(out_dir, exist_ok=True)
        fp = os.path.join(out_dir, "train_pool_f.pt")
        if os.path.exists(fp):
            print(f"{name:<12} cached")
            continue
        items = capped_train(name, args.cap)
        f, y = encode_pool(items, model, args.device, args.augment_epoch,
                           args.batch_size, args.workers)
        torch.save(f, fp)
        torch.save(y, os.path.join(out_dir, "train_pool_l.pt"))
        print(f"{name:<12} {tuple(f.shape)}  "
              f"min/class={y.bincount().min().item()}")


if __name__ == "__main__":
    main()
