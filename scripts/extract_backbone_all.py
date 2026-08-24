"""val / test / cache-pool features for every (backbone, dataset) cell.

The prototype-collapse experiment needs three things per cell, and only RN50 on
three datasets currently has all of them:

  val    hyperparameter search set for the alpha/beta grid
  test   evaluation
  pool   per-image augmented-mean train features, from which any K-shot draw is
         an exact cache (see extract_train_cache_pool.py for why)

Caps keep this affordable without distorting the comparison: every arm in the
experiment sees the identical val/test/pool, so a cap changes the absolute
numbers slightly but not the tip-vs-prototype contrast that is being measured.
Pool cap 32 still leaves real draw variation at K=16.

Writes caches/<dataset>/collapse/<tag>_{val,test,pool}_{f,l}.pt. Deliberately NOT
caches/<dataset>/backbones/, which holds the calibration experiment's test
features built from a different item selection -- reusing those would silently
misalign features against labels here.
"""

import argparse
import math
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
from scripts.extract_train_cache_pool import TRAIN_TFM

# Dataset root. Override with the LEGDA_DATA_ROOT environment variable;
# see the README for the expected CoOp-style directory layout.
ROOT = os.environ.get("LEGDA_DATA_ROOT", "data/tip-adapter")
DATASETS = ["caltech101", "dtd", "eurosat", "food101", "oxford_pets"]
BACKBONES = ["RN50", "RN101", "ViT-B/32", "ViT-B/16"]

POOL_CAP = 32          # >= max K of 16, with room for draw variation
EVAL_TARGET = 3000     # rough per-split size target; cap derived per dataset


class ImgList(Dataset):
    def __init__(self, items, tfm):
        self.items, self.tfm = list(items), tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d = self.items[i]
        return self.tfm(Image.open(d.impath).convert("RGB")), d.label


def cap_per_class(items, cap, seed=0):
    by_cls = defaultdict(list)
    for d in items:
        by_cls[d.label].append(d)
    rng = random.Random(seed)
    out = []
    for c in sorted(by_cls):
        pool = by_cls[c]
        out.extend(pool if len(pool) <= cap else rng.sample(pool, cap))
    out.sort(key=lambda d: d.impath)
    return out


def splits_for(dataset, full_test=False):
    """{split: items}, capped. `full_test` returns the STANDARD test split.

    `full_test` is an explicit argument and NOT a module constant on purpose.
    `build_calibrated` once read K from a module constant and would have
    silently inflated every non-16-shot cache term; a cap that can be toggled
    invisibly is the same failure with a bigger blast radius, because the
    capped and uncapped splits differ in LENGTH and every downstream tensor is
    indexed by it.

    WHY THIS EXISTS. EVAL_TARGET=3000 was correct for the compact-cache
    experiment it was written for -- a within-experiment contrast where only
    the difference between arms mattered, and every arm saw the identical
    split. LE-GDA inherited it with the caches and then put the numbers in a
    HEAD-TO-HEAD against published literature, where it is not correct at all:
    10 of 11 datasets are subsets (food101 3030 vs 30300, sun397 11910 vs
    19850), so a table mixing our rows with published ones would partly be
    measuring which images were evaluated. Measured against the GDA-CLIP
    paper's own Tip-Adapter row, our re-runs differ by up to 2.38 points on
    caltech101 -- 2.4x the +1.01 increment being claimed.

    ONLY the test cap is lifted. Capping VALIDATION is a defensible
    search-budget choice and is not a comparability threat, since no paper
    reports validation accuracy; `pool` is the few-shot support set and is
    capped by construction at K.
    """
    # num_shots=-1 makes generate_fewshot_dataset return the FULL train list,
    # so this works for every builder (split-json datasets, stanford_cars's
    # dataset_dir-rooted split, fgvc's txt splits) without special-casing.
    ds = build_dataset(dataset, ROOT, -1)
    train, val, test = ds.train_x, ds.val, ds.test
    n_cls = len(ds.classnames)
    eval_cap = max(30, math.ceil(EVAL_TARGET / n_cls))
    # the uncapped path still sorts by impath, so the two splits are in the
    # same order on the items they share and dtd (where the cap binds on
    # nothing) must reproduce the capped file exactly -- a free alignment check
    test_items = sorted(test, key=lambda d: d.impath) if full_test \
        else cap_per_class(test, eval_cap)
    return {
        "val": cap_per_class(val, eval_cap),
        "test": test_items,
        "pool": cap_per_class(train, POOL_CAP),
    }, ds


def add_test_split_args(ap):
    """--full-test / --test-tag, shared by all three extractors.

    The tag renames only the OUTPUT FILE; the in-memory split key stays
    "test" so every downstream `src["test"]` keeps working. Writing the
    uncapped split to a parallel namespace rather than over the existing one
    is deliberate: §3 and §5c of the write-up are fitted on the capped
    files, and silently rewriting features that published numbers rest on is
    the exact failure extract_layers.py's "never overwrite" comment guards
    against.
    """
    ap.add_argument("--full-test", action="store_true",
                    help="use the STANDARD (uncapped) test split")
    ap.add_argument("--test-tag", default=None,
                    help="filename token for the test split "
                         "(default: 'test', or 'testfull' with --full-test)")
    return ap


def resolve_test_tag(args):
    tag = args.test_tag or ("testfull" if args.full_test else "test")
    assert not (args.full_test and tag == "test"), (
        "--full-test with --test-tag=test would OVERWRITE the capped features "
        "that the write-up §3/§5c are fitted on. Use a separate tag.")
    return tag


@torch.no_grad()
def encode(items, model, tfm, dev, bs, workers, epochs=1):
    loader = DataLoader(ImgList(items, tfm), batch_size=bs, shuffle=False,
                        num_workers=workers, pin_memory=True)
    total, labels = None, None
    for e in range(epochs):
        feats, ys = [], []
        for x, y in tqdm(loader, leave=False):
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
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--augment-epoch", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    add_test_split_args(ap)
    args = ap.parse_args()
    ttag = resolve_test_tag(args)
    fname = {"test": ttag}          # split key -> filename token

    _, pre = clip.load("RN50", device="cpu")   # eval transform, shared shape

    for bb in args.backbones:
        model, preprocess = clip.load(bb, device=args.device)
        model.eval()
        tag = bb.replace("/", "-")
        for name in args.datasets:
            out_dir = os.path.join("caches", name, "collapse")
            os.makedirs(out_dir, exist_ok=True)
            done = all(os.path.exists(
                os.path.join(out_dir, f"{tag}_{fname.get(s, s)}_f.pt"))
                for s in ("val", "test", "pool"))
            if done:
                print(f"{bb:<10}{name:<14}cached", flush=True)
                continue
            src, _ = splits_for(name, full_test=args.full_test)
            for split, items in src.items():
                out_split = fname.get(split, split)
                fp = os.path.join(out_dir, f"{tag}_{out_split}_f.pt")
                if os.path.exists(fp):
                    continue
                tfm = TRAIN_TFM if split == "pool" else preprocess
                ep = args.augment_epoch if split == "pool" else 1
                f, y = encode(items, model, tfm, args.device,
                              args.batch_size, args.workers, ep)
                torch.save(f, fp)
                torch.save(y, os.path.join(out_dir, f"{tag}_{out_split}_l.pt"))
                print(f"{bb:<10}{name:<14}{out_split:<9}{tuple(f.shape)}",
                      flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
