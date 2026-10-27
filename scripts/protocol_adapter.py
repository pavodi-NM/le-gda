"""Our evaluation protocol, exposed to another method's training code.

Every method in the paper is evaluated under the same protocol:

  P1  Splits. Standard CoOp test splits, uncapped (`testfull`). ImageNet: 40
      images per class of the official validation set as test, 10 per class
      as validation.
  P2  Support set. K shots per class drawn by `draw(pool_y, K, seed)` from a
      fixed 32-per-class training pool, seeds 0, 1, 2; identical draws for
      every method.
  P3  Model selection on validation only: no test-set epoch selection and no
      best-on-test reporting.
  P4  Fusion weights (and Tip-Adapter's beta) searched on validation over the
      method's own grid.
  P5  Zero-shot head: one prompt template per dataset, ImageNet's 7-template
      ensemble (as in Tip-Adapter / CoOp).
  P6  Reporting: 3-seed mean; `±` is the std over seeds of the 11-dataset mean.

This module lets an original `main.py` follow P1-P6 without touching its
model, loss, optimiser, schedule, batch size or augmentation: the original
loaders are swapped for the calls below, and the test-set selection is removed
(see patches/tip_adapter_our_protocol.patch).

The original trainer needs images rather than cached feature rows, so the
item list behind the cached pool is rebuilt deterministically:

    build_dataset(ds, ROOT, -1).train_x        # num_shots=-1 -> full train
      -> cap_per_class(train, POOL_CAP=32)     # random.Random(0).sample, then
                                               # sort by impath
      -> draw(pool_y, K, seed)                 # torch.Generator(seed), per class

`assert_pool_matches` checks the rebuilt list against the cached label tensor.

Note: the pool is 32 per class, so the K=16 draw is from a subsample of the
full training split, whereas Tip-Adapter's own `generate_fewshot_dataset`
draws from the full split.
"""

import json
import os
from statistics import mean, pstdev

import torch

from datasets import build_dataset
from scripts.extract_backbone_all import ROOT, POOL_CAP, cap_per_class, EVAL_TARGET
from scripts.run_prototype_collapse_seeds import draw
import math


def _pool_items(dataset):
    """The item list whose order the cached `_pool_*` tensors were built in."""
    ds = build_dataset(dataset, ROOT, -1)
    return cap_per_class(ds.train_x, POOL_CAP), ds


def support_items(dataset, K, seed):
    """P2: the images behind our seeded K-shot draw, in draw() order.

    Returns (items, labels, pool_len).
    """
    pool, ds = _pool_items(dataset)
    pool_y = torch.tensor([d.label for d in pool])
    idx = draw(pool_y, K, seed, "cpu").tolist()
    items = [pool[i] for i in idx]
    return items, [d.label for d in items], len(pool)


def eval_items(dataset, split):
    """P1: our val and test item lists.

    `split="testfull"` is the standard uncapped CoOp test split, sorted by
    impath. `split="val"` is capped at max(30, ceil(3000/n_cls)) per class,
    which leaves most datasets' validation split unchanged
    (see val_is_upstream_val).
    """
    ds = build_dataset(dataset, ROOT, -1)
    n_cls = len(ds.classnames)
    cap = max(30, math.ceil(EVAL_TARGET / n_cls))
    if split == "testfull":
        return sorted(ds.test, key=lambda d: d.impath)
    if split == "test":
        return cap_per_class(ds.test, cap)
    if split == "val":
        return cap_per_class(ds.val, cap)
    raise ValueError(split)


def val_is_upstream_val(dataset):
    """True iff our val cap binds on nothing, i.e. our val == `dataset.val`."""
    ds = build_dataset(dataset, ROOT, -1)
    return len(eval_items(dataset, "val")) == len(ds.val)


def text_head(dataset, clip_model, device="cuda"):
    """P5: our zero-shot classifier -- single template, ImageNet's 7-ensemble."""
    from utils import clip_classifier
    from scripts.ds_meta import class_template
    classnames, template = class_template(dataset)
    return clip_classifier(classnames, template, clip_model, device=device)


def assert_pool_matches(dataset, backbone_tag):
    """Reconstruction check: our rebuilt pool must equal the cached one.

    Compares length and the full label vector against
    `caches/<ds>/collapse/<tag>_pool_l.pt`, so the trainer is guaranteed to see
    the same support set as every other method.
    """
    pool, _ = _pool_items(dataset)
    p = f"caches/{dataset}/collapse/{backbone_tag}_pool_l.pt"
    cached = torch.load(p, map_location="cpu")
    got = torch.tensor([d.label for d in pool])
    if len(got) != len(cached):
        raise AssertionError(f"{dataset}/{backbone_tag}: pool length "
                             f"{len(got)} != cached {len(cached)}")
    if not torch.equal(got, cached.cpu()):
        raise AssertionError(f"{dataset}/{backbone_tag}: pool LABEL ORDER "
                             f"differs from the cached tensor")
    return len(got)


def report(cells, out, meta=None):
    """P6: write results as {"meta": ..., "rows": [...]}.

    `cells` is a list of dicts each carrying at least
    {backbone, dataset, K, test_split, arms: {name: [v_seed0, v_seed1, ...]}}.
    Per-seed values are kept; `acc` is their mean and `std` the population
    std over seeds.
    """
    rows = []
    for c in cells:
        arms = c.pop("arms")
        rows.append({**c,
                     "per_seed": arms,
                     "acc": {k: mean(v) for k, v in arms.items()},
                     "std": {k: pstdev(v) for k, v in arms.items()}})
    payload = {"meta": meta or {}, "rows": rows}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    return payload


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="self-check the reconstruction")
    ap.add_argument("--datasets", nargs="+",
                    default=["stanford_cars", "dtd", "caltech101"])
    ap.add_argument("--backbone-tag", default="RN50")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    for ds in a.datasets:
        n = assert_pool_matches(ds, a.backbone_tag)
        sizes = [len(support_items(ds, a.k, s)[0]) for s in a.seeds]
        sets = [tuple(d.impath for d in support_items(ds, a.k, s)[0])
                for s in a.seeds]
        vfull = val_is_upstream_val(ds)
        print(f"  {ds:15s} pool={n:5d} OK | support={sizes} | "
              f"distinct draws={len(set(sets))}/{len(a.seeds)} | "
              f"val==upstream val: {vfull} | "
              f"testfull={len(eval_items(ds, 'testfull'))}")
