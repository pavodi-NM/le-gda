"""Our protocol, exposed to somebody else's training code.

Protocol of record: the clauses P1-P6 defined in this module, and the
README section *Comparative methods under our protocol*.

The point of this module is to make an upstream `main.py` obey clauses P1-P6
WITHOUT touching its model, loss, optimizer, schedule, batch size or
augmentation. Each upstream repo then needs a ~20-line diff that swaps its
loaders for these four calls and deletes its test-selection block. The diff
stays auditable at a glance, which is the whole reason this is a module and
not a fork.

WHY THIS IS RECONSTRUCTION AND NOT RE-EXTRACTION. Our cached features live in
`caches/<ds>/collapse/<tag>_*` and the indices `draw()` returns address rows of
that pool. An upstream trainer needs IMAGES, not rows, because live
augmentation through the encoder is exactly the thing under test (section 1.2).
So the item list behind those rows has to be rebuilt. It is fully
deterministic:

    build_dataset(ds, ROOT, -1).train_x        # num_shots=-1 -> FULL train
      -> cap_per_class(train, POOL_CAP=32)     # random.Random(0).sample, then
                                               # sort by impath
      -> draw(pool_y, K, seed)                 # torch.Generator(seed), per class

`assert_pool_matches` below checks the reconstruction against the cached label
tensor, so a drift in either extractor is caught rather than silently
producing a different support set than every other row of the table.

P2 NOTE, and it matters. Our pool is capped at 32/class, so our K=16 draw is
from a 32/class subsample of the full train split. Upstream's own
`generate_fewshot_dataset(train, num_shots=16)` draws from the FULL split.
Handing upstream OUR items is the point -- it is what makes the support set
identical to every other row -- but it is a real protocol difference and it is
part of what `B - A` measures.
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

    Returns (items, labels, pool_len). `items[i]` is the Datum for the i-th row
    of what every other arm in the table fits on.
    """
    pool, ds = _pool_items(dataset)
    pool_y = torch.tensor([d.label for d in pool])
    idx = draw(pool_y, K, seed, "cpu").tolist()
    items = [pool[i] for i in idx]
    return items, [d.label for d in items], len(pool)


def eval_items(dataset, split):
    """P1: our val and test item lists.

    `split="testfull"` is the standard uncapped CoOp test split, sorted by
    impath. `split="val"` is capped at max(30, ceil(3000/n_cls)) per class --
    that cap binds on nothing for most datasets but MUST be checked per dataset
    rather than assumed, because a val set that differs from upstream's is a
    silent protocol change (P1).
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
    """P5: our zero-shot classifier -- single template, ImageNet's 7-ensemble.

    Only for a matched-prompt variant. An upstream method that ships its own
    prompts (ClusterAdapter's CuPL banks) keeps them in the primary row; see
    decision D1. Do not substitute this silently.
    """
    from utils import clip_classifier
    from scripts.ds_meta import class_template
    classnames, template = class_template(dataset)
    return clip_classifier(classnames, template, clip_model, device=device)


def assert_pool_matches(dataset, backbone_tag):
    """Reconstruction check: our rebuilt pool must equal the cached one.

    Compares length and the full label vector against
    `caches/<ds>/collapse/<tag>_pool_l.pt`. Cheap, and it is the only thing
    standing between "upstream trained on our support set" and "upstream
    trained on a different 16 images that happen to be the right size".
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
    """P6: our JSON shape -- per_seed kept, mean and std derived.

    `cells` is a list of dicts each carrying at least
    {backbone, dataset, K, test_split, arms: {name: [v_seed0, v_seed1, ...]}}.
    `std` is the population std over seeds, matching
    run_calibrated_ft.py's pstdev. Collapsing per_seed is what left three arms
    without a `±` for the head-to-head table; do not do it again.
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
