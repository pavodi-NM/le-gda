"""One-stop setup for LE-GDA: CLIP weights, dataset check, feature extraction.

The heavy artefacts of this project are NOT in the repository and cannot be:
the extracted feature caches are ~20 GB and the image datasets are far larger.
Everything heavy is *derived*, so this script rebuilds it instead of shipping it.

    python scripts/download_and_prepare.py --weights        # CLIP checkpoints
    python scripts/download_and_prepare.py --check          # dataset layout
    python scripts/download_and_prepare.py --extract        # build the caches
    python scripts/download_and_prepare.py --all            # all three, in order

Set LEGDA_DATA_ROOT to wherever the images live (default: ./data/tip-adapter).
Feature caches are always written under ./caches/ relative to the repository
root, which is where every runner looks for them.

WHAT --extract COSTS. Measured on one RTX 4060 (8 GB), 11 datasets x 4 CLIP
backbones: base projected features ~35 min, layer/stage features ~50 min, and
about 20 GB of disk. ImageNet dominates both. Extraction is resumable: files
that already exist are skipped, so an interrupted run is restarted with the
same command.
"""

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("LEGDA_DATA_ROOT", "data/tip-adapter")

BACKBONES = ["RN50", "RN101", "ViT-B/32", "ViT-B/16"]

TEN = ["caltech101", "dtd", "eurosat", "fgvc", "food101", "oxford_flowers",
       "oxford_pets", "stanford_cars", "sun397", "ucf101"]

# (dataset key, directory under $LEGDA_DATA_ROOT, a path that must exist inside
# it, where to get it). The directory names are the ones datasets/*.py expect --
# they are CoOp's names, and renaming them breaks the loaders, not just this
# check. Splits are CoOp's `split_zhou_*.json` files; see the CoOp DATASET.md
# (https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) for the download
# links, which are the same splits every baseline in the paper is fitted on.
LAYOUT = [
    # ImageNet is used as a 2-3 GB subset, not the 155 GB original: the official
    # val set at 50/class (split into 40 test + 10 val) plus 32/class of train
    # for the support pool. scripts/fetch_imagenet_subset.py downloads exactly
    # that from a public mirror.
    ("imagenet",       "imagenet_subset", "val",
     "python scripts/fetch_imagenet_subset.py   (~2-3 GB, no ImageNet account)"),
    ("caltech101",     "caltech-101",    "101_ObjectCategories",
     "http://www.vision.caltech.edu/Image_Datasets/Caltech101/"),
    ("oxford_pets",    "oxford_pets",    "images",
     "https://www.robots.ox.ac.uk/~vgg/data/pets/"),
    ("stanford_cars",  "stanford_cars",  "split_zhou_StanfordCars.json",
     "http://ai.stanford.edu/~jkrause/car196/"),
    ("oxford_flowers", "oxford_flowers", "jpg",
     "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"),
    ("food101",        "food-101",       "images",
     "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/"),
    ("fgvc",           "fgvc_aircraft",  "images",
     "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/"),
    # The official SUN397 hosts are dead; reconstruct_sun397_from_hf.py rebuilds
    # the exact original tree from a HF mirror so split_zhou_SUN397.json applies.
    ("sun397",         "sun397",         "SUN397",
     "python scripts/reconstruct_sun397_from_hf.py   (~39 GB)"),
    ("dtd",            "dtd",            "images",
     "https://www.robots.ox.ac.uk/~vgg/data/dtd/"),
    ("eurosat",        "eurosat",        "2750",
     "http://madm.dfki.de/files/sentinel/EuroSAT.zip"),
    ("ucf101",         "ucf101",         "UCF-101-midframes",
     "UCF-101-midframes.zip, per the CoOp instructions"),
]


def weights():
    """Fetch the four CLIP checkpoints into ~/.cache/clip (clip's own cache)."""
    # The names are private to clip.clip, so `import *` does not re-export
    # them; import the module directly rather than going through the package.
    from clip.clip import _MODELS, _download
    cache = os.path.expanduser("~/.cache/clip")
    for bb in BACKBONES:
        print(f"[weights] {bb}")
        _download(_MODELS[bb], cache)
    print(f"[weights] done -> {cache}")


def check():
    """Report which datasets are ready. Returns the list that is NOT."""
    print(f"[check] LEGDA_DATA_ROOT = {os.path.abspath(ROOT)}\n")
    missing = []
    for key, d, probe, where in LAYOUT:
        ok = os.path.exists(os.path.join(ROOT, d, probe))
        print(f"  {'OK   ' if ok else 'MISS '} {key:<15} {d}/{probe}")
        if not ok:
            missing.append((key, d, where))
    if missing:
        print("\n[check] missing datasets, and where they come from:")
        for key, d, where in missing:
            print(f"  {key:<15} -> $LEGDA_DATA_ROOT/{d}\n"
                  f"  {'':<15}    {where}")
        print("\n  Every dataset also needs its CoOp split_zhou_*.json:")
        print("  https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md")
    else:
        print("\n[check] all 11 datasets present")
    return [k for k, _, _ in missing]


def run(cmd):
    print("\n$ " + " ".join(cmd))
    env = dict(os.environ, PYTHONPATH=REPO)
    r = subprocess.run(cmd, cwd=REPO, env=env)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}")


def extract(datasets, backbones, with_imagenet):
    """Rebuild caches/. Order matters: the layer extractors check themselves
    against the base features written by step 1 and abort if those are absent,
    which is what keeps a wrong readout from silently producing weak features
    instead of an error."""
    py = sys.executable
    ten = [d for d in datasets if d != "imagenet"]
    vits = [b for b in backbones if b.startswith("ViT")]
    rns = [b for b in backbones if b.startswith("RN")]

    # 1. base projected features on the standard test split.
    if ten:
        run([py, "scripts/extract_backbone_all.py", "--backbones", *backbones,
             "--datasets", *ten, "--full-test"])
    # 2. per-layer (ViT) and per-stage (ResNet) features.
    if vits and ten:
        run([py, "scripts/extract_layers.py", "--backbones", *vits,
             "--datasets", *ten, "--with-pre", "--full-test"])
    if rns and ten:
        run([py, "scripts/extract_layers_rn.py", "--backbones", *rns,
             "--datasets", *ten, "--full-test"])
    # 3. ImageNet last and separately, since it is the largest. Its base
    #    features come from extract_imagenet_subset.py, and it has a single
    #    test split (no --full-test).
    if with_imagenet:
        run([py, "scripts/extract_imagenet_subset.py", "--backbones", *backbones])
        if vits:
            run([py, "scripts/extract_layers.py", "--backbones", *vits,
                 "--datasets", "imagenet", "--with-pre"])
        if rns:
            run([py, "scripts/extract_layers_rn.py", "--backbones", *rns,
                 "--datasets", "imagenet"])
    print("\n[extract] done -> caches/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", action="store_true", help="download CLIP checkpoints")
    ap.add_argument("--check", action="store_true", help="verify the dataset layout")
    ap.add_argument("--extract", action="store_true", help="build caches/ from the images")
    ap.add_argument("--all", action="store_true", help="--weights --check --extract")
    ap.add_argument("--datasets", nargs="+", default=TEN + ["imagenet"])
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    args = ap.parse_args()

    if not (args.weights or args.check or args.extract or args.all):
        ap.error("pick at least one of --weights / --check / --extract / --all")

    if args.weights or args.all:
        weights()
    missing = []
    if args.check or args.all:
        missing = check()
    if args.extract or args.all:
        todo = [d for d in args.datasets if d not in missing]
        skipped = [d for d in args.datasets if d in missing]
        if skipped:
            print(f"\n[extract] skipping (dataset not installed): {', '.join(skipped)}")
        if not todo:
            sys.exit("[extract] nothing to extract -- install the datasets first")
        extract(todo, args.backbones, "imagenet" in todo)


if __name__ == "__main__":
    main()
