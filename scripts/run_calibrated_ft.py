"""The cache-model baseline arms of the head-to-head, trained and training-free.

Produces the Tip-Adapter, Tip-Adapter-F and Proto-Adapter-F rows of the main
table. Every arm sees Tip-Adapter's unmodified alpha/beta grid and
size-calibrated values V[j,c] = K/m_c, so `alpha` means the same thing at every
cache size (see run_size_calibrated.py for why that matters).

Arms at K=16:

    TIP       every shot a key, training-free    -> "Tip-Adapter"
    TIP-F     the same cache, keys trained        -> "Tip-Adapter-F"
    CC-F-1    1 trainable key/class, prototype init -> "Proto-Adapter-F"

`CC-F-1` is a historical key name kept because the evidence files and the
table generators agree on it; the row it produces is Proto-Adapter-F.

All trained arms use EPOCHS = 20 -- Tip-Adapter-F's own config default --
applied uniformly, so training length cannot differ between a 16-key and a
1-key adapter. That uniformity is what makes the "LE-GDA beats both trained
arms while training nothing" comparison honest.

LE-GDA does not appear here: it has no cache term, no beta, and nothing to
train. It is run by run_layer_ensemble.py / run_nonlinear_fusion.py.
"""

import argparse
import json
from statistics import mean, median, pstdev

import torch
import torch.nn.functional as F

import clip
from utils import clip_classifier
from scripts.ds_meta import class_template
from scripts.run_cache_retrieval_null import load_cfg, grid, acc, search
from scripts.run_prototype_collapse_seeds import draw
from scripts.run_compact_cache import kmeans, acc_chunked
from scripts.run_compact_cache_ft import train_keys, LRS
from scripts.run_size_calibrated import build_calibrated

DATASETS = ["caltech101", "dtd", "eurosat", "food101", "oxford_pets",
            "oxford_flowers", "ucf101", "stanford_cars", "fgvc", "sun397",
            "imagenet"]
BACKBONES = ["RN50", "RN101", "ViT-B/32", "ViT-B/16"]
SEEDS = [0, 1, 2]
K = 16

# Tip-Adapter-F's own default: configs/*.yaml set train_epoch = 20 for every
# dataset except EuroSAT (100). We use 20 UNIFORMLY -- the same budget for
# every arm and every dataset -- so training length cannot differ between a
# 16-key and a 1-key adapter. The previous 200 (run_compact_cache_ft.EPOCHS)
# was 10x this, with no LR schedule and no best-epoch selection, which
# penalises the arm with more parameters exactly in the direction of the
# "compactness is regularization" claim. EuroSAT's config value of 100 is
# deliberately NOT honoured here: a per-dataset budget would reintroduce the
# same asymmetry across cells.
EPOCHS = 20


def ft_arm_calibrated(keys0, V, f_tr, y_tr, clip_tr, f_val, y_val, f_te, y_te,
                      clip_val, clip_te, betas, alphas, epochs=EPOCHS):
    """ft_arm from run_compact_cache_ft, except that V is size-calibrated.

    Also returns the mean L2 norm of the selected trained keys (T2): keys are
    unit-norm at init and train_keys never renormalises, so any growth is the
    mechanism by which a trained arm escapes the alpha ceiling.

    Third return value is SUPPORT-SET (train) accuracy at the same selected
    keys/beta/alpha -- item (5). Every figure on the regularization axis so far
    plots test accuracy against capacity, which is consistent with
    regularization but does not demonstrate it: a monotone test curve is
    equally consistent with the large cache simply being a worse estimator.
    The train/test gap separates the two. Computing it consumes no RNG and
    happens after training, so every pre-existing number is unchanged.
    """
    b0, a0 = search(betas, alphas, f_val @ keys0, V, clip_val, y_val)
    best = (-1.0, None)
    for lr in LRS:
        keys = train_keys(keys0, V, f_tr, y_tr, clip_tr, a0, b0, lr,
                          epochs=epochs)
        b, a = search(betas, alphas, f_val @ keys, V, clip_val, y_val)
        a_val = acc(clip_val + a * (((-1) * (b - b * (f_val @ keys))).exp()
                                    @ V), y_val)
        if a_val > best[0]:
            best = (a_val, (keys, b, a))
    keys, b, a = best[1]
    norm = float(keys.norm(dim=0).mean())          # keys is [D, m*C]
    tr = acc(clip_tr + a * (((-1) * (b - b * (f_tr @ keys))).exp() @ V), y_tr)
    return acc_chunked(keys, V, f_te, y_te, clip_te, b, a), norm, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--ks", nargs="+", type=int, default=[K],
                    help="shot counts to sweep. K enters BOTH the draw and the "
                         "size calibration V[j,c] = K/m_c, so it is threaded "
                         "into build_calibrated rather than left at 16.")
    ap.add_argument("--device", default="cuda")
    # "test" is the capped eval split every grid result uses;
    # "testfull" is the uncapped split, for head-to-head runs
    # against an external implementation (scripts/extract_full_test.py).
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help="training budget for every trained arm "
                         "(20 = Tip-Adapter-F's config default; "
                         "200 = the budget all earlier "
                         "evidence used)")
    ap.add_argument("--out", default="evidence/calibrated_ft.json")
    args = ap.parse_args()
    dev = args.device

    cells = []
    for bb in args.backbones:
        tag = bb.replace("/", "-")
        model, _ = clip.load(bb, device=dev)
        model.eval()
        for name in args.datasets:
            cfg = load_cfg(name)
            betas, alphas = grid(cfg)
            classnames, template = class_template(name)
            p = f"caches/{name}/collapse/{tag}"
            f_val = torch.load(f"{p}_val_f.pt", map_location=dev).float()
            y_val = torch.load(f"{p}_val_l.pt", map_location=dev)
            f_te = torch.load(f"{p}_{args.test_split}_f.pt",
                              map_location=dev).float()
            y_te = torch.load(f"{p}_{args.test_split}_l.pt", map_location=dev)
            pool_f = torch.load(f"{p}_pool_f.pt", map_location=dev).float()
            pool_y = torch.load(f"{p}_pool_l.pt", map_location=dev)
            W = clip_classifier(classnames, template, model, device=dev).float()
            clip_val, clip_te = 100.0 * f_val @ W, 100.0 * f_te @ W
            zs = acc(clip_te, y_te)
            n_cls = int(pool_y.max()) + 1

            for Ksh in args.ks:
                arms = {}
                for s in args.seeds:
                    idx = draw(pool_y.cpu(), Ksh, s, dev)
                    f_tr, y_tr = pool_f[idx], pool_y[idx]
                    clip_tr = 100.0 * f_tr @ W

                    # `full` = every shot as a key (Tip-Adapter's cache).
                    # `km1`  = one prototype per class (Proto-Adapter's cache).
                    full = build_calibrated(f_tr, y_tr, n_cls,
                                            lambda X: F.normalize(X, dim=-1),
                                            shots=Ksh)
                    km1 = build_calibrated(f_tr, y_tr, n_cls,
                                           lambda X: kmeans(X, 1, seed=s),
                                           shots=Ksh)

                    for label, (kb, Vb) in (("TIP", full),):
                        b, a = search(betas, alphas, f_val @ kb, Vb, clip_val, y_val)
                        arms.setdefault(label, []).append(
                            acc_chunked(kb, Vb, f_te, y_te, clip_te, b, a))

                    for label, (kb, Vb) in (("TIP-F", full),
                                            ("CC-F-1", km1)):
                        r, nrm, tr = ft_arm_calibrated(
                            kb, Vb, f_tr, y_tr, clip_tr, f_val, y_val, f_te, y_te,
                            clip_val, clip_te, betas, alphas, epochs=args.epochs)
                        arms.setdefault(label, []).append(r)
                        arms.setdefault(f"norm::{label}", []).append(nrm)
                        arms.setdefault(f"train::{label}", []).append(tr)
                    del full, km1
                    torch.cuda.empty_cache()

                means = {k: mean(v) for k, v in arms.items()}
                stds = {k: pstdev(v) for k, v in arms.items()}
                # PER-SEED VALUES ARE NOW KEPT. Collapsing them here is what
                # left Tip-Adapter, Tip-Adapter-F and Proto-Adapter-F without a
                # +- in the main table: the paper's uncertainty is sd of the
                # 11-DATASET MEAN over seeds, which cannot be reconstructed
                # from per-cell sds (measured: the independence estimator is
                # off by 0.21x-1.63x on the arms where the truth is known).
                # Strict addition -- no reported number changes.
                cells.append({"backbone": bb, "dataset": name, "zs": zs,
                              "test_split": args.test_split,
                              "K": Ksh, "epochs": args.epochs,
                              "seeds": list(args.seeds),
                              "per_seed": {k: list(v) for k, v in arms.items()},
                              "C": means["TIP"] - zs, "acc": means, "std": stds})
                print(f"{bb:<10}{name:<15}C={means['TIP'] - zs:+6.2f}  "
                      f"TIP={means['TIP']:5.2f}  "
                      f"TIP-F={means['TIP-F']:5.2f}  "
                      f"CC-F-1={means['CC-F-1']:5.2f}"
                      f"({means['CC-F-1'] - means['TIP-F']:+.2f})  "
                      f"|k|: TIPF={means['norm::TIP-F']:.2f} "
                      f"CCF1={means['norm::CC-F-1']:.2f}", flush=True)
                json.dump(cells, open(args.out, "w"), indent=2)
        del model
        torch.cuda.empty_cache()

    json.dump(cells, open(args.out, "w"), indent=2)


# The per-arm scoring that used to live here belonged to a separate study and
# was removed: this module's job is to produce the baseline cells and write
# them to --out. The paper's tables are built from that JSON by
# make_le_gda_head_to_head.py and make_backbone_main_table.py.


if __name__ == "__main__":
    main()
