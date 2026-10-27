"""Per-dataset greedy selection and standalone accuracy, one figure per backbone.

Two heatmaps sharing the dataset rows:
  (a)  times greedy selection picks each candidate space, summed over the
       3 seeds (K=16; up to 4 picks per run, so 0..12);
  (b)  standalone top-1 of the GDA classifier fitted in that space alone,
       minus the one fitted in the final embedding, averaged over seeds
       (colour scale clipped at +-3 points, --clip).
Rows are the 11 datasets ordered by LE-GDA's gain over GDA-CLIP averaged over
the four backbones (Table 2 of the paper), largest first, so the order is the
same in all four figures.

Selection is LE-GDA's (per-query standardised fusion): `rules.F1.subset` in
evidence/nonlinear_fusion_full_*.json. Standalone accuracies and `gda_last`
come from the evidence/layer_ensemble_*_testfull.json cells; LE-GDA's accuracy
is `rules.F1.test`. The two sources are joined cell by cell.

The script also prints, per dataset, the share of the selected multiset that
is not the final embedding and its Spearman correlation with the gain.

    PYTHONPATH=. python scripts/plot_le_gda_selection_heatmap.py
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

INK, INK2 = "#1a1a1a", "#4a4a4a"

LE_FILES = ["vitb16", "vitb32", "imagenet_b16", "imagenet_b32",
            "RN50", "RN50_imagenet", "RN101", "RN101_imagenet"]
NF_GLOB = "evidence/nonlinear_fusion_full_*.json"

BACKBONES = ["RN50", "RN101", "ViT-B/32", "ViT-B/16"]
VIT = ["L4", "L6", "L8", "L10", "L11", "pre", "last"]
RN = ["R1", "R2", "R3", "R4", "last"]
MAX_PICKS = 4           # greedy runs T=4 steps, so one space is picked <= 4x

# plain names on the canvas
NICE_SPACE = {"L4": "block 4", "L6": "block 6", "L8": "block 8",
              "L10": "block 10", "L11": "block 11", "pre": "pre-proj.",
              "last": "final", "R1": "stage 1", "R2": "stage 2",
              "R3": "stage 3", "R4": "stage 4"}
NICE_DS = {"caltech101": "Caltech101", "dtd": "DTD", "eurosat": "EuroSAT",
           "fgvc": "FGVCAircraft", "food101": "Food101", "imagenet": "ImageNet",
           "oxford_flowers": "Flowers102", "oxford_pets": "OxfordPets",
           "stanford_cars": "StanfordCars", "sun397": "SUN397",
           "ucf101": "UCF101"}


# Table 2 of the paper: K=16 gain over GDA-CLIP averaged over the four
# backbones, printed in the row labels. main() checks each against the gain
# recomputed from the evidence files.
TABLE2_GAIN = {"caltech101": 1.13, "dtd": 0.68, "eurosat": 4.86, "fgvc": 4.69,
               "food101": -0.03, "imagenet": 0.16, "oxford_flowers": 1.20,
               "oxford_pets": -0.16, "stanford_cars": 1.00, "sun397": 0.90,
               "ucf101": 2.33}


def spaces(bb):
    return VIT if bb.startswith("ViT") else RN


def load(k):
    cells = {}
    for tag in LE_FILES:
        for c in json.load(open(f"evidence/layer_ensemble_{tag}_testfull.json"))["cells"]:
            if c["K"] == k:
                key = (c["backbone"], c["dataset"], c["seed"])
                assert key not in cells, f"duplicate cell {key}"
                cells[key] = c
    rows = {}
    for fp in sorted(glob.glob(NF_GLOB)):
        for r in json.load(open(fp))["rows"]:
            if r["K"] == k:
                key = (r["backbone"], r["dataset"], r["seed"])
                assert key not in rows, f"duplicate fusion row {key}"
                rows[key] = r
    assert set(cells) == set(rows), "fusion rows and layer_ensemble cells differ"
    assert len(rows) == 132, f"{len(rows)} cells at K={k}, expected 132"
    return cells, rows


def measure(cells, rows):
    """Per backbone: picks-per-run and solo-difference matrices, and gains."""
    out, gain = {}, defaultdict(dict)
    datasets = sorted({k[1] for k in rows})
    for bb in BACKBONES:
        sp = spaces(bb)
        picks = np.zeros((len(datasets), len(sp)))
        solo = np.zeros((len(datasets), len(sp)))
        for i, ds in enumerate(datasets):
            keys = [k for k in rows if k[0] == bb and k[1] == ds]
            assert len(keys) == 3, f"{bb}/{ds}: {len(keys)} seeds"
            cnt = Counter(s for k in keys for s in rows[k]["rules"]["F1"]["subset"])
            assert set(cnt) <= set(sp), f"{bb}/{ds}: unknown space {set(cnt) - set(sp)}"
            picks[i] = [cnt[s] / len(keys) for s in sp]
            solo[i] = [mean(cells[k]["acc"][f"gda_{s}"] - cells[k]["acc"]["gda_last"]
                            for k in keys) for s in sp]
            gain[bb][ds] = mean(rows[k]["rules"]["F1"]["test"] - cells[k]["acc"]["gda_last"]
                                for k in keys)
        out[bb] = (datasets, picks, solo)
    return out, gain


def annotate(ax, mat, im, fmt, skip_zero=False, skip_col=None):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if (skip_zero and v == 0) or j == skip_col:
                continue
            v = v + 0.0 if round(v, 1) != 0 else 0.0     # no "-0.0"
            r, g, b, _ = im.cmap(im.norm(v))
            dark = 0.299 * r + 0.587 * g + 0.114 * b < 0.5
            # values >= 10 use 6 pt in the 7-column ViT panels so they fit
            small = abs(v) >= 9.95 and mat.shape[1] > 5
            ax.text(j, i, format(v, fmt), ha="center", va="center",
                    fontsize=6 if small else 7,
                    color="white" if dark else INK)


def frame(ax, xt, yt):
    ax.set_xticks(range(len(xt)), xt, rotation=40, ha="right", fontsize=8,
                  color=INK2)
    ax.set_yticks(range(len(yt)), yt, fontsize=8, color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def draw(bb, datasets, picks, solo, order, pooled, out, width, dpi, lim):
    idx = [datasets.index(d) for d in order]
    # totals over the 3 seeds
    picks, solo = np.rint(picks[idx] * 3), solo[idx]
    xt = [NICE_SPACE[s] for s in spaces(bb)]
    # rows are ordered by this gain, shown next to the name
    yt = [f"{NICE_DS[d]} ({pooled[d]:+.2f})" for d in order]

    fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.56),
                             facecolor="white", sharey=True)
    ax = axes[0]
    # purple here, so blue only means "better than final" in panel (b)
    im = ax.imshow(picks, cmap="Purples", vmin=0, vmax=3 * MAX_PICKS,
                   aspect="auto")
    annotate(ax, picks, im, ".0f", skip_zero=True)
    frame(ax, xt, yt)
    ax.set_title("(a) Greedy selection", fontsize=9.5, color=INK, loc="left")
    # backbone name in the corner
    ax.text(-0.02, 1.0, bb, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, color=INK, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03,
                      ticks=range(0, 3 * MAX_PICKS + 1, 3))
    cb.set_label("times selected", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7.5, colors=INK2)
    cb.outline.set_visible(False)

    ax = axes[1]
    # clipped so differences of a few points stay visible; large deficits
    # saturate
    im = ax.imshow(solo, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    j = spaces(bb).index("last")
    annotate(ax, solo, im, "+.1f", skip_col=j)
    for i in range(solo.shape[0]):
        ax.text(j, i, "ref.", ha="center", va="center", fontsize=7,
                color=INK2, style="italic")
    frame(ax, xt, yt)
    ax.tick_params(labelleft=False)
    ax.set_title("(b) Standalone accuracy", fontsize=9.5, color=INK, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, extend="both")
    cb.set_label("top-1 difference from final (points)", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7.5, colors=INK2)
    cb.outline.set_visible(False)

    fig.tight_layout()
    stem = f"fig_le_gda_selection_heatmap_{bb.replace('/', '-')}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/{stem}.{ext}", dpi=dpi, facecolor="white",
                    bbox_inches="tight")
    plt.close(fig)
    return stem


def report(m, gain, order):
    print("\nshare of the selected multiset that is not the final embedding")
    print(f"  {'dataset':<14}" + "".join(f"{bb:>18}" for bb in BACKBONES))
    shares = defaultdict(dict)
    for bb in BACKBONES:
        datasets, picks, _ = m[bb]
        j = spaces(bb).index("last")
        for i, ds in enumerate(datasets):
            shares[bb][ds] = 1 - picks[i, j] / picks[i].sum()
    for ds in order:
        print(f"  {ds:<14}" + "".join(
            f"   {shares[bb][ds]:5.2f} ({gain[bb][ds]:+5.2f})" for bb in BACKBONES))
    print("\nSpearman(non-final share, gain over GDA-CLIP), n = 11 datasets")
    for bb in BACKBONES:
        ds = sorted(shares[bb])
        rho, p = spearmanr([shares[bb][d] for d in ds], [gain[bb][d] for d in ds])
        print(f"  {bb:<10} rho {rho:+.3f}  p {p:.4f}")
    a = [shares[bb][d] for bb in BACKBONES for d in order]
    b = [gain[bb][d] for bb in BACKBONES for d in order]
    rho, p = spearmanr(a, b)
    print(f"  pooled 44  rho {rho:+.3f}  p {p:.2e}  (settings, not independent)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/le_gda")
    ap.add_argument("--width", type=float, default=6.875)   # WACV \textwidth
    ap.add_argument("--clip", type=float, default=3.0,
                    help="colour limit of panel (b), points")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--only", choices=BACKBONES, help="draw one backbone")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cells, rows = load(args.k)
    m, gain = measure(cells, rows)
    pooled = {d: mean(gain[bb][d] for bb in BACKBONES) for d in gain["RN50"]}
    off = {d: pooled[d] - TABLE2_GAIN[d] for d in pooled}
    assert all(abs(v) <= 0.015 for v in off.values()), \
        f"evidence disagrees with Table 2 beyond rounding: {off}"
    order = sorted(TABLE2_GAIN, key=lambda d: -TABLE2_GAIN[d])
    assert order == sorted(pooled, key=lambda d: -pooled[d]), \
        "Table 2 and the evidence give different row orders"
    for bb in ([args.only] if args.only else BACKBONES):
        stem = draw(bb, *m[bb], order, TABLE2_GAIN, args.out, args.width, args.dpi,
                    args.clip)
        print(f"wrote {args.out}/{stem}.pdf (+ .png)")
    print("\npooled gain over GDA-CLIP (row order): "
          + ", ".join(f"{d} {pooled[d]:+.2f}" for d in order))
    report(m, gain, order)


if __name__ == "__main__":
    main()
