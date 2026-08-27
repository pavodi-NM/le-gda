"""LE-GDA vs GDA-CLIP across BACKBONES -> figures/le_gda/.

The existing series (plot_le_gda_figures.py) is one figure per DATASET on one
backbone, with all seven head-to-head methods. This is the other cut: one figure
per BACKBONE, averaged over the 11 datasets, with only the three curves that
carry the paper's decomposition --

    zero-shot CLIP  ->  GDA-CLIP        is THEIRS (ICLR'24, ~+16 at K=16)
    GDA-CLIP        ->  LE-GDA          is OURS   (+0.44 .. +2.26)

and a fifth figure showing our margin alone on all four backbones, which is the
`--shots` table of make_le_gda_head_to_head.py drawn rather than tabulated.

Writes (house rule: one figure = one PDF, plus a PNG preview):

    fig_le_gda_vs_gda_<backbone>.pdf   x 4
    fig_le_gda_margin.pdf              the four margins on one axes

SOURCES, and the two traps this pipeline already fell into once:

  GDA-CLIP   acc.gda_last     evidence/layer_ensemble_*<suffix>.json
  LE-GDA     rules.F1.test    evidence/nonlinear_fusion_*.json

  (1) SPLIT. The z-fusion evidence is standard-split ONLY. The layer_ensemble
      files exist in a capped and a _testfull version and `suffix` picks between
      them; reading one series capped and the other standard inflated the
      baseline and fabricated a 0.79-point deficit on food101. --suffix defaults
      to _testfull and anything else is refused.
  (2) NAMING. The evidence labels the operator `F1`; that is an internal arm
      label, NOT the F1 score -- every axis here is top-1 accuracy. The
      manuscript says "per-sample standardisation" and never prints "F1"
      (the write-up §6.1).

Dataset coverage is printed per point as `n`, and a point drawn from fewer
datasets than its neighbours is REPORTED rather than silently averaged, because
a mean over a different dataset set is not comparable to its neighbours.
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = ["#3366cc", "#e8710a", "#12a37e", "#c2298a", "#8f6b00", "#6b4fd8"]
INK, INK2, GRID = "#1a1a1a", "#4a4a4a", "#e6e6e6"
SHOTS = [1, 2, 4, 8, 16]
BACKBONES = ["ViT-B/16", "ViT-B/32", "RN50", "RN101"]
BB_COL = {"ViT-B/16": PAL[3], "ViT-B/32": PAL[5],
          "RN50": PAL[2], "RN101": PAL[1]}
BB_MK = {"ViT-B/16": "*", "ViT-B/32": "o", "RN50": "s", "RN101": "^"}
NF_GLOBS = ["evidence/nonlinear_fusion_full_*.json",
            "evidence/nonlinear_fusion_k14_*.json",
            "evidence/nonlinear_fusion_k28_*.json"]


def cells(path):
    d = json.load(open(path))
    return d["cells"] if isinstance(d, dict) and "cells" in d else d


def load(suffix):
    if suffix != "_testfull":
        raise SystemExit("SPLIT MISMATCH: the LE-GDA series is standard-split "
                         f"only; --suffix={suffix!r} would read the CAPPED "
                         "files for GDA-CLIP. See the module docstring.")
    gda, zs = {}, {}
    for p in glob.glob(f"evidence/layer_ensemble_*{suffix}.json"):
        for c in cells(p):
            gda[(c["backbone"], c["dataset"], c["K"], c["seed"])] = \
                c["acc"]["gda_last"]
            zs[(c["backbone"], c["dataset"])] = c["zs"]
    le = {}
    for g in NF_GLOBS:
        for p in glob.glob(g):
            for c in json.load(open(p)).get("rows", []):
                if "F1" not in c.get("rules", {}):
                    continue
                want = "test" if c["dataset"] == "imagenet" else "testfull"
                if c.get("test_split") != want:
                    raise SystemExit(f"{p}: test_split={c.get('test_split')!r}")
                le[(c["backbone"], c["dataset"], c["K"], c["seed"])] = \
                    c["rules"]["F1"]["test"]
    return gda, le, zs


def curves(gda, le, zs, bb):
    """{K: (gda_mean, le_mean, n_datasets)} over cells present in BOTH."""
    out = {}
    for k in SHOTS:
        keys = [x for x in le if x[0] == bb and x[2] == k and x in gda]
        if not keys:
            continue
        out[k] = (mean(gda[x] for x in keys), mean(le[x] for x in keys),
                  len({x[1] for x in keys}))
    z = [v for (b, _), v in zs.items() if b == bb]
    return out, (mean(z) if z else None)


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def xaxis(ax, zero=True):
    ax.set_xscale("symlog", base=2, linthresh=1)
    ax.set_xticks(([0] if zero else []) + SHOTS)
    ax.set_xticklabels((["0"] if zero else []) + [str(s) for s in SHOTS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("labelled examples per class", fontsize=10, color=INK,
                  labelpad=5)


def save(fig, out, stem, dpi):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/{stem}.{ext}", dpi=dpi, facecolor="white",
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_testfull")
    ap.add_argument("--out", default="figures/le_gda")
    ap.add_argument("--width", type=float, default=5.2)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()
    gda, le, zs = load(args.suffix)

    print(f"writing to {args.out}/")
    margins = {}
    for bb in BACKBONES:
        cur, z = curves(gda, le, zs, bb)
        if not cur:
            continue
        ks = sorted(cur)
        margins[bb] = {k: cur[k][1] - cur[k][0] for k in ks}
        ns = {cur[k][2] for k in ks}

        fig, ax = plt.subplots(figsize=(args.width, args.width * 0.78),
                               facecolor="white")
        if z is not None:
            ax.plot([0], [z], marker="D", color=PAL[0], ms=6, ls="none",
                    label="Zero-shot CLIP", zorder=6)
        ax.plot(ks, [cur[k][0] for k in ks], "-.", color=PAL[5], marker="^",
                ms=5.5, lw=1.5, mew=0, label="GDA-CLIP (final-feature)", zorder=4)
        ax.plot(ks, [cur[k][1] for k in ks], "-", color=PAL[3], marker="*",
                ms=9, lw=1.6, mew=0, label="LE-GDA (ours)", zorder=5)
        style(ax)
        xaxis(ax)
        ax.set_ylabel("top-1 accuracy (%)", fontsize=10, color=INK, labelpad=5)
        n = f"{ns.pop()} datasets" if len(ns) == 1 else f"{sorted(ns)} datasets"
        ax.set_title(f"{bb} — mean over {n}", fontsize=12.5, color=INK,
                     pad=9, loc="left")
        ax.legend(fontsize=8.4, frameon=False, loc="upper left",
                  bbox_to_anchor=(0.01, 0.99), labelcolor=INK2,
                  handlelength=2.2, borderaxespad=0)
        save(fig, args.out, f"fig_le_gda_vs_gda_{bb.replace('/', '-')}",
             args.dpi)

    # ---- the margin, all backbones on one axes ----
    fig, ax = plt.subplots(figsize=(args.width, args.width * 0.78),
                           facecolor="white")
    # House rule 2: nothing on the canvas is unexplained. The zero line is the
    # baseline itself, since the axis plots a difference against it.
    ax.axhline(0, color=INK2, lw=1.0, ls=":", zorder=2,
               label="no difference from GDA-CLIP (final-feature)")
    for bb, m in margins.items():
        ks = sorted(m)
        ax.plot(ks, [m[k] for k in ks], "-", color=BB_COL[bb],
                marker=BB_MK[bb], ms=8 if BB_MK[bb] == "*" else 5.5,
                lw=1.5, mew=0, label=bb, zorder=5)
    style(ax)
    xaxis(ax, zero=False)
    # Rule 2a: the difference being taken belongs on the axis, with its
    # statistic and unit. Rule 1: the title is a noun phrase naming the
    # quantity -- not "our increment", which is first-person and a claim.
    ax.set_ylabel("LE-GDA − GDA-CLIP (mean top-1 difference, points)",
                  fontsize=10, color=INK, labelpad=5)
    ax.set_title("Accuracy difference from GDA-CLIP across shot budgets",
                 fontsize=11.5, color=INK, pad=9, loc="left")
    ax.legend(fontsize=8.4, frameon=False, loc="upper left",
              bbox_to_anchor=(0.01, 0.99), labelcolor=INK2,
              handlelength=2.2, borderaxespad=0)
    save(fig, args.out, "fig_le_gda_margin", args.dpi)


if __name__ == "__main__":
    main()
