"""Accuracy versus shots figures -> figures/le_gda/, one PDF (+ PNG) each.

Writes 12 figures for one backbone (default ViT-B/16): one per dataset and the
average over datasets.

    fig_le_gda_shots_average.pdf
    fig_le_gda_shots_<dataset>.pdf   x 11

Series and their sources (K in {1, 2, 4, 8, 16}; zero-shot at K=0):

  Zero-shot CLIP     zs                  layer_ensemble_*.json
  Tip-Adapter        TIP                 runB/, runB_shots/   (original code)
  Tip-Adapter-F      TIP-F               runB/, runB_shots/   (original code)
  Proto-Adapter      PROTO               proto_adapter/ours_*.json
  Proto-Adapter-F    PROTO-F             proto_adapter/ours_*.json
  GPCache            gpcache16           baselines_e4_*.json
  GDA-CLIP           gda_last            layer_ensemble_*.json
  LE-GDA (ours)      rules.F1.test       nonlinear_fusion_*.json

LE-GDA is the per-query standardised fusion (rule key "F1", a label, not the
F1 score). At K < 16, GPCache uses all K shots as keys, like Tip-Adapter.

All sources use the same cached features, so their zero-shot accuracies must
agree; --strict stops on a disagreement (beyond ~0.02 on ImageNet, from the
fp16 text-classifier rebuild) or on a missing point.
"""

import argparse
import glob
import json
import os
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = ["#3366cc", "#e8710a", "#12a37e", "#c2298a", "#8f6b00", "#6b4fd8"]
INK, INK2 = "#1a1a1a", "#4a4a4a"
GRID = "#e6e6e6"

SHOTS = [1, 2, 4, 8, 16]
NICE = {"caltech101": "Caltech101", "dtd": "DTD", "eurosat": "EuroSAT",
        "fgvc": "FGVCAircraft", "food101": "Food101", "imagenet": "ImageNet",
        "oxford_flowers": "Flowers102", "oxford_pets": "OxfordPets",
        "stanford_cars": "StanfordCars", "sun397": "SUN397",
        "ucf101": "UCF101"}
ORDER = ["imagenet", "caltech101", "oxford_pets", "stanford_cars",
         "oxford_flowers", "food101", "fgvc", "sun397", "dtd", "eurosat",
         "ucf101"]

LE_FILES = ["vitb16", "shots_b16", "imagenet_b16", "shots_in_b16"]
# LE-GDA: K=16 in full_*, K in {1,4} in k14_*, K in {2,8} in k28_*
NF_GLOBS = ["evidence/nonlinear_fusion_full_*.json",
            "evidence/nonlinear_fusion_k14_*.json",
            "evidence/nonlinear_fusion_k28_*.json"]
# Tip-Adapter (original code, our protocol): K=16 in runB/, K in {1,2,4,8} in
# runB_shots/. Keep K<16 files out of runB/: rebuild_backbone_main.py reads
# every file there as K=16.
RUNB_GLOBS = ["evidence/runB/runB_*.json",
              "evidence/runB_shots/runB_*.json"]
# Proto-Adapter: K=16 in ours_<tag>.json, the sweep in ours_<tag>_K<k>.json
PROTO_GLOBS = ["evidence/proto_adapter/ours_*.json"]
# GPCache (the --suffix is appended to each name)
E4_FILES = ["evidence/baselines_e4.json", "evidence/baselines_e4_phaseA.json",
            "evidence/baselines_e4_shots_b16.json"]

ZS = "Zero-shot CLIP"
# (label, colour, marker, linestyle). Drawing order == legend order.
# Linestyle: dashed = gradient-trained, dash-dot = GDA-CLIP, solid = LE-GDA,
# dotted = the other training-free methods. Colour and marker identify the
# method; the two Proto-Adapter variants share a grey.
SERIES = [
    (ZS, PAL[0], "D", "none"),
    ("Tip-Adapter", PAL[2], "o", ":"),
    ("Proto-Adapter", "#adadad", "X", ":"),
    ("GPCache (W+C)", PAL[4], "v", ":"),
    ("Tip-Adapter-F", PAL[1], "s", "--"),
    ("Proto-Adapter-F", "#7a7a7a", "P", "--"),
    ("GDA-CLIP (final-feature)", PAL[5], "^", "-."),
    ("LE-GDA (ours)", PAL[3], "*", "-"),
]
IMAGENET_ZS_TOL = 0.05


def cells(path):
    """Row list, whatever the file calls it.

    `layer_ensemble`/`baselines_e4` write `cells`; `runB` and `proto_adapter`
    go through `protocol_adapter.report`, which writes
    `{"meta": ..., "rows": [...]}`.
    """
    d = json.load(open(path))
    if isinstance(d, dict):
        for k in ("cells", "rows"):
            if k in d:
                return d[k]
    return d


def collect(backbone, strict, suffix=""):
    """{dataset: {series: {K: acc}}}, {dataset: zero-shot}."""
    out, zs, zs_alt = {}, {}, {}

    def sfx(p):
        return p[:-5] + suffix + ".json" if suffix else p

    def add(ds, lab, k, v):
        out.setdefault(ds, {}).setdefault(lab, {}).setdefault(k, []).append(v)

    for f in LE_FILES:
        p = sfx(f"evidence/layer_ensemble_{f}.json")
        if not os.path.exists(p):
            continue
        for c in cells(p):
            if c["backbone"] != backbone:
                continue
            zs[c["dataset"]] = c["zs"]
            add(c["dataset"], "GDA-CLIP (final-feature)", c["K"],
                c["acc"]["gda_last"])
    # LE-GDA: per-query standardised fusion (rule key "F1")
    seen_nf = 0
    for g in NF_GLOBS:
        for p in sorted(glob.glob(g)):   # standard split only: no suffix
            for c in json.load(open(p)).get("rows", []):
                if c["backbone"] != backbone or "F1" not in c.get("rules", {}):
                    continue
                # LE-GDA results are on the standard split ("testfull";
                # ImageNet "test"), so every other series must be too.
                want = "test" if c["dataset"] == "imagenet" else "testfull"
                if c.get("test_split") != want:
                    raise SystemExit(
                        f"{p}: test_split={c.get('test_split')!r}, expected "
                        f"{want!r} for {c['dataset']}")
                if suffix != "_testfull":
                    raise SystemExit(
                        "the LE-GDA series is on the standard test split; pass "
                        "--suffix _testfull (the default)")
                add(c["dataset"], "LE-GDA (ours)", c["K"],
                    c["rules"]["F1"]["test"])
                seen_nf += 1
    if not seen_nf and strict:
        raise SystemExit(f"no LE-GDA results found for {backbone} in "
                         "evidence/nonlinear_fusion_*")
    # Tip-Adapter and Proto-Adapter: standard split only, no suffix
    for g in RUNB_GLOBS + PROTO_GLOBS:
        for p in sorted(glob.glob(g)):
            proto = "proto_adapter" in p
            for r in cells(p):
                if r["backbone"] != backbone:
                    continue
                if r.get("test_split") != "testfull":
                    raise SystemExit(
                        f"{p}: test_split={r.get('test_split')!r}, expected "
                        "'testfull'")
                if proto:
                    zs_alt.setdefault(r["dataset"], r["zs"])
                for arm, lab in (("TIP", "Tip-Adapter"),
                                 ("TIP-F", "Tip-Adapter-F"),
                                 ("PROTO", "Proto-Adapter"),
                                 ("PROTO-F", "Proto-Adapter-F")):
                    if arm in r.get("per_seed", {}):
                        for v in r["per_seed"][arm]:
                            add(r["dataset"], lab, r["K"], v)
    for p in map(sfx, E4_FILES):
        if not os.path.exists(p):
            continue
        for c in cells(p):
            if c["backbone"] != backbone or "gpcache16" not in c["acc"]:
                continue
            zs_alt.setdefault(c["dataset"], c["zs"])
            add(c["dataset"], "GPCache (W+C)", c.get("K", 16), c["acc"]["gpcache16"])

    bad = [(d, zs[d], zs_alt[d]) for d in zs if d in zs_alt
           and abs(zs[d] - zs_alt[d]) > (IMAGENET_ZS_TOL if d == "imagenet"
                                         else 1e-6)]
    for d, a, b in bad:
        print(f"  MISALIGNED {d}: zero-shot {a:.4f} vs {b:.4f}")
    if bad and strict:
        raise SystemExit("zero-shot disagrees between sources -> not the same "
                         "splits; refusing to plot them together")

    for d in out.values():                       # seeds -> mean
        for lab in d:
            d[lab] = {k: mean(v) for k, v in d[lab].items()}
    return out, zs


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def one_figure(name, data, zs, out, width, dpi, subtitle=None):
    fig, ax = plt.subplots(figsize=(width, width * 0.78), facecolor="white")
    for lab, col, mk, ls in SERIES:
        if lab == ZS:
            if zs is None:
                continue
            ax.plot([0], [zs], marker=mk, color=col, ms=6, ls="none",
                    label=lab, zorder=6)
            continue
        pts = sorted((k, v) for k, v in data.get(lab, {}).items()
                     if k in SHOTS)
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=col,
                marker=mk, ms=5.5 if mk != "*" else 9, lw=1.5, mew=0,
                label=lab, zorder=5 if lab.endswith("(ours)") else 4)
    style(ax)
    ax.set_xscale("symlog", base=2, linthresh=1)
    ax.set_xticks([0] + SHOTS)
    ax.set_xticklabels(["0"] + [str(s) for s in SHOTS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("labelled examples per class", fontsize=10, color=INK,
                  labelpad=5)
    ax.set_ylabel("top-1 accuracy (%)", fontsize=10, color=INK, labelpad=5)
    ax.set_title(subtitle or NICE.get(name, name), fontsize=12.5, color=INK,
                 pad=9, loc="left")
    # fixed upper-left legend: the curves rise with K, so that corner is empty,
    # and a fixed place keeps the twelve figures consistent
    ax.legend(fontsize=7.6, frameon=False, loc="upper left",
              handlelength=2.0, labelspacing=0.30, borderpad=0.2)
    fig.tight_layout()
    stem = f"fig_le_gda_shots_{name}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/{stem}.{ext}", dpi=dpi, facecolor="white")
    plt.close(fig)
    return stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="ViT-B/16")
    ap.add_argument("--out", default="figures/le_gda")
    ap.add_argument("--width", type=float, default=4.6)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--strict", action="store_true",
                    help="fail if any series is missing a shot, or if the "
                         "sources disagree on zero-shot")
    ap.add_argument("--suffix", default="_testfull",
                    help="evidence-file suffix; _testfull = the standard "
                         "test splits")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data, zs = collect(args.backbone, args.strict, args.suffix)
    have = [d for d in ORDER if d in data]

    # ---- completeness, reported per (dataset, series) -----------------------
    want = {lab: SHOTS for lab, _, _, _ in SERIES[1:]}
    gaps = []
    for d in have:
        for lab, ks in want.items():
            miss = [k for k in ks if k not in data[d].get(lab, {})]
            if miss:
                gaps.append((d, lab, miss))
    if gaps:
        print("INCOMPLETE — these (dataset, method) pairs are missing shots:")
        for d, lab, miss in gaps:
            print(f"  {d:<16}{lab:<18}missing K={miss}")
        if args.strict:
            raise SystemExit("refusing to plot an incomplete sweep "
                             "(drop --strict to render anyway)")
    else:
        print("complete: every method has every shot on every dataset")

    # ---- the average: one fixed dataset set for every point ----------------
    avg_ds = [d for d in have
              if all(k in data[d].get(lab, {})
                     for lab, ks in want.items() for k in ks)]
    dropped = [d for d in have if d not in avg_ds]
    if dropped:
        print(f"  average excludes {dropped} (a series is missing a shot)")
    written = []
    if avg_ds:
        avg = {lab: {k: mean(data[d][lab][k] for d in avg_ds) for k in ks}
               for lab, ks in want.items()}
        written.append(one_figure(
            "average", avg, mean(zs[d] for d in avg_ds), args.out,
            args.width, args.dpi,
            subtitle=f"{args.backbone} - Average over {len(avg_ds)} datasets"))
    for d in have:
        # the backbone leads the title, as in the average panel
        written.append(one_figure(d, data[d], zs.get(d), args.out, args.width,
                                  args.dpi,
                                  subtitle=f"{args.backbone} - {NICE.get(d, d)}"))
    print(f"\nwrote {len(written)} figures to {args.out}/ "
          f"(.pdf + .png each) [{args.backbone}]")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
