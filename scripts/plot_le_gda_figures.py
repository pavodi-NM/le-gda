"""LE-GDA paper figures -> figures/le_gda/. ONE PDF (+ PNG preview) PER FIGURE.

Writes 12 figures: one per dataset, plus the average over datasets.

    fig_le_gda_shots_average.pdf
    fig_le_gda_shots_<dataset>.pdf   x 11

House style: scripts/plot_paper_figures.py and memory/paper-figure-style.md.
Rule 6 -- one figure = one PDF -- is why these are separate files and not a
3x4 grid: "this is a paper, not a draft".

SERIES, and the single source of each. Every method in the sec.5b.6
head-to-head appears, at every shot count it is defined at:

  Zero-shot CLIP          K=0 marker            layer_ensemble_*.json (`zs`)
  Tip-Adapter             TIP        ORIGINAL   runB/, runB_shots/
  Proto-Adapter           PROTO                 proto_adapter/ours_*.json
  GPCache (16 keys)       gpcache16             baselines_e4_*.json
  Tip-Adapter-F           TIP-F      ORIGINAL   runB/, runB_shots/
  Proto-Adapter-F         PROTO-F    TRAINED    proto_adapter/ours_*.json
  GDA-CLIP                gda_last              layer_ensemble_*.json
  LE-GDA (ours)           rules.F1.test         nonlinear_fusion_*.json

RETIRED 2026-08-26 (the README, Comparative methods under our protocol): the
first four rows used to be read out of `calibrated_ft_*.json`, and three of
them were wrong.

  * `Tip-Adapter`/`Tip-Adapter-F` were OUR reimplementation. TIP is faithful to
    +0.04, but TIP-F ran on our frozen-feature trainer, worth **-2.30**
    grid-wide (the shot-sweep measurement) -- the single biggest defect in the old figure.
  * `Proto-Adapter-F` was our `CC-F-1`, which is NOT Proto-Adapter-F: plain CE
    instead of ArcFace, no channel-wise normalization, a different calibration
    rule, and the weak trainer again (the Proto-Adapter design note).
  * There was no training-free `Proto-Adapter` series at all.

They now come from the same evidence `rebuild_backbone_main.py` builds
the per-backbone main table from, so the figure and the table can no longer disagree
about the same row. `calibrated_ft_*.json` is still read, but ONLY for its
zero-shot values, which feed the alignment assert below.

IMAGENET'S SPLIT LABEL DIFFERS BETWEEN SOURCES AND THE IMAGES DO NOT. The
layer_ensemble / nonlinear_fusion files label it `test`; runB and
proto_adapter label it `testfull`. Both are the same 40 000-image set: the
zero-shot accuracies are bit-identical (67.84499883651733 from either), and
40000*zs is an integer while 30000*zs is not. `test` there is a
cached-tensor namespace, not `protocol_adapter.eval_items(...,"test")`, which
caps at 30/class. The alignment assert is what actually guards this -- the
labels are checked per source, the numbers are checked against each other.

LE-GDA IS PLOTTED UNDER PER-SAMPLE STANDARDISATION, NOT THE UNIFORM MEAN.
the write-up §5e supersedes the fusion rule: each space's score vector is
standardised across the CLASS axis for every test sample before averaging, which
takes KC-S1 from +0.534 to +1.136. The evidence files label that operator `F1`
and the old uniform mean `F0` -- internal arm labels only, NOT the F1 score.
Everything plotted here is top-1 accuracy. See the write-up §6.1:
the manuscript must never print the token "F1"; the operator is named
"per-sample standardisation" in prose.

THERE IS NO FALLBACK TO THE UNIFORM MEAN. If a (dataset, K) has no
per-sample-standardised number, the point is DROPPED and --strict fails. Mixing
the two fusion rules inside one curve is forbidden by the direction file, and a
silent fallback is exactly how that would happen.

The trained arms come from run_calibrated_ft at a uniform 20 epochs and uniform
size-calibrated values at EVERY K -- not from run_shot_sweep, which trains CC-F
at m = K/4 and therefore cannot produce K=1 or K=2 at all.

At K < 16 "GPCache (16 keys)" uses all K shots as keys, because build_keys
falls back to the full support set when m >= K. That is the same convention
Tip-Adapter follows, so the two remain comparable at every K.

ALIGNMENT IS ASSERTED: all three sources are fitted on the same
`caches/<ds>/collapse/<tag>` tensors, so zero-shot must agree. --strict fails
the build on any disagreement beyond imagenet's fp16 text-encoder rebuild.
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
# LE-GDA under per-sample standardisation. Globs, because the shot counts were
# run in separate passes: K=16 (full_*), K in {1,4} (k14_*), K in {2,8} (k28_*),
# with imagenet always in its own file on its own split.
NF_GLOBS = ["evidence/nonlinear_fusion_full_*.json",
            "evidence/nonlinear_fusion_k14_*.json",
            "evidence/nonlinear_fusion_k28_*.json"]
# Zero-shot only now; the four method series moved to RUNB_GLOBS/PROTO_GLOBS.
FT_FILES = ["evidence/calibrated_ft_e20.json",
            "evidence/calibrated_ft_shots_b16.json"]
# Upstream Tip-Adapter under our protocol. K=16 in runB/, K in {1,2,4,8} in
# runB_shots/ -- a separate directory because `update_runB_status.py` and
# `rebuild_backbone_main.py` both glob `evidence/runB/runB_*.json` and neither
# filters on K, so a K<16 file dropped in there would silently corrupt the
# Tip-Adapter rows of the per-backbone main table.
RUNB_GLOBS = ["evidence/runB/runB_*.json",
              "evidence/runB_shots/runB_*.json"]
# Proto-Adapter from the Sensors paper. K=16 is ours_<tag>.json, the sweep is
# ours_<tag>_K<k>.json; both carry `K`, so one glob is safe here.
PROTO_GLOBS = ["evidence/proto_adapter/ours_*.json"]
E4_FILES = ["evidence/baselines_e4.json", "evidence/baselines_e4_phaseA.json",
            "evidence/baselines_e4_phaseA2.json",
            "evidence/baselines_e4_shots_b16.json"]

ZS = "Zero-shot CLIP"
# (label, colour, marker, linestyle). Drawing order == legend order.
# LINESTYLE CARRIES THE `Grad.` COLUMN OF the per-backbone main table.
#   dashed `--`  == gradient-based, and NOTHING else is dashed
#   dash-dot `-.`== GDA-CLIP, kept separable: it is the baseline the paper's
#                   decomposition is measured against and the whole content of
#                   fig_le_gda_margin
#   solid `-`    == ours
#   dotted `:`   == the remaining training-free arms
# Colour and marker carry identity; linestyle carries the property. This matters
# at K=1, where the crossing is between the two dashed curves.
# Grey pairs the two Proto-Adapter arms; order follows the per-backbone main table.
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

    `layer_ensemble`/`baselines_e4`/`calibrated_ft` write `cells` or a bare
    list; `runB` and `proto_adapter` go through `protocol_adapter.report`,
    which writes `{"meta": ..., "rows": [...]}`.
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
    # LE-GDA: per-sample standardisation only (sec.5e), never the uniform mean.
    seen_nf = 0
    for g in NF_GLOBS:
        for p in sorted(glob.glob(g)):   # NEVER sfx: see SPLIT CONSISTENCY
            for c in json.load(open(p)).get("rows", []):
                if c["backbone"] != backbone or "F1" not in c.get("rules", {}):
                    continue
                # SPLIT CONSISTENCY. The z-fusion evidence is on the STANDARD
                # split ("testfull"; imagenet on its own "test", sec.7.1). The
                # other series come from layer_ensemble/calibrated_ft/baselines
                # files that exist in BOTH a capped and a _testfull version, and
                # `suffix` is what selects between them. Reading LE-GDA from the
                # standard split while the baselines come from the ~3000-image
                # capped subsets inflates every baseline -- food101 GDA-CLIP
                # reads 87.92 capped against 87.16 standard, which fabricated a
                # 0.79-point deficit for our own method. Fail loudly instead.
                want = "test" if c["dataset"] == "imagenet" else "testfull"
                if c.get("test_split") != want:
                    raise SystemExit(
                        f"{p}: test_split={c.get('test_split')!r}, expected "
                        f"{want!r} for {c['dataset']}")
                if suffix != "_testfull":
                    raise SystemExit(
                        "SPLIT MISMATCH: the LE-GDA series is on the standard "
                        "test split, but --suffix=" + repr(suffix) + " selects "
                        "the CAPPED files for every other series. Pass "
                        "--suffix _testfull (the default).")
                add(c["dataset"], "LE-GDA (ours)", c["K"],
                    c["rules"]["F1"]["test"])
                seen_nf += 1
    if not seen_nf and strict:
        raise SystemExit("no per-sample-standardised LE-GDA cells found for "
                         f"{backbone}; refusing to fall back to the uniform "
                         "mean (see module docstring)")
    for p in map(sfx, FT_FILES):
        if not os.path.exists(p):
            continue
        for c in cells(p):
            if c["backbone"] != backbone:
                continue
            # ZERO-SHOT ONLY. The TIP / TIP-F / CC-F-1 arms in these files
            # are retired: see the module docstring.
            zs_alt.setdefault(c["dataset"], c["zs"])
    # Upstream Tip-Adapter and Proto-Adapter. NEVER sfx: both were run on the
    # standard split only, exactly like the LE-GDA evidence.
    for g in RUNB_GLOBS + PROTO_GLOBS:
        for p in sorted(glob.glob(g)):
            proto = "proto_adapter" in p
            for r in cells(p):
                if r["backbone"] != backbone:
                    continue
                if r.get("test_split") != "testfull":
                    raise SystemExit(
                        f"{p}: test_split={r.get('test_split')!r}, expected "
                        "'testfull' (see SPLIT CONSISTENCY)")
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
    # Upper-left, not lower-right: every curve here rises monotonically in K,
    # so the top-left corner is reliably empty while the bottom-right is where
    # the weakest method lands (Tip-Adapter's imagenet curve ran straight
    # through a lower-right legend). Fixed placement, not "best", so the twelve
    # figures of the series stay visually consistent.
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
    # "_testfull" selects the runs made on the STANDARD uncapped test split.
    # Kept as a suffix rather than a replacement so the capped figure set
    # stays reproducible alongside it.
    ap.add_argument("--suffix", default="_testfull",
                    help="evidence-file suffix selecting the test split. "
                         "DEFAULT _testfull = the standard splits. Passing "
                         "'' selects the capped subsets and is refused, "
                         "because the LE-GDA series is standard-split only.")
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

    # ---- the average: one fixed dataset set for every point (rule 7) --------
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
        # Same convention as the average panel: the backbone leads the title,
        # built from args.backbone so it cannot claim ViT-B/16 on a run that
        # was not ViT-B/16.
        written.append(one_figure(d, data[d], zs.get(d), args.out, args.width,
                                  args.dpi,
                                  subtitle=f"{args.backbone} - {NICE.get(d, d)}"))
    print(f"\nwrote {len(written)} figures to {args.out}/ "
          f"(.pdf + .png each) [{args.backbone}]")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
