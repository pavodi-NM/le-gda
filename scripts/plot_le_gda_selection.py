"""LE-GDA Figure: individual quality vs greedy selection frequency, by readout.

The write-up (§5) names this figure: "per-space deficit vs greedy
selection frequency, split by readout -- clause (ii) in one panel". Clause (ii)
is the ViT/ResNet contrast of the write-up §3, carried by the
tables of the write-up §5d.4:

  on ViT CLS tokens the most-selected intermediate space (`L6`) is 5.60 points
  WORSE alone than the final embedding, while on ResNet pooled stages the
  workhorses (`R3`, `R4`) sit at near-parity and the final embedding is itself
  the most-selected space.

REGENERATED, NEVER HAND-COPIED. Both axes are recomputed here from the same
evidence files and the same eligibility rule that `analyze_layer_ensemble.py`
uses (K=16, cache gain C >= 3), and --strict asserts the recomputed numbers
against the §5d.4 table before anything is drawn. If the evidence moves, the
assert fires rather than the figure quietly disagreeing with the text.

WHICH SELECTION RULE IS PLOTTED. The greedy subsets drawn here are the ones
recorded by `run_layer_ensemble.py`, i.e. selection under UNIFORM averaging of
the per-space scores -- the arm the §5d.4 tables and clause (ii) are stated
on. The deployed fusion rule is per-sample standardisation (write-up §5e),
whose greedy selections are recorded separately in
`evidence/nonlinear_fusion_full_*.json` and are NOT the same; --rule standardised
draws those instead. The rule is named in the caption either way, because the
two give materially different ViT orderings.

House style: the project figure-style convention. One figure = one PDF (+ PNG preview).
"""

import argparse
import glob
import json
import os
from collections import Counter
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same palette as scripts/plot_le_gda_figures.py, so the figure set is one look.
PAL = ["#3366cc", "#e8710a", "#12a37e", "#c2298a", "#8f6b00", "#6b4fd8"]
INK, INK2 = "#1a1a1a", "#4a4a4a"
GRID = "#e6e6e6"

LE_FILES = ["vitb16", "vitb32", "imagenet_b16", "imagenet_b32",
            "RN50", "RN50_imagenet", "RN101", "RN101_imagenet"]
NF_GLOB = "evidence/nonlinear_fusion_full_*.json"

# Rule 11: no coined vocabulary on the canvas. `L6`/`R3`/`pre`/`last` are
# internal codes; the reader gets the plain phrase and the caption carries the
# code so the link to the notation section survives.
NICE = {"L4": "block 4", "L6": "block 6", "L8": "block 8", "L10": "block 10",
        "L11": "block 11", "pre": "pre-projection", "last": "final embedding",
        "R1": "stage 1", "R2": "stage 2", "R3": "stage 3", "R4": "stage 4"}

# Where each label sits relative to its point: (dx in points, dy in points, ha).
# Hand-placed -- these eleven points collide badly under any automatic rule.
OFFSET = {
    ("ViT", "L4"): (0, 10, "center"), ("ViT", "L6"): (0, 10, "center"),
    ("ViT", "L8"): (0, -13, "center"), ("ViT", "L10"): (-8, 0, "right"),
    ("ViT", "L11"): (9, 0, "left"), ("ViT", "pre"): (9, 0, "left"),
    ("ViT", "last"): (9, 0, "left"),
    ("RN", "R1"): (0, 10, "center"), ("RN", "R2"): (0, 10, "center"),
    ("RN", "R3"): (-9, 5, "right"), ("RN", "R4"): (-9, -6, "right"),
    ("RN", "last"): (9, 0, "left"),
}
# The standardised rule moves the points, so a few labels need moving with them.
OFFSET_STD = {
    ("ViT", "L8"): (0, 10, "center"), ("ViT", "L10"): (-9, 0, "right"),
    ("ViT", "pre"): (9, 4, "left"), ("ViT", "last"): (9, -4, "left"),
    ("RN", "R1"): (0, 10, "center"), ("RN", "R2"): (0, -13, "center"),
    ("RN", "R3"): (-9, 2, "right"), ("RN", "R4"): (-9, 0, "right"),
}
# Where the empty corner is differs between the two rules.
LEGEND_LOC = {"uniform": "lower left", "standardised": "upper left"}

# The write-up (§5d.4), the record. --strict refuses to draw anything that
# disagrees with it. Values are (median accuracy difference, greedy picks).
RECORD = {
    ("ViT", "L4"): (-6.90, 25), ("ViT", "L6"): (-5.60, 45),
    ("ViT", "L8"): (-4.22, 19), ("ViT", "L10"): (-1.29, 14),
    ("ViT", "L11"): (+0.15, 23), ("ViT", "pre"): (+0.12, 34),
    ("RN", "R1"): (-10.24, 19), ("RN", "R2"): (-5.26, 7),
    ("RN", "R3"): (-0.43, 38), ("RN", "R4"): (-0.12, 36),
}
RECORD_LAST_PICKS = 74      # §5d.4 pools `last` over both readouts

FAMILIES = [
    # key, legend label, colour, marker, the order spaces run in
    ("ViT", "ViT CLS tokens", PAL[5], "o",
     ["L4", "L6", "L8", "L10", "L11", "pre", "last"]),
    ("RN", "ResNet pooled stages", PAL[1], "s",
     ["R1", "R2", "R3", "R4", "last"]),
]


def family(backbone):
    return "ViT" if backbone.startswith("ViT") else "RN"


def load_cells(k, elig):
    """The eligible cells of the scorer of record, pooled over its eight files."""
    cells = []
    for tag in LE_FILES:
        fp = f"evidence/layer_ensemble_{tag}_testfull.json"
        cells += json.load(open(fp))["cells"]
    seen = {(c["backbone"], c["dataset"], c["K"], c["seed"]) for c in cells}
    assert len(seen) == len(cells), "duplicate cells would double-count picks"
    return [c for c in cells if c["K"] == k and c["C"] >= elig]


def standardised_subsets(cells):
    """Greedy subsets under per-sample standardisation, joined on the same cells.

    The fusion evidence records no `C`, so eligibility is taken from the
    layer-ensemble cells it is joined against -- the identical population.
    """
    keep = {(c["backbone"], c["dataset"], c["K"], c["seed"]) for c in cells}
    out = {}
    for fp in sorted(glob.glob(NF_GLOB)):
        for r in json.load(open(fp))["rows"]:
            key = (r["backbone"], r["dataset"], r["K"], r["seed"])
            if key in keep:
                out[key] = r["rules"]["F1"]["subset"]
    missing = keep - set(out)
    assert not missing, f"{len(missing)} eligible cells have no fusion row"
    return out


def measure(cells, rule):
    """(median accuracy difference, picks, n) per (family, space)."""
    subs = standardised_subsets(cells) if rule == "standardised" else None
    out, ncells = {}, {}
    for fam, _, _, _, spaces in FAMILIES:
        sub = [c for c in cells if family(c["backbone"]) == fam]
        ncells[fam] = len(sub)
        # WITH multiplicity, as §5d.4 counts them: greedy selects with
        # replacement (§2.5), a repeat is a weight on a space it already
        # holds, and dropping repeats would understate exactly the spaces the
        # figure is about. So a setting can contribute more than one pick.
        if subs is None:
            picks = Counter(s for c in sub
                            for s in c["sel"]["le_gda"]["subset"])
        else:
            picks = Counter(
                s for c in sub for s in
                subs[(c["backbone"], c["dataset"], c["K"], c["seed"])])
        for sp in spaces:
            arm = f"gda_{sp}"
            d = [c["acc"][arm] - c["acc"]["gda_last"]
                 for c in sub if arm in c["acc"]]
            assert len(d) == len(sub), f"{fam}/{sp} missing from some cells"
            out[(fam, sp)] = (median(d), picks.get(sp, 0), len(sub))
    return out, ncells


def check_record(m):
    """Rule: the figure may not disagree with the write-up §5d.4."""
    bad = []
    for key, (want_d, want_n) in RECORD.items():
        got_d, got_n, _ = m[key]
        if abs(got_d - want_d) > 0.005 or got_n != want_n:
            bad.append(f"{key}: recomputed ({got_d:+.2f}, {got_n}) vs "
                       f"§5d.4 ({want_d:+.2f}, {want_n})")
    pooled = m[("ViT", "last")][1] + m[("RN", "last")][1]
    if pooled != RECORD_LAST_PICKS:
        bad.append(f"`last` picks pooled: {pooled} vs §5d.4 "
                   f"{RECORD_LAST_PICKS}")
    if bad:
        raise SystemExit("EVIDENCE DISAGREES WITH THE WRITE-UP (§5d.4):\n  "
                         + "\n  ".join(bad))
    print("strict: every point matches the write-up §5d.4")


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def draw(m, ncells, out, width, dpi, stem, rule):
    fig, ax = plt.subplots(figsize=(width, width * 0.78), facecolor="white")
    # The final embedding is the reference of the x axis, so it sits at 0 by
    # construction. Drawn, because it is one of the selectable spaces.
    ax.axvline(0.0, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2,
               label="the final embedding's own accuracy")
    for fam, lab, col, mk, spaces in FAMILIES:
        xs = [m[(fam, s)][0] for s in spaces]
        ys = [m[(fam, s)][1] / ncells[fam] * 100 for s in spaces]
        ax.scatter(xs, ys, s=46, color=col, marker=mk, lw=0, zorder=5,
                   label=f"{lab} ({ncells[fam]} settings)")
        for s, x, y in zip(spaces, xs, ys):
            off = OFFSET if rule == "uniform" else\
                {**OFFSET, **OFFSET_STD}
            dx, dy, ha = off[(fam, s)]
            ax.annotate(NICE[s], (x, y), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, va="center", fontsize=7.6,
                        color=col, zorder=6)
    style(ax)
    ax.set_xlim(-11.8, 3.4)
    # A setting can contribute several picks to one space, so the rate is not
    # bounded by 100% -- under the standardised rule `R3` reaches 140%.
    top = max(104, max(m[(f, s)][1] / ncells[f] * 100
                       for f, _, _, _, sp in FAMILIES for s in sp) + 9)
    ax.set_ylim(-5, top)
    ax.set_yticks([t for t in range(0, int(top) + 1, 20)])
    ax.set_xlabel("median top-1 difference from the final embedding (points)",
                  fontsize=10, color=INK, labelpad=5)
    ax.set_ylabel("greedy picks per setting (%)", fontsize=10,
                  color=INK, labelpad=5)
    ax.set_title("Individual accuracy and selection frequency per space",
                 fontsize=11.5, color=INK, pad=9, loc="left")
    # Under the uniform rule the two most-selected spaces sit high and their
    # labels reach into the top-left corner, so the legend goes low; under the
    # standardised rule a space drops to 0% and the empty corner swaps.
    ax.legend(fontsize=7.6, frameon=False, loc=LEGEND_LOC[rule],
              handlelength=1.6, labelspacing=0.34, borderpad=0.2,
              scatterpoints=1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/{stem}.{ext}", dpi=dpi, facecolor="white")
    plt.close(fig)
    return stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/le_gda")
    ap.add_argument("--width", type=float, default=6.0)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--elig", type=float, default=3.0)
    ap.add_argument("--rule", default="uniform",
                    choices=["uniform", "standardised"],
                    help="which greedy selection to plot. `uniform` is the arm "
                         "the write-up §5d.4 and clause (ii) are stated on; "
                         "`standardised` is the deployed fusion rule (§5e).")
    ap.add_argument("--strict", action="store_true",
                    help="refuse to draw if the recomputed numbers disagree "
                         "with the §5d.4 table (uniform rule only)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cells = load_cells(args.k, args.elig)
    m, ncells = measure(cells, args.rule)
    print(f"{len(cells)} eligible cells at K={args.k}, C >= {args.elig}: "
          + ", ".join(f"{f} {n}" for f, n in ncells.items()))
    print(f"selection rule: {args.rule}")
    for fam, lab, _, _, spaces in FAMILIES:
        print(f"  {lab}")
        for s in spaces:
            d, n, tot = m[(fam, s)]
            print(f"    {NICE[s]:<18}{d:+7.2f}   {n:>3}/{tot}  "
                  f"{n / tot * 100:5.1f}%")
    if args.strict:
        if args.rule != "uniform":
            raise SystemExit("--strict checks the §5d.4 table, which is "
                             "stated on the uniform rule only")
        check_record(m)

    stem = ("fig_le_gda_selection" if args.rule == "uniform"
            else "fig_le_gda_selection_standardised")
    draw(m, ncells, args.out, args.width, args.dpi, stem, args.rule)
    print(f"\nwrote {args.out}/{stem}.pdf (+ .png)")


if __name__ == "__main__":
    main()
