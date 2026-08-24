"""The main backbone table (tab:backbone_main), regenerated from evidence.

Every entry is a mean over 11 datasets; the +- is the SD ACROSS THE THREE SEEDS
OF THAT 11-DATASET MEAN -- run-to-run variability of the whole number, NOT the
spread across datasets, which is an order of magnitude larger and is not an
uncertainty on the mean. Same convention as make_le_gda_head_to_head.spread().

WHY THIS SCRIPT EXISTS. Tip-Adapter, Tip-Adapter-F and Proto-Adapter-F had no
+- because run_calibrated_ft.py collapsed its per-seed list into a mean and a
per-cell sd at write time. A per-cell sd CANNOT be turned into the sd of the
11-dataset mean: assuming independence across datasets and using
sqrt(sum sd_d^2)/n_d was tested against the three arms where per-seed values do
exist and is off by 0.21x-1.63x. So the values were recomputed, and the re-run
reproduces every recorded mean BIT-IDENTICALLY (d = 0.0e+00) -- no reported
accuracy moves; only the previously-discarded dispersion is recovered.

ZERO-SHOT CLIP HAS NO +- AND MUST NOT BE GIVEN ONE. It uses no support set, so
there is nothing for a seed to vary; its three "seeds" would be the same number
three times. An error bar there would be a fabrication, not a measurement.
"""

import argparse
import glob
import json
from statistics import mean, stdev

from scripts.make_le_gda_head_to_head import E4_FILES, LE_FILES, NF_GLOBS, cells

BBS = ["RN50", "RN101", "ViT-B/32", "ViT-B/16"]
DATASETS = ["caltech101", "dtd", "eurosat", "fgvc", "food101",
            "oxford_flowers", "oxford_pets", "stanford_cars", "sun397",
            "ucf101", "imagenet"]
# label -> (source, key). Order is the table's order.
ROWS = [("Zero-shot CLIP", "--", "zs", "zs"),
        ("Tip-Adapter", "--", "ft", "TIP"),
        ("GPCache", "--", "e4", "gpcache16"),
        ("Tip-Adapter-F", r"$\checkmark$", "ft", "TIP-F"),
        ("Proto-Adapter-F", r"$\checkmark$", "ft", "CC-F-1"),
        ("Final-feature GDA", "--", "le", "gda_last"),
        (r"\textbf{LE-GDA}", "--", "nf", "F1")]


def build(ft_files, k=16):
    """-> per[(row_key, backbone, dataset, seed)] = accuracy."""
    per = {}
    for f in LE_FILES:
        for c in cells(f"evidence/layer_ensemble_{f}_testfull.json"):
            if c["K"] == k:
                per[("gda_last", c["backbone"], c["dataset"], c["seed"])] = \
                    c["acc"]["gda_last"]
                # zero-shot is deterministic; recorded once per cell
                per[("zs", c["backbone"], c["dataset"], c["seed"])] = c["zs"]
    for p in E4_FILES:
        try:
            src = cells(p[:-5] + "_testfull.json")
        except FileNotFoundError:
            continue
        for c in src:
            if c.get("K") == k and "gpcache16" in c.get("acc", {}):
                per[("gpcache16", c["backbone"], c["dataset"], c["seed"])] = \
                    c["acc"]["gpcache16"]
    for g in NF_GLOBS:
        for p in sorted(glob.glob(g)):
            for c in json.load(open(p)).get("rows", []):
                if c["K"] == k and "F1" in c.get("rules", {}):
                    per[("F1", c["backbone"], c["dataset"], c["seed"])] = \
                        c["rules"]["F1"]["test"]
    for p in ft_files:
        for c in cells(p):
            if c.get("K") != k:
                continue
            assert "per_seed" in c, (
                f"{p}: no per_seed -- re-run scripts/run_calibrated_ft.py, "
                "which now retains it")
            for arm, vals in c["per_seed"].items():
                for s, v in zip(c["seeds"], vals):
                    per[(arm, c["backbone"], c["dataset"], s)] = v
    return per


def spread(per, key, bb, dss, deterministic=False):
    """mean over datasets, then mean +- sample sd over seeds. None if incomplete."""
    seeds = sorted({s for (a, b, d, s) in per
                    if a == key and b == bb and d in dss})
    if not seeds:
        return None
    vals = []
    for s in seeds:
        got = [per.get((key, bb, d, s)) for d in dss]
        if any(v is None for v in got):
            return None
        vals.append(mean(got))
    if deterministic:
        assert max(vals) - min(vals) < 1e-9, (
            f"{key}/{bb}: zero-shot varies across seeds -- it must not")
        return mean(vals), None
    return (mean(vals), stdev(vals) if len(vals) > 1 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-files", nargs="+", default=[
        "evidence/calibrated_ft_e20_ten_testfull_seeded.json",
        "evidence/calibrated_ft_e20_in_testfull_seeded.json"])
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()
    per = build(args.ft_files, args.k)

    out, incomplete = [], []
    for label, grad, _src, key in ROWS:
        det = key == "zs"
        cellsv = [spread(per, key, bb, DATASETS, det) for bb in BBS]
        if any(c is None for c in cellsv):
            incomplete.append((label, [b for b, c in zip(BBS, cellsv)
                                       if c is None]))
        txt = []
        for c in cellsv:
            if c is None:
                txt.append("--")
            elif c[1] is None:
                txt.append(f"{c[0]:.2f}")
            else:
                txt.append(f"{c[0]:.2f}$\\pm${c[1]:.2f}".replace("0.", "."))
        m = ([c[0] for c in cellsv] if all(c is not None for c in cellsv)
             else None)
        out.append([label, grad, txt,
                    f"{mean(m):.2f}" if m else "--",
                    [c[0] if c else None for c in cellsv],
                    mean(m) if m else None])

    # Bold the best entry in each accuracy column, and in the Mean column.
    # Computed from the raw values, never from the rounded strings.
    for j in range(len(BBS)):
        vals = [(r[4][j], i) for i, r in enumerate(out) if r[4][j] is not None]
        if vals:
            i = max(vals)[1]
            out[i][2][j] = r"\textbf{" + out[i][2][j] + "}"
    mv = [(r[5], i) for i, r in enumerate(out) if r[5] is not None]
    if mv:
        i = max(mv)[1]
        out[i][3] = r"\textbf{" + out[i][3] + "}"

    print("% regenerate: python -m scripts.make_backbone_main_table")
    print(r"\begin{tabular}{lcrrrrr}")
    print(r"\toprule")
    print(r"Method & Grad. & RN50 & RN101 & ViT-B/32 & ViT-B/16 & Mean\\")
    print(r"\midrule")
    for label, grad, txt, mn, _v, _m in out:
        print(f"{label} &{grad}&" + "&".join(txt) + f"&{mn}\\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    if incomplete:
        print("\n% INCOMPLETE ROWS (reported as '--', never estimated):")
        for label, bbs in incomplete:
            print(f"%   {label}: missing/short on {', '.join(bbs)}")


if __name__ == "__main__":
    main()
