"""The §3.2 head-to-head table, at Phase A scale: 11 datasets x the two ViTs.

Emits markdown so the write-up is never hand-copied from a terminal.

WHERE EACH ROW COMES FROM, because they are different runs and that is the
whole risk in this table:

    zero-shot                  either source (they agree, and this asserts it)
    Tip-Adapter    TIP         evidence/runB/       UPSTREAM CODE, our protocol
    Tip-Adapter-F  TIP-F       evidence/runB/       UPSTREAM CODE, trained 20 ep
    Proto-Adapter  PROTO       evidence/proto_adapter/
    Proto-Adapter-F PROTO-F    evidence/proto_adapter/            trained, 20 ep
    GPCache        gpcache16   evidence/baselines_e4*.json
    GDA-CLIP       gda_last    evidence/layer_ensemble_*.json
    LE-GDA         rules.F1.test  evidence/nonlinear_fusion_*.json

LE-GDA IS THE PER-SAMPLE STANDARDISED FUSION, NOT THE UNIFORM MEAN. The write-up
§5e supersedes the fusion rule: each selected space's score vector is
standardised across the CLASS axis for every test sample before averaging, which
takes KC-S1 from +0.534 to +1.136. The evidence files label that operator `F1`
and the old uniform mean `F0` -- INTERNAL ARM LABELS, not the F1 score; every
number in this table is top-1 accuracy (the write-up §6.1, which
also forbids printing "F1" in the manuscript).

There is NO fallback to the uniform mean: a missing standardised cell raises.
The write-up (§2.0) -- "never mix the two fusion rules in one
table" -- is the reason, and a silent fallback is how it would be broken.

ALIGNMENT IS ASSERTED, NOT ASSUMED. Every source is fitted on the same
`caches/<ds>/collapse/<tag>` tensors, so their zero-shot accuracies must agree.
The check below fails loudly on any cell where they do not -- with one measured
exception, imagenet, where the two runs differ by ~0.02 points because the text
classifier is rebuilt in fp16 per run (the image features are the same file).
That is ~9 images of 40k. IMAGENET_ZS_TOL exists for exactly that and nothing
else; it is not a licence to widen the check.

Trained arms are 3-seed means with `std` recorded in the source; the layer
files carry per-seed cells and are averaged here.
"""

import argparse
import glob
import json
from statistics import mean, stdev

LE_FILES = ["vitb16", "vitb32", "imagenet_b16", "imagenet_b32",
            "RN50", "RN50_imagenet", "RN101", "RN101_imagenet"]
E4_FILES = ["evidence/baselines_e4.json", "evidence/baselines_e4_phaseA.json",
            "evidence/baselines_e4_phaseA2.json",
            "evidence/baselines_e4_phaseA3.json",
            "evidence/baselines_e4_rn.json"]
# RETIRED as a source for the comparative rows. The Tip-Adapter arms are now
# the UPSTREAM implementation run under our protocol (evidence/runB/), and the
# Proto-Adapter arms are implemented from the Sensors paper
# (evidence/proto_adapter/). The previous source, calibrated_ft_e20.json, held
# OUR reimplementations: TIP was faithful to +0.04 but TIP-F ran on a
# frozen-feature trainer worth -2.30, and `CC-F-1` is not Proto-Adapter-F at
# all (plain CE instead of ArcFace, no channel-wise normalization, a different
# calibration rule). See the README, Comparative methods under our protocol.
RUNB_GLOB = "evidence/runB/runB_*.json"
PROTO_GLOB = "evidence/proto_adapter/ours_*.json"
# LE-GDA under per-sample standardisation. Globbed because the shot counts were
# separate passes; imagenet always sits in its own file on its own split.
NF_GLOBS = ["evidence/nonlinear_fusion_full_*.json",
            "evidence/nonlinear_fusion_k14_*.json",
            "evidence/nonlinear_fusion_k28_*.json"]
BACKBONES = ["ViT-B/16", "ViT-B/32", "RN50", "RN101"]
DATASETS = ["caltech101", "dtd", "eurosat", "fgvc", "food101", "imagenet",
            "oxford_flowers", "oxford_pets", "stanford_cars", "sun397",
            "ucf101"]
SHORT = {"caltech101": "caltech", "oxford_flowers": "flowers",
         "oxford_pets": "pets", "stanford_cars": "cars", "imagenet": "in1k"}
ZS_TOL = 1e-6
IMAGENET_ZS_TOL = 0.05      # measured 0.02; fp16 text-encoder rebuild only


def _missing(bb, d, k):
    raise SystemExit(
        f"no per-sample-standardised LE-GDA cell for {bb}/{d}/K={k}. Refusing "
        "to fall back to the uniform mean (see module docstring).")


def cells(path):
    """Row list, whatever the file calls it.

    layer_ensemble / baselines_e4 / calibrated_ft write `cells` or a bare list;
    runB and proto_adapter go through protocol_adapter.report, which writes
    {"meta": ..., "rows": [...]}.
    """
    d = json.load(open(path))
    if isinstance(d, dict):
        for k in ("cells", "rows"):
            if k in d:
                return d[k]
    return d


def load(suffix=""):
    def sfx(p):
        return p[:-5] + suffix + ".json" if suffix else p

    le = sum((cells(sfx(f"evidence/layer_ensemble_{f}.json"))
              for f in LE_FILES), [])
    # NEVER sfx: both sources were run on the standard split only.
    ft = {}
    for g in (RUNB_GLOB, PROTO_GLOB):
        for path in sorted(glob.glob(g)):
            for c in cells(path):
                if c.get("K") != 16:
                    continue
                d = ft.setdefault((c["backbone"], c["dataset"]), {"acc": {}})
                d["acc"].update(c["acc"])
                # runB carries no zero-shot; proto_adapter does, and it feeds
                # the alignment assert below.
                if "zs" in c:
                    d["zs"] = c["zs"]
    e4 = {}
    for p in map(sfx, E4_FILES):
        try:
            for c in cells(p):
                e4.setdefault((c["backbone"], c["dataset"]), []).append(c)
        except FileNotFoundError:
            print(f"  (note: {p} absent -- GPCache will show '--' there)")
    # LE-GDA rows: per-sample standardisation, and ALWAYS the standard split,
    # so these globs never take the suffix.
    nf, nf_seed = {}, {}
    for g in NF_GLOBS:
        for path in sorted(glob.glob(g)):
            for c in json.load(open(path)).get("rows", []):
                if "F1" not in c.get("rules", {}):
                    continue
                want = "test" if c["dataset"] == "imagenet" else "testfull"
                if c.get("test_split") != want:
                    raise SystemExit(f"{path}: test_split="
                                     f"{c.get('test_split')!r} != {want!r}")
                nf.setdefault((c["backbone"], c["dataset"], c["K"]),
                              []).append(c["rules"]["F1"]["test"])
                nf_seed[(c["backbone"], c["dataset"], c["K"],
                         c["seed"])] = c["rules"]["F1"]["test"]
    if suffix != "_testfull":
        raise SystemExit(
            "SPLIT MISMATCH: the LE-GDA row is standard-split only, but "
            f"--suffix={suffix!r} selects the CAPPED files for every other "
            "row. Pass --suffix _testfull (the default). This exact mismatch "
            "fabricated a 0.79-point deficit for our own method on food101 in "
            "the figure pipeline.")
    return le, ft, e4, nf, nf_seed


def shot_margin(nf, suffix):
    """LE-GDA - GDA-CLIP at EVERY shot count, paired per cell.

    Exists because a claim I made repeatedly was wrong and only a table could
    show it: §5e.6 says the FUSION RULE (per-sample standardisation minus the
    uniform mean) is a K=16 phenomenon, and that got restated as "the method
    only gains at K=16". Those are different quantities. This one -- the
    method's margin over its baseline -- is positive at EVERY K.
    """
    # LE_FILES is the K=16 grid; the shot sweeps live in their own files, so
    # GDA-CLIP at K=2/8 is only there.
    # LE_FILES is the K=16 grid; the shot sweeps live in their own
    # layer_ensemble_shots_* files, one per backbone, so they are globbed rather
    # than listed -- a hard-coded list is how ViT-B/32 would silently stay a
    # hyphen after its run landed.
    gd = {}
    shot_files = [f.split("layer_ensemble_")[1].rsplit(".json", 1)[0]
                  .replace(suffix, "")
                  for f in glob.glob(f"evidence/layer_ensemble_shots_*{suffix}.json")]
    for f in LE_FILES + sorted(set(shot_files)):
        try:
            for c in cells(f"evidence/layer_ensemble_{f}{suffix}.json"):
                gd[(c["backbone"], c["dataset"], c["K"], c["seed"])] = \
                    c["acc"]["gda_last"]
        except FileNotFoundError:
            continue
    ks = sorted({k[2] for k in nf})
    out = ["**LE-GDA − GDA-CLIP at every shot count** — paired per cell, means "
           "over the datasets and seeds present (`n` = datasets).", "",
           "| backbone | " + " | ".join(f"K={k}" for k in ks) + " | **mean** |",
           "|---" * (len(ks) + 2) + "|"]
    percol = {k: [] for k in ks}
    for bb in BACKBONES:
        row, alld = [], []
        for k in ks:
            d = [v - gd[key] for key, v in nf.items()
                 if key[0] == bb and key[2] == k and key in gd]
            nds = len({key[1] for key, v in nf.items()
                       if key[0] == bb and key[2] == k and key in gd})
            row.append(f"**{mean(d):+.2f}** ({nds})" if d else "—")
            if d:
                percol[k] += d
                alld += d
        out.append(f"| {bb} | " + " | ".join(row)
                   + (f" | **{mean(alld):+.2f}** |" if alld else " | — |"))
    out.append("| **mean over backbones** | "
               + " | ".join(f"**{mean(percol[k]):+.2f}**" if percol[k] else "—"
                            for k in ks)
               + f" | **{mean(sum(percol.values(), [])):+.2f}** |")
    block = "\n".join(out)
    print("\n" + block)
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    # evidence-filename suffix, so the capped and full-test grids can coexist
    # instead of one silently replacing the other. "_testfull" selects the runs
    # made against the STANDARD uncapped test split -- the ones that may be
    # compared against published tables. Without it this emits the capped grid,
    # which is internally consistent but NOT cross-paper comparable.
    ap.add_argument("--suffix", default="_testfull")
    ap.add_argument("--inject", default=None,
                    help="markdown file whose marked shot-margin block to "
                         "replace in place. A PASTED table goes stale: the one "
                         "in the write-up still showed hyphens an hour after the "
                         "runs that filled them landed.")
    ap.add_argument("--shots", action="store_true",
                    help="also emit the per-shot-count margin table")
    args = ap.parse_args()
    le, ft, e4, nf, nf_seed = load(args.suffix)

    # ---- alignment: same splits => same zero-shot. Asserted, not assumed. ----
    zs = {}
    for c in le:
        zs.setdefault((c["backbone"], c["dataset"]), c["zs"])
    bad = []
    for k, v in sorted(zs.items()):
        if k in ft:
            tol = IMAGENET_ZS_TOL if k[1] == "imagenet" else ZS_TOL
            if abs(v - ft[k]["zs"]) > tol:
                bad.append((k, v, ft[k]["zs"]))
    if bad:
        for k, a, b in bad:
            print(f"  MISALIGNED {k}: zs {a:.4f} (layers) vs {b:.4f} (ft)")
        raise SystemExit("zero-shot disagrees -> the runs are not on the same "
                         "splits and the rows may not be put in one table")
    print(f"alignment OK: zero-shot agrees on {len(zs)} cells "
          f"(imagenet within {IMAGENET_ZS_TOL}, fp16 text encoder)\n")

    # ---- per-seed index, for the seed spread of the cross-backbone means ----
    # A seed changes the SUPPORT DRAW, so the spread computed from these is the
    # run-to-run variability of the whole 11-dataset mean. It is NOT the spread
    # across datasets, which is an order of magnitude larger and is not an
    # uncertainty on the mean.
    gda_seed, gp_seed = {}, {}
    for c in le:
        key = (c["backbone"], c["dataset"], c["K"], c["seed"])
        if key in gda_seed and gda_seed[key] != c["acc"]["gda_last"]:
            raise SystemExit(f"GDA-CLIP cell {key} appears twice with "
                             "different values -- pooling would fake the n")
        gda_seed[key] = c["acc"]["gda_last"]
    for (bb_, d_), cs in e4.items():
        for c in cs:
            if "gpcache16" not in c["acc"]:
                continue
            key = (bb_, d_, c["K"], c["seed"])
            if key in gp_seed and gp_seed[key] != c["acc"]["gpcache16"]:
                raise SystemExit(f"GPCache cell {key} appears twice with "
                                 "different values -- pooling would fake the n")
            gp_seed[key] = c["acc"]["gpcache16"]

    def spread(bb, dss, table):
        """mean ± sd over seeds of the mean over `dss`, or None if incomplete.

        Every seed must cover the SAME dataset set, otherwise the three
        numbers being compared are means of different things.
        """
        seeds = sorted({s for (b, d, k, s) in table
                        if b == bb and k == args.k and d in dss})
        if len(seeds) < 2:
            return None
        per = []
        for s in seeds:
            vals = [table.get((bb, d, args.k, s)) for d in dss]
            if any(v is None for v in vals):
                return None
            per.append(mean(vals))
        return mean(per), stdev(per), len(per)

    def paired_spread(bb, dss):
        """LE-GDA − GDA-CLIP, paired within seed before the spread is taken."""
        seeds = sorted({s for (b, d, k, s) in nf_seed
                        if b == bb and k == args.k and d in dss})
        if len(seeds) < 2:
            return None
        per = []
        for s in seeds:
            pairs = [(nf_seed.get((bb, d, args.k, s)),
                      gda_seed.get((bb, d, args.k, s))) for d in dss]
            if any(a is None or b is None for a, b in pairs):
                return None
            per.append(mean(a - b for a, b in pairs))
        return mean(per), stdev(per), len(per)

    def fmt(t, sign=False):
        if t is None:
            return "—"
        mu, sd, _ = t
        return f"{mu:+.2f} ± {sd:.2f}" if sign else f"{mu:.2f} ± {sd:.2f}"

    gaps = []
    summary = {}
    for bb in BACKBONES:
        rows = {}
        for d in DATASETS:
            s = [c for c in le if c["backbone"] == bb and c["dataset"] == d
                 and c["K"] == args.k]
            if not s:
                continue
            f = ft.get((bb, d), {}).get("acc", {})
            g = e4.get((bb, d), [])
            gp = [c["acc"]["gpcache16"] for c in g if "gpcache16" in c["acc"]]
            rows[d] = {
                "zero-shot": s[0]["zs"],
                "Tip-Adapter": f.get("TIP"),
                "Proto-Adapter": f.get("PROTO"),
                "GPCache (16 keys)": mean(gp) if gp else None,
                "Tip-Adapter-F": f.get("TIP-F"),
                "Proto-Adapter-F": f.get("PROTO-F"),
                "GDA-CLIP": mean(c["acc"]["gda_last"] for c in s),
                "LE-GDA (ours)": mean(nf[(bb, d, args.k)])
                if (bb, d, args.k) in nf else _missing(bb, d, args.k),
            }
        if not rows:
            continue
        order = list(next(iter(rows.values())))
        hdr = [SHORT.get(d, d) for d in rows]
        print(f"**{bb}** — K={args.k}, 3-seed means, all {len(rows)} datasets\n")
        print("| method | " + " | ".join(hdr) + " | **mean** | trained? |")
        print("|---" * (len(hdr) + 3) + "|")
        TR = {"Tip-Adapter-F": "**yes (20 ep)**",
              "Proto-Adapter-F": "**yes (20 ep)**",
              "zero-shot": "—"}
        for m in order:
            vals = [rows[d][m] for d in rows]
            got = [v for v in vals if v is not None]
            # a mean over a DIFFERENT set of datasets is not comparable, so an
            # incomplete row reports its own n rather than a bare number
            if len(got) == len(vals):
                mu = f"{mean(got):.2f}"
            elif got:
                # a mean over a DIFFERENT dataset set is not comparable to the
                # complete rows, so it is marked, never quoted bare
                miss = [SHORT.get(d, d) for d, v in zip(rows, vals) if v is None]
                mu = f"{mean(got):.2f} *(n={len(got)}, not run: {', '.join(miss)})*"
                gaps.append((m, bb, miss))
            else:
                mu = "—"
            cellstr = " | ".join("—" if v is None else f"{v:.2f}" for v in vals)
            print(f"| {m} | {cellstr} | **{mu}** | {TR.get(m, 'no')} |")
        gd = mean(rows[d]["GDA-CLIP"] for d in rows)
        le_ = mean(rows[d]["LE-GDA (ours)"] for d in rows)
        gpv = [rows[d]["GPCache (16 keys)"] for d in rows]
        summary[bb] = (mean(gpv) if all(v is not None for v in gpv) else None,
                       gd, le_, list(rows))
        print(f"\n`LE-GDA − GDA-CLIP` = **{le_ - gd:+.2f}** "
              f"— that increment is ours; everything below GDA-CLIP is "
              f"GDA-CLIP's lead, not ours.\n")

    # cross-backbone summary: the three training-free arms and our increment.
    # Emitted from the same means as the per-dataset tables above, so the two
    # can never disagree.
    print("\n**Cross-backbone summary** — 11-dataset means, K=16, 3 seeds\n")
    print("| backbone | GPCache | GDA-CLIP | LE-GDA | ours |")
    print("|---|---|---|---|---|")
    for bb in BACKBONES:
        if bb not in summary:
            continue
        gp, gd, le_, dss = summary[bb]
        sg = spread(bb, dss, gp_seed)
        sd_ = spread(bb, dss, gda_seed)
        sl = spread(bb, dss, nf_seed)
        so = paired_spread(bb, dss)
        # the ± is decoration on the mean the per-dataset tables already
        # printed, so the centres must agree with them exactly
        for t, want in ((sd_, gd), (sl, le_), (so, le_ - gd)):
            if t is not None and abs(t[0] - want) > 5e-9:
                raise SystemExit(f"{bb}: seed-mean {t[0]} disagrees with the "
                                 f"per-dataset table's {want}")
        print(f"| {bb} | {'—' if gp is None else fmt(sg)} | {fmt(sd_)} | "
              f"**{fmt(sl)}** | **{fmt(so, sign=True)}** |")
    ns = {t[2] for bb in summary
          for t in (spread(bb, summary[bb][3], nf_seed),) if t}
    print(f"\n`±` is one standard deviation over the "
          f"{'/'.join(str(n) for n in sorted(ns))} seeds, of the "
          f"11-dataset mean. A seed redraws the K-shot support set, so this "
          f"is run-to-run variability of the whole column, NOT the spread "
          f"across datasets. `ours` is paired within seed before the "
          f"spread is taken. With n=3 the estimate is crude.\n")

    if args.shots:
        block = shot_margin(nf_seed, args.suffix)
        if args.inject:
            BEG = "<!-- BEGIN shot-margin (generated: "\
                  "make_le_gda_head_to_head.py --shots --inject) -->"
            END = "<!-- END shot-margin -->"
            doc = open(args.inject).read()
            if BEG not in doc or END not in doc:
                raise SystemExit(f"{args.inject}: markers {BEG} / {END} not "
                                 "found -- add them around the table first")
            i, j = doc.index(BEG), doc.index(END)
            open(args.inject, "w").write(
                doc[:i] + BEG + "\n\n" + block + "\n\n" + doc[j:])
            print(f"\ninjected into {args.inject}")

    if gaps:
        print("\n### Runs still missing (hyphens above)\n")
        for m, bb, miss in gaps:
            print(f"- **{m}**, {bb}: `{' '.join(miss)}`")


if __name__ == "__main__":
    main()
