"""Per-backbone results tables (markdown): 11 datasets x 4 backbones, K=16.

Where each row comes from:

    zero-shot                     evidence/layer_ensemble_*.json (and checked
                                  against evidence/proto_adapter/)
    Tip-Adapter     TIP           evidence/runB/          original code, our protocol
    Tip-Adapter-F   TIP-F         evidence/runB/          trained, 20 epochs
    Proto-Adapter   PROTO         evidence/proto_adapter/
    Proto-Adapter-F PROTO-F       evidence/proto_adapter/ trained, 20 epochs
    GPCache         gpcache16     evidence/baselines_e4*.json
    GDA-CLIP        gda_last      evidence/layer_ensemble_*.json
    LE-GDA          rules.F1.test evidence/nonlinear_fusion_*.json

LE-GDA is the per-query standardised fusion (rule key "F1" in the result files;
this is a label, not the F1 score). Every number is top-1 accuracy.

All sources use the same cached features and splits, so their zero-shot
accuracies must agree; the script stops if they do not. On ImageNet they may
differ by ~0.02 points because the text classifier is rebuilt in fp16 in each
run (IMAGENET_ZS_TOL).
"""

import argparse
import glob
import json
from statistics import mean, stdev

LE_FILES = ["vitb16", "vitb32", "imagenet_b16", "imagenet_b32",
            "RN50", "RN50_imagenet", "RN101", "RN101_imagenet"]
# GPCache results (the --suffix is appended to each name)
E4_FILES = ["evidence/baselines_e4.json", "evidence/baselines_e4_phaseA.json"]
RUNB_GLOB = "evidence/runB/runB_*.json"
PROTO_GLOB = "evidence/proto_adapter/ours_*.json"
# LE-GDA: K=16 in full_*, K in {1,4} in k14_*, K in {2,8} in k28_*
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
IMAGENET_ZS_TOL = 0.05      # fp16 text-classifier rebuild, ~0.02 measured


def _missing(bb, d, k):
    raise SystemExit(
        f"no LE-GDA result for {bb}/{d}/K={k} in evidence/nonlinear_fusion_*")


def cells(path):
    """Row list, whatever the file calls it.

    layer_ensemble / baselines_e4 write `cells`; runB and proto_adapter go
    through protocol_adapter.report, which writes {"meta": ..., "rows": [...]}.
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
    # runB and proto_adapter were run on the standard split only: no suffix
    ft = {}
    for g in (RUNB_GLOB, PROTO_GLOB):
        for path in sorted(glob.glob(g)):
            for c in cells(path):
                if c.get("K") != 16:
                    continue
                d = ft.setdefault((c["backbone"], c["dataset"]), {"acc": {}})
                d["acc"].update(c["acc"])
                # runB carries no zero-shot; proto_adapter does (checked below)
                if "zs" in c:
                    d["zs"] = c["zs"]
    e4 = {}
    for p in map(sfx, E4_FILES):
        try:
            for c in cells(p):
                e4.setdefault((c["backbone"], c["dataset"]), []).append(c)
        except FileNotFoundError:
            print(f"  (note: {p} absent -- GPCache will show '--' there)")
    # LE-GDA results are on the standard split only: no suffix
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
            "the LE-GDA results are on the standard test split; pass "
            "--suffix _testfull (the default) so every row uses it")
    return le, ft, e4, nf, nf_seed


def shot_margin(nf, suffix):
    """LE-GDA - GDA-CLIP at every shot count, paired per cell."""
    # LE_FILES is the K=16 grid; the shot sweeps are in the
    # layer_ensemble_shots_* files, one per backbone
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
    # evidence-filename suffix; "_testfull" = the standard test split
    ap.add_argument("--suffix", default="_testfull")
    ap.add_argument("--shots", action="store_true",
                    help="also emit the per-shot-count margin table")
    args = ap.parse_args()
    le, ft, e4, nf, nf_seed = load(args.suffix)

    # ---- alignment: same splits => same zero-shot ----
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
    # A seed redraws the support set, so this spread is the run-to-run
    # variability of the 11-dataset mean, not the spread across datasets.
    gda_seed, gp_seed = {}, {}
    for c in le:
        key = (c["backbone"], c["dataset"], c["K"], c["seed"])
        if key in gda_seed and gda_seed[key] != c["acc"]["gda_last"]:
            raise SystemExit(f"GDA-CLIP cell {key} appears twice with "
                             "different values")
        gda_seed[key] = c["acc"]["gda_last"]
    for (bb_, d_), cs in e4.items():
        for c in cs:
            if "gpcache16" not in c["acc"]:
                continue
            key = (bb_, d_, c["K"], c["seed"])
            if key in gp_seed and gp_seed[key] != c["acc"]["gpcache16"]:
                raise SystemExit(f"GPCache cell {key} appears twice with "
                                 "different values")
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
            # an incomplete row reports its own n, since its mean is over a
            # different set of datasets
            if len(got) == len(vals):
                mu = f"{mean(got):.2f}"
            elif got:
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
        print(f"\n`LE-GDA − GDA-CLIP` = **{le_ - gd:+.2f}**\n")

    # cross-backbone summary, from the same means as the tables above
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
        # the seed-wise means must agree with the per-dataset tables
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
          f"is run-to-run variability of the whole column, not the spread "
          f"across datasets. `ours` is paired within seed before the "
          f"spread is taken.\n")

    if args.shots:
        shot_margin(nf_seed, args.suffix)

    if gaps:
        print("\n### Missing results (hyphens above)\n")
        for m, bb, miss in gaps:
            print(f"- **{m}**, {bb}: `{' '.join(miss)}`")


if __name__ == "__main__":
    main()
