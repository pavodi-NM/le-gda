"""Summarise the leave-one-dataset-out configuration-transfer ablation.

Reads the output of scripts/run_config_transfer.py (--phase eval) and joins
full LE-GDA (arm4_le_gda) from evidence/nonlinear_fusion_*.json. Arm
definitions are in run_config_transfer.py. Contrasts reported:

  arm3 - arm1        transferred ensemble vs. single-space GDA, both without
                     validation
  arm2 - gda_last    transferred multiset vs. GDA-CLIP, both with validation
                     for the ridge and alpha
  arm4 - arm2        LE-GDA vs. the transferred multiset
  arm2 - arm3        cost of fixing ridges and alpha, split into
                     arm2 - arm3a (ridges) and arm3a - arm3 (alpha)
  arm4 - arm3        what validation access is worth overall

Tables quote means over datasets; each contrast is paired, with an exact sign
test and a Wilcoxon signed-rank test.

    --unit setting   seeds averaged per (backbone, dataset): 44 units
    --unit dataset   seeds and backbones averaged: 11 units (conservative)
    --unit cell      all 132 cells
"""

import argparse
import collections
import json
import math
from statistics import median, mean, stdev

from scipy.stats import wilcoxon

from scripts.analyze_evidence_gate import sign_test

ARMS = ["arm0_zs", "arm1_gda_eb", "arm2_loo_multiset", "arm3_valfree",
        "arm3a_loolam_valalpha", "arm4_le_gda"]


def to_settings(cells):
    """Average the 3 seeds within each (backbone, dataset) -> 44 settings.

    Seeds of one dataset share its classes, split and prompts, so they are
    repeated measurements rather than independent trials.
    """
    by = collections.defaultdict(list)
    for c in cells:
        by[(c["backbone"], c["dataset"], c["K"])].append(c)
    out = []
    for (bb, ds, K), g in sorted(by.items()):
        arms = set.intersection(*(set(c["acc"]) for c in g))
        out.append({"backbone": bb, "dataset": ds, "K": K, "n_seeds": len(g),
                    "cfg": g[0]["cfg"],
                    "acc": {a: mean([c["acc"][a] for c in g]) for a in arms}})
    return out


LABEL = {"arm0_zs": "0 zero-shot", "arm1_gda_eb": "1 GDA val-free",
         "arm2_loo_multiset": "2 LOO multiset", "arm3_valfree": "3 val-free",
         "arm3a_loolam_valalpha": "3a LOO lam", "arm4_le_gda": "4 LE-GDA"}


def sd(v):
    return stdev(v) if len(v) > 1 else float("nan")


def attach_arm4(cells, k):
    """Join full LE-GDA (rules.F1.test in the fusion results) by cell."""
    import glob
    idx = {}
    for f in glob.glob("evidence/nonlinear_fusion_*.json"):
        for r in json.load(open(f)).get("rows", []):
            idx[(r["backbone"], r["dataset"], r["K"], r["seed"])] = r
    n = 0
    for c in cells:
        r = idx.get((c["backbone"], c["dataset"], c["K"], c["seed"]))
        assert r is not None, f"no LE-GDA result for {c['backbone']}/{c['dataset']}"
        c["acc"]["arm4_le_gda"] = r["rules"]["F1"]["test"]
        n += 1
    print(f"  joined arm 4 (LE-GDA) onto {n} cells")


def to_datasets(cells):
    """Average over backbones as well as seeds -> 11 dataset-level units.

    The conservative unit: when the sign of a contrast is the same on every
    backbone for a dataset (see SIGN CONSISTENCY below), its four backbone
    measurements are not independent.
    """
    by = collections.defaultdict(list)
    for c in cells:
        by[(c["dataset"], c["K"])].append(c)
    out = []
    for (ds, K), g in sorted(by.items()):
        arms = set.intersection(*(set(c["acc"]) for c in g))
        out.append({"backbone": "pooled", "dataset": ds, "K": K,
                    "n_backbones": len(g), "cfg": g[0]["cfg"],
                    "acc": {a: mean([c["acc"][a] for c in g]) for a in arms}})
    return out


def paired(cells, a, b, label):
    """Paired per-unit contrast; the sd is over the differences."""
    d = [c["acc"][a] - c["acc"][b] for c in cells
         if a in c["acc"] and b in c["acc"]]
    if not d:
        print(f"    {label:<52} (not available)")
        return None, None, None, None
    pos, neg, p = sign_test(d)
    md, mn = median(d), mean(d)
    sd = stdev(d) if len(d) > 1 else float("nan")
    # the sign test uses only the signs; the Wilcoxon signed-rank test also
    # uses the magnitudes
    try:
        pw = float(wilcoxon(d, alternative="two-sided").pvalue)
    except ValueError:                       # all-zero differences
        pw = 1.0
    print(f"    {label:<52} mean {mn:+6.3f} +-{sd:6.3f}  median {md:+6.3f}  "
          f"win {pos:>3}/{pos + neg:<3} sign p={p:.2e} "
          f"wilcox p={pw:.2e} "
          f"{'sig' if p < 0.05 else 'n.s.'}")
    return mn, sd, md, p, pw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="evidence/config_transfer_f1.json")
    ap.add_argument("--derive", default="evidence/config_transfer_derive_f1.json")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--unit", choices=["setting", "cell", "dataset"],
                    default="setting",
                    help="`setting`: seeds averaged per (backbone, dataset), "
                         "44 units. `dataset`: also averaged over backbones, "
                         "11 units. `cell`: all 132 cells.")
    args = ap.parse_args()

    cells = [c for c in json.load(open(args.file))["cells"] if c["K"] == args.k]
    print("=" * 78)
    print(f"LOO CONFIGURATION TRANSFER — K={args.k}, {len(cells)} cells")
    print("=" * 78)
    attach_arm4(cells, args.k)
    if args.unit == "setting":
        cells = to_settings(cells)
        print(f"  unit of analysis: {len(cells)} SETTINGS "
              f"(3 seeds averaged within each backbone x dataset)")
    elif args.unit == "dataset":
        cells = to_datasets(to_settings(cells))
        print(f"  unit of analysis: {len(cells)} DATASETS "
              f"(seeds and backbones averaged)")

    bys = collections.defaultdict(list)
    for c in cells:
        bys[c["backbone"]].append(c)

    # ---------------- the table ----------------
    print("\n" + "=" * 78)
    print("TABLE — mean accuracy over the held-out datasets "
          "(mean of per-dataset means; 3 seeds each)")
    print("=" * 78)
    print("  mean +- sd, where the sd is ACROSS THE PER-DATASET MEANS")
    hdr = "".join(f"{LABEL[a]:>17}" for a in ARMS)
    print(f"  {'backbone':<11}{hdr}{'D(3-1)':>17}{'D(4-3)':>17}")
    for bb in sorted(bys):
        cs = bys[bb]
        dss = sorted({c['dataset'] for c in cs})
        # sd across the per-dataset means: the between-dataset spread of
        # accuracy, not a measure of whether one arm beats another
        row, sdv = {}, {}
        arms_here = [a for a in ARMS if all(a in c["acc"] for c in cs)]
        for a in arms_here:
            per = [mean([c["acc"][a] for c in cs if c["dataset"] == ds])
                   for ds in dss]
            row[a], sdv[a] = mean(per), stdev(per)
        d31 = [mean([c["acc"]["arm3_valfree"] - c["acc"]["arm1_gda_eb"]
                     for c in cs if c["dataset"] == ds]) for ds in dss]
        d43 = [mean([c["acc"]["arm4_le_gda"] - c["acc"]["arm3_valfree"]
                     for c in cs if c["dataset"] == ds]) for ds in dss]
        print(f"  {bb:<11}"
              + "".join(f"{row[a]:>9.2f}+-{sdv[a]:<6.2f}" if a in row
                        else f"{'—':>17}" for a in ARMS)
              + f"{mean(d31):>+8.2f}+-{stdev(d31):<6.2f}"
              + f"{mean(d43):>+8.2f}+-{stdev(d43):<6.2f}")

    # ---------------- paired contrasts ----------------
    print("\n" + "=" * 78)
    print(f"PAIRED CONTRASTS — per {args.unit}, exact sign test")
    print("=" * 78)
    for scope, cs in [("ALL", cells)] + sorted(bys.items()):
        print(f"\n  {scope}  (n={len(cs)})")
        # arm2 is compared with gda_last rather than arm1, so that both sides
        # use validation for the same quantities
        paired(cs, "arm3_valfree", "arm1_gda_eb",
               "ensemble     3 val-free - 1 val-free GDA  [no validation]")
        paired(cs, "arm2_loo_multiset", "gda_last_val",
               "multiset     2 LOO mset - GDA val-searched [val fixed]")
        paired(cs, "arm4_le_gda", "arm2_loo_multiset",
               "cost of mset 4 LE-GDA   - 2 LOO mset       [val fixed]")
        paired(cs, "arm2_loo_multiset", "arm3_valfree",
               "cost of lam+a 2 LOO mset - 3 val-free      [mset fixed]")
        paired(cs, "arm2_loo_multiset", "arm3a_loolam_valalpha",
               "  of which lam 2 LOO mset - 3a LOO lam     [alpha searched]")
        paired(cs, "arm3a_loolam_valalpha", "arm3_valfree",
               "  of which a   3a LOO lam - 3 val-free     [lam fixed both]")
        paired(cs, "arm4_le_gda", "arm3_valfree",
               "price of val 4 LE-GDA   - 3 val-free")
        paired(cs, "arm1_gda_eb", "gda_last_val",
               "val-free tax 1 val-free GDA - GDA val-searched")
        paired(cs, "arm1_gda_eb", "arm0_zs",
               "floor check  1 val-free GDA - 0 zero-shot")

    # ---------------- per dataset ----------------
    print("\n" + "=" * 78)
    print("PER DATASET — Delta(3 - 1) and Delta(4 - 3)")
    print("=" * 78)
    dss = sorted({c["dataset"] for c in cells})
    print("  per-backbone cells are means over 3 seeds; the sd on the mean "
          "columns is ACROSS THE 4 BACKBONE MEANS")
    print(f"  {'dataset':<16}" + "".join(f"{bb:>12}" for bb in sorted(bys))
          + f"{'mean D(3-1)':>18}{'mean D(4-3)':>18}")
    for ds in dss:
        v, w = [], []
        for bb in sorted(bys):
            d = [c["acc"]["arm3_valfree"] - c["acc"]["arm1_gda_eb"]
                 for c in bys[bb] if c["dataset"] == ds]
            e = [c["acc"]["arm4_le_gda"] - c["acc"]["arm3_valfree"]
                 for c in bys[bb] if c["dataset"] == ds]
            v.append(mean(d) if d else float("nan"))
            w.append(mean(e) if e else float("nan"))
        print(f"  {ds:<16}" + "".join(f"{x:>+12.2f}" for x in v)
              + f"{mean(v):>+10.2f}+-{sd(v):<6.2f}"
              + f"{mean(w):>+10.2f}+-{sd(w):<6.2f}")

    # ---------------- is the effect dataset-driven or backbone-driven? -----
    if args.unit != "dataset":
        print("\n" + "=" * 78)
        print("SIGN CONSISTENCY — is Delta(3-1) a dataset fact or a backbone "
              "fact?")
        print("=" * 78)
        print("  If every backbone agrees on the sign for a dataset, the 44 "
              "settings are not\n  independent; report --unit dataset "
              "alongside.")
        agree = 0
        for ds in sorted({c["dataset"] for c in cells}):
            sg = [1 if mean([c["acc"]["arm3_valfree"] - c["acc"]["arm1_gda_eb"]
                             for c in bys[bb] if c["dataset"] == ds]) > 0
                  else -1 for bb in sorted(bys)]
            agree += abs(sum(sg)) == len(sg)
        print(f"\n  datasets where ALL {len(bys)} backbones agree on the sign: "
              f"{agree}/{len(set(c['dataset'] for c in cells))}")
        if agree == len({c["dataset"] for c in cells}):
            print("  all datasets agree across backbones; see --unit dataset "
                  "(n=11)")

    # ---------------- what the transferred configuration IS ----------------
    print("\n" + "=" * 78)
    print("THE TRANSFERRED CONFIGURATION — one row per (backbone, held-out ds)")
    print("=" * 78)
    for bb in sorted(bys):
        seen = {}
        for c in bys[bb]:
            seen[c["dataset"]] = c["cfg"]
        print(f"\n  {bb}")
        print(f"    {'held-out':<16}{'multiset':<34}{'a_fuse':>10}"
              f"{'a_arm1':>10}")
        for ds in sorted(seen):
            g = seen[ds]
            ms = " ".join(f"{s}x{m}" for s, m in zip(g["spaces"], g["mult"]))
            print(f"    {ds:<16}{ms:<34}{g['alpha_fuse']:>10.3g}"
                  f"{g['alpha_arm1']:>10.3g}")

    # ---------------- alpha plateau ----------------
    # Validation accuracy lost by using the transferred alpha instead of each
    # run's own best alpha; derivation records only, no test labels.
    print("\n" + "=" * 78)
    print("ALPHA PLATEAU — validation cost of the LOO alpha vs each run's own "
          "argmax")
    print("=" * 78)
    recs = [r for r in json.load(open(args.derive))["records"]
            if r["K"] == args.k]
    from scripts.run_config_transfer import ALPHAS, loo_config
    byb = collections.defaultdict(list)
    for r in recs:
        byb[r["backbone"]].append(r)
    # heavy-tailed, so order statistics first
    print(f"  {'backbone':<11}{'n':>4}{'median':>10}{'p90':>10}{'max':>9}"
          f"{'mean +- sd':>18}")
    for bb in sorted(byb):
        loss = []
        for r in byb[bb]:
            g = loo_config(recs, bb, r["dataset"])
            i = min(range(len(ALPHAS)),
                    key=lambda j: abs(math.log(ALPHAS[j] / g["alpha_fuse"])))
            cur = r["d"]["alpha_curve"]
            loss.append(max(cur) - cur[i])
        loss.sort()
        print(f"  {bb:<11}{len(loss):>4}{median(loss):>10.3f}"
              f"{loss[int(0.9 * (len(loss) - 1))]:>10.3f}{loss[-1]:>9.3f}"
              f"{mean(loss):>11.3f}+-{stdev(loss):<6.3f}")


if __name__ == "__main__":
    main()
