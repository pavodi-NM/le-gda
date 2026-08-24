"""Score the LOO configuration-transfer ablation. Report: the ablation write-up.

Written before the numbers exist. The question is CLAP's: how much of LE-GDA's
gain survives when the configuration is not searched on the held-out dataset's
validation split?

  Delta(3 - 1)  does the transferred ensemble beat a validation-free GDA
                baseline? This is the claim.
  Delta(2 - 1)  does the MULTISET transfer, when lambda and alpha are still
                searched? Without it a disappointing arm 3 is uninterpretable.
  Delta(4 - 3)  the price of going validation-free -- what validation access
                actually buys, which nobody in this literature reports.
  Delta(3b - 3) the anchor control. §2.2's anchor is a VALIDATION median, so
                arm 3 substitutes a support-set anchor; 3b is arm 3 with the
                published anchor and nothing else changed.

REPORTING RULE, carried from gotcha (c): the table quotes MEANS over datasets
because that is what the literature quotes, but every verdict is a PAIRED
per-cell contrast with an exact sign test. An unpaired median of accuracy
across heterogeneous datasets produced a wrong verdict once already.
"""

import argparse
import collections
import json
import math
from statistics import median, mean, pstdev, stdev

from scipy.stats import wilcoxon

from scripts.analyze_evidence_gate import sign_test

ARMS = ["arm0_zs", "arm1_gda_eb", "arm2_loo_multiset", "arm3_valfree",
        "arm3a_loolam_valalpha", "arm3b_valanchor", "arm4_le_gda"]


def to_settings(cells):
    """Average the 3 seeds within each (backbone, dataset) -> 44 SETTINGS.

    THE UNIT OF ANALYSIS IS THE SETTING, NOT THE CELL. Three seeds of one
    dataset share its classes, its split and its text prompts; they are
    repeated measurements, not independent trials, and a sign test over 132
    cells silently triples its own n. The write-up's main text reports 36/44, so
    scoring this supplement over cells would have it contradict the protocol
    the paper states.
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
         "arm3a_loolam_valalpha": "3a LOO lam", "arm3b_valanchor": "3b val-anc",
         "arm4_le_gda": "4 LE-GDA"}
ALPHA_STEP_LOG10 = 6 / 29


def attach_arm4_f1(cells, k):
    """Arm 4 for the F1 run is §5e's record (`rules.F1.test`)."""
    import glob
    idx = {}
    for f in glob.glob("evidence/nonlinear_fusion_*.json"):
        if "preregistration" in f:
            continue
        for r in json.load(open(f)).get("rows", []):
            idx[(r["backbone"], r["dataset"], r["K"], r["seed"])] = r
    n = 0
    for c in cells:
        r = idx.get((c["backbone"], c["dataset"], c["K"], c["seed"]))
        assert r is not None, f"no F1 record for {c['backbone']}/{c['dataset']}"
        c["acc"]["arm4_le_gda"] = r["rules"]["F1"]["test"]
        n += 1
    print(f"  joined arm 4 (§5e F1) onto {n} cells")


def attach_arm4(cells, k):
    """Arm 4 is the sweep of record, joined on the cell key.

    It is NOT recomputed: recomputing would re-pay the full lambda search for a
    number that already exists, and joining makes the pairing exact -- same
    draw, same seed, same items. The zero-shot column is the alignment assert,
    the same one make_le_gda_head_to_head.py uses across its three sources.
    """
    import glob
    idx = {}
    for f in glob.glob("evidence/layer_ensemble_*testfull.json"):
        for c in json.load(open(f))["cells"]:
            idx[(c["backbone"], c["dataset"], c["K"], c["seed"])] = c
    n = 0
    for c in cells:
        r = idx.get((c["backbone"], c["dataset"], c["K"], c["seed"]))
        assert r is not None, f"no record cell for {c['backbone']}/{c['dataset']}"
        assert abs(r["zs"] - c["acc"]["arm0_zs"]) < 1e-6, (
            f"SPLIT MISALIGNMENT {c['backbone']}/{c['dataset']}: zero-shot "
            f"{r['zs']:.4f} in the record vs {c['acc']['arm0_zs']:.4f} here")
        c["acc"]["arm4_le_gda"] = r["acc"]["le_gda"]
        c["acc"]["gda_last_val"] = r["acc"]["gda_last"]
        n += 1
    print(f"  joined arm 4 onto {n} cells; zero-shot aligned on all of them")


def to_datasets(cells):
    """Average over BACKBONES as well as seeds -> 11 dataset-level units.

    THE CONSERVATIVE UNIT, and §8 is why it is needed. The sign of
    Delta(3-1) is identical on all four backbones for every one of the 11
    datasets: seven datasets are positive four times out of four and four are
    negative four times out of four. The four backbone measurements of a
    dataset are therefore not four independent trials of the sign -- they are
    one dataset-level fact measured four times, and a test over 44 settings
    treats a perfectly correlated replicate as new evidence.

    Reported ALONGSIDE the 44-setting test, never instead of it: 44 is the
    convention the main text uses, 11 is the number that is not
    anti-conservative. Where the two disagree, the 11 is the one to believe.
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


def paired(cells, a, b, label, bar=None):
    """Paired per-cell contrast. The sd is over the DELTAS, not over accuracy.

    That is the only spread that means anything for a paired contrast: the sd
    of accuracy across heterogeneous datasets is dominated by how hard the
    datasets are, which the pairing has already removed.
    """
    d = [c["acc"][a] - c["acc"][b] for c in cells
         if a in c["acc"] and b in c["acc"]]
    if not d:
        print(f"    {label:<52} (not defined for this fusion rule)")
        return None, None, None, None
    pos, neg, p = sign_test(d)
    md, mn = median(d), mean(d)
    sd = stdev(d) if len(d) > 1 else float("nan")
    # WILCOXON SIGNED-RANK alongside the sign test. The sign test discards the
    # magnitudes, which is the least powerful use of a paired design; the
    # signed-rank test uses them. Both are reported because they answer
    # slightly different questions -- the sign test asks whether the arm wins
    # more often, the signed-rank test whether it wins by more -- and because
    # a verdict that depends on which one is quoted is a verdict that should
    # be reported as borderline rather than resolved by choosing.
    try:
        pw = float(wilcoxon(d, alternative="two-sided").pvalue)
    except ValueError:                       # all-zero differences
        pw = 1.0
    tag = "" if bar is None else ("  PASS" if (md >= bar and p < 0.05)
                                  else "  FAIL")
    print(f"    {label:<52} mean {mn:+6.3f} +-{sd:6.3f}  median {md:+6.3f}  "
          f"win {pos:>3}/{pos + neg:<3} sign p={p:.2e} "
          f"wilcox p={pw:.2e} "
          f"{'sig' if p < 0.05 else 'n.s.'}{tag}")
    return mn, sd, md, p, pw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="evidence/config_transfer.json")
    ap.add_argument("--derive", default="evidence/config_transfer_derive.json")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--unit", choices=["setting", "cell", "dataset"],
                    default="setting",
                    help="`setting` averages the 3 seeds within a (backbone, "
                         "dataset) -> 44 units, the convention the write-up's "
                         "main text uses. `dataset` also averages over "
                         "backbones -> 11 units, the conservative unit (§8: "
                         "the sign of Delta(3-1) is backbone-independent, so "
                         "the 4 backbones are not independent replicates). "
                         "`cell` keeps 132 and is for diagnosis only.")
    args = ap.parse_args()

    cells = [c for c in json.load(open(args.file))["cells"] if c["K"] == args.k]
    fusion = cells[0].get("fusion", "F0")
    print("=" * 78)
    print(f"LOO CONFIGURATION TRANSFER — fusion {fusion}, K={args.k}, "
          f"{len(cells)} cells")
    if fusion == "F1":
        print("  Eq. 7's per-query standardisation — THE PAPER'S METHOD")
    else:
        print("  §2.4's uniform mean — a VARIANT, not the submitted method")
    print("=" * 78)
    (attach_arm4_f1 if fusion == "F1" else attach_arm4)(cells, args.k)
    if args.unit == "setting":
        cells = to_settings(cells)
        print(f"  unit of analysis: {len(cells)} SETTINGS "
              f"(3 seeds averaged within each backbone x dataset)")
    elif args.unit == "dataset":
        cells = to_datasets(to_settings(cells))
        print(f"  unit of analysis: {len(cells)} DATASETS "
              f"(seeds AND backbones averaged) — the conservative unit")

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
        # sd is ACROSS THE 11 PER-DATASET MEANS -- the between-dataset spread
        # of accuracy. It is large by construction (eurosat and imagenet are
        # not the same problem) and says nothing about whether an arm beats
        # another; the paired table below is what does that.
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

    # ---------------- the verdicts, paired ----------------
    print("\n" + "=" * 78)
    print(f"PAIRED CONTRASTS — per {args.unit}, exact sign test")
    print("=" * 78)
    for scope, cs in [("ALL", cells)] + sorted(bys.items()):
        print(f"\n  {scope}  (n={len(cs)})")
        # ORDER MATTERS: the firewalled contrast is first because it is the
        # only one that speaks to validation-freeness. `arm2 - arm1` is NOT in
        # this list -- arm 2 searches lambda and alpha on validation and arm 1
        # does not, so that contrast conflates "ensemble vs single space" with
        # "validation vs none" and cannot be quoted as multiset evidence.
        # `arm2 - gda_last` is its honest replacement: validation access is
        # held fixed on both sides.
        paired(cs, "arm3_valfree", "arm1_gda_eb",
               "THE CLAIM    3 val-free - 1 val-free GDA  [both firewalled]",
               bar=0.0)
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
        paired(cs, "arm3b_valanchor", "arm3_valfree",
               "anchor ctrl  3b val-anc - 3 val-free")
        paired(cs, "arm1_gda_eb", "gda_last_val",
               "val-free tax 1 val-free GDA - GDA val-searched")
        paired(cs, "arm1_gda_eb", "arm0_zs",
               "floor check  1 val-free GDA - 0 zero-shot")

    # ---------------- per dataset, the claim only ----------------
    print("\n" + "=" * 78)
    print("PER DATASET — Delta(3 - 1), the claim, and Delta(4 - 3), the price")
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
              + f"{mean(v):>+10.2f}+-{stdev(v):<6.2f}"
              + f"{mean(w):>+10.2f}+-{stdev(w):<6.2f}")

    # ---------------- is the effect dataset-driven or backbone-driven? -----
    if args.unit != "dataset":
        print("\n" + "=" * 78)
        print("SIGN CONSISTENCY — is Delta(3-1) a dataset fact or a backbone "
              "fact?")
        print("=" * 78)
        print("  If every backbone agrees on the sign for a dataset, then the "
              "4 backbone\n  measurements are one fact counted four times, and "
              "an n=44 test is\n  anti-conservative. This is scored so that "
              "cannot be quietly skipped.")
        agree = 0
        for ds in sorted({c["dataset"] for c in cells}):
            sg = [1 if mean([c["acc"]["arm3_valfree"] - c["acc"]["arm1_gda_eb"]
                             for c in bys[bb] if c["dataset"] == ds]) > 0
                  else -1 for bb in sorted(bys)]
            agree += abs(sum(sg)) == len(sg)
        print(f"\n  datasets where ALL {len(bys)} backbones agree on the sign: "
              f"{agree}/{len(set(c['dataset'] for c in cells))}")
        if agree == len({c["dataset"] for c in cells}):
            print("  *** UNANIMOUS. The effect is DATASET-DRIVEN and "
                  "BACKBONE-INDEPENDENT. The n=44")
            print("      test counts each dataset 4 times; quote --unit "
                  "dataset (n=11) beside it. ***")

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

    # ---------------- the alpha-plateau diagnostic ----------------
    # The preflight measures the DISPERSION of the argmax alpha in log space.
    # What a fixed alpha actually costs is the DEPTH of the validation curve
    # away from that argmax, and a flat curve makes a dispersed argmax cheap.
    # This is computed from the derivation records alone -- no test labels, no
    # extra fits -- and it is the honest correction to the preflight's verdict.
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
    # order statistics lead here because the quantity is heavy-tailed; the
    # mean +- sd is carried alongside rather than instead, since a mean over a
    # heavy tail is not the number to reason with.
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

    # anchor substitution: how far is the support anchor from the val anchor?
    print("\n  support-anchor / validation-anchor ratio (arms 1 and 3 must "
          "substitute it)")
    print(f"  {'backbone':<11}{'median':>10}{'p10':>10}{'p90':>10}"
          f"{'mean +- sd':>18}")
    for bb in sorted(byb):
        rt = sorted(v for r in byb[bb] for v in r["d"]["anchor_ratio"].values())
        print(f"  {bb:<11}{median(rt):>10.3f}{rt[int(0.1 * (len(rt) - 1))]:>10.3f}"
              f"{rt[int(0.9 * (len(rt) - 1))]:>10.3f}"
              f"{mean(rt):>11.3f}+-{stdev(rt):<6.3f}")


if __name__ == "__main__":
    main()
