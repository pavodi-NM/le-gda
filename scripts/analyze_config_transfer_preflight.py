"""Preflight for the LOO configuration-transfer ablation (the ablation write-up).

Decides whether the four-arm sweep is worth running, from evidence that already
exists -- no GPU, no re-fit. Two quantities, both per backbone at K=16:

  (1) MULTISET CONCENTRATION. Table 4 pools selection counts over everything.
      Transfer needs the ranking to agree ACROSS DATASETS within a backbone, so
      here the count is tabulated per (backbone, dataset) and the agreement of
      the per-dataset top-ranked space is reported.

  (2) ALPHA DISPERSION. The fusion alpha grid is log-spaced with a fixed ratio,
      so dispersion is reported in GRID STEPS -- the only unit in which "a
      single fixed alpha will cost you" is a statement about this instrument.
      Bar carried from the plan: more than ~2 grid steps of spread and arm 3
      (fully validation-free) is expected to be the weak arm.

Read the DECISION line at the bottom. It is a gate on scope, not on truth: a
dispersed alpha does not invalidate arms 1-2, it just means the fusion strength
is the part that does not transfer.
"""

import argparse
import collections
import glob
import json
import math
from statistics import median, pstdev

# the grid of record, from run_layer_ensemble.py -- 40 points EXTENDING the old
# 30 at the identical ratio, so one "grid step" is well defined.
ALPHA_STEP_LOG10 = 6 / 29
LAM_LO, LAM_HI, N_LAM = 1e-8, 1e2, 12
LAM_STEP_LOG10 = math.log10(LAM_HI / LAM_LO) / (N_LAM - 1)


def load_derive(fp, K):
    """Selections from a `--phase derive` record instead of the F0 sweep.

    THE PREFLIGHT MUST BE ABOUT THE OPERATOR BEING ABLATED. Read from the F0
    sweep it describes the uniform mean of §2.4; under Eq. 7's per-query
    standardisation the greedy search sees z-scored scores and picks a
    different multiset and a different alpha, so the F0 preflight would be
    answering a question about a variant that is not in the submission.
    """
    out = []
    for r in json.load(open(fp))["records"]:
        if r["K"] != K:
            continue
        out.append({"backbone": r["backbone"], "dataset": r["dataset"],
                    "K": r["K"], "seed": r["seed"],
                    "sel": {"le_gda": {"subset": r["d"]["subset_sup"],
                                       "alpha": r["d"]["alpha_sup"]},
                            **{f"gda_{sp}": {"reg": lam}
                               for sp, lam in r["d"]["lam_sup"].items()}}})
    return out


def load(files, K):
    """Cells at shot K from the standard-split sweep files.

    The split is taken from the FILENAME, not from the cell's `test_split`
    field, because imagenet is the one dataset whose full test split could not
    be re-extracted (a data limit, not a code one -- memory
    `capped-test-splits-blocker`). Its cells therefore record `test_split:
    "test"` while living in the `*_testfull.json` run of record, and filtering
    on the field silently drops the largest dataset in the sweep.
    """
    cells = []
    for f in files:
        assert "testfull" in f, f"{f} is not a standard-split sweep file"
        for c in json.load(open(f))["cells"]:
            if c["K"] == K:
                cells.append(c)
    return cells


def spread_steps(vals, step_log10):
    """IQR and full range of log10(v), expressed in grid steps."""
    lg = sorted(math.log10(v) for v in vals)
    n = len(lg)
    q1, q3 = lg[n // 4], lg[(3 * n) // 4]
    return ((q3 - q1) / step_log10, (lg[-1] - lg[0]) / step_log10,
            pstdev(lg) / step_log10, 10 ** median(lg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", nargs="+",
                    default=sorted(glob.glob(
                        "evidence/layer_ensemble_*testfull.json")))
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--derive", default=None,
                    help="a run_config_transfer --phase derive record; use it "
                         "to preflight the F1 operator rather than F0's.")
    args = ap.parse_args()

    cells = (load_derive(args.derive, args.k) if args.derive
             else load(args.file, args.k))
    bys = collections.defaultdict(list)
    for c in cells:
        bys[c["backbone"]].append(c)

    print("=" * 78)
    print(f"PREFLIGHT — LOO configuration transfer, K={args.k}, "
          f"standard splits, operator "
          f"{'F1 (Eq. 7, the paper)' if args.derive else 'F0 (§2.4, uniform mean)'}")
    print("=" * 78)

    verdict = {}
    for bb in sorted(bys):
        cs = bys[bb]
        dss = sorted({c["dataset"] for c in cs})
        print(f"\n{'-' * 78}\n{bb}   ({len(cs)} runs, {len(dss)} datasets)"
              f"\n{'-' * 78}")

        # ---------- (1) multiset concentration ----------
        pooled = collections.Counter()
        per_ds_top, sizes = {}, collections.Counter()
        for ds in dss:
            cnt = collections.Counter()
            for c in cs:
                if c["dataset"] != ds:
                    continue
                sub = c["sel"]["le_gda"]["subset"]
                sizes[len(sub)] += 1
                cnt.update(sub)
                pooled.update(sub)
            per_ds_top[ds] = cnt.most_common()

        rank = [s for s, _ in pooled.most_common()]
        print(f"  pooled selection count: "
              + "  ".join(f"{s}:{pooled[s]}" for s in rank))
        print(f"  |L*| histogram: "
              + "  ".join(f"{k}:{v}" for k, v in sorted(sizes.items()))
              + f"   modal |L*| = {sizes.most_common(1)[0][0]}")
        print(f"\n  {'dataset':<16}{'top-ranked space (count)':<28}"
              f"{'rank of pooled-top':>20}")
        agree = 0
        for ds in dss:
            mc = per_ds_top[ds]
            top = f"{mc[0][0]} ({mc[0][1]})"
            order = [s for s, _ in mc]
            r = order.index(rank[0]) + 1 if rank[0] in order else 99
            agree += (mc[0][0] == rank[0])
            print(f"  {ds:<16}{top:<28}{r:>20}")
        print(f"\n  per-dataset top == pooled top ({rank[0]}) on "
              f"{agree}/{len(dss)} datasets")

        # ---------- (2) alpha dispersion ----------
        al = [c["sel"]["le_gda"]["alpha"] for c in cs]
        iqr, rng, sd, med = spread_steps(al, ALPHA_STEP_LOG10)
        print(f"\n  fusion alpha over {len(al)} runs: median {med:.3g}"
              f"   IQR {iqr:.2f} steps   sd {sd:.2f}   full range {rng:.2f}")
        # per-dataset alpha medians: is the dispersion BETWEEN datasets or
        # within (seed noise)? Only the between part is what a fixed alpha pays.
        dsm = {ds: median([c["sel"]["le_gda"]["alpha"]
                           for c in cs if c["dataset"] == ds]) for ds in dss}
        biqr, brng, bsd, _ = spread_steps(list(dsm.values()), ALPHA_STEP_LOG10)
        print(f"  between-dataset (medians of {len(dsm)}): IQR {biqr:.2f} steps"
              f"   sd {bsd:.2f}   full range {brng:.2f}")
        print("   " + "  ".join(f"{ds[:7]}:{v:.3g}" for ds, v in
                                sorted(dsm.items(), key=lambda kv: kv[1])))

        # ---------- (2b) per-space lambda dispersion ----------
        print(f"\n  {'space':<8}{'n':>4}{'median lam':>13}{'IQR steps':>12}"
              f"{'range':>9}")
        for s in rank:
            lv = [c["sel"][f"gda_{s}"]["reg"] for c in cs]
            i2, r2, _, m2 = spread_steps(lv, LAM_STEP_LOG10)
            print(f"  {s:<8}{len(lv):>4}{m2:>13.3g}{i2:>12.2f}{r2:>9.2f}")

        tight_multiset = agree >= 9 * len(dss) / 11
        tight_alpha = biqr <= 2.0
        verdict[bb] = (agree, len(dss), biqr, tight_multiset, tight_alpha)

    print("\n" + "=" * 78)
    print("DECISION")
    print("=" * 78)
    print(f"  {'backbone':<12}{'top-space agree':>17}{'alpha IQR (steps)':>20}"
          f"{'verdict':>22}")
    n_tm = n_ta = 0
    for bb, (a, n, biqr, tm, ta) in sorted(verdict.items()):
        n_tm += tm
        n_ta += ta
        v = ("run everything" if tm and ta else
             "arms 1-2 (alpha dispersed)" if tm else
             "multiset dispersed" if ta else "both dispersed")
        print(f"  {bb:<12}{f'{a}/{n}':>17}{biqr:>20.2f}{v:>22}")
    print(f"\n  multiset concentrated on {n_tm}/4 backbones; "
          f"alpha tight on {n_ta}/4")
    if n_tm >= 3 and n_ta >= 3:
        print("  -> RUN ALL FOUR ARMS.")
    elif n_tm >= 3:
        print("  -> RUN ARMS 1-2 as the primary claim; arm 3 is expected weak "
              "and is still worth measuring, since Delta(4-3) is the price of "
              "validation access and nobody reports it.")
    else:
        print("  -> The useful representation set is dataset-dependent. Run "
              "the arms, expect the supplement framing.")


if __name__ == "__main__":
    main()
