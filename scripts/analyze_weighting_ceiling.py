"""Score W -- pre-registered before the code was written; kill conditions in the paper.

WRITTEN WHILE THE GRID WAS STILL RUNNING, deliberately, so the analysis choices
are not selected after seeing the answer.

The three arms are all ceilings with the weights AND alpha chosen on test, so
none of them is an achievable accuracy. The only quantities that mean anything
are the DIFFERENCES from `le_gda`, and the primary is `w_oracle_S - le_gda`:
the headroom a combiner bolted on top of LE-GDA could occupy if it were handed
the test labels. The statistic is the MEDIAN over cells, as registered.

`w_oracle_S >= le_gda` holds by construction (greedy's quarter-multiplicities
are on the Q=8 grid and alpha is also oracle'd), so KC-W3 checks it as an
instrument assertion. A violation is a bug, not a finding.

A cell that selected ONE distinct space has no weighting freedom at all -- the
simplex over one point is a single vector -- so its `w_oracle_S - le_gda` is
exactly the alpha-only headroom. Those cells are KEPT, not filtered: they are
20% of the grid and dropping the zero-headroom end after seeing it is the
post-hoc exclusion this project forbids. They are also reported separately, so
the reader can see the split rather than only the pooled number.
"""

import argparse
import glob
import json
from statistics import median

from scripts.analyze_evidence_gate import sign_test

VIT = ("ViT-B/16", "ViT-B/32")
BAR = 0.30


def rows_of(paths):
    out = []
    for p in paths:
        d = json.load(open(p))
        out += d["rows"] if isinstance(d, dict) and "rows" in d else d
    return out


def report(label, rows, key, indent="  "):
    if not rows:
        print(f"{indent}{label:<26} (no cells)")
        return None
    d = [r[key] - r["le_gda"] for r in rows]
    pos, neg, p = sign_test(d)
    print(f"{indent}{label:<26} median {median(d):+.3f}  mean {sum(d)/len(d):+.3f}  "
          f"max {max(d):+.3f}  >0 in {pos:>2}/{len(d):<3} p={p:.2e}  n={len(rows)}")
    return median(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", nargs="+",
                    default=sorted(glob.glob("evidence/weighting_ceiling*.json")))
    args = ap.parse_args()
    rows = rows_of(args.file)
    bbs = sorted({r["backbone"] for r in rows})
    dss = sorted({r["dataset"] for r in rows})
    print(f"{len(rows)} cells: {len(dss)} datasets x {len(bbs)} backbones, K=16")
    print(f"backbones: {bbs}\ndatasets:  {dss}\n")

    print("=" * 78)
    print("CEILINGS — all differences from le_gda; weights AND alpha on TEST")
    print("=" * 78)
    m1 = report("w_oracle_S  (PRIMARY)", rows, "w_oracle_S")
    m2 = report("w_oracle_all", rows, "w_oracle_all")
    report("sel_oracle  (context)", rows, "sel_oracle")

    print("\n  by readout family:")
    for lbl, rs in (("ViT (CLS)", [r for r in rows if r["backbone"] in VIT]),
                    ("ResNet (GAP)", [r for r in rows if r["backbone"] not in VIT])):
        report(f"{lbl} w_oracle_S", rs, "w_oracle_S", indent="    ")
        report(f"{lbl} w_oracle_all", rs, "w_oracle_all", indent="    ")

    print("\n  by how many DISTINCT spaces the method selected:")
    for n in sorted({r["n_distinct"] for r in rows}):
        rs = [r for r in rows if r["n_distinct"] == n]
        report(f"{n} distinct space(s)", rs, "w_oracle_S", indent="    ")

    print("\n  per dataset (w_oracle_S):")
    for ds in dss:
        report(ds, [r for r in rows if r["dataset"] == ds], "w_oracle_S",
               indent="    ")

    print("\n" + "=" * 78)
    print("KILL CONDITIONS")
    print("=" * 78)
    ok1 = m1 is not None and m1 >= BAR
    print(f"  KC-W1  median(w_oracle_S - le_gda) = {m1:+.3f} >= {BAR} -> "
          f"{'PASS' if ok1 else 'FAIL'}")
    ok2 = m2 is not None and m2 >= BAR
    print(f"  KC-W2  median(w_oracle_all - le_gda) = {m2:+.3f} >= {BAR} -> "
          f"{'PASS' if ok2 else 'FAIL'}")
    # TOLERANCE, and why it is not 1e-9. Accuracy here is a count over test
    # items rendered in fp32, so the SMALLEST REAL violation is one item --
    # 0.030 on fgvc (n=3333), 0.0025 on the largest split we run. A tie between
    # two algebraically identical weightings computed by different paths
    # (`stack().mean(0)` vs a weighted sum) lands ~1e-6 apart in fp32. 1e-9 flags
    # that FP noise as a bug: the first Q=12 run tripped on RN50/fgvc/seed 1 at
    # -7.2e-07, where the oracle had found LE-GDA's OWN vector (0.667, 0.333).
    # 1e-4 sits two orders below any real one-item violation and two above the
    # noise. This is the `max|d|` lesson from the write-up again: a
    # tolerance must be set by the granularity of the metric, not by float dtype.
    TOL = 1e-4
    bad = [r for r in rows if r["w_oracle_S"] < r["le_gda"] - TOL]
    bad2 = [r for r in rows if r["w_oracle_all"] < r["w_oracle_S"] - TOL]
    ok3 = not bad and not bad2
    print(f"  KC-W3  w_oracle_S >= le_gda on {len(rows)-len(bad)}/{len(rows)} "
          f"cells, w_oracle_all >= w_oracle_S on {len(rows)-len(bad2)}/{len(rows)}"
          f" (both must be 100%) -> {'PASS' if ok3 else 'FAIL — BUG'}")
    for r in bad[:5]:
        print(f"         VIOLATION {r['backbone']} {r['dataset']} s={r['seed']}: "
              f"{r['w_oracle_S']:.4f} < {r['le_gda']:.4f}")

    print("\n" + "=" * 78)
    if not ok3:
        print("VERDICT: INSTRUMENT FAILURE — fix the bug, score nothing.")
    elif ok1:
        print("VERDICT: KC-W1 PASSES — there is real headroom over uniform "
              "fusion.\n  Expand to the full 132 cells (imagenet included) "
              "before building anything.")
    elif ok2:
        print("VERDICT: KC-W1 FAILS, KC-W2 PASSES — the headroom is in WHICH "
              "spaces are\n  chosen, not in how they are weighted. A combiner "
              "is dead; selection is not.")
    else:
        print("VERDICT: BOTH FAIL — the entire weighting family is closed.\n"
              "  No combiner over these spaces can pay, latent or otherwise. "
              "Do not build it.")
    print("=" * 78)


if __name__ == "__main__":
    main()
