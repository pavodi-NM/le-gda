"""Score N -- pre-registered before the code was written; kill conditions in the paper.

WRITTEN WHILE THE GRID WAS RUNNING, so the analysis choices are not selected
after seeing the answer.

KC-N1 asks whether a nonlinear fusion rule is USEFUL. KC-N2 asks whether it is
genuinely OUTSIDE the bounded family, by requiring the validation-selected rule
to beat `w_oracle_S` -- the TEST-ORACLE linear ceiling from the weighting run on
the identical cells. A val-selected rule beating an oracle linear rule cannot be
linear weighting in disguise. Failing KC-N2 while passing KC-N1 means the gain
sits inside what weighting could already have bought, and the honest claim is
"another route to the same +0.15", not a new mechanism.

Four rules were registered together, so the best is a selection over four: if
more than one clears KC-N1 the pre-registration binds us to the SIMPLEST, and
the adopted rule must be confirmed on 132 cells before it enters the paper.
"""

import argparse
import glob
import json
from statistics import median

from scripts.analyze_evidence_gate import sign_test

RULES = ["F1", "F2", "F3", "F4"]
NAMES = {"F1": "per-sample z-score", "F2": "per-sample rank",
         "F3": "median over spaces", "F4": "softmax, mean of probs"}
BAR_N1 = 0.10


def rows_of(paths):
    out = []
    for p in paths:
        d = json.load(open(p))
        out += d["rows"] if isinstance(d, dict) and "rows" in d else d
    return out


def line(label, d, indent="  "):
    if not d:
        print(f"{indent}{label:<30} (none)")
        return None
    pos, neg, p = sign_test(d)
    print(f"{indent}{label:<30} median {median(d):+.3f}  mean {sum(d)/len(d):+.3f}"
          f"  >0 in {pos:>2}/{pos+neg:<3} p={p:.2e} "
          f"{'sig' if p < 0.05 else 'n.s.'}  n={len(d)}")
    return median(d), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", nargs="+",
                    default=sorted(glob.glob("evidence/nonlinear_fusion*.json")))
    args = ap.parse_args()
    rows = rows_of(args.file)
    print(f"{len(rows)} cells: {len({r['dataset'] for r in rows})} datasets x "
          f"{len({r['backbone'] for r in rows})} backbones, K=16\n")

    print("=" * 78)
    print("NONLINEAR FUSION — test accuracy vs F0 (uniform mean, the method today)")
    print("  every rule selected on VALIDATION, identical greedy budget")
    print("=" * 78)
    res = {}
    for rl in RULES:
        res[rl] = line(f"{rl}  {NAMES[rl]}",
                       [r["rules"][rl]["test"] - r["rules"]["F0"]["test"]
                        for r in rows])

    # join to the weighting ceiling on identical cells -- asserted, not assumed
    W = {}
    for f in sorted(glob.glob("evidence/weighting_ceiling*.json")):
        for w in json.load(open(f))["rows"]:
            W[(w["backbone"], w["dataset"], w["seed"])] = w
    joined = [(r, W[(r["backbone"], r["dataset"], r["seed"])]) for r in rows
              if (r["backbone"], r["dataset"], r["seed"]) in W]
    drift = max((abs(w["le_gda"] - r["rules"]["F0"]["test"]) for r, w in joined),
                default=None)
    print(f"\n  joined to the weighting ceiling on {len(joined)}/{len(rows)} cells;"
          f" max |le_gda drift| = {drift:.2e}" if joined else "\n  NO JOIN")

    print("\n" + "=" * 78)
    print("KC-N2 — vs the TEST-ORACLE linear ceiling (w_oracle_S)")
    print("=" * 78)
    orc = line("w_oracle_S - F0  (the bound)",
               [w["w_oracle_S"] - r["rules"]["F0"]["test"] for r, w in joined])
    n2 = {}
    for rl in RULES:
        n2[rl] = line(f"{rl} - w_oracle_S",
                      [r["rules"][rl]["test"] - w["w_oracle_S"] for r, w in joined])

    print("\n" + "=" * 78)
    print("KILL CONDITIONS")
    print("=" * 78)
    winners = [rl for rl in RULES if res[rl][0] >= BAR_N1 and res[rl][1] < 0.05]
    best = max(RULES, key=lambda rl: res[rl][0])
    ok1 = bool(winners)
    print(f"  KC-N1  best rule {best} at {res[best][0]:+.3f} "
          f"(bar {BAR_N1:+.2f}, p<0.05) -> {'PASS' if ok1 else 'FAIL'}")
    if winners:
        print(f"         clearing the bar: {winners} -> ADOPT THE SIMPLEST "
              f"({winners[0]}), then confirm on 132 cells")
    ok2 = ok1 and n2[winners[0]][0] > 0 if winners else False
    if ok1:
        print(f"  KC-N2  {winners[0]} - w_oracle_S = {n2[winners[0]][0]:+.3f} > 0 "
              f"-> {'PASS' if ok2 else 'FAIL'}")
    else:
        print("  KC-N2  not evaluated — KC-N1 failed, there is nothing to adopt")

    print("\n" + "=" * 78)
    if not ok1:
        print("VERDICT: NONLINEAR FUSION IS DEAD. The linear family was already\n"
              "  capped and the escape outside it is empty — the fusion stage is\n"
              "  finished. Do not propose a sixth combiner.")
    elif ok2:
        print(f"VERDICT: {winners[0]} PASSES BOTH. A validation-selected rule beats\n"
              "  the test-oracle LINEAR ceiling, so this is genuinely outside the\n"
              "  bounded family. Confirm on 132 cells, then it is the method.")
    else:
        print(f"VERDICT: {winners[0]} is useful (KC-N1) but sits INSIDE the linear\n"
              "  bound (KC-N2 fails). Claim it as another route to the ~+0.15\n"
              "  weighting already had, not as a new mechanism.")
    print("=" * 78)


if __name__ == "__main__":
    main()
