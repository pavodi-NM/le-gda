"""Adjudicate the kill conditions KC-S1..S4 (write-up §6.4) for the 11-dataset x 4-backbone sweep.

The §3 verdict on 4 datasets was computed ad hoc. This is the scorer of
record for the scale-up, and it is written BEFORE the numbers exist.

ORDER IS MANDATED (plan §0.3 step 4, and analyze_allocation.py): arm health
is printed and gated FIRST. The specific way this instrument died once is
documented in the write-up §4 -- a matched null whose members had collapsed to
chance, which KC-L2 then "passed" against by +58 points. So the null's own
accuracy is checked against `gda_last` here before any gate is scored: an
information-equivalent null that lands far BELOW the final layer is broken, not
beaten.

KC-S3 is not a pass/fail gate. It is an instruction about how to report: if the
mean is carried by <= 2 datasets, the aggregate may not be quoted alone. It is
scored here so that instruction cannot be quietly skipped.
"""

import argparse
import json
from statistics import median

from scripts.analyze_evidence_gate import sign_test

KC_S1_BAR = 0.30       # le_gda - gda_last, generalisation
KC_S2_BAR = 0.30       # le_gda - null_ridge, decisive
KC_S4_MIN = 3          # backbones that must show the effect, of 4
NULL_FLOOR = -1.0      # null_ridge may not sit this far below gda_last


def rep(cells, a, b, label=None, bar=None, indent="  "):
    """Paired median + exact sign test for arms a - b over `cells`."""
    d = [c["acc"][a] - c["acc"][b] for c in cells
         if a in c["acc"] and b in c["acc"]]
    if not d:
        print(f"{indent}{label or f'{a} - {b}':<34} (no cells)")
        return None, None, False
    pos, neg, p = sign_test(d)
    md = median(d)
    ok = (md >= bar and p < 0.05) if bar is not None else None
    tag = "" if bar is None else ("  PASS" if ok else "  FAIL")
    print(f"{indent}{(label or f'{a} - {b}'):<34} median {md:+6.3f}  "
          f"win {pos:>3}/{pos + neg:<3} p={p:.2e} "
          f"{'sig' if p < 0.05 else 'n.s.'}{tag}")
    return md, p, bool(ok)


def health(cells):
    """Is the null alive, and is any arm degenerate? Printed before any gate."""
    print("=" * 78)
    print("ARM HEALTH — read this before any verdict below")
    print("  the write-up §4: a matched null whose members collapse to chance")
    print("  makes KC-S2 meaningless. null_ridge is information-equivalent to")
    print("  gda_last, so it must land NEAR it, not far below.")
    print("=" * 78)
    live = True
    print(f"  {'arm':<14}{'n':>5}{'mean acc':>10}{'min':>9}{'vs gda_last':>13}")
    arms = sorted({a for c in cells for a in c["acc"]})
    for a in arms:
        v = [c["acc"][a] for c in cells if a in c["acc"]]
        d = [c["acc"][a] - c["acc"]["gda_last"] for c in cells if a in c["acc"]]
        flag = ""
        if a == "null_ridge" and median(d) < NULL_FLOOR:
            flag, live = "  <-- NULL IS BROKEN (§4)", False
        if min(v) < 5.0:
            flag, live = "  <-- AT CHANCE (E10)", False
        print(f"  {a:<14}{len(v):>5}{sum(v) / len(v):>10.2f}{min(v):>9.2f}"
              f"{median(d):>+13.3f}{flag}")

    # a greedy selection that never leaves the final layer is not an ensemble
    subs = [c["sel"]["le_gda"]["subset"] for c in cells if "le_gda" in c["sel"]]
    if subs:
        only_last = sum(set(s) == {"last"} for s in subs) / len(subs)
        print(f"\n  le_gda subsets that are `last` alone: {only_last:.1%}"
              + ("  <-- METHOD IS INERT" if only_last > 0.5 else ""))
        if only_last > 0.5:
            live = False
    if not live:
        print("\n  *** AN ARM IS DEGENERATE. Kill conditions below are NOT "
              "interpretable. ***")
    return live


def concentration(cells, a, b):
    """KC-S3: per-dataset deltas, and how few datasets carry the mean."""
    per = {}
    for ds in sorted({c["dataset"] for c in cells}):
        d = [c["acc"][a] - c["acc"][b] for c in cells if c["dataset"] == ds]
        per[ds] = (median(d), len(d))
    for ds, (m, n) in sorted(per.items(), key=lambda kv: -kv[1][0]):
        print(f"  {ds:<16}(n={n:>2})  {m:+6.2f}" + ("   <-- negative"
                                                    if m < 0 else ""))
    tot = sum(m for m, _ in per.values())
    pos = sorted((m for m, _ in per.values() if m > 0), reverse=True)
    carry, run = 0, 0.0
    while carry < len(pos) and run < 0.5 * tot:
        run += pos[carry]
        carry += 1
    print(f"\n  {carry} of {len(per)} datasets carry half the total effect "
          f"({tot:+.2f} summed)")
    if carry <= 2:
        print("  *** KC-S3: the mean is CARRIED BY <= 2 DATASETS. Report this ")
        print("      as a TARGETED method. The aggregate may not be quoted "
              "alone. ***")
    return carry, per


def main():
    ap = argparse.ArgumentParser()
    # imagenet is swept into its own file (it is the only cell that can OOM),
    # so the scorer takes several and pools the cells.
    ap.add_argument("--file", nargs="+",
                    default=["evidence/layer_ensemble_full.json"])
    ap.add_argument("--elig", type=float, default=3.0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--force", action="store_true",
                    help="score the gates on a dead instrument, for inspection")
    args = ap.parse_args()
    cells, pfs = [], {}
    for fp in args.file:
        d = json.load(open(fp))
        cells += d["cells"] if isinstance(d, dict) else d
        if isinstance(d, dict) and "preflight" in d:
            pfs[fp] = d["preflight"]
    seen = {(c["backbone"], c["dataset"], c["K"], c["seed"]) for c in cells}
    assert len(seen) == len(cells), (
        f"{len(cells) - len(seen)} duplicate cells across {args.file} -- "
        f"pooling the same cell twice would fake the sign test's n")

    bbs = sorted({c["backbone"] for c in cells})
    dss = sorted({c["dataset"] for c in cells})
    print(f"\n{len(cells)} cells: {len(dss)} datasets x {len(bbs)} backbones x "
          f"{len({c['seed'] for c in cells})} seeds x "
          f"K in {sorted({c['K'] for c in cells}, reverse=True)}")
    print(f"backbones: {bbs}")
    print(f"datasets:  {dss}\n")

    K = [c for c in cells if c["K"] == args.k]
    live = health(K)
    E = [c for c in K if c["C"] >= args.elig]
    print(f"\neligible at K={args.k}: {len(E)}/{len(K)} cells with "
          f"C >= {args.elig}")
    drop = sorted({c["dataset"] for c in K} - {c["dataset"] for c in E})
    if drop:
        print(f"  datasets with NO eligible cell: {drop}")
    if not live and not args.force:
        print("\nREFUSING TO SCORE KILL CONDITIONS on a degenerate instrument.")
        print("Fix the arm, re-run, then score. --force for inspection only.")
        return

    print("\n" + "=" * 78)
    print(f"KILL CONDITIONS (write-up §6.4), K={args.k}, C >= {args.elig}")
    print(f"  KC-S1 generalisation  le_gda - gda_last   >= +{KC_S1_BAR}, p<0.05")
    print(f"  KC-S2 the null        le_gda - null_ridge >= +{KC_S2_BAR}, p<0.05"
          "   <-- decisive")
    print("=" * 78)
    s1 = rep(E, "le_gda", "gda_last", "KC-S1  le_gda - gda_last", KC_S1_BAR)
    s2 = rep(E, "le_gda", "null_ridge", "KC-S2  le_gda - null_ridge", KC_S2_BAR)
    print()
    rep(E, "null_ridge", "gda_last", "(null_ridge - gda_last)")
    rep(E, "le_gda", "gda_pre", "(le_gda - gda_pre, best single)")

    print("\n" + "=" * 78)
    print("KC-S3 concentration — a median over datasets hides a single carrier")
    print("=" * 78)
    carry, _ = concentration(E, "le_gda", "gda_last")

    print("\n" + "=" * 78)
    print(f"KC-S4 backbone — the effect must appear on >= {KC_S4_MIN} of 4")
    print("=" * 78)
    hits = 0
    for bb in bbs:
        sub = [c for c in E if c["backbone"] == bb]
        m, p, _ = rep(sub, "le_gda", "gda_last", f"{bb:<12}le_gda - gda_last")
        if m is not None and m >= KC_S1_BAR and p < 0.05:
            hits += 1
    print(f"\n  {hits}/{len(bbs)} backbones tested show the effect at the "
          f"KC-S1 bar")
    if len(bbs) < 4:
        print(f"  *** KC-S4 IS NOT DECIDABLE YET — only {len(bbs)} of 4 "
              f"backbones present.")
        print(f"      Phase A tests ViTs only; the ResNets need the GAP readout "
              f"(§6.1). ***")
    elif hits < KC_S4_MIN:
        print("  *** KC-S4 FAILS. A backbone-specific effect is a SCOPE LIMIT "
              "and belongs in the abstract. ***")

    print("\n" + "=" * 78)
    print("Other K (not gated; KC-S1 was n.s. at K=4 on the 4-dataset set)")
    print("  UNFILTERED: the C >= 3 eligibility is registered for K=16 only, "
          "and\n  applying it at K=1 would drop cells for having a small "
          "cache gain")
    print("=" * 78)
    for k in sorted({c["K"] for c in cells}, reverse=True):
        sub = [c for c in cells if c["K"] == k]
        rep(sub, "le_gda", "gda_last", f"K={k:<3} le_gda - gda_last "
                                       f"(n={len(sub)})")

    print("\n" + "=" * 78)
    print("Per-layer accuracy — KC-L3 said single intermediate layers are FAR")
    print("worse. Does that hold at scale?")
    print("=" * 78)
    spaces = [a[4:] for a in sorted({a for c in E for a in c["acc"]})
              if a.startswith("gda_")]
    for s in spaces:
        rep(E, f"gda_{s}", "gda_last", f"gda_{s} - gda_last")
    freq = {}
    for c in E:
        for s in c["sel"]["le_gda"]["subset"]:
            freq[s] = freq.get(s, 0) + 1
    print("\n  greedy selection frequency: "
          + "  ".join(f"{s}={n}" for s, n in
                      sorted(freq.items(), key=lambda kv: -kv[1])))

    if pfs:
        print("\n" + "=" * 78)
        print("PRE-FLIGHT (from each run's own record)")
        print("=" * 78)
        for fp, pfd in pfs.items():
            bad = [f"{a}.{p}" for a, ps in pfd.get("pinning", {}).items()
                   for p, v in ps.items() if max(v["lo"], v["hi"]) > 0.5]
            print(f"  {fp}")
            print("    nesting: " + "; ".join(
                f"{n['label']} max|d|={n['max_abs_diff']:.1e} "
                f"{'OK' if n['ok'] else 'FAIL'}"
                for n in pfd.get("nesting", [])))
            print(f"    edge-pinned arms (>50%, report as BOUNDS): "
                  f"{bad if bad else 'none'}")

    print("\n" + "=" * 78)
    verdict = "PASS" if (s1[2] and s2[2]) else "FAIL"
    print(f"VERDICT: KC-S1 {'PASS' if s1[2] else 'FAIL'}, "
          f"KC-S2 {'PASS' if s2[2] else 'FAIL'} -> {verdict}")
    if verdict == "PASS" and carry <= 2:
        print("  ... but KC-S3 says TARGETED, not general. Quote the "
              "per-dataset table, never the mean alone.")
    if verdict == "FAIL":
        print("  LE-GDA is a fine-grained special case, not a method "
              "(§6.4 KC-S1).")
    print("=" * 78)


if __name__ == "__main__":
    main()
