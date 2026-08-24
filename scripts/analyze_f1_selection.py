"""Selection frequencies under Eq. 7 (F1), recounted from the run of record.

WHY THIS EXISTS. The write-up (§5d.4) tabulates greedy selection counts under
§2.4's UNIFORM MEAN. §5e replaced that fusion operator, and the greedy
search under F1 sees z-scored scores -- so it can and does pick a different
multiset. There was no F1 selection table anywhere in the write-up, which is
exactly how the F0 counts survived into a draft that describes the F1 method.

The data live in `evidence/nonlinear_fusion_*.json` under `rules.F1.subset`,
one multiset per cell. No markdown holds them; this script is the only path
from the record to a table.

CONTROL. The same files carry `rules.F0.subset`. Counting F0 here and checking
it against §5d.4's published counts establishes that the counting convention
is the same one -- otherwise a difference between the F0 and F1 tables could be
a difference in how they were counted rather than in what was selected.
"""

import argparse
import collections
import glob
import json

VIT = ["L4", "L6", "L8", "L10", "L11", "pre", "last"]
RN = ["R1", "R2", "R3", "R4", "last"]


# The record, named in the write-up §5e's head-to-head row. A bare
# `nonlinear_fusion_*` glob also matches two SUPERSEDED 4-dataset pilots
# (`nonlinear_fusion_{RN50,vitb16}.json`), which silently double-count
# dtd/fgvc/oxford_flowers/stanford_cars and inflate every ViT count. Measured:
# the bare glob gives 156 cells at K=16 instead of 132, and moves `L6`'s
# ViT-B/16 count from 7 to 31.
RECORD = ["evidence/nonlinear_fusion_full_*.json",
          "evidence/nonlinear_fusion_k14_*.json",
          "evidence/nonlinear_fusion_k28_*.json"]


def load(ks):
    rows, seen = [], set()
    for pat in RECORD:
        for f in glob.glob(pat):
            for r in json.load(open(f)).get("rows", []):
                key = (r["backbone"], r["dataset"], r["K"], r["seed"])
                if r["K"] in ks and "F1" in r.get("rules", {}):
                    assert key not in seen, f"duplicate cell {key} in {f}"
                    seen.add(key)
                    rows.append(r)
    for k in ks:
        n = sum(1 for r in rows if r["K"] == k)
        assert n == 132, f"K={k}: {n} cells, expected 132 (11 x 4 x 3)"
    return rows


def control(rows):
    """F0 subsets here must equal the layer_ensemble record, cell for cell.

    If they do not, a difference between the F0 and F1 tables below could be a
    difference in counting rather than in what the greedy search selected.
    """
    idx = {}
    for f in glob.glob("evidence/layer_ensemble_*testfull*.json"):
        for c in json.load(open(f)).get("cells", []):
            idx[(c["backbone"], c["dataset"], c["K"], c["seed"])] = \
                c["sel"]["le_gda"]["subset"]
    n, bad = 0, 0
    for r in rows:
        k = (r["backbone"], r["dataset"], r["K"], r["seed"])
        if k in idx:
            n += 1
            bad += (idx[k] != r["rules"]["F0"]["subset"])
    print(f"CONTROL: F0 subsets match the layer_ensemble record on "
          f"{n - bad}/{n} cells" + ("" if bad == 0 else f"  <-- {bad} DISAGREE"))
    assert bad == 0, "counting convention differs from the F0 record"


def counts(rows, rule):
    per_bb = collections.defaultdict(collections.Counter)
    sizes = collections.defaultdict(collections.Counter)
    pooled = collections.Counter()
    for r in rows:
        sub = r["rules"][rule]["subset"]
        per_bb[r["backbone"]].update(sub)
        sizes[r["backbone"]][len(sub)] += 1
        pooled.update(sub)
    return per_bb, sizes, pooled


def show(rows, rule, label):
    per_bb, sizes, pooled = counts(rows, rule)
    print(f"\n{'=' * 78}\n{label}   ({len(rows)} cells)\n{'=' * 78}")
    print("  pooled greedy frequency: "
          + ", ".join(f"`{s}` {n}" for s, n in pooled.most_common()))
    print(f"\n  {'backbone':<11}{'n':>4}   selection count, most-selected first")
    for bb in sorted(per_bb):
        c = per_bb[bb]
        n = sum(1 for r in rows if r["backbone"] == bb)
        print(f"  {bb:<11}{n:>4}   "
              + "  ".join(f"{s}:{v}" for s, v in c.most_common()))
    print(f"\n  {'backbone':<11}|L*| histogram (1..4)          modal |L*|")
    for bb in sorted(sizes):
        h = sizes[bb]
        print(f"  {bb:<11}"
              + "  ".join(f"{k}:{h[k]}" for k in sorted(h))
              + f"          {h.most_common(1)[0][0]}")
    return per_bb, pooled


# Solo deficit vs `gda_last`, the write-up §5d.4. Measured under F0 anchoring
# -- they are the numbers the paper quotes, and no F1 single-space table
# exists. Flagged wherever used.
SOLO = {"L4": -6.90, "L6": -5.60, "L8": -4.22, "L10": -1.29, "L11": +0.16,
        "pre": +0.12, "last": 0.00, "R1": -10.24, "R2": -5.26, "R3": -0.43,
        "R4": -0.12}


def spearman(a, b):
    def rk(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb, n = rk(a), rk(b), len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) *
           sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def richness(f0_bb, f1_bb):
    """Does selection track SOLO STRENGTH? The mechanism claim lives here.

    §5b.7's headline -- and the working title -- rest on the most-selected
    space being one that is much WORSE alone ("complementarity without
    richness"). That was measured under F0. If F1 selection instead tracks solo
    strength, the sentence does not survive the change of fusion operator.
    """
    print(f"\n{'=' * 78}")
    print("DOES SELECTION TRACK SOLO STRENGTH? Spearman(solo deficit, count)")
    print("  positive rho = the STRONGER a space is alone, the more it is "
          "selected")
    print("  (solo deficits are §5d.4's, measured under F0 -- no F1 "
          "single-space table exists)")
    print("=" * 78)
    print(f"  {'backbone':<11}{'F0 rho':>10}{'F1 rho':>10}   most-selected "
          f"space (its solo deficit)")
    for bb in sorted(f1_bb):
        sp = sorted(f1_bb[bb])
        r0 = spearman([SOLO[x] for x in sp], [f0_bb[bb][x] for x in sp])
        r1 = spearman([SOLO[x] for x in sp], [f1_bb[bb][x] for x in sp])
        t0 = f0_bb[bb].most_common(1)[0][0]
        t1 = f1_bb[bb].most_common(1)[0][0]
        print(f"  {bb:<11}{r0:>+10.3f}{r1:>+10.3f}   F0 `{t0}` "
              f"({SOLO[t0]:+.2f})  ->  F1 `{t1}` ({SOLO[t1]:+.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="+", type=int, default=[16])
    args = ap.parse_args()
    rows = load(set(args.ks))
    assert rows, "no F1 rows found"
    control(rows)

    f0_bb, f0_pool = show(rows, "F0", "F0 — §2.4 uniform mean (the CONTROL, "
                                     "matches §5d.4's convention)")
    f1_bb, f1_pool = show(rows, "F1", "F1 — Eq. 7 per-query standardisation "
                                      "(THE PAPER'S METHOD)")

    print(f"\n{'=' * 78}\nWHAT CHANGED — F1 minus F0, pooled\n{'=' * 78}")
    spaces = sorted(set(f0_pool) | set(f1_pool),
                    key=lambda s: -(f1_pool[s] - f0_pool[s]))
    print(f"  {'space':<8}{'F0':>6}{'F1':>6}{'delta':>8}")
    for s in spaces:
        print(f"  {s:<8}{f0_pool[s]:>6}{f1_pool[s]:>6}"
              f"{f1_pool[s] - f0_pool[s]:>+8}")

    richness(f0_bb, f1_bb)

    print(f"\n  per backbone, rank of each space (F0 -> F1)")
    for bb in sorted(f1_bb):
        o0 = [s for s, _ in f0_bb[bb].most_common()]
        o1 = [s for s, _ in f1_bb[bb].most_common()]
        moves = []
        for s in o1:
            r0 = o0.index(s) + 1 if s in o0 else None
            moves.append(f"{s}:{r0 or '-'}→{o1.index(s) + 1}")
        print(f"  {bb:<11}" + "  ".join(moves))


if __name__ == "__main__":
    main()
