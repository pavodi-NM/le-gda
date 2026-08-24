"""Adjudicate the pre-registered kill conditions for the evidence-gated cache.

Pre-registered before the code was written. Paired per-cell contrasts and
sign tests only -- never unpaired medians of accuracy across heterogeneous
datasets (the cache-size protocol note).
"""

import argparse
import json
from math import comb
from statistics import median

FINE = ["stanford_cars", "fgvc", "oxford_flowers", "dtd"]   # H1 targets


def sign_test(deltas, tol=1e-9):
    """Two-sided exact binomial sign test on non-zero deltas."""
    pos = sum(d > tol for d in deltas)
    neg = sum(d < -tol for d in deltas)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return pos, neg, min(1.0, p)


def paired(cells, a, b):
    return [c["acc"][a] - c["acc"][b] for c in cells]


def report(cells, a, b, label):
    d = paired(cells, a, b)
    pos, neg, p = sign_test(d)
    print(f"  {label:<34} median {median(d):+6.3f}  win {pos:>3}/{pos+neg:<3} "
          f"p={p:.2e}  {'sig' if p < 0.05 else 'n.s.'}")
    return median(d), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="evidence/evidence_gate_killtest.json")
    ap.add_argument("--elig", type=float, default=3.0)
    args = ap.parse_args()
    cells = json.load(open(args.file))["cells"]

    # eligibility: C = fixed - zeroshot >= 3, evaluated at m=16 and applied to
    # both m values of the same (backbone, dataset, seed) triple.
    key = lambda c: (c["backbone"], c["dataset"], c["seed"])
    cbig = {key(c): c["C"] for c in cells if c["m"] == 16}
    elig = [c for c in cells if cbig.get(key(c), -99) >= args.elig]
    print(f"cells: {len(cells)} total, {len(elig)} eligible (C >= {args.elig})")

    for m in sorted({c["m"] for c in elig}, reverse=True):
        E = [c for c in elig if c["m"] == m]
        if not E:
            continue
        print("\n" + "=" * 74)
        print(f"m = {m}   ({len(E)} eligible cells)")
        print("=" * 74)

        # ---------------- KC1: is the gate non-degenerate? ----------------
        iqr = [c["gamma_iqr"] for c in E]
        xcl = [c["gamma_xclass_top10"] for c in E]
        n_ok = sum(v >= 0.05 for v in iqr)
        print(f"\nKC1 gate non-degenerate")
        print(f"  IQR(gamma) at predicted class: median {median(iqr):.3f}, "
              f"{n_ok}/{len(E)} cells >= 0.05")
        print(f"  spread across top-10 classes : median {median(xcl):.3f}")
        kc1 = not (median(iqr) < 0.05 and n_ok < 0.5 * len(E))
        print(f"  -> {'SURVIVES' if kc1 else 'KILL'}  "
              f"(kill iff median < 0.05 AND < 50% of cells >= 0.05)")

        # ---------------- KC2: does it change predictions? ----------------
        flip = [c["flip_gate_vs_fixed"] for c in E]
        print(f"\nKC2 gate consequential")
        print(f"  argmax flips vs fixed-alpha: median {median(flip):.2f}%  "
              f"range {min(flip):.2f}-{max(flip):.2f}%")
        kc2 = median(flip) >= 1.0
        print(f"  -> {'SURVIVES' if kc2 else 'KILL-TO-ALLOCATION'} "
              f"(kill iff median < 1.0%)")

        # ---------------- accuracy contrasts ----------------
        print(f"\nPaired accuracy contrasts (all eligible cells)")
        report(E, "gate", "fixed", "gate - fixed (Tip-Adapter)")
        report(E, "amu_raw", "fixed", "amu_raw - fixed")
        report(E, "amu_norm", "fixed", "amu_norm - fixed")
        report(E, "loglin", "fixed", "loglin - fixed")

        # KC3: distinguishable from AMU?
        best_amu = [max(c["acc"]["amu_raw"], c["acc"]["amu_norm"]) for c in E]
        d = [c["acc"]["gate"] - b for c, b in zip(E, best_amu)]
        pos, neg, p = sign_test(d)
        print(f"\nKC3 gate vs best-of-AMU")
        print(f"  median {median(d):+6.3f}  win {pos}/{pos+neg}  p={p:.2e}")
        kc3 = abs(median(d)) >= 0.10 and p < 0.05
        print("  -> " + ("DISTINGUISHABLE" if kc3 else
                         "PIVOT (indistinguishable from published heuristic)"))

        # KC4: does the text-evidence term earn its place?
        print(f"\nKC4 text-evidence term (the §2.4 punchline)")
        md, p4 = report(E, "gate", "gate_noeta", "gate - gate_noeta (eta=0)")
        etas = [c["sel"]["gate"]["eta"] for c in E]
        z = sum(e == 0.0 for e in etas)
        print(f"  selected eta: 0 on {z}/{len(E)} cells; "
              f"distribution " + str({e: etas.count(e)
                                      for e in sorted(set(etas))}))
        kc4 = md >= 0.05 and z <= 0.5 * len(E)
        print(f"  -> {'SUPPORTED' if kc4 else 'NOT SUPPORTED'}")

        # THE decisive ablation: is the sigmoid inert?
        print(f"\nDECISIVE ABLATION  is the sigmoid doing anything?")
        md5, p5 = report(E, "gate", "loglin", "gate - loglin (saturated limit)")
        print(f"  -> sigmoid is {'LOAD-BEARING' if abs(md5) >= 0.10 and p5 < 0.05 else 'INERT'}")

        # gate saturation diagnostic
        gmed = [c["gamma_med"] for c in E]
        print(f"  median gamma at predicted class: {median(gmed):.3f} "
              f"(>0.9 == operating in the log-linear regime)")

        # per-dataset breakdown
        print(f"\nPer-dataset (median paired delta vs fixed-alpha)")
        print(f"  {'dataset':<17}{'n':>3}{'gate':>8}{'loglin':>8}{'amu_n':>8}"
              f"{'IQRg':>8}{'flip%':>8}")
        for ds in sorted({c["dataset"] for c in E}):
            D = [c for c in E if c["dataset"] == ds]
            print(f"  {ds:<17}{len(D):>3}"
                  f"{median(paired(D, 'gate', 'fixed')):>+8.2f}"
                  f"{median(paired(D, 'loglin', 'fixed')):>+8.2f}"
                  f"{median(paired(D, 'amu_norm', 'fixed')):>+8.2f}"
                  f"{median(c['gamma_iqr'] for c in D):>8.3f}"
                  f"{median(c['flip_gate_vs_fixed'] for c in D):>8.2f}")

    # ---------------- instrument check I5: edge pinning ----------------
    print("\n" + "=" * 74)
    print("I5 edge pinning (fraction of ALL cells pinned at a grid edge)")
    print("=" * 74)
    for arm in cells[0]["sel"]:
        sp = sum(c["sel"][arm]["s_pin"] for c in cells) / len(cells)
        bp = sum(c["sel"][arm]["b_pin"] for c in cells) / len(cells)
        flag = "  <-- REPORT AS A BOUND" if max(sp, bp) > 0.5 else ""
        print(f"  {arm:<12} offset {sp:5.1%}   beta {bp:5.1%}{flag}")

    # eligibility sensitivity
    print("\nEligibility sensitivity (median gate - fixed, m=16)")
    for t in (1.0, 2.0, 3.0, 5.0):
        E = [c for c in cells if c["m"] == 16 and cbig.get(key(c), -99) >= t]
        if E:
            print(f"  C >= {t}: n={len(E):>3}  "
                  f"gate-fixed {median(paired(E, 'gate', 'fixed')):+.3f}   "
                  f"gate-loglin {median(paired(E, 'gate', 'loglin')):+.3f}")


if __name__ == "__main__":
    main()


def reference_check(path="evidence/evidence_gate_killtest.json",
                    ref="evidence/compact_cache_all.json"):
    """Compare against the STOCK-grid Tip-Adapter number from paper 1.

    Our `fixed` arm is Tip-Adapter searched over the matched 85-point budget
    every arm gets. That budget reaches alpha-equivalents far above the stock
    grid's cap of 10, and on some cells it finds a higher-validation,
    lower-test configuration -- i.e. the matched budget mildly OVERFITS a small
    val set, so `fixed` can sit below the published-style number. The
    arm-vs-arm contrasts are unaffected (shared budget), but any claim against
    Tip-Adapter as normally run needs this second reference.
    """
    import json
    from statistics import mean, median
    cells = json.load(open(path))["cells"]
    stock = {(c["backbone"], c["dataset"]): c["acc"]["TIP"]
             for c in json.load(open(ref)) if c.get("K") == 16}
    rows = {}
    for c in cells:
        if c["m"] != 16:
            continue
        rows.setdefault((c["backbone"], c["dataset"]), []).append(c)
    print("\n" + "=" * 74)
    print("REFERENCE CHECK vs stock-grid Tip-Adapter (paper 1, seed-mean)")
    print("=" * 74)
    d_fix, d_gate, d_lin = [], [], []
    for k, cs in sorted(rows.items()):
        if k not in stock:
            continue
        t = stock[k]
        f = mean(c["acc"]["fixed"] for c in cs)
        g = mean(c["acc"]["gate"] for c in cs)
        l = mean(c["acc"]["loglin"] for c in cs)
        if mean(c["C"] for c in cs) < 3.0:
            continue
        d_fix.append(f - t); d_gate.append(g - t); d_lin.append(l - t)
        print(f"  {k[0]:<10}{k[1]:<16}stock={t:6.2f}  fixed{f - t:+6.2f}  "
              f"gate{g - t:+6.2f}  loglin{l - t:+6.2f}")
    for lab, d in (("fixed", d_fix), ("gate", d_gate), ("loglin", d_lin)):
        pos, neg, p = sign_test(d)
        print(f"  {lab:<8} vs stock Tip-Adapter: median {median(d):+.3f}  "
              f"win {pos}/{pos+neg}  p={p:.2e}")
