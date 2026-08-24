"""Standing pre-flight checks for every experiment run script.

Mandated by the project pre-registration protocol, written after the
2026-08-13 audit of the 2026-08-12 session (plan §0.2, errors E1-E14). Four of
those fourteen errors were instrument defects that a run script could have
caught itself, and each was found by the luck of a later check rather than by a
standard:

    E10  the matched null was inactive on 48/48 cells and controlled for
         nothing -- a null that silently no-ops looks exactly like a null that
         passes, because both produce "no difference".
    E11  a scale-broken w put a hard activation threshold where nothing
         cleared it (see plan §3.2.1).
    E12  fp16 storage of a quantity that is later exponentiated moved top-1 by
         up to 0.24 points.
    E13  grid edge-pinning was discovered three separate times, on alpha, on
         beta and on the gate offset, and prevented none of them.

The rule these encode: **an arm that cannot move cannot be evidence, and that
applies to the CONTROL arms exactly as much as to the treatment arms.**

Usage -- collect during the run, adjudicate before any kill condition:

    from scripts.preflight import Preflight
    pf = Preflight("allocation_stage1")
    pf.assert_fp32(f_val=f_val, f_te=f_te)              # once, up front
    ...
    pf.grid(cell, arm, "beta", beta_sel, betas)         # per arm per cell
    pf.grid(cell, arm, "alpha", alpha_sel, alphas)
    pf.activation(cell, arm, "w", frac_active, frac_at_bound)
    ...
    pf.nesting("R=0 == calibrated Tip-Adapter", max_abs_diff, tol=1e-3)
    ok = pf.report()          # prints the table; False blocks the verdict

`report()` returns False if any check fails. **Print it BEFORE the kill
conditions and refuse to score them if it is False.** A verdict computed on top
of a dead instrument is not a result -- that is E1's whole lesson.
"""

from collections import defaultdict

import torch

# An arm pinned at a grid edge on more than this fraction of cells has its
# number reported as a BOUND, not a value (logpool prereg §3, allocation §7).
PIN_BOUND = 0.50
# An arm whose mechanism is active on fewer than this fraction of its own
# parameters is degenerate: it cannot serve as a treatment OR as a control.
ACTIVE_MIN = 0.50


class Preflight:
    def __init__(self, name, pin_bound=PIN_BOUND, active_min=ACTIVE_MIN):
        self.name = name
        self.pin_bound = pin_bound
        self.active_min = active_min
        self.pins = defaultdict(lambda: defaultdict(list))   # arm -> param -> []
        self.acts = defaultdict(lambda: defaultdict(list))   # arm -> mech  -> []
        self.nests = []
        self.dtypes = []

    # ---- fp32: E12 --------------------------------------------------------
    def assert_fp32(self, **tensors):
        """Every tensor whose values reach an exponent must be fp32.

        fp16 near a cosine of 1.0 resolves to ~5e-4; multiplied by beta up to
        80 that is a ~2% error in the cache term, measured at 0.24 points of
        top-1 in the allocation run and 0.12 in the gate run.
        """
        for k, t in tensors.items():
            ok = isinstance(t, torch.Tensor) and t.dtype == torch.float32
            self.dtypes.append((k, str(getattr(t, "dtype", type(t))), ok))
            if not ok:
                raise AssertionError(
                    f"[preflight/{self.name}] {k} is "
                    f"{getattr(t, 'dtype', type(t))}, must be torch.float32 "
                    f"(E12: fp16 is unsafe anywhere a cosine is exponentiated)")

    # ---- grid edges: E13 --------------------------------------------------
    def grid(self, cell, arm, param, selected, values, tol=1e-9,
             lo_boundary=False, hi_boundary=False):
        """Record whether a val-selected parameter landed on a grid edge.

        `lo_boundary` / `hi_boundary` mark that end of the grid as the
        parameter's DOMAIN boundary rather than an arbitrary cutoff -- e.g. a
        variance or a ridge coefficient at 0, which is the unregularized limit
        and a genuine member of the model family. Selecting a domain boundary
        is INFORMATIVE, not a defect: widening the grid cannot move it, because
        there is nowhere to widen to. It is still reported, just not failed.

        Everything else stays a failure. E13's lesson is that a grid edge means
        the number is a bound; that lesson does not apply when the edge is the
        edge of the parameter itself.
        """
        lo, hi = min(values), max(values)
        self.pins[arm][param].append(
            (cell, selected <= lo + tol, selected >= hi - tol,
             lo_boundary, hi_boundary))

    # ---- mechanism activation: E10, E11 -----------------------------------
    def activation(self, cell, arm, mech, frac_active, frac_at_bound=0.0):
        """Record that `arm`'s mechanism actually did something on this cell.

        Call this for EVERY arm that has a mechanism, especially the matched
        null. `frac_active` is the fraction of the arm's own parameters that
        are non-degenerate (e.g. |w| > 1e-6); `frac_at_bound` the fraction
        sitting on a clamp, which is the opposite failure.
        """
        self.acts[arm][mech].append((cell, frac_active, frac_at_bound))

    # ---- nesting: the degenerate setting must reproduce the baseline ------
    def nesting(self, label, max_abs_diff, tol=1e-3, acc_diff=0.0,
                acc_tol=1e-6):
        """Assert the degenerate parameterization reproduces a known baseline.

        Plan §4.6 step 4, "the step people skip and regret": if R=0 / w=0 /
        m=K does not reproduce the published number, every downstream contrast
        is against an unknown quantity.
        """
        ok = max_abs_diff <= tol and abs(acc_diff) <= acc_tol
        self.nests.append((label, max_abs_diff, tol, acc_diff, ok))
        return ok

    # ---- adjudication -----------------------------------------------------
    def report(self, verbose=True):
        ok = True
        p = print if verbose else (lambda *a, **k: None)
        p("\n" + "=" * 78)
        p(f"PRE-FLIGHT — {self.name}")
        p("  no kill condition may be scored while any line below reads FAIL")
        p("=" * 78)

        if self.dtypes:
            bad = [k for k, _, o in self.dtypes if not o]
            p(f"\nfp32 (E12): {len(self.dtypes)} tensors checked, "
              f"{'all fp32' if not bad else 'NOT fp32: ' + ', '.join(bad)}")
            ok &= not bad

        if self.nests:
            p("\nnesting — the degenerate setting reproduces the baseline "
              "(plan §4.6):")
            for lab, d, tol, da, o in self.nests:
                p(f"  {lab:<52} max|d|={d:.2e} (tol {tol:.0e}) "
                  f"dacc={da:+.6f}  {'OK' if o else 'FAIL'}")
                ok &= o
        else:
            p("\nnesting: NONE DECLARED — FAIL. Every run script must show that "
              "\n  its degenerate setting reproduces the published baseline.")
            ok = False

        if self.pins:
            p("\ngrid edge-pinning (E13) — per arm, fraction of cells "
              "val-selected AT an edge:")
            p(f"  {'arm':<12}{'param':<10}{'n':>5}{'lo%':>8}{'hi%':>8}   verdict")
            for arm in sorted(self.pins):
                for prm in sorted(self.pins[arm]):
                    rows = self.pins[arm][prm]
                    n = len(rows)
                    flo = sum(r[1] for r in rows) / n
                    fhi = sum(r[2] for r in rows) / n
                    lo_dom = any(r[3] for r in rows)
                    hi_dom = any(r[4] for r in rows)
                    # a domain boundary cannot be widened away, so it is
                    # reported but never failed
                    bad = (max(fhi if not hi_dom else 0.0,
                               flo if not lo_dom else 0.0) > self.pin_bound)
                    note = "ok"
                    if bad:
                        note = "BOUND, not a value — widen the grid and re-run"
                    elif (lo_dom and flo > self.pin_bound) or (
                            hi_dom and fhi > self.pin_bound):
                        note = ("at the parameter's DOMAIN boundary — "
                                "informative, not a defect")
                    p(f"  {arm:<12}{prm:<10}{n:>5}{flo:>7.1%}{fhi:>7.1%}   "
                      + note)
                    ok &= not bad

        if self.acts:
            p("\nmechanism activation (E10/E11) — EVERY arm, controls included:")
            p(f"  {'arm':<12}{'mech':<10}{'n':>5}{'active':>9}{'at bound':>10}"
              "   verdict")
            for arm in sorted(self.acts):
                for mech in sorted(self.acts[arm]):
                    rows = self.acts[arm][mech]
                    n = len(rows)
                    a = sum(r[1] for r in rows) / n
                    b = sum(r[2] for r in rows) / n
                    bad = a < self.active_min
                    p(f"  {arm:<12}{mech:<10}{n:>5}{a:>8.1%}{b:>9.1%}   "
                      + ("DEGENERATE — this arm cannot be evidence, and if it "
                         "is a null it controls for nothing" if bad else "ok"))
                    ok &= not bad
        else:
            p("\nmechanism activation: NONE DECLARED. If any arm has a "
              "mechanism\n  (a weight, a gate, a direction set), this is a FAIL "
              "— see E10.")

        p("\n" + ("PRE-FLIGHT PASSED — kill conditions may now be scored"
                  if ok else
                  "PRE-FLIGHT FAILED — DO NOT SCORE ANY KILL CONDITION. "
                  "Fix the instrument first;\n  a verdict on a dead instrument "
                  "measures the implementation, not the idea (E1)."))
        p("=" * 78 + "\n")
        return ok

    def to_dict(self):
        """Serialisable summary, to be stored alongside the cells in the JSON."""
        return {
            "name": self.name,
            "fp32": [{"tensor": k, "dtype": d, "ok": o} for k, d, o in
                     self.dtypes],
            "nesting": [{"label": l, "max_abs_diff": d, "tol": t,
                         "acc_diff": a, "ok": o}
                        for l, d, t, a, o in self.nests],
            "pinning": {arm: {prm: {
                "n": len(rows),
                "lo": sum(r[1] for r in rows) / len(rows),
                "hi": sum(r[2] for r in rows) / len(rows),
                "lo_domain": any(r[3] for r in rows),
                "hi_domain": any(r[4] for r in rows)}
                for prm, rows in prms.items()}
                for arm, prms in self.pins.items()},
            "activation": {arm: {mech: {
                "n": len(rows),
                "active": sum(r[1] for r in rows) / len(rows),
                "at_bound": sum(r[2] for r in rows) / len(rows)}
                for mech, rows in mechs.items()}
                for arm, mechs in self.acts.items()},
        }
