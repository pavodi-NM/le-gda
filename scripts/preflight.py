"""Sanity checks recorded by every experiment run script.

Each run collects these during the sweep and prints a summary at the end; the
summary is also stored in the output JSON under "preflight".

    fp32        tensors that are later exponentiated must be float32
    grid edges  how often a validation-selected hyperparameter sits at the
                edge of its search grid (if often, the grid is too narrow and
                the reported number is a bound)
    activation  the mechanism of an arm is actually active on its parameters
    nesting     a degenerate setting of the method reproduces the baseline it
                reduces to (e.g. LE-GDA restricted to the final layer is GDA-CLIP)

Usage:

    from scripts.preflight import Preflight
    pf = Preflight("layer_ensemble")
    pf.assert_fp32(f_val=f_val, f_te=f_te)              # once, up front
    ...
    pf.grid(cell, arm, "alpha", alpha_sel, alphas)      # per arm per cell
    ...
    pf.nesting("restricted to final == GDA-CLIP", max_abs_diff, tol=1e-3)
    ok = pf.report()          # prints the summary; False if any check fails
"""

from collections import defaultdict

import torch

# An arm pinned at a grid edge on more than this fraction of cells has its
# number reported as a bound, not a value.
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

    # ---- fp32 -------------------------------------------------------------
    def assert_fp32(self, **tensors):
        """Every tensor whose values reach an exponent must be fp32.

        fp16 near a cosine of 1.0 resolves to ~5e-4; multiplied by beta up to
        80 that is a ~2% error in the cache term, enough to move top-1 by a
        few tenths of a point.
        """
        for k, t in tensors.items():
            ok = isinstance(t, torch.Tensor) and t.dtype == torch.float32
            self.dtypes.append((k, str(getattr(t, "dtype", type(t))), ok))
            if not ok:
                raise AssertionError(
                    f"[preflight/{self.name}] {k} is "
                    f"{getattr(t, 'dtype', type(t))}, must be torch.float32 "
                    f"(fp16 is unsafe anywhere a cosine is exponentiated)")

    # ---- grid edges -------------------------------------------------------
    def grid(self, cell, arm, param, selected, values, tol=1e-9,
             lo_boundary=False, hi_boundary=False):
        """Record whether a val-selected parameter landed on a grid edge.

        `lo_boundary` / `hi_boundary` mark that end of the grid as the
        parameter's DOMAIN boundary rather than an arbitrary cutoff -- e.g. a
        variance or a ridge coefficient at 0, which is the unregularized limit
        and a genuine member of the model family. Selecting a domain boundary
        is informative, not a defect: widening the grid cannot move it, because
        there is nowhere to widen to. It is still reported, just not failed.
        """
        lo, hi = min(values), max(values)
        self.pins[arm][param].append(
            (cell, selected <= lo + tol, selected >= hi - tol,
             lo_boundary, hi_boundary))

    # ---- mechanism activation ---------------------------------------------
    def activation(self, cell, arm, mech, frac_active, frac_at_bound=0.0):
        """Record that `arm`'s mechanism actually did something on this cell.

        Call this for every arm that has a mechanism, control arms included.
        `frac_active` is the fraction of the arm's own parameters that
        are non-degenerate (e.g. |w| > 1e-6); `frac_at_bound` the fraction
        sitting on a clamp, which is the opposite failure.
        """
        self.acts[arm][mech].append((cell, frac_active, frac_at_bound))

    # ---- nesting: the degenerate setting must reproduce the baseline ------
    def nesting(self, label, max_abs_diff, tol=1e-3, acc_diff=0.0,
                acc_tol=1e-6):
        """Assert the degenerate parameterization reproduces a known baseline.

        If the degenerate setting does not reproduce the baseline, every
        downstream comparison is against an unknown quantity.
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
        p("=" * 78)

        if self.dtypes:
            bad = [k for k, _, o in self.dtypes if not o]
            p(f"\nfp32: {len(self.dtypes)} tensors checked, "
              f"{'all fp32' if not bad else 'NOT fp32: ' + ', '.join(bad)}")
            ok &= not bad

        if self.nests:
            p("\nnesting — the degenerate setting reproduces the baseline:")
            for lab, d, tol, da, o in self.nests:
                p(f"  {lab:<52} max|d|={d:.2e} (tol {tol:.0e}) "
                  f"dacc={da:+.6f}  {'OK' if o else 'FAIL'}")
                ok &= o
        else:
            p("\nnesting: none declared — FAIL. Every run script must show that "
              "\n  its degenerate setting reproduces the baseline.")
            ok = False

        if self.pins:
            p("\ngrid edge-pinning — per arm, fraction of cells "
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
            p("\nmechanism activation — every arm, controls included:")
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
                      + ("DEGENERATE — this arm is inactive" if bad else "ok"))
                    ok &= not bad
        else:
            p("\nmechanism activation: none declared.")

        p("\n" + ("PRE-FLIGHT PASSED" if ok else
                  "PRE-FLIGHT FAILED — check the lines marked above before "
                  "using the results."))
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
