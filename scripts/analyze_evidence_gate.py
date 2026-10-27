"""Exact two-sided sign test, used by scripts/analyze_config_transfer.py."""

from math import comb


def sign_test(deltas, tol=1e-9):
    """Two-sided exact binomial sign test on non-zero deltas.

    Returns (number positive, number negative, p-value).
    """
    pos = sum(d > tol for d in deltas)
    neg = sum(d < -tol for d in deltas)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return pos, neg, min(1.0, p)
