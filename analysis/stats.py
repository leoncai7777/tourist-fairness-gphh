"""
Statistics of the main comparison (Step 6 plan, section 6)
==========================================================

For two methods, the test hypervolumes of runs with the same run number are
paired. Everything here works on one set of paired differences d, computed once
and rounded to 12 decimal places, so that the test, the zeros, the effect size
and the interval all use the same numbers.

  - signed_rank(d): the two-sided signed-rank test. The p-value is exact for any
    number of non-zero differences, with or without ties: the distribution of
    T+ over all sign patterns of the given ranks is counted by dynamic
    programming on the doubled ranks, which are whole numbers. No normal
    approximation is used, so there is no continuity correction to choose.
  - holm(p): Holm's step-down adjustment over the three pairs.
  - median_interval(d): the median of all differences, zeros included, with the
    distribution-free interval from order statistics.

Plain Python and numpy only; SciPy is used in the tests, to check the exact
p-value against scipy.stats.wilcoxon on cases without ties.
"""

import math
from fractions import Fraction

import numpy as np

DECIMALS = 12


def paired_differences(x, y):
    """x and y map run number -> value. Returns the run numbers present in both, in order, and the
    differences x - y rounded to 12 decimal places. A run missing from either method is left out."""
    runs = sorted(set(x) & set(y))
    d = np.array([round(float(x[s]) - float(y[s]), DECIMALS) for s in runs])
    if not np.isfinite(d).all():
        raise ValueError("a non-finite paired difference")
    return runs, d


def mean_ranks(values):
    """Ranks 1..m of the values, ties taking the mean of the ranks they share."""
    values = np.asarray(values, float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1  # the mean of ranks i+1 .. j+1
        i = j + 1
    return ranks


def null_counts(doubled):
    """counts[s] = number of the 2^m sign patterns whose doubled T+ is s, for the doubled ranks given."""
    counts = [1] + [0] * sum(doubled)
    for w in doubled:  # each rank is either in T+ or not
        for s in range(len(counts) - 1, w - 1, -1):
            counts[s] += counts[s - w]
    return counts


def signed_rank(d):
    """The two-sided signed-rank test of whether the differences d are symmetric about zero.

    Zero differences are dropped (zero_method "wilcox"). Returns n, the number of zeros, m, T+ and T-,
    the exact p-value and the matched-pairs rank-biserial correlation r = (T+ - T-) / (T+ + T-).
    With no non-zero difference, p = 1 and r is not defined (None)."""
    d = np.asarray(d, float)
    nonzero = d[d != 0]
    n, m = len(d), len(nonzero)
    out = {"n": n, "zeros": n - m, "m": m}
    if m == 0:
        return {**out, "T+": 0.0, "T-": 0.0, "p": 1.0, "r": None, "note": "all differences zero"}
    doubled = [int(round(2 * r)) for r in mean_ranks(np.abs(nonzero))]  # mean ranks are multiples of 1/2
    t_plus2 = sum(w for w, v in zip(doubled, nonzero) if v > 0)
    t_minus2 = sum(doubled) - t_plus2
    counts = null_counts(doubled)
    lower, upper = sum(counts[:t_plus2 + 1]), sum(counts[t_plus2:])
    p = min(Fraction(1), 2 * Fraction(min(lower, upper), 2 ** m))
    return {**out, "T+": t_plus2 / 2, "T-": t_minus2 / 2, "p": float(p),
            "r": (t_plus2 - t_minus2) / (t_plus2 + t_minus2), "note": ""}


def holm(p):
    """Holm-adjusted p-values, in the order given: sorted p(1) <= ... <= p(k), adjusted
    p~(i) = max over j <= i of min(1, (k - j + 1) p(j))."""
    k = len(p)
    order = sorted(range(k), key=lambda i: p[i])
    adjusted, running = [0.0] * k, 0.0
    for j, i in enumerate(order):
        running = max(running, min(1.0, (k - j) * p[i]))
        adjusted[i] = running
    return adjusted


def binomial_cdf(k, n):
    """P(Bin(n, 1/2) <= k), exactly."""
    return Fraction(sum(math.comb(n, i) for i in range(k + 1)), 2 ** n)


def interval_rank(n, tail=Fraction(1, 40)):
    """The largest k >= 1 with P(Bin(n, 1/2) <= k - 1) <= 0.025, or None if there is none."""
    k = None
    for candidate in range(1, n + 1):
        if binomial_cdf(candidate - 1, n) <= tail:
            k = candidate
        else:
            break
    return k


def median_interval(d):
    """The median of all the differences, zeros included, and the distribution-free 95% interval
    [d(k), d(n-k+1)] with its coverage; no interval when n <= 5."""
    d = np.sort(np.asarray(d, float))
    n = len(d)
    if n == 0:
        return {"n": 0, "median": None, "low": None, "high": None, "coverage": None}
    k = interval_rank(n)
    if k is None:
        return {"n": n, "median": float(np.median(d)), "low": None, "high": None, "coverage": None}
    coverage = 1 - 2 * binomial_cdf(k - 1, n)
    return {"n": n, "median": float(np.median(d)), "low": float(d[k - 1]), "high": float(d[n - k]),
            "coverage": float(coverage)}


def compare(x, y):
    """One pair of methods: the paired differences x - y, the signed-rank test and the median interval."""
    runs, d = paired_differences(x, y)
    return {"runs": runs, **signed_rank(d), **{f"median {k}": v for k, v in median_interval(d).items() if k != "n"}}
