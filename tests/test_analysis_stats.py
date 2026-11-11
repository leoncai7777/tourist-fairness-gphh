"""Tests of analysis/stats.py (Step 6 plan, section 6 and requirement 1 of section 3a)."""

import itertools
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from stats import (compare, holm, interval_rank, mean_ranks, median_interval, null_counts,  # noqa: E402
                   paired_differences, signed_rank)


def brute_p(d):
    """The two-sided p-value by listing every sign pattern: the definition the counting must match."""
    nonzero = [v for v in d if v != 0]
    ranks = mean_ranks(np.abs(nonzero))
    t = sum(r for r, v in zip(ranks, nonzero) if v > 0)
    lower = upper = 0
    for signs in itertools.product((0, 1), repeat=len(nonzero)):
        s = sum(r for r, sign in zip(ranks, signs) if sign)
        lower += s <= t + 1e-9
        upper += s >= t - 1e-9
    return min(1.0, 2 * min(lower, upper) / 2 ** len(nonzero))


def test_mean_ranks_share_ties():
    assert list(mean_ranks([3, 1, 3, 2])) == [3.5, 1, 3.5, 2]
    assert list(mean_ranks([5, 5, 5])) == [2, 2, 2]


def test_the_counts_cover_every_sign_pattern():
    for doubled in ([2, 4, 6], [3, 3, 6, 8], [2] * 7, [2, 4, 7, 7, 10, 12]):
        counts = null_counts(doubled)
        assert sum(counts) == 2 ** len(doubled)
        assert counts == counts[::-1]  # flipping every sign mirrors the distribution


@pytest.mark.parametrize("seed", range(150))
def test_the_exact_p_value_agrees_with_listing_every_sign_pattern(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 13))  # m <= 12, with zeros and ties
    d = rng.integers(-4, 5, size=n) * 0.25
    assert signed_rank(d)["p"] == pytest.approx(brute_p(d), abs=1e-12)


def test_the_cases_at_the_edges():
    zero = signed_rank([0.0] * 30)
    assert zero["p"] == 1.0 and zero["r"] is None and zero["m"] == 0 and zero["zeros"] == 30
    one = signed_rank([0.0, 0.0, 0.3])
    assert one["p"] == 1.0 and one["r"] == 1.0 and one["m"] == 1
    equal = signed_rank([0.5] * 6)  # all |d| equal and positive: only the all-plus pattern reaches T+
    assert equal["p"] == pytest.approx(2 / 2 ** 6)
    assert equal["r"] == 1.0
    mixed = signed_rank([0.5, -0.5, 0.5, -0.5])
    assert mixed["p"] == 1.0 and mixed["r"] == 0.0


def test_the_sign_of_the_effect_follows_the_differences():
    d = np.array([0.1, 0.2, 0.3, -0.05, 0.4, 0.25])
    a, b = signed_rank(d), signed_rank(-d)
    assert a["p"] == pytest.approx(b["p"]) and a["r"] == pytest.approx(-b["r"])
    assert a["T+"] + a["T-"] == 6 * 7 / 2  # the rank-biserial uses both rank sums


def test_the_exact_p_value_agrees_with_scipy_without_ties():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    for n in (5, 10, 20, 30):
        for _ in range(20):
            x, y = rng.normal(size=n), rng.normal(size=n)
            d = np.round(x - y, 12)
            if len(set(np.abs(d))) < n or (d == 0).any():
                continue
            ours = signed_rank(d)["p"]
            theirs = scipy_stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", method="exact").pvalue
            assert ours == pytest.approx(theirs, rel=1e-9)


def test_floating_point_noise_counts_as_a_zero_and_as_a_tie():
    x = {0: 0.3, 1: 0.6, 2: 0.7}
    y = {0: 0.1 + 0.2, 1: 0.6 - 1e-14, 2: 0.5}
    runs, d = paired_differences(x, y)
    assert runs == [0, 1, 2] and d[0] == 0 and d[1] == 0 and d[2] == pytest.approx(0.2)
    assert signed_rank(d)["zeros"] == 2


def test_a_missing_run_drops_its_pair():
    runs, d = paired_differences({0: 1.0, 1: 2.0, 3: 4.0}, {0: 0.5, 2: 1.0, 3: 3.0})
    assert runs == [0, 3] and list(d) == [0.5, 1.0]
    with pytest.raises(ValueError):
        paired_differences({0: math.nan}, {0: 1.0})


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 1.0, 0.001]) == pytest.approx([1.0, 1.0, 0.003])
    assert holm([1.0, 1.0, 1.0]) == [1.0, 1.0, 1.0]


def test_the_median_interval():
    assert interval_rank(30) == 10
    assert interval_rank(5) is None and interval_rank(6) == 1
    d = np.arange(1, 31) / 100
    out = median_interval(d[::-1])  # order of the input does not matter
    assert out["low"] == d[9] and out["high"] == d[20]
    assert out["coverage"] == pytest.approx(0.95723, abs=1e-5)
    assert median_interval([0.0, 0.0, 0.1, 0.2, 0.3])["low"] is None  # n <= 5: the median only
    with_zeros = median_interval([0.0] * 20 + [0.1] * 10)
    assert with_zeros["median"] == 0.0  # zeros stay in the interval's sample


def test_compare_puts_it_together():
    x = {s: 0.6 + 0.01 * s for s in range(30)}
    y = {s: 0.6 for s in range(30)}
    out = compare(x, y)
    assert out["n"] == 30 and out["zeros"] == 1 and out["r"] == 1.0
    assert out["p"] == pytest.approx(float(Fraction(2, 2 ** 29)))
