"""
Tests of evaluating a rule (Step 4b). Section 9, stage D, of search_design.md.

Run:  python3 -m pytest tests -q   (from the repository root)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from day import PROCESSED  # noqa: E402
import evaluate as evaluate_module  # noqa: E402
from evaluate import (Days, budget_cuts, evaluate, grouped, histogram_bin, hourly_peaks,  # noqa: E402
                      share_at_least, visitor_rows)
from gp_rule import from_text  # noqa: E402

needs_data = pytest.mark.skipif(not (PROCESSED / "manifest.json").exists(), reason="processed tables not built")
RULE = "sub(AV, div(TA, RemT))"


# ---------------------------------------------------------------------- #
# 1. Histogram and grouped outputs, by hand
# ---------------------------------------------------------------------- #

def test_histogram_bins_at_the_edges():
    below, above = np.nextafter(0.8, 0), np.nextafter(0.8, 1)
    r = np.array([0.8, below, above, 0.9, np.nextafter(0.9, 0), -2.0, -2.5, 1.99, 2.0, 7.0])
    assert histogram_bin(r).tolist() == [16, 15, 16, 18, 17, -40, -41, 39, 40, 40]
    # Every edge splits exactly as a direct comparison does, also for random values near the edges.
    rng = np.random.default_rng(0)
    edges = np.arange(-40, 41) / 20
    near = (edges[:, None] + rng.integers(-3, 4, (81, 50)) * np.spacing(edges)[:, None]).ravel()
    bins = histogram_bin(near)
    for k in range(-40, 41):
        assert ((bins >= k) == (near >= k / 20)).all()


def hand_rows(**changes):
    rows = pd.DataFrame({"visitor": list("abcdef"), "follows": [True] * 5 + [False],
                         "arrival": [550.0, 560.0, 700.0, 710.0, 900.0, 555.0],
                         "budget": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                         "observed stops": [3, 3, 4, 4, 2, 3], "stops": [3, 3, 4, 4, 2, 3],
                         "R_v": [0.8, np.nextafter(0.8, 0), 0.9, 1.2, -3.0, 1.0],
                         "change A": [0.1] * 6, "change C": [0.2] * 6, "change T": [0.0] * 6,
                         "end reason": ["limit"] * 5 + ["not a follower"], "overruns": [0, 1, 0, 2, 0, 0]})
    for column, values in changes.items():
        rows[column] = values
    return rows


def test_grouped_outputs_by_hand():
    table = grouped(hand_rows())
    assert table["visitors"].sum() == 6 and table["overrun records"].sum() == 3
    assert table.loc[~table["follows"], "visitors"].sum() == 1  # the follows split
    first = table[(table["follows"]) & (table["arrival group"] == 0) & (table["stop group"] == 3)].iloc[0]
    assert first["visitors"] == 2 and first["sum R_v"] == pytest.approx(0.8 + np.nextafter(0.8, 0))
    # Shares at the bin edges are exact: R_v >= 0.8 for a, c, d, f; >= 0.9 for c, d, f.
    assert share_at_least(table, 0.8) == 4 / 6 and share_at_least(table, 0.9) == 3 / 6


def test_same_lost_stops_but_different_early_endings():
    # Two lost stops: one visitor losing both, or two visitors losing one each.
    one = grouped(hand_rows(stops=[1, 3, 4, 4, 2, 3]))
    two = grouped(hand_rows(stops=[2, 2, 4, 4, 2, 3]))
    lost = lambda t: (t["observed stops"] - t["stops"]).sum()  # noqa: E731
    assert lost(one) == lost(two) == 2
    assert one["ended early"].sum() == 1 and two["ended early"].sum() == 2


# ---------------------------------------------------------------------- #
# 2. Evaluation on real days
# ---------------------------------------------------------------------- #

@needs_data
def test_the_same_rule_day_and_replicate_give_the_same_objectives():
    days = Days()
    rule = from_text(f"GP-adaptive | h1: {RULE} | h2: mul(GDAY, 0.4)")
    one, two = evaluate(rule, "2019-12-18", 3, days), evaluate(rule, "2019-12-18", 3, days)
    assert (one.mean_r, one.g_hat) == (two.mean_r, two.g_hat)
    other = evaluate(rule, "2019-12-18", 4, days)
    assert (other.mean_r, other.g_hat) != (one.mean_r, one.g_hat)  # another replicate, other stays


@needs_data
def test_adaptive_with_a_constant_h2_has_the_objectives_of_fixed():
    days = Days()
    for c in (0.05, 3.0):
        fixed = evaluate(from_text(f"GP-fixed | h1: {RULE} | c: {c!r}"), "2019-12-18", 0, days)
        adaptive = evaluate(from_text(f"GP-adaptive | h1: {RULE} | h2: {c!r}"), "2019-12-18", 0, days)
        assert (fixed.mean_r, fixed.g_hat) == (adaptive.mean_r, adaptive.g_hat)
        assert fixed.visitors.equals(adaptive.visitors)


def test_thresholds_outside_the_exact_range_are_refused():
    r = np.array([-3.0, 0.8, 0.9, 2.1, 7.0])
    rows = hand_rows(**{"R_v": list(r) + [1.0]}).iloc[:5]
    table = grouped(rows)
    for k in range(-40, 41):  # every edge from -2 to 2 matches the direct count
        assert share_at_least(table, k / 20) == (r >= k / 20).mean()
    for theta in (2.05, 2.5, 7.0, -2.05, -3.0, 0.82, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            share_at_least(table, theta)


@needs_data
def test_a_test_day_or_a_date_of_unknown_split_is_refused_before_loading(monkeypatch):
    def must_not_load(*args, **kwargs):
        raise AssertionError("load_day was called")
    monkeypatch.setattr(evaluate_module, "load_day", must_not_load)
    cases = {"2019-10-16": "test day",         # an even ISO week
             "2019-12-25": "no known split",   # not in the 2019 day table
             "2020-06-01": "no known split",   # no day table for 2020
             "2022-07-14": "test day" if (PROCESSED / "days_2022.csv").exists() else "no known split"}
    for date, message in cases.items():
        with pytest.raises(ValueError, match=message):
            Days().get(date)
    # A label that is not train, validation or test is unknown too.
    monkeypatch.setattr(evaluate_module, "split_of", lambda year: {"2019-12-18": "holdout"})
    with pytest.raises(ValueError, match="no known split"):
        Days().get("2019-12-18")
    monkeypatch.undo()
    assert Days().get("2019-12-18")[0].date == pd.Timestamp("2019-12-18")  # a training day still loads


@needs_data
def test_the_cache_keeps_settings_apart():
    plain, floored, span = Days(), Days(peak_floor=10.0), Days(budget="span")
    a, b, c = plain.get("2019-12-18")[0], floored.get("2019-12-18")[0], span.get("2019-12-18")[0]
    assert plain.get("2019-12-18")[0] is a  # loaded once
    assert (b.usual_peak > a.usual_peak).any() and b.options["peak_floor"] == 10.0
    assert (c.visitors["budget_limit"] < a.visitors["budget_limit"]).any()


def test_a_days_setting_cannot_be_changed():
    days = Days()
    with pytest.raises(TypeError):
        days.options["peak_floor"] = 10.0
    for name in ("options", "follows", "test"):
        with pytest.raises(AttributeError):
            setattr(days, name, None)


@needs_data
def test_changing_the_callers_follows_list_changes_nothing():
    first = Days().get("2019-12-18")[0]
    evaluated = first.visitors["visitor"][first.visitors["evaluated"]]
    chosen = list(evaluated.iloc[:5])
    days = Days(follows=chosen)
    chosen.extend(evaluated.iloc[5:10])  # the caller changes its own list afterwards
    assert days.get("2019-12-18")[0].visitors["follows"].sum() == 5


@needs_data
def test_the_stop_limit_switch_reaches_the_simulation():
    from rules import Pop
    days = Days()
    with_limit = evaluate(Pop(), "2019-12-18", 0, days)
    without = evaluate(Pop(), "2019-12-18", 0, days, stop_limit=False)
    assert (with_limit.visitors["stops"] <= with_limit.visitors["observed stops"]).all()
    assert (without.visitors["stops"] > without.visitors["observed stops"]).any()
    assert without.run.settings["stop_limit"] is False


@needs_data
def test_grouped_outputs_add_up_to_the_visitors():
    days = Days()
    day, _ = days.get("2019-12-18")
    result = evaluate(from_text(f"GP-fixed | h1: {RULE} | c: 1.0"), "2019-12-18", 0, days)
    rows = visitor_rows(day, result)
    table = grouped(rows)
    assert table["visitors"].sum() == len(rows) == day.visitors["evaluated"].sum()
    assert table["sum R_v"].sum() == pytest.approx(rows["R_v"].sum())
    assert table["sum R_v"].sum() / table["visitors"].sum() == pytest.approx(result.mean_r)
    assert table["ended early"].sum() == (rows["stops"] < rows["observed stops"]).sum()
    assert table[[f"end: {r}" for r in ("limit", "none feasible")]].to_numpy().sum() == len(rows)
    assert table["overrun records"].sum() == len(result.run.overruns)
    assert share_at_least(table, 0.8) == pytest.approx((rows["R_v"] >= 0.8).mean())
    peaks = hourly_peaks(day, result.crowding)
    assert (peaks["peak C"].to_numpy() == result.crowding.max(axis=1)).all()


@needs_data
def test_budget_cuts_come_from_the_training_days():
    visitors = pd.read_csv(PROCESSED / "visitors_2019.csv.gz")
    train = visitors[(visitors["split"] == "train") & (visitors["stops"] >= 2)]
    low, high = budget_cuts()
    assert low < high and abs((train["budget"] < low).mean() - 1 / 3) < 0.02
