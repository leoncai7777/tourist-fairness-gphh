"""
Tests of the day input and the observed replay (Steps 3a and 3b).

Run:  python3 -m pytest tests -q   (from the repository root)
The tests on real days are skipped when data/verona/processed/ is not there.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from day import PROCESSED, Day, load_day, month_hours, visitor_frame  # noqa: E402
from replay import baseline_value, replay  # noqa: E402
from scoring import hourly_person_minutes, relative_value, route_parts, route_value  # noqa: E402

PILOT_DAYS = {"2019-12-18": (383, 14), "2019-10-09": (1006, 14), "2019-03-17": (1960, 14),
              "2019-04-26": (3887, 14), "2019-06-03": (680, 11)}  # visits, available sites
needs_data = pytest.mark.skipif(not (PROCESSED / "manifest.json").exists(), reason="processed tables not built")


def small_day():
    """Three sites; visitor 0 goes 0 -> 1, visitor 1 visits site 2 only."""
    visitors = pd.DataFrame({"visitor": ["a_1", "b_1"], "stops": [2, 1], "evaluated": [True, False],
                             "follows": [True, False]})
    fixed = pd.DataFrame({"visitor": [0, 0, 1], "entry_index": [0, 1, 0], "site": [0, 1, 2],
                          "entry": [600.0, 650.0, 610.0], "exit": [640.0, 700.0, 630.0]})
    walk = np.array([[0, 10, 20], [10, 0, 15], [20, 15, 0]], float)
    return Day(date=pd.Timestamp("2019-01-15"), day_type="other", site_ids=np.array([1, 2, 3]),
               available=np.array([True, True, True]), attractiveness=np.array([1.0, 0.5, 0.2]),
               stay=np.array([40.0, 50.0, 20.0]), usual_peak=np.array([2.0, 1.0, 1.0]),
               opening=np.zeros(3), closing=np.full(3, 1200.0), walk=walk,
               usual_gday=np.ones(781), usual_arrivals=np.ones(781), visitors=visitors,
               routes=[np.array([0, 1]), np.array([2])], entries=[np.array([600.0, 650.0]), np.array([610.0])],
               fixed=fixed, data_version="test")


# ---------------------------------------------------------------------- #
# 1. Crowding and route value
# ---------------------------------------------------------------------- #

def test_person_minutes_split_exactly_at_an_hour_boundary():
    minutes = hourly_person_minutes([0], [659.8], [660.2], 1)
    assert np.isclose(minutes[0, 10], 0.2) and np.isclose(minutes[0, 11], 0.2)


def test_late_and_zero_length_intervals():
    minutes = hourly_person_minutes([0, 0], [1250.0, 700.0], [1330.0, 700.0], 1)
    assert minutes[0, 20] == 10 and minutes[0, 21] == 60 and minutes[0, 22] == 10  # not cut at 21:00
    assert minutes.sum() == 80  # the zero-length interval adds nothing


def test_route_parts_by_hand():
    day = small_day()
    c = np.zeros((3, 24))
    c[0, 10], c[1, 10] = 0.5, 2.0
    a, c_sum, t = route_parts([0, 1], [600.0, 650.0], c, day.attractiveness, day.walk)
    assert a == 1.5 and c_sum == 2.5 and np.isclose(t, 10 / 60)
    assert np.isclose(route_value(a, c_sum, t, 0.25, 0.5), 1.5 - 0.625 - 5 / 60)


# ---------------------------------------------------------------------- #
# 2. Replay on a hand-built day
# ---------------------------------------------------------------------- #

def test_replay_by_hand():
    result = replay(small_day())
    assert result.entries_per_site.tolist() == [1, 1, 1]
    # site 0: 40 person-minutes in hour 10, usual peak 2; site 1: 10 in hour 10, 40 in hour 11
    assert np.isclose(result.crowding[0, 10], 40 / 60 / 2)
    assert np.isclose(result.crowding[1, 10], 10 / 60) and np.isclose(result.crowding[1, 11], 40 / 60)
    assert result.person_minutes.sum() == 40 + 50 + 20
    b = result.baseline.loc[0]
    assert b["A"] == 1.5 and np.isclose(b["C"], 40 / 120 + 10 / 60) and np.isclose(b["T"], 10 / 60)
    assert list(result.baseline.index) == [0]  # only evaluated visitors


def test_baseline_value_by_hand():
    """Checked against the formula, not against itself: a test of R_v = 1 alone would pass
    even if both sides read the wrong column."""
    result = replay(small_day())
    value = baseline_value(result, 0.25, 0.5).loc[0]
    assert np.isclose(value, 1.5 - 0.25 * (40 / 120 + 10 / 60) - 0.5 * (10 / 60))
    assert relative_value(value, value, 1.5) == 1


# ---------------------------------------------------------------------- #
# 3. Input rules
# ---------------------------------------------------------------------- #

def test_missing_month_uses_yearly_hours():
    sites = pd.DataFrame({"o": [500.0, 520.0], "c": [1100.0, 1050.0]}, index=[10, 20])
    hours = pd.DataFrame({"site": [10], "month": [3], "o": [540.0], "c": [1000.0]})
    opening, closing = month_hours(hours, sites, 3)
    assert opening.tolist() == [540.0, 520.0] and closing.tolist() == [1000.0, 1050.0]


def test_following_changes_neither_the_evaluated_set_nor_the_numbers():
    visitors = pd.DataFrame({"visitor": ["c_1", "a_1", "b_1"], "stops": [3, 2, 1]})
    everyone = visitor_frame(visitors, None)
    some = visitor_frame(visitors, ["c_1"])
    assert everyone.visitor.tolist() == some.visitor.tolist() == ["a_1", "b_1", "c_1"]
    assert everyone.evaluated.tolist() == some.evaluated.tolist() == [True, False, True]
    assert some.follows.tolist() == [False, False, True]
    with pytest.raises(AssertionError):
        visitor_frame(visitors, ["b_1"])  # a one-stop visitor cannot follow


# ---------------------------------------------------------------------- #
# 4. Real days
# ---------------------------------------------------------------------- #

@needs_data
def test_pilot_days_match_the_tables():
    days = pd.read_csv(PROCESSED / "days_2019.csv", parse_dates=["date"]).set_index("date")
    for date, (visits, open_sites) in PILOT_DAYS.items():
        day = load_day(date)
        row = days.loc[pd.Timestamp(date)]
        assert row.split == "train" and row.visits == visits and row.open_sites == open_sites
        assert len(day.fixed) == visits and day.available.sum() == open_sites
        assert sum(len(r) for r in day.routes) == visits
        assert (day.visitors.evaluated == (day.visitors.stops >= 2)).all()
        assert replay(day).entries_per_site.sum() == visits


@needs_data
def test_options_for_the_sensitivity_runs():
    main = load_day("2019-06-03")
    span = load_day("2019-06-03", budget="span")
    assert (span.visitors.budget_limit <= main.visitors.budget_limit + 1e-9).all()
    assert (span.visitors.budget_limit < main.visitors.budget_limit).any()
    same_day = load_day("2019-06-03", same_day=True)
    with_entries = {day_site for day_site in same_day.fixed.site}
    expected = [k in with_entries and site != 300 for k, site in enumerate(same_day.site_ids)]
    assert same_day.available.tolist() == expected
    with pytest.raises(ValueError):
        load_day("2019-06-03", budget="corected")


@needs_data
def test_partial_following_keeps_rows_on_a_real_day():
    day = load_day("2019-10-09")
    half = day.visitors.visitor[day.visitors.evaluated].iloc[::2]
    other = load_day("2019-10-09", follows=half)
    assert day.visitors.visitor.tolist() == other.visitors.visitor.tolist()
    assert day.visitors.evaluated.tolist() == other.visitors.evaluated.tolist()
    assert other.visitors.follows.sum() == len(half)
