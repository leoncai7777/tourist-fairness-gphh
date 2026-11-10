"""Checks of the data preparation on small hand-built data. Run from the repository root: python3 -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import preprocess as p  # noqa: E402


def raw(rows):
    return pd.DataFrame([{"id_veronacard": c, "date": pd.Timestamp(d), "t": t, "sito_nome": s} for c, d, t, s in rows])


def visits(rows):
    frame = p.clean(raw(rows), "test")
    return p.add_stays(frame, {(a, b): 10 for a in p.SITE_ID.values() for b in p.SITE_ID.values()})


def test_a_repeat_keeps_the_first_entry_and_other_sites_are_removed():
    d = p.clean(raw([("a", "2019-01-14", 600, "Arena"), ("a", "2019-01-14", 601, "Arena"), ("a", "2019-01-14", 700, "Eataly Verona"),
                     ("a", "2019-01-14", 800, "Duomo")]), "test")
    assert d.t.tolist() == [600, 800] and d.stop.tolist() == [1, 2]


def test_stay_is_the_gap_less_the_walk_and_a_last_stop_has_none():
    d = visits([("a", "2019-01-14", 600, "Arena"), ("a", "2019-01-14", 680, "Duomo")])
    assert d.stay.iloc[0] == 70 and np.isnan(d.stay.iloc[1])


def test_non_positive_stays_are_left_out_of_the_median():
    d = visits([("a", "2019-01-14", 600, "Arena"), ("a", "2019-01-14", 605, "Duomo"),
                ("b", "2019-01-14", 600, "Arena"), ("b", "2019-01-14", 650, "Duomo")])
    assert p.site_statistics(d[d.site == 49].assign(next_t=d.next_t, walk=d.walk)).d[49] == 40


def test_occupancy_ends_before_the_next_entry():
    d = visits([("a", "2019-01-14", 600, "Arena"), ("a", "2019-01-14", 620, "Duomo")])
    arena = d[d.site == 49]
    assert abs(p.usual_peak(arena, pd.Series({49: 60.0})).loc[49] - 10 / 60) < 1e-9


def test_budget_is_widened_only_when_the_route_does_not_fit():
    d = visits([("a", "2019-01-14", 600, "Arena"), ("a", "2019-01-14", 610, "Duomo"),
                ("b", "2019-01-14", 600, "Arena"), ("b", "2019-01-14", 800, "Duomo")])
    s = pd.DataFrame({"d": {49: 60.0, 52: 30.0}, "o": {49: 0.0, 52: 0.0}, "c": {49: 1400.0, 52: 1400.0}})
    v = p.visitor_table(d, s, {}, {(a, b): 10 for a in (49, 52) for b in (49, 52)}).set_index("visitor")
    assert v.span["a_20190114"] == 40 and v.budget["a_20190114"] == 100
    assert v.budget["b_20190114"] == v.span["b_20190114"] == 230


def test_split_follows_the_weeks():
    days = p.split_days(pd.to_datetime(["2019-01-07", "2019-01-14", "2019-01-28"])).set_index("date").split
    assert days[pd.Timestamp("2019-01-07")] == "test"  # week 2
    assert days[pd.Timestamp("2019-01-14")] == "train"  # week 3
    assert days[pd.Timestamp("2019-01-28")] == "validation"  # week 5


def test_changing_test_days_does_not_change_any_training_statistic():
    rows = [("a", "2019-01-14", 600, "Arena"), ("a", "2019-01-14", 700, "Duomo"), ("t", "2019-01-07", 600, "Arena"), ("t", "2019-01-07", 900, "Duomo")]
    changed = rows[:2] + [("t", "2019-01-07", 500, "Duomo"), ("t", "2019-01-07", 1000, "San Zeno")]

    def statistics(data):
        d = visits(data)
        days = p.split_days(d.date.unique())
        fit = d.merge(days[["date", "split"]], on="date")
        fit = fit[fit.split == "train"]
        ids = [49, 52, 63]
        site_table = p.site_statistics(fit)
        shares = p.weekday_shares(fit, ids)
        opened = p.weekday_opening(shares, [pd.Timestamp(x) for x in fit.date.unique()])
        return site_table, p.site_hours(fit, site_table), shares, p.usual_gday(fit, opened), p.usual_arrivals(fit)

    for one, two in zip(statistics(rows), statistics(changed)):
        pd.testing.assert_frame_equal(one, two)


def test_occupancy_intervals_sum_to_the_person_minutes_in_the_hours():
    d = visits([("a", "2019-01-14", 600, "Arena"), ("a", "2019-01-14", 620, "Duomo")])
    d_i = pd.Series({49: 60.0, 52: 30.0})
    i = p.observed_occupancy_intervals(d, d_i)
    assert i.exit.tolist() == [610, 650]  # next entry less the walk; a last stop stays d
    hours = sum(float((i.exit.clip(upper=60 * (h + 1)) - i.entry.clip(lower=60 * h)).clip(lower=0).sum()) for h in range(8, 22))
    assert hours == (i.exit - i.entry).sum()


def test_same_day_sensitivity_keeps_low_coverage_sites_out():
    d = visits([("a", "2019-01-14", 600, "Arena"), ("b", "2019-01-14", 600, "Museo Miniscalchi")])
    table = p.day_table(d, {pd.Timestamp("2019-01-14"): [49]})
    assert np.isnan(table.G_hat_same_day.iloc[0])  # one site left, so no Gini; Miniscalchi not counted


def test_gini_hat_and_availability():
    assert p.gini_hat([3, 3]) == 0 and p.gini_hat([0, 2]) == 1
    shares = pd.DataFrame({d: [1.0, 0.4, 1.0] for d in range(7)}, index=[49, 52, 300])
    assert p.weekday_opening(shares, [pd.Timestamp("2019-01-14")])[pd.Timestamp("2019-01-14")] == [49]


def test_usual_denominators_use_the_same_minute_as_their_numerators():
    d = visits([("a", "2019-01-14", 600, "Arena"), ("b", "2019-01-14", 610, "Duomo"), ("c", "2019-01-15", 620, "Arena")])
    u = p.usual_arrivals(d).set_index(["day_type", "minute"]).usual_arrivals
    assert u["monday", 605] == 1 and u["monday", 610] == 2 and u["monday", 659] == 2 and u["monday", 660] == 1  # 60 minutes up to the minute
    assert u["other", 619] == 0 and u["other", 620] == 1
    g = p.usual_gday(d, {pd.Timestamp("2019-01-14"): [49, 52], pd.Timestamp("2019-01-15"): [49, 52]}).set_index(["day_type", "minute"]).usual_gday
    assert g["monday", 600] == 1 and g["monday", 610] == 0  # one site visited, then both once


def test_entries_with_seconds_count_from_the_next_minute():
    d = visits([("a", "2019-01-14", 590, "Duomo"), ("b", "2019-01-14", 600.8, "Arena")])
    u = p.usual_arrivals(d).set_index(["day_type", "minute"]).usual_arrivals
    assert u["monday", 600] == 1 and u["monday", 601] == 2  # 600.8 is not up to minute 600
    assert u["monday", 649] == 2 and u["monday", 650] == 1  # 590 leaves the 60 minutes up to 650
    assert u["monday", 660] == 1 and u["monday", 661] == 0  # 600.8 is still within (600, 660], gone by 661
    g = p.usual_gday(d, {pd.Timestamp("2019-01-14"): [49, 52]}).set_index(["day_type", "minute"]).usual_gday
    assert g["monday", 600] == 1 and g["monday", 601] == 0


def test_occupancy_edge_cases():
    d = visits([("a", "2019-01-14", 640, "Arena"), ("a", "2019-01-14", 710, "Duomo"),
                ("b", "2019-01-14", 600, "Arena"), ("b", "2019-01-14", 605, "Duomo"),
                ("c", "2019-01-14", 650, "San Zeno")])
    i = p.observed_occupancy_intervals(d, pd.Series({49: 60.0, 52: 30.0, 63: 20.0})).set_index(["visitor", "site"])
    assert i.loc[("a_20190114", 49), "exit"] == 700  # crosses 11:00
    assert i.loc[("b_20190114", 49), "exit"] == 600  # next entry less the walk is earlier: zero length
    assert i.loc[("c_20190114", 63), "exit"] == 670  # one-stop visitor stays d
    hours = sum(float((i.exit.clip(upper=60 * (h + 1)) - i.entry.clip(lower=60 * h)).clip(lower=0).sum()) for h in range(8, 22))
    assert hours == (i.exit - i.entry).sum()
