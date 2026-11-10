"""
Tests of the event engine and the observed-order control (Steps 3c and 3d).

Run:  python3 -m pytest tests -q   (from the repository root)
Hand-built days first, each worked out in its comments; then real days, skipped
when data/verona/processed/ is not there.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from day import PROCESSED, Day, load_day  # noqa: E402
from engine import Nearest, ObservedOrder, simulate, stay_table  # noqa: E402
from replay import baseline_value, replay  # noqa: E402
from scoring import crowding, hourly_person_minutes, relative_value, route_parts  # noqa: E402

needs_data = pytest.mark.skipif(not (PROCESSED / "manifest.json").exists(), reason="processed tables not built")


def hand_day(visitors, walk=None, opening=(0, 0, 0), closing=(1000, 1000, 1000), stay=(30, 20, 10),
             available=(True, True, True)):
    """Three sites on a line, 10 minutes apart. Each visitor: route, entries, follows, budget,
    and for visitors who do not follow, the exit of each fixed interval."""
    walk = np.array(walk if walk is not None else [[0, 10, 20], [10, 0, 10], [20, 10, 0]], float)
    rows, fixed = [], []
    for k, v in enumerate(visitors):
        rows.append({"visitor": f"v{k}", "stops": len(v["route"]), "evaluated": len(v["route"]) >= 2,
                     "follows": v.get("follows", True), "budget_limit": v.get("budget", 400.0),
                     "stop_limit": len(v["route"])})
        for i, (site, entry) in enumerate(zip(v["route"], v["entries"])):
            exit_ = v.get("exits", v["entries"])[i] if not v.get("follows", True) else entry + stay[site]
            fixed.append((k, i, site, float(entry), float(exit_)))
    return Day(date=pd.Timestamp("2019-01-15"), day_type="other", site_ids=np.array([10, 20, 30]),
               available=np.array(available), attractiveness=np.ones(3), stay=np.array(stay, float),
               usual_peak=np.ones(3), opening=np.array(opening, float), closing=np.array(closing, float),
               walk=walk, usual_gday=np.ones(781), usual_arrivals=np.ones(781), visitors=pd.DataFrame(rows),
               routes=[np.array(v["route"]) for v in visitors],
               entries=[np.array(v["entries"], float) for v in visitors],
               fixed=pd.DataFrame(fixed, columns=["visitor", "entry_index", "site", "entry", "exit"]),
               data_version="test")


class Recorder(Nearest):
    """Nearest, but keeps what each decision saw."""
    def __init__(self):
        self.seen = []

    def choose(self, engine, k, candidates):
        self.seen.append({"visitor": k, "time": engine.now, "inside": engine.inside.tolist(),
                          "on_the_way": engine.on_the_way.tolist(), "visits": engine.visits.tolist()})
        return super().choose(engine, k, candidates)


def stays(rows):
    return np.array(rows, float)


# ---------------------------------------------------------------------- #
# 1. One follower, by hand
# ---------------------------------------------------------------------- #

def test_one_follower_two_sites_by_hand():
    # Starts at site 0 at 600 and stays 40; at 640 the nearest site is 1 (walk 10), arrival 650,
    # stays 25 until 675; the stop limit (2) ends the day there.
    day = hand_day([{"route": [0, 1], "entries": [600, 650]}])
    run = simulate(day, Nearest(), stays=stays([[40, 25, 15]]), check=True)
    assert run.routes[0] == [(0, 600.0, 40.0), (1, 650.0, 25.0)]
    assert run.end_reason[0] == "limit" and run.events == 7
    assert run.visits.tolist() == [1, 1, 0]


def test_waiting_for_opening():
    # Site 1 opens at 700: the visitor arrives at 650, waits, and starts at 700.
    day = hand_day([{"route": [0, 1], "entries": [600, 650]}], opening=(0, 700, 0))
    run = simulate(day, Nearest(), stays=stays([[40, 25, 15]]), trace=True, check=True)
    assert run.routes[0][1] == (1, 700.0, 25.0)
    arrival = [line for line in run.trace if "arrival" in line][0]
    assert "inside [0, 0, 0]" in arrival and "on the way [0, 0, 0]" in arrival  # waiting: neither inside nor on the way


def test_stop_limit_switch():
    day = hand_day([{"route": [0, 1], "entries": [600, 650]}])
    with_limit = simulate(day, Nearest(), stays=stays([[40, 25, 15]]))
    without = simulate(day, Nearest(), stays=stays([[40, 25, 15]]), stop_limit=False)
    assert len(with_limit.routes[0]) == 2 and len(without.routes[0]) == 3


def test_closing_and_budget_are_checked_with_expected_stays():
    # From site 0 at 640: site 1 would end at 650 + 20 = 670, after its closing at 660; site 2 would end
    # at 660 + 10 = 670, after the budget end 600 + 60. Nothing is feasible.
    day = hand_day([{"route": [0, 1], "entries": [600, 650], "budget": 60}], closing=(1000, 660, 1000))
    run = simulate(day, Nearest(), stays=stays([[40, 25, 15]]), check=True)
    assert run.routes[0] == [(0, 600.0, 40.0)] and run.end_reason[0] == "none feasible"


def test_first_stop_is_kept_even_at_an_unavailable_site():
    day = hand_day([{"route": [0, 1], "entries": [600, 650]}], available=(False, True, True))
    run = simulate(day, Nearest(), stays=stays([[40, 25, 15]]), check=True)
    assert run.routes[0][0] == (0, 600.0, 40.0) and run.routes[0][1][0] == 1


def test_overrun_is_kept_in_full_and_adds_no_budget():
    # Site 1 is chosen at 640 since 650 + 20 <= 690, but the sampled stay is 100: the visit runs to 750.
    day = hand_day([{"route": [0, 1], "entries": [600, 650], "budget": 120}], closing=(1000, 690, 1000))
    run = simulate(day, Nearest(), stays=stays([[40, 100, 15]]), check=True)
    assert run.routes[0][1] == (1, 650.0, 100.0)
    assert (1, 650.0, 750.0) in run.intervals
    assert run.overruns == [(0, 1, "closing"), (0, 1, "budget")]


def test_first_stop_overrun_is_not_counted():
    # The first stop is given, not chosen: its stay of 40 runs past closing at 620, but it is no overrun.
    day = hand_day([{"route": [0, 1], "entries": [600, 650]}], closing=(620, 1000, 1000))
    run = simulate(day, Nearest(), stays=stays([[40, 25, 15]]), check=True)
    assert run.routes[0][0] == (0, 600.0, 40.0) and run.overruns == []


# ---------------------------------------------------------------------- #
# 2. Several visitors at the same time
# ---------------------------------------------------------------------- #

def test_same_time_and_a_zero_minute_walk():
    # Sites 0 and 1 are 0 minutes apart. Both followers end their first visit at 640. v0 decides first,
    # walks 0 minutes to site 1, and that arrival and start re-enter the heap before v1's decision,
    # so v1 sees one person inside site 1 and nobody on the way.
    walk = [[0, 0, 20], [0, 0, 10], [20, 10, 0]]
    day = hand_day([{"route": [0, 1], "entries": [600, 650]}, {"route": [2, 1], "entries": [600, 650]}], walk=walk)
    policy = Recorder()
    simulate(day, policy, stays=stays([[40, 25, 15], [40, 25, 40]]), check=True)
    v1 = [seen for seen in policy.seen if seen["visitor"] == 1][0]
    assert v1["time"] == 640 and v1["inside"] == [0, 1, 0] and v1["on_the_way"] == [0, 0, 0]


def test_zero_length_fixed_entry_counts_a_visit_but_no_one_inside():
    # A visitor who does not follow enters site 2 at 640 with a zero-length interval, at the moment v0 decides.
    day = hand_day([{"route": [0, 1], "entries": [600, 650]},
                    {"route": [2, 0], "entries": [640, 700], "follows": False, "exits": [640, 730]}])
    policy = Recorder()
    run = simulate(day, policy, stays=stays([[40, 25, 15], [1, 1, 1]]), check=True)
    seen = policy.seen[0]
    assert seen["visits"][2] == 1 and seen["inside"][2] == 0
    assert run.visits.tolist() == [2, 1, 1]


def test_fixed_visitors_keep_their_times_and_everyone_leaves():
    day = hand_day([{"route": [0, 1], "entries": [600, 650]},
                    {"route": [1, 2], "entries": [610, 660], "follows": False, "exits": [655, 700]},
                    {"route": [2], "entries": [620], "follows": False, "exits": [640]}])
    run = simulate(day, Nearest(), stays=stays([[40, 25, 15], [9, 9, 9], [9, 9, 9]]), check=True)
    assert (1, 610.0, 655.0) in run.intervals and (2, 620.0, 640.0) in run.intervals
    assert sorted(run.routes) == [0]  # only the follower has a simulated route


def check_background_matches_replay(day):
    """With no followers, the engine only replays the fixed events, so it must give the replay's
    entries, person-minutes, crowding and A, C, T, V of every evaluated visitor, and R_v = 1."""
    run = simulate(day, Nearest(), check=True)
    expected = replay(day)
    assert run.routes == {} and np.array_equal(run.visits, expected.entries_per_site)
    for j in range(len(day.site_ids)):
        engine_times = sorted(t for site, t in run.entries if site == j)
        assert engine_times == sorted(day.fixed["entry"][day.fixed["site"] == j])
    site, start, end = (np.array(column) for column in zip(*run.intervals))
    minutes = hourly_person_minutes(site, start, end, len(day.site_ids))
    assert np.allclose(minutes, expected.person_minutes, rtol=0, atol=1e-9)
    c = crowding(minutes, day.usual_peak)
    for k, (a, c_sum, t) in expected.baseline.iterrows():
        assert np.allclose(route_parts(day.routes[k], day.entries[k], c, day.attractiveness, day.walk),
                           (a, c_sum, t), rtol=0, atol=1e-9)
    value = baseline_value(expected, 0.25, 0.5)
    assert np.allclose(relative_value(value, value, expected.baseline["A"]), 1.0)


def test_no_followers_gives_the_replay_on_a_hand_day():
    day = hand_day([{"route": [0, 1], "entries": [600, 650], "follows": False, "exits": [640, 660]},
                    {"route": [1, 2], "entries": [610, 660], "follows": False, "exits": [655, 700]},
                    {"route": [2], "entries": [620], "follows": False, "exits": [620]}])
    check_background_matches_replay(day)


def test_bad_inputs_are_refused():
    day = hand_day([{"route": [0, 1], "entries": [600, 650]}])
    for shape in ((1, 2), (2, 3)):
        with pytest.raises(ValueError):
            simulate(day, Nearest(), stays=np.full(shape, 10.0))
    for value in (np.nan, 0.5):
        with pytest.raises(ValueError):
            simulate(day, Nearest(), stays=stays([[40, value, 15]]))
    for field_name, value in (("walk", np.nan), ("walk", -1.0), ("stay", 0.0), ("usual_peak", np.nan),
                              ("closing", np.inf)):
        broken = hand_day([{"route": [0, 1], "entries": [600, 650]}])
        getattr(broken, field_name)[0] = value
        with pytest.raises(ValueError):
            simulate(broken, Nearest())
        with pytest.raises(ValueError):
            replay(broken)
    with pytest.raises(ValueError):  # a fixed interval that ends before it starts
        replay(hand_day([{"route": [0, 1], "entries": [600, 650], "follows": False, "exits": [590, 660]}]))


def test_a_run_keeps_its_settings():
    day = hand_day([{"route": [0, 1], "entries": [600, 650]}])
    run = simulate(day, Nearest(), replicate=3, stop_limit=False)
    assert run.settings["policy"] == "Nearest" and run.settings["replicate"] == 3
    assert run.settings["stop_limit"] is False and run.settings["stays given"] is False


# ---------------------------------------------------------------------- #
# 3. The observed-order control
# ---------------------------------------------------------------------- #

def test_observed_order_skips_with_the_first_reason():
    # Observed route 0, 1, 2; site 1 closes at 660, so at 640 it is passed over for closing and site 2 is taken.
    day = hand_day([{"route": [0, 1, 2], "entries": [600, 650, 700]}], closing=(1000, 660, 1000))
    run = simulate(day, ObservedOrder(), stays=stays([[40, 25, 15]]), check=True)
    assert [stop[0] for stop in run.routes[0]] == [0, 2]
    assert run.skips == [(0, 1, "closing")]  # recorded once, not re-judged at the end


def test_observed_order_records_stops_left_at_the_end():
    # A budget of 50 leaves no time after site 0: sites 1 and 2 are both lost, each once.
    day = hand_day([{"route": [0, 1, 2], "entries": [600, 650, 700], "budget": 50}])
    run = simulate(day, ObservedOrder(), stays=stays([[40, 25, 15]]), check=True)
    assert run.routes[0] == [(0, 600.0, 40.0)]
    assert run.skips == [(0, 1, "budget"), (0, 2, "budget")]


def test_one_observed_order_policy_used_twice_gives_the_same_day():
    day = hand_day([{"route": [0, 1, 2], "entries": [600, 650, 700]}], closing=(1000, 660, 1000))
    policy = ObservedOrder()
    first = simulate(day, policy, stays=stays([[40, 25, 15]]))
    second = simulate(day, policy, stays=stays([[40, 25, 15]]))
    assert first.routes == second.routes and first.skips == second.skips and first.end_reason == second.end_reason


def test_a_rule_cannot_stop_a_visitor():
    class Stopper:
        def choose(self, engine, k, candidates):
            return None

    day = hand_day([{"route": [0, 1], "entries": [600, 650]}])
    with pytest.raises(ValueError):
        simulate(day, Stopper(), stays=stays([[40, 25, 15]]))


# ---------------------------------------------------------------------- #
# 4. Random stays
# ---------------------------------------------------------------------- #

def test_stays_are_keyed_and_floored():
    day = hand_day([{"route": [0, 1], "entries": [600, 650]}, {"route": [2, 1], "entries": [600, 650]}])
    table = stay_table(day, 0)
    assert np.array_equal(table, stay_table(day, 0)) and not np.array_equal(table, stay_table(day, 1))
    # With d = 1 minute, about half the draws fall below 1 and must be set to exactly 1.
    short = hand_day([{"route": [0, 1], "entries": [600, 650]}] * 20, stay=(1, 1, 1))
    table = stay_table(short, 0)
    assert table.min() == 1.0 and (table == 1.0).mean() > 0.3


# ---------------------------------------------------------------------- #
# 5. Real days
# ---------------------------------------------------------------------- #

def check_routes(day, run):
    """Every later stop met closing and budget under expected stays when chosen; one place at a time."""
    start = np.array([e[0] for e in day.entries])
    budget_end = start + day.visitors.budget_limit.to_numpy()
    for k, route in run.routes.items():
        assert len(route) <= day.visitors.stop_limit[k]
        for (a, t_a, stay_a), (b, t_b, _) in zip(route, route[1:]):
            assert t_b >= t_a + stay_a + day.walk[a, b] - 1e-9  # leaves a, walks, then starts b
            assert t_b + day.stay[b] <= day.closing[b] + 1e-9 and t_b + day.stay[b] <= budget_end[k] + 1e-9


@needs_data
def test_real_day_runs_are_consistent_and_repeatable():
    day = load_day("2019-10-09")
    nearest = simulate(day, Nearest(), check=True)
    again = simulate(day, Nearest(), check=True)
    assert nearest.routes == again.routes and nearest.intervals == again.intervals
    observed = simulate(day, ObservedOrder(), check=True)
    for run in (nearest, observed):
        check_routes(day, run)
        assert run.visits.sum() == len(run.entries)
    table = stay_table(day, 0)
    for run in (nearest, observed):  # both rules read the same stay for the same visitor and site
        for k, route in run.routes.items():
            assert all(stay == table[k, j] for j, _, stay in route)


@needs_data
def test_no_followers_gives_the_replay_on_a_real_day():
    check_background_matches_replay(load_day("2019-10-09", follows=[]))


@needs_data
def test_load_day_refuses_what_it_cannot_build():
    with pytest.raises(ValueError):
        load_day("2019-10-09", year=2022)
    with pytest.raises(ValueError):
        load_day("2019-02-30")  # not a date at all
    with pytest.raises(ValueError):
        load_day("2020-06-01")  # no tables for that year
    if (PROCESSED / "visits_2023.csv.gz").exists():
        with pytest.raises(ValueError):
            load_day("2023-06-01")  # the 2023 tables stop at February
    with pytest.raises(ValueError):
        load_day("2019-10-09", follows=["no such card"])
    one_stop = load_day("2019-10-09").visitors
    with pytest.raises(ValueError):
        load_day("2019-10-09", follows=one_stop["visitor"][~one_stop["evaluated"]].iloc[:1])
    assert load_day("2019-10-09", budget="span").options == {"budget": "span", "same_day": False,
                                                                   "peak_floor": None}


@needs_data
def test_followers_and_fixed_visitors_together_on_a_real_day():
    day = load_day("2019-10-09")
    half = day.visitors.visitor[day.visitors.evaluated].iloc[::2]
    run = simulate(load_day("2019-10-09", follows=half), Nearest(), check=True)
    assert len(run.routes) == len(half)
    assert run.visits.sum() == len(run.entries)
