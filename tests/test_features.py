"""
Tests of the state inputs and the decision log (Step 3e).

Run:  python3 -m pytest tests -q   (from the repository root)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from day import PROCESSED, load_day  # noqa: E402
from engine import Nearest, ObservedOrder, simulate, stay_table  # noqa: E402
from features import (H1_TERMINALS, H2_TERMINALS, best, finite, h1_inputs, h2_inputs,  # noqa: E402
                      log_decision, visit_share)
from test_engine import hand_day, stays  # noqa: E402

needs_data = pytest.mark.skipif(not (PROCESSED / "manifest.json").exists(), reason="processed tables not built")


class Snapshot(Nearest):
    """Nearest, keeping every input it could have read at each decision."""
    def __init__(self):
        self.seen = []

    def choose(self, engine, k, candidates):
        self.seen.append({"visitor": k, "time": engine.now, "candidates": candidates.tolist(),
                          "h1": dict(zip(H1_TERMINALS, h1_inputs(engine, k, candidates))),
                          "h2": dict(zip(H2_TERMINALS, h2_inputs(engine, k))),
                          "VS": visit_share(engine, candidates)})
        return super().choose(engine, k, candidates)


# ---------------------------------------------------------------------- #
# 1. Each terminal on a hand-built state
# ---------------------------------------------------------------------- #

def test_each_terminal_by_hand():
    # v0 follows: site 0 from 600 for 40 minutes, decides at 640 between sites 1 and 2.
    # v1 is fixed: in site 1 from 610 to 700. v2 is fixed: in site 2 from 630 to 650.
    # Site 1 opens at 660 and closes at 900; usual peaks 2, 4, 5; d = 30, 20, 10.
    day = hand_day([{"route": [0, 1], "entries": [600, 650], "budget": 200.0},
                    {"route": [1, 2], "entries": [610, 700], "follows": False, "exits": [700, 720]},
                    {"route": [2], "entries": [630], "follows": False, "exits": [650]}],
                   opening=(0, 660, 0), closing=(1000, 900, 950))
    day.usual_peak = np.array([2.0, 4.0, 5.0])
    day.usual_arrivals = np.full(781, 2.0)
    day.usual_gday = np.full(781, 0.5)
    policy = Snapshot()
    simulate(day, policy, stays=stays([[40, 25, 15], [9, 9, 9], [9, 9, 9]]), check=True)
    seen = policy.seen[0]
    assert seen["time"] == 640 and seen["candidates"] == [1, 2]
    h1 = seen["h1"]
    assert h1["AV"].tolist() == [1, 1] and h1["DUR"].tolist() == [20, 10]
    assert h1["TA"].tolist() == [10, 20]
    assert h1["TO"].tolist() == [20, -640] and h1["TC"].tolist() == [260, 310]
    assert h1["TSV"].tolist() == [20, 20] and h1["TFV"].tolist() == [40, 30]
    assert h1["SL"].tolist() == [250, 290] and h1["RemT"].tolist() == [160, 160]
    assert h1["CR"].tolist() == [1 / 4, 1 / 5]   # v0 has left site 0; v1 inside 1, v2 inside 2
    assert h1["HEAD"].tolist() == [0, 0]
    h2 = seen["h2"]
    assert h2["TLD"] == 1000 - 640
    assert h2["ARR"] == 3 / 2                     # entries at 600, 610, 630
    assert h2["MCR"] == pytest.approx((0 + 1 / 4 + 1 / 5) / 3)
    x = np.array([1, 1, 1])                      # one visit at each site: G_hat 0
    assert h2["GDAY"] == 0 and x.sum() == 3
    assert h2["RemT"] == 160 and h2["STOPS"] == 1
    assert seen["VS"].tolist() == [1, 1]


def test_head_counts_a_visitor_on_the_way():
    # v0 and v1 both start at 600; v0 leaves site 0 at 620 and walks 20 minutes to site 2.
    # When v1 decides at 630, v0 is on the way to site 2.
    day = hand_day([{"route": [0, 2], "entries": [600, 640]}, {"route": [1, 2], "entries": [600, 640]}],
                   walk=[[0, 50, 20], [50, 0, 10], [20, 10, 0]])
    policy = Snapshot()
    simulate(day, policy, stays=stays([[20, 9, 9], [9, 30, 9]]), check=True)
    second = policy.seen[1]
    assert second["visitor"] == 1 and second["time"] == 630
    assert second["h1"]["HEAD"].tolist() == [0, 1 / 1]  # candidates 0 and 2; usual peak 1


def test_vs_and_gday_before_any_visit_and_after():
    day = hand_day([{"route": [0, 1], "entries": [600, 650]}])
    day.usual_gday = np.full(781, 0.5)
    policy = Snapshot()
    run = simulate(day, policy, stays=stays([[40, 25, 15]]))
    # At 640 the only visit so far is v0's at site 0: x = (1, 0, 0), mean 1/3.
    assert policy.seen[0]["VS"].tolist() == [0, 0]
    assert policy.seen[0]["h2"]["GDAY"] == pytest.approx(2.0)  # G_hat 1 over usual 0.5
    assert run.fallbacks == {}


# ---------------------------------------------------------------------- #
# 2. Fallbacks
# ---------------------------------------------------------------------- #

def test_fallbacks_by_reason():
    # Before 08:00 both usual curves are out of range.
    early = hand_day([{"route": [0, 1], "entries": [400, 450]}])
    policy = Snapshot()
    run = simulate(early, policy, stays=stays([[40, 25, 15]]))
    assert policy.seen[0]["h2"]["ARR"] == 1 and policy.seen[0]["h2"]["GDAY"] == 1
    assert run.fallbacks == {"ARR: outside 08:00-21:00": 1, "GDAY: outside 08:00-21:00": 1}

    # Visits only at an unavailable site: GDAY has nothing to measure; VS is 1.
    closed = hand_day([{"route": [0, 1], "entries": [600, 650]}], available=(False, True, True))
    policy = Snapshot()
    run = simulate(closed, policy, stays=stays([[40, 25, 15]]))
    assert policy.seen[0]["h2"]["GDAY"] == 1 and policy.seen[0]["VS"].tolist() == [1, 1]
    assert run.fallbacks == {"GDAY: no visits yet at the available sites": 1}

    # A usual value of zero: ARR falls back. With no recent entries and a positive usual value
    # ARR is 0, not a fallback.
    zero = hand_day([{"route": [0, 1], "entries": [600, 650]}])
    zero.usual_arrivals = np.zeros(781)
    run = simulate(zero, Snapshot(), stays=stays([[40, 25, 15]]))
    assert run.fallbacks == {"ARR: usual value not positive": 1}
    quiet = hand_day([{"route": [0, 1], "entries": [600, 700]}])
    policy = Snapshot()
    run = simulate(quiet, policy, stays=stays([[90, 25, 15]]))  # decides at 690, the entry was at 600
    assert policy.seen[0]["h2"]["ARR"] == 0 and run.fallbacks == {}


def test_usual_curves_are_interpolated():
    # Decision at 640.5; usual ARR is 2 at minute 640 and 4 at 641, so 3 at 640.5.
    day = hand_day([{"route": [0, 1], "entries": [600, 650]}])
    curve = np.full(781, 2.0)
    curve[641 - 480:] = 4.0
    day.usual_arrivals = curve
    policy = Snapshot()
    simulate(day, policy, stays=stays([[40.5, 25, 15]]))
    assert policy.seen[0]["h2"]["ARR"] == pytest.approx(1 / 3)


def test_non_finite_priorities_become_zero_and_are_counted():
    day = hand_day([{"route": [0, 1], "entries": [600, 650]}])
    run = simulate(day, Nearest(), stays=stays([[40, 25, 15]]))

    class Holder:
        pass
    engine = Holder()
    engine.run, engine.day = run, day
    values = finite(engine, [np.nan, 2.0, np.inf], "h1")
    assert values.tolist() == [0, 2, 0] and run.fallbacks == {"h1: not finite": 2}


# ---------------------------------------------------------------------- #
# 3. An independent ledger of the counts at every decision
# ---------------------------------------------------------------------- #

class Ledger:
    """Wraps a policy. At every decision it recomputes people inside, on the way, visits so far
    and the recent entries from what has already happened, without the engine's running counts,
    and compares them with what the state inputs read."""
    def __init__(self, inner):
        self.inner, self.walks, self.checked = inner, [], 0
        self.may_stop = getattr(inner, "may_stop", False)

    def reset(self):
        self.walks = []
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    def finish(self, engine, k):
        if hasattr(self.inner, "finish"):
            self.inner.finish(engine, k)

    def choose(self, engine, k, candidates):
        day, t, n = engine.day, engine.now, len(engine.day.site_ids)
        follows = day.visitors["follows"].to_numpy()
        fixed = day.fixed[~follows[day.fixed["visitor"].to_numpy()]]
        site, entry, exit_ = (fixed[c].to_numpy() for c in ("site", "entry", "exit"))
        inside = np.bincount(site[(entry <= t) & (t < exit_)], minlength=n)
        x = np.bincount(site[entry <= t], minlength=n)
        starts = list(entry[entry <= t])
        for route in engine.run.routes.values():  # followers' visits that have started
            for j, start, stay in route:
                inside[j] += start <= t < start + stay
                x[j] += 1
                starts.append(start)
        on_the_way = np.zeros(n, int)
        for j, arrival in self.walks:
            on_the_way[j] += arrival > t
        recent = sum(t - 60 < s <= t for s in starts)

        assert np.array_equal(inside, engine.inside) and np.array_equal(on_the_way, engine.on_the_way)
        assert np.array_equal(x, engine.visits)
        h1 = dict(zip(H1_TERMINALS, h1_inputs(engine, k, candidates)))
        h2 = dict(zip(H2_TERMINALS, h2_inputs(engine, k)))
        assert np.allclose(h1["CR"], inside[candidates] / day.usual_peak[candidates])
        assert np.allclose(h1["HEAD"], on_the_way[candidates] / day.usual_peak[candidates])
        usual = np.interp(t, np.arange(480, 1261), day.usual_arrivals) if 480 <= t <= 1260 else np.nan
        assert h2["ARR"] == pytest.approx(recent / usual if usual > 0 else 1.0)
        self.checked += 1

        j = self.inner.choose(engine, k, candidates)
        if j is not None:
            self.walks.append((j, t + day.walk[engine.place[k], j]))
        return j


def test_ledger_on_a_hand_day_with_zero_walks_and_zero_length_entries():
    day = hand_day([{"route": [0, 1], "entries": [600, 650]},
                    {"route": [0, 2], "entries": [600, 650]},
                    {"route": [1, 2], "entries": [640, 700], "follows": False, "exits": [640, 720]},
                    {"route": [2], "entries": [620], "follows": False, "exits": [660]}],
                   walk=[[0, 0, 10], [0, 0, 10], [10, 10, 0]])
    policy = Ledger(Nearest())
    simulate(day, policy, stays=stays([[40, 25, 15], [40, 25, 15], [9, 9, 9], [9, 9, 9]]), check=True)
    assert policy.checked == 2  # both decide at 640; their second stops reach the stop limit


@needs_data
@pytest.mark.parametrize("inner", [Nearest, ObservedOrder])
def test_ledger_on_a_real_day(inner):
    day = load_day("2019-10-09")
    policy = Ledger(inner())
    run = simulate(day, policy, check=True)
    assert policy.checked > 500 and run.events > 0  # about 710 decisions on this day


# ---------------------------------------------------------------------- #
# 4. No input reads the future
# ---------------------------------------------------------------------- #

def test_inputs_do_not_change_with_what_happens_later():
    # The two days are the same until 700; afterwards a fixed visitor's later entries differ.
    base = [{"route": [0, 1, 2], "entries": [600, 650, 700]}]
    one = hand_day(base + [{"route": [1, 2], "entries": [610, 900], "follows": False, "exits": [650, 950]}])
    two = hand_day(base + [{"route": [1, 0], "entries": [610, 800], "follows": False, "exits": [650, 990]}])
    table = stays([[40, 25, 15], [9, 9, 9]])
    first, second = Snapshot(), Snapshot()
    simulate(one, first, stays=table)
    simulate(two, second, stays=table)
    early = [(a, b) for a, b in zip(first.seen, second.seen) if a["time"] < 700]
    assert early
    for a, b in early:
        assert all(np.array_equal(a["h1"][name], b["h1"][name]) for name in H1_TERMINALS)
        assert a["h2"] == b["h2"]


# ---------------------------------------------------------------------- #
# 5. Ties and the decision log
# ---------------------------------------------------------------------- #

def test_ties_go_to_the_lowest_site_id_whatever_the_order():
    day = hand_day([{"route": [0, 1], "entries": [600, 650]}])
    day.site_ids = np.array([30, 20, 10])  # index 2 has the lowest id

    class Holder:
        pass
    engine = Holder()
    engine.day = day
    assert best(engine, np.array([0, 1, 2]), np.array([5.0, 5.0, 5.0])) == 2
    assert best(engine, np.array([2, 0, 1]), np.array([5.0, 5.0, 1.0])) == 2
    assert best(engine, np.array([0, 1]), np.array([5.0, 5.0])) == 1


class Fair:
    """h1 = AV - TA / 100, w = 0.5; logs its decisions when asked."""
    def __init__(self, log):
        self.log = log

    def choose(self, engine, k, candidates):
        h1_values = dict(zip(H1_TERMINALS, h1_inputs(engine, k, candidates)))
        h1 = h1_values["AV"] - h1_values["TA"] / 100
        vs = visit_share(engine, candidates)
        if self.log:
            return log_decision(engine, k, candidates, h1, 0.5, vs, h2_inputs(engine, k))
        return best(engine, candidates, h1 - 0.5 * vs)


@needs_data
def test_shadow_choice_changes_neither_routes_nor_random_numbers():
    day = load_day("2019-12-18")
    with_log = simulate(day, Fair(True))
    without = simulate(day, Fair(False))
    assert with_log.routes == without.routes and with_log.intervals == without.intervals
    assert without.decisions == [] and len(with_log.decisions) > 100
    table = stay_table(day, 0)
    assert all(stay == table[k, j] for k, route in with_log.routes.items() for j, _, stay in route)
    row = with_log.decisions[0]
    assert set(H2_TERMINALS) <= set(row) and row["changed"] == (row["j0"] != row["jw"])
    changed = [r for r in with_log.decisions if r["changed"]]
    assert changed and all(r["h1 given up"] >= 0 for r in with_log.decisions)


# ---------------------------------------------------------------------- #
# 6. The usual-peak floor
# ---------------------------------------------------------------------- #

@needs_data
def test_usual_peak_floor_is_an_option():
    plain = load_day("2019-10-09")
    floored = load_day("2019-10-09", peak_floor=10.0)
    assert (floored.usual_peak == np.maximum(plain.usual_peak, 10.0)).all()
    assert (floored.usual_peak > plain.usual_peak).any() and floored.options["peak_floor"] == 10.0


# ---------------------------------------------------------------------- #
# 7. Hand rules (Step 3g)
# ---------------------------------------------------------------------- #

def rule_choice(rule, walk=None):
    """v0 leaves site 0 at 640 and chooses between sites 1 and 2. A = (1, 0.4, 0.6); walks 10 and 20
    minutes; a one-stop visitor entered site 2 at 620, so x = (1, 0, 1); every site closes at 1000."""
    from rules import RULES
    day = hand_day([{"route": [0, 1], "entries": [600, 650]},
                    {"route": [2], "entries": [620], "follows": False, "exits": [630]}], walk=walk)
    day.attractiveness = np.array([1.0, 0.4, 0.6])
    run = simulate(day, RULES[rule](), stays=stays([[40, 25, 15], [9, 9, 9]]), check=True)
    return run.routes[0][1][0]


def test_hand_rules_by_hand():
    assert rule_choice("POP") == 2      # 0.6 > 0.4
    assert rule_choice("LEAST") == 1    # no visit yet at site 1
    assert rule_choice("SOT") == 1      # 0.4 / 10 > 0.6 / 20
    assert rule_choice("SOS") == 2      # slack 350 and 340: 0.4 / 350 < 0.6 / 340


def test_sot_with_a_zero_minute_walk_uses_protected_division():
    # Site 1 is 0 minutes away: A / 0 gives 1 (Step 1's protected division), more than 0.6 / 20.
    assert rule_choice("SOT", walk=[[0, 0, 20], [0, 0, 10], [20, 10, 0]]) == 1
