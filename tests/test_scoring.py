"""
Tests of scoring a simulated day (Step 3f).

Run:  python3 -m pytest tests -q   (from the repository root)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from day import PROCESSED, load_day  # noqa: E402
from engine import Nearest, ObservedOrder, simulate  # noqa: E402
from replay import replay  # noqa: E402
from scoring import score_day, summary  # noqa: E402
from test_engine import hand_day, stays  # noqa: E402

needs_data = pytest.mark.skipif(not (PROCESSED / "manifest.json").exists(), reason="processed tables not built")


class LowestSite:
    """Always the feasible site with the lowest id."""
    def choose(self, engine, k, candidates):
        return candidates[0]


def two_visitors():
    """v0 does not follow: site 1 from 650 to 690, then site 2 from 700 to 710.
    v1 follows. Observed: site 2 at 600 (10 minutes), then site 1 at 650 (20 minutes).
    Simulated: site 2 from 600 to 610, then site 0, walking 20 minutes, from 630 to 660.
    Usual peaks and A are all 1; walks 10 minutes between neighbouring sites."""
    day = hand_day([{"route": [1, 2], "entries": [650, 700], "follows": False, "exits": [690, 710]},
                    {"route": [2, 1], "entries": [600, 650]}])
    run = simulate(day, LowestSite(), stays=stays([[9, 9, 9], [30, 20, 10]]), check=True)
    return day, run


def test_a_visitor_who_keeps_the_route_gains_when_another_leaves():
    day, run = two_visitors()
    assert run.routes[1] == [(2, 600.0, 10.0), (0, 630.0, 30.0)]
    v, g_hat, _ = score_day(day, run, replay(day), 0.25, 0.5)
    # v0. Observed C: site 1 in hour 10 holds v0 for 10 and v1 for 10 minutes (20/60), site 2 in
    # hour 11 holds v0 for 10 (10/60). Simulated, v1 is not at site 1: 10/60 + 10/60.
    assert v.loc[0, "C"] == pytest.approx(20 / 60) and v.loc[0, "change C"] == pytest.approx(-10 / 60)
    assert v.loc[0, "R_v"] == pytest.approx(1 + 0.25 * (10 / 60) / 2)
    # v1. Observed C: site 2 in hour 10 (10/60) + site 1 in hour 10 (20/60); walk 10 minutes.
    # Simulated C: site 2 (10/60) + site 0 (30/60); walk 20 minutes.
    assert v.loc[1, "C"] == pytest.approx(40 / 60) and v.loc[1, "T"] == pytest.approx(20 / 60)
    assert v.loc[1, "R_v"] == pytest.approx(1 + (-0.25 * (10 / 60) - 0.5 * (10 / 60)) / 2)
    # Visits (1, 1, 2): G_hat = (2 / 2) / (9 * 4/3) / (2/3) = 1/4.
    assert g_hat == pytest.approx(0.25)


def test_summary_pools_the_visitors_and_the_split_adds_up():
    day, run = two_visitors()
    v, _, _ = score_day(day, run, replay(day), 0.25, 0.5)
    out = summary(v, 0.25, 0.5)
    assert out["all"]["visitors"] == 2 and out["all"]["mean R_v"] == pytest.approx(v["R_v"].mean())
    assert out["followers"]["visitors"] == 1 and out["non-followers"]["visitors"] == 1
    assert out["followers"]["end reasons"] == {"limit": 1}
    assert out["non-followers"]["end reasons"] == {"not a follower": 1}
    for group in out.values():
        assert group["split adds up"] and group["completion"] == 1 and group["stops lost"] == 0


def test_without_the_crowding_term_crowding_changes_nothing():
    day, run = two_visitors()
    v, _, _ = score_day(day, run, replay(day), 0.0, 0.5)
    assert v.loc[0, "R_v"] == 1  # same route, same walk
    assert v.loc[1, "R_v"] == pytest.approx(1 - 0.5 * (10 / 60) / 2)


def test_lost_stops_and_overruns_are_counted():
    # v0 follows with a budget of 45 minutes: after 40 minutes at site 0 nothing fits.
    day = hand_day([{"route": [0, 1], "entries": [600, 650], "budget": 45.0}])
    run = simulate(day, Nearest(), stays=stays([[40, 25, 15]]), check=True)
    v, _, _ = score_day(day, run, replay(day), 0.25, 0.5)
    out = summary(v, 0.25, 0.5)["all"]
    assert out["completion"] == 0.5 and out["ended early"] == 1 and out["stops lost"] == 1
    assert out["end reasons"] == {"none feasible": 1}
    # R_v divides by A of the observed route (2), not of the new one (1). Observed: C = 30/60 at
    # site 0 + 10/60 at site 1 (from 650), walk 10 minutes. New: C = 40/60, no walk.
    change = (1 - 0.25 * 40 / 60) - (2 - 0.25 * 40 / 60 - 0.5 * 10 / 60)
    assert v.loc[0, "R_v"] == pytest.approx(1 + change / 2)


def test_completion_is_pooled_over_visitors():
    # v0 keeps 1 of 2 stops, v1 all 3: pooled 4/5, not the mean of 1/2 and 1.
    day = hand_day([{"route": [0, 1], "entries": [600, 650], "budget": 45.0},
                    {"route": [2, 1, 0], "entries": [600, 620, 650]}])
    run = simulate(day, Nearest(), stays=stays([[40, 25, 15], [40, 25, 15]]), check=True)
    v, _, _ = score_day(day, run, replay(day), 0.25, 0.5)
    assert summary(v, 0.25, 0.5)["all"]["completion"] == pytest.approx(4 / 5)


@needs_data
def test_a_baseline_from_another_day_or_peak_is_refused():
    day = load_day("2019-10-09")
    run = simulate(day, Nearest())
    for other in (load_day("2019-10-10"), load_day("2019-10-09", peak_floor=10.0)):
        with pytest.raises(ValueError):
            score_day(day, run, replay(other), 0.25, 0.5)
    # Who follows and the budget do not change the baseline, so they may differ.
    span = load_day("2019-10-09", budget="span", follows=[])
    score_day(span, simulate(span, Nearest()), replay(day), 0.25, 0.5)


@needs_data
def test_compliance_sets_are_told_apart():
    day = load_day("2019-10-09")
    half = day.visitors["visitor"][day.visitors["evaluated"]].iloc[::2]
    other = day.visitors["visitor"][day.visitors["evaluated"]].iloc[1::2]
    ids = {simulate(load_day("2019-10-09", follows=f), Nearest()).settings["followers"] for f in (half, other, None)}
    assert len(ids) == 3


@needs_data
def test_with_no_followers_every_r_v_is_one_and_g_hat_is_observed():
    day = load_day("2019-10-09", follows=[])
    run = simulate(day, Nearest())
    observed = replay(day)
    v, g_hat, c = score_day(day, run, observed, 0.25, 0.5)
    assert np.allclose(v["R_v"], 1, rtol=0, atol=1e-12) and np.allclose(c, observed.crowding)
    from preprocess import gini_hat
    assert g_hat == gini_hat(observed.entries_per_site[day.available])


@needs_data
def test_a_real_day_scores_every_evaluated_visitor():
    day = load_day("2019-10-09")
    v, g_hat, _ = score_day(day, simulate(day, ObservedOrder()), replay(day), 0.25, 0.5)
    assert len(v) == day.visitors["evaluated"].sum() and np.isfinite(v["R_v"]).all()
    out = summary(v, 0.25, 0.5)
    assert 0 < g_hat < 1 and 0.8 < out["all"]["completion"] <= 1 and out["all"]["split adds up"]
