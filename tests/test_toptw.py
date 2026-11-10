"""Hand-built checks of the single-tourist simulation. Run from the repository root: python3 -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toptw import Instance, load_instance, protected_div, sample_durations, solve, sos, sot  # noqa: E402


def line_instance():
    """Depot at 0, sites at 10 and 20 on a line, then a site too far to return from."""
    x = np.array([0.0, 10.0, 20.0, 500.0])
    travel = np.abs(x[:, None] - x[None, :])
    return Instance("line", np.array([0.0, 5.0, 7.0, 100.0]), np.array([0.0, 2.0, 3.0, 1.0]),
                    np.array([0.0, 0.0, 30.0, 0.0]), np.array([100.0, 100.0, 100.0, 100.0]), travel)


def highest_score(score, *rest):
    return score


def test_infeasible_site_is_never_chosen():
    instance = line_instance()
    total, stops = solve(instance, highest_score, instance.duration)
    assert 3 not in stops and total == 12.0


def test_waits_for_opening_and_keeps_order():
    instance = line_instance()
    seen = []

    def record(score, dur, to, tc, ta, tr, tsv, tfv, sl, rem):
        seen.append((score.tolist(), to.tolist(), ta.tolist(), tsv.tolist()))
        return -ta

    total, stops = solve(instance, record, instance.duration)
    assert stops == [1, 2] and total == 12.0
    assert seen[0][1] == [0.0, 30.0] and seen[0][3] == [10.0, 30.0]  # site 2 cannot start before it opens
    assert seen[1][1] == [18.0] and seen[1][2] == [10.0]  # clock is 12 after visiting site 1


def test_second_day_restarts_at_the_depot():
    instance = line_instance()
    instance.closing[0] = 45.0
    instance.opening[2] = 0.0

    def nearest_then_stop(score, dur, to, tc, ta, tr, tsv, tfv, sl, rem):
        return -ta

    assert solve(instance, nearest_then_stop, instance.duration, days=2)[1] == [1, 2]


def test_sampled_duration_moves_the_clock_but_not_feasibility():
    instance = line_instance()
    instance.closing[0] = 36.0  # expected route ends exactly within budget only without a long first visit
    long_first = instance.duration.copy()
    long_first[1] = 60.0
    assert solve(instance, lambda score, *r: -r[3], long_first)[1] == [1]


def test_protected_division_and_handwritten_rules():
    assert protected_div(np.array([2.0, 3.0]), np.array([0.0, 3.0])).tolist() == [1.0, 1.0]
    score, sl, ta = np.array([4.0]), np.array([2.0]), np.array([0.0])
    zeros = np.zeros(1)
    assert sos(score, zeros, zeros, zeros, ta, zeros, zeros, zeros, sl, zeros) == 2.0
    assert sot(score, zeros, zeros, zeros, ta, zeros, zeros, zeros, sl, zeros) == 1.0


def test_routes_on_benchmarks_pass_an_independent_check():
    """Recompute 1,080 routes outside solve(): visits once, time windows and budget hold, and the score agrees."""
    folder = Path(__file__).resolve().parents[1] / "data" / "toptw"
    if not folder.exists():
        pytest.skip("TOPTW instances are not in the repository")
    names = "c101 c103 c201 c203 r101 r105 r202 r205 rc101 rc105 rc201 rc205 pr01 pr02 pr03 pr11 pr12 pr14".split()
    for name in names:
        instance = load_instance(folder / f"{name}.txt")
        rng, choices = np.random.default_rng(1), np.random.default_rng(2)

        def random_rule(score, *rest):
            return choices.random(score.size)

        for policy in (sot, random_rule):
            for _ in range(30):
                durations = sample_durations(instance, rng)
                total, stops = solve(instance, policy, durations)
                assert len(set(stops)) == len(stops) and 0 not in stops
                assert abs(total - instance.score[stops].sum()) < 1e-9
                now, here = instance.opening[0], 0
                for stop in stops:
                    arrive = now + instance.travel[here, stop]
                    start = max(arrive, instance.opening[stop])
                    assert arrive <= instance.closing[stop] + 1e-9
                    assert start + instance.duration[stop] + instance.travel[stop, 0] <= instance.closing[0] + 1e-9
                    now, here = start + durations[stop], stop
