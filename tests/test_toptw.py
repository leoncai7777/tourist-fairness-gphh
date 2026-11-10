"""
Tests of the single-tourist simulation and the GP operators (Step 1).

Run:  python3 -m pytest tests -q   (from the repository root)
The benchmark check is skipped when data/toptw/ is not there.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest
from deap import gp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import single_gphh as s  # noqa: E402
from toptw import Instance, counters, load_instance, protected_div, sample_durations, solve, sos, sot  # noqa: E402

BENCHMARKS = Path(__file__).resolve().parents[1] / "data" / "toptw"


def line_instance():
    """Depot at 0, sites at 10 and 20 on a line, and a site too far away to return from."""
    x = np.array([0.0, 10.0, 20.0, 500.0])
    travel = np.abs(x[:, None] - x[None, :])
    score = np.array([0.0, 5.0, 7.0, 100.0])
    duration = np.array([0.0, 2.0, 3.0, 1.0])
    opening = np.array([0.0, 0.0, 30.0, 0.0])
    closing = np.array([100.0, 100.0, 100.0, 100.0])
    return Instance("line", score, duration, opening, closing, travel)


def highest_score(score, *rest):
    return score


def nearest(score, dur, to, tc, ta, tr, tsv, tfv, sl, rem):
    return -ta


# ---------------------------------------------------------------------- #
# 1. Simulation on a hand-built instance
# ---------------------------------------------------------------------- #

def test_infeasible_site_is_never_chosen():
    instance = line_instance()
    total, stops = solve(instance, highest_score, instance.duration)
    assert 3 not in stops and total == 12.0


def test_waits_for_opening_and_keeps_order():
    instance = line_instance()
    seen = []

    def record(score, dur, to, tc, ta, tr, tsv, tfv, sl, rem):
        seen.append({"to": to.tolist(), "ta": ta.tolist(), "tsv": tsv.tolist()})
        return -ta

    total, stops = solve(instance, record, instance.duration)
    assert stops == [1, 2] and total == 12.0
    assert seen[0]["to"] == [0.0, 30.0] and seen[0]["tsv"] == [10.0, 30.0]  # site 2 cannot start before it opens
    assert seen[1]["to"] == [18.0] and seen[1]["ta"] == [10.0]  # the clock is 12 after visiting site 1


def test_second_day_restarts_at_the_depot():
    instance = line_instance()
    instance.closing[0] = 45.0
    instance.opening[2] = 0.0
    assert solve(instance, nearest, instance.duration, days=2)[1] == [1, 2]


def test_sampled_duration_moves_the_clock_but_not_feasibility():
    instance = line_instance()
    instance.closing[0] = 36.0  # the expected route fits only without a long first visit
    long_first = instance.duration.copy()
    long_first[1] = 60.0
    assert solve(instance, nearest, long_first)[1] == [1]


def test_ties_go_to_the_earliest_point():
    instance = line_instance()
    assert solve(instance, lambda score, *rest: np.zeros(score.size), instance.duration)[1] == [1, 2]


def test_non_finite_priorities_are_set_to_zero_and_counted():
    instance = line_instance()
    counters["non_finite"] = 0
    stops = solve(instance, lambda score, *rest: np.full(score.size, np.nan), instance.duration)[1]
    assert stops == [1, 2] and counters["non_finite"] == 3  # two candidates, then one


def test_protected_division_and_handwritten_rules():
    assert protected_div(np.array([2.0, 3.0]), np.array([0.0, 3.0])).tolist() == [1.0, 1.0]
    score, sl, ta, zeros = np.array([4.0]), np.array([2.0]), np.array([0.0]), np.zeros(1)
    assert sos(score, zeros, zeros, zeros, ta, zeros, zeros, zeros, sl, zeros) == 2.0
    assert sot(score, zeros, zeros, zeros, ta, zeros, zeros, zeros, sl, zeros) == 1.0


# ---------------------------------------------------------------------- #
# 2. Routes on the benchmark instances, checked outside solve()
# ---------------------------------------------------------------------- #

def check_route(instance, stops, durations, total):
    """Each point once; each selection feasible under expected durations; the score agrees.
    Realised sampled durations may exceed the nominal return budget, as in the paper."""
    assert len(set(stops)) == len(stops) and 0 not in stops
    assert abs(total - instance.score[stops].sum()) < 1e-9
    now, here = instance.opening[0], 0
    for stop in stops:
        arrive = now + instance.travel[here, stop]
        start = max(arrive, instance.opening[stop])
        assert arrive <= instance.closing[stop] + 1e-9
        assert start + instance.duration[stop] + instance.travel[stop, 0] <= instance.closing[0] + 1e-9
        now, here = start + durations[stop], stop


def test_routes_on_benchmarks_pass_an_independent_check():
    """1,080 routes: 18 instances, two rules, 30 samples each."""
    if not BENCHMARKS.exists():
        pytest.skip("TOPTW instances are not in the repository")
    names = "c101 c103 c201 c203 r101 r105 r202 r205 rc101 rc105 rc201 rc205 pr01 pr02 pr03 pr11 pr12 pr14".split()
    for name in names:
        instance = load_instance(BENCHMARKS / f"{name}.txt")
        rng, choices = np.random.default_rng(1), np.random.default_rng(2)

        def random_rule(score, *rest):
            return choices.random(score.size)

        for policy in (sot, random_rule):
            for _ in range(30):
                durations = sample_durations(instance, rng)
                total, stops = solve(instance, policy, durations)
                check_route(instance, stops, durations, total)


# ---------------------------------------------------------------------- #
# 3. GP operators
# ---------------------------------------------------------------------- #

def test_grow_may_stop_early_and_respects_the_height():
    random.seed(3)
    trees = [gp.PrimitiveTree(s.toolbox.subtree(pset=s.pset)) for _ in range(3000)]
    assert {t.height for t in trees} == {0, 1, 2, 3, 4}  # a full tree would always have height 4
    assert np.mean([len(t) for t in trees]) < 10


def test_initial_trees_are_ramped_within_the_limit():
    random.seed(4)
    trees = [gp.PrimitiveTree(s.toolbox.expr()) for _ in range(3000)]
    assert max(t.height for t in trees) == 5 and min(t.height for t in trees) == 0
    full = [t for t in trees if t.height == 5 and len(t) == 2 ** 6 - 1]
    assert full  # the full half of ramped half-and-half


def test_offspring_never_exceed_the_depth_limit():
    random.seed(5)
    population = [s.toolbox.individual() for _ in range(200)]
    for individual in population:
        individual.fitness.values = (random.random(),)
    assert max(x.height for x in s.breed(population)) <= s.MAX_HEIGHT


def test_every_individual_is_scored_afresh_each_generation():
    """Elites included: no fitness from an earlier sample may carry over."""
    random.seed(6)
    instance = line_instance()
    population = [s.toolbox.individual() for _ in range(30)]
    s.evaluate(population, instance, instance.duration)
    offspring = s.breed(population)
    longer = instance.duration * 20
    s.evaluate(offspring, instance, longer)
    for individual in offspring:
        assert individual.fitness.values[0] == solve(instance, gp.compile(individual, s.pset), longer)[0]
