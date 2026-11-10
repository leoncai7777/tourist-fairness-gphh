"""
Tests of the GP rules (Step 4a). Section 9, stage D, of search_design.md.

Run:  python3 -m pytest tests -q   (from the repository root)
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from day import PROCESSED, load_day  # noqa: E402
from engine import simulate  # noqa: E402
from features import H2_TERMINALS, choose_site, visit_share  # noqa: E402
from gp_rule import (GPRule, METHODS, constants, draw_constant, from_text, identity, new_individual,  # noqa: E402
                     size, to_text)
from rules import Pop  # noqa: E402
from test_engine import hand_day, stays  # noqa: E402

needs_data = pytest.mark.skipif(not (PROCESSED / "manifest.json").exists(), reason="processed tables not built")


def terminal_day():
    """The state of test_features.test_each_terminal_by_hand: v0 decides at 640 between sites 1
    and 2 with AV (1, 1), TA (10, 20), DUR (20, 10), ARR 3/2, GDAY 0; VS (1, 1)."""
    day = hand_day([{"route": [0, 1], "entries": [600, 650], "budget": 200.0},
                    {"route": [1, 2], "entries": [610, 700], "follows": False, "exits": [700, 720]},
                    {"route": [2], "entries": [630], "follows": False, "exits": [650]}],
                   opening=(0, 660, 0), closing=(1000, 900, 950))
    day.usual_peak = np.array([2.0, 4.0, 5.0])
    day.usual_arrivals = np.full(781, 2.0)
    day.usual_gday = np.full(781, 0.5)
    return day, stays([[40, 25, 15], [9, 9, 9], [9, 9, 9]])


class Seen(GPRule):
    """GPRule, keeping what parts() gave at each decision."""
    def __init__(self, rule, log=False):
        super().__init__(rule, log)
        self.seen = []

    def parts(self, engine, k, candidates):
        out = super().parts(engine, k, candidates)
        self.seen.append(out)
        return out


def first_parts(text):
    day, table = terminal_day()
    policy = Seen(from_text(text))
    run = simulate(day, policy, stays=table, check=True)
    return policy.seen[0], run


# ---------------------------------------------------------------------- #
# 1. Compiled trees by hand
# ---------------------------------------------------------------------- #

def test_compiled_trees_by_hand():
    (h1, w, vs, _), _ = first_parts("GP-adaptive | h1: sub(AV, div(TA, DUR)) | h2: add(ARR, 0.5)")
    assert h1.tolist() == [1 - 10 / 20, 1 - 20 / 10] and w == 3 / 2 + 0.5 and vs.tolist() == [1, 1]
    (h1, w, _, _), _ = first_parts("GP-fixed | h1: max(TSV, HEAD) | c: 0.25")
    assert h1.tolist() == [20, 20] and w == 0.25
    (_, w, _, _), _ = first_parts("GP-prop | h1: AV | c: 4.0")
    assert w == 4.0 * 0  # GDAY is 0 here: equal visits at every site
    (h1, w, _, _), _ = first_parts("GP-single | tree: mul(VS, CR)")
    assert h1.tolist() == [1 / 4, 1 / 5] and w == 0


def test_protected_division_and_a_scalar_tree():
    (h1, _, _, _), _ = first_parts("GP-fixed | h1: div(TA, sub(DUR, DUR)) | c: 1.0")
    assert h1.tolist() == [1, 1]  # dividing by zero gives 1
    # A GP-single tree of scalar terminals gives one value, broadcast by choose_site:
    # equal priorities, so the tie rule takes site 1.
    (h1, _, _, _), run = first_parts("GP-single | tree: add(ARR, STOPS)")
    assert np.ndim(h1) == 0 and h1 == 3 / 2 + 1 and run.routes[0][1][0] == 1


def test_non_finite_outputs_still_give_a_determined_choice():
    big = "RemT"
    for _ in range(9):
        big = f"mul({big}, {big})"  # RemT ** 512 overflows to inf
    (h1, _, _, _), run = first_parts(f"GP-fixed | h1: sub({big}, {big}) | c: 1.0")
    assert np.isnan(h1).all() and run.routes[0][1][0] == 1  # set to 0, then the lowest site id
    assert run.fallbacks == {"h1: not finite": 2}


# ---------------------------------------------------------------------- #
# 2. Text, identity and constants
# ---------------------------------------------------------------------- #

def test_a_one_node_constant_h2_is_saved_rebuilt_and_compiled():
    text = "GP-adaptive | h1: AV | h2: 0.123456789012345"
    rule = from_text(text)
    assert to_text(rule) == text and size(rule) == [(1, 0), (1, 0)] and constants(rule[1]) == [0]
    (_, w, _, _), _ = first_parts(text)
    assert w == 0.123456789012345


def test_random_rules_rebuild_from_text_with_the_same_decisions():
    day, table = terminal_day()
    random.seed(4)
    for method in METHODS:
        for _ in range(5):
            rule = new_individual(method)
            again = from_text(to_text(rule))
            assert to_text(again) == to_text(rule) and identity(again) == identity(rule)
            one, two = simulate(day, GPRule(rule), stays=table), simulate(day, GPRule(again), stays=table)
            assert one.routes == two.routes and one.fallbacks == two.fallbacks
            assert all(1e-3 <= part <= 1e3 for part in rule if isinstance(part, float))
            assert all(h <= 5 for _, h in size(rule))


def test_constants_are_log_uniform_in_range_and_plain_floats():
    random.seed(0)
    draws = [draw_constant() for _ in range(2000)]
    assert all(type(c) is float and 1e-3 <= c <= 1e3 for c in draws)
    logs = np.log10(draws)
    assert logs.min() < -2.9 and logs.max() > 2.9 and abs(np.mean(logs)) < 0.15  # spread over the whole range
    random.seed(1)
    for _ in range(50):
        h2 = new_individual("GP-adaptive")[1]
        assert all(1e-3 <= h2[i].value <= 1e3 for i in constants(h2))


def test_the_run_records_the_rule():
    rule = from_text("GP-fixed | h1: AV | c: 2.0")
    day, table = terminal_day()
    run = simulate(day, GPRule(rule), stays=table)
    assert run.settings["policy"] == f"GP-fixed:{identity(rule)}"


# ---------------------------------------------------------------------- #
# 3. The methods against each other
# ---------------------------------------------------------------------- #

class Tuned:
    """The tuned weighted rule of section 8: A - lambda * VS through choose_site."""
    def __init__(self, weight):
        self.weight = weight

    def choose(self, engine, k, candidates):
        return choose_site(engine, k, candidates, engine.day.attractiveness[candidates], self.weight,
                           visit_share(engine, candidates))


@needs_data
def test_adaptive_with_a_constant_h2_is_fixed_with_that_c():
    day = load_day("2019-12-18")
    h1 = "sub(AV, div(TA, RemT))"
    for c in (0.01, 0.37, 25.0):
        fixed = simulate(day, GPRule(from_text(f"GP-fixed | h1: {h1} | c: {c!r}")))
        adaptive = simulate(day, GPRule(from_text(f"GP-adaptive | h1: {h1} | h2: {c!r}")))
        assert fixed.routes == adaptive.routes and fixed.intervals == adaptive.intervals


@needs_data
def test_fixed_with_av_is_the_tuned_weighted_rule():
    day = load_day("2019-12-18")
    for weight in (0.01, 0.1, 1.0, 10.0, 100.0):
        rule = simulate(day, GPRule(from_text(f"GP-fixed | h1: AV | c: {weight!r}")))
        assert rule.routes == simulate(day, Tuned(weight)).routes
    assert simulate(day, Tuned(0.0)).routes == simulate(day, Pop()).routes  # lambda = 0 is POP


def test_prop_weight_moves_with_gday():
    # Hand day with unequal visits, so GDAY differs between decisions.
    day = hand_day([{"route": [0, 1], "entries": [600, 650]}, {"route": [0, 2], "entries": [600, 650]},
                    {"route": [1, 2], "entries": [700, 750]}])
    day.usual_gday = np.full(781, 0.5)
    run = simulate(day, GPRule(from_text("GP-prop | h1: AV | c: 3.0"), log=True),
                   stays=stays([[40, 25, 15], [50, 25, 15], [40, 25, 15]]))
    gday = [row["GDAY"] for row in run.decisions]
    assert len(set(gday)) > 1 and all(row["w"] == pytest.approx(3.0 * row["GDAY"]) for row in run.decisions)


def test_gp_single_reads_vs():
    # Unequal visits, so VS is not 1 everywhere; the tree AV - VS must follow it.
    day = hand_day([{"route": [0, 1], "entries": [600, 650]}, {"route": [0, 2], "entries": [600, 650]},
                    {"route": [1, 2], "entries": [700, 750]}])
    policy = Seen(from_text("GP-single | tree: sub(AV, VS)"))
    simulate(day, policy, stays=stays([[40, 25, 15], [50, 25, 15], [40, 25, 15]]))
    assert any((vs != 1).any() for _, _, vs, _ in policy.seen)
    assert all(np.allclose(h1, 1 - vs) for h1, _, vs, _ in policy.seen)


def test_every_method_logs_the_same_fields():
    day, table = terminal_day()
    fields = set()
    for text in ("GP-fixed | h1: AV | c: 1.0", "GP-prop | h1: AV | c: 1.0", "GP-adaptive | h1: AV | h2: GDAY",
                 "GP-single | tree: AV"):
        run = simulate(day, GPRule(from_text(text), log=True), stays=table)
        fields.add(frozenset(run.decisions[0]))
    assert len(fields) == 1 and set(H2_TERMINALS) <= set(next(iter(fields)))


def test_the_log_does_not_change_the_choices():
    day, table = terminal_day()
    random.seed(7)
    for method in METHODS:
        rule = new_individual(method)
        with_log = simulate(day, GPRule(rule, log=True), stays=table)
        assert with_log.routes == simulate(day, GPRule(rule), stays=table).routes
