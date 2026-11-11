"""
Tests of the search (Step 4c). Section 9, stage E, of search_design.md.

Run:  python3 -m pytest tests -q   (from the repository root)
"""

import copy
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from deap import gp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import search  # noqa: E402
from day import PROCESSED  # noqa: E402
from gp_rule import H1_SET, H2_SET, constants, from_text, make_rule, to_text, tree_from_text, tree_sets  # noqa: E402
from search import (Run, breed, cross_h1, cross_h2, crossover, mutate_tree, mutation, rank, schedule,  # noqa: E402
                    search_seed, step_constant, training_dates)

needs_data = pytest.mark.skipif(not (PROCESSED / "manifest.json").exists(), reason="processed tables not built")
SMALL = {"population": 6, "generations": 3, "crossover": 0.8, "mutation": 0.15, "monitor every": 1,
         "validation replicates": 1}
H1_NAMES = set(H1_SET.arguments)
H2_NAMES = set(H2_SET.arguments)


def names(tree):
    """Terminal names of a tree; constants as 'constant'."""
    out = set()
    for node in tree:
        if isinstance(node, gp.Terminal):
            out.add(node.value if isinstance(node.value, str) else "constant")
    return out


def counts():
    return {f"{kind} {what}": 0 for kind in ("crossover", "mutation", "reproduction")
            for what in ("children", "new texts")} | {"reverts": 0, "constant cuts": 0}


# ---------------------------------------------------------------------- #
# 1. Operators
# ---------------------------------------------------------------------- #

def test_mutation_can_choose_the_root(monkeypatch):
    tree = tree_from_text("add(AV, TA)", H1_SET)
    random.seed(0)
    monkeypatch.setattr(random, "randrange", lambda n: 0)  # force the root as the mutation point
    child = mutate_tree(copy.deepcopy(tree), H1_SET)
    assert str(child) != "add(AV, TA)" and names(child) <= H1_NAMES


def test_constants_read_from_text_can_be_mutated():
    # The bug this guards: DEAP types a constant read from text as float, and mutating it then fails.
    random.seed(11)
    rule = from_text("GP-adaptive | h1: AV | h2: 0.5")
    assert rule[1][0].ret is H2_SET.ret
    for _ in range(20):
        mutate_tree(copy.deepcopy(rule[1]), H2_SET)


def test_single_node_trees_can_be_mutated():
    random.seed(1)
    h1 = [mutate_tree(tree_from_text("AV", H1_SET), H1_SET) for _ in range(20)]
    h2 = [mutate_tree(tree_from_text("0.5", H2_SET), H2_SET) for _ in range(20)]
    assert any(str(t) != "AV" for t in h1) and any(str(t) != "0.5" for t in h2)
    assert all(names(t) <= H1_NAMES for t in h1) and all(names(t) <= H2_NAMES | {"constant"} for t in h2)


def test_mutation_grows_subtrees_of_the_designed_heights(monkeypatch):
    # h2's new subtree has DEAP height 0 to 4; h1's is Step 1's, height 4 (section 4.4).
    seen = []

    def recording_grow(pset, min_, max_, type_=None):
        seen.append((pset.name, min_, max_))
        return [pset.terminals[pset.ret][0]]
    monkeypatch.setattr(search, "grow", recording_grow)
    random.seed(12)
    mutate_tree(tree_from_text("add(GDAY, ARR)", H2_SET), H2_SET)
    mutate_tree(tree_from_text("add(AV, TA)", H1_SET), H1_SET)
    assert seen == [("H2", 0, 4), ("H1", 4, 4)]


def test_h2_mutation_can_produce_a_one_node_tree():
    random.seed(2)
    trees = [mutate_tree(tree_from_text("add(GDAY, mul(ARR, 2.0))", H2_SET), H2_SET)
             for _ in range(300)]
    assert any(len(t) == 1 for t in trees) and max(t.height for t in trees) <= 6


def test_h2_crossover_for_every_kind_of_tree():
    random.seed(3)
    cases = [("0.5", "2.0"), ("0.5", "add(GDAY, ARR)"), ("add(GDAY, 3.0)", "mul(ARR, MCR)")]
    for left, right in cases:
        a, b = (tree_from_text(t, H2_SET) for t in (left, right))
        a, b = cross_h2(a, b)
        assert names(a) | names(b) <= H2_NAMES | {"constant"}
        assert len(a) + len(b) == len(tree_from_text(left, H2_SET)) + len(tree_from_text(right, H2_SET))
    # Two constants: the only points are the roots, so the trees are exchanged whole.
    a, b = cross_h2(tree_from_text("0.5", H2_SET), tree_from_text("2.0", H2_SET))
    assert (str(a), str(b)) == ("2.0", "0.5")


def test_h1_crossover_never_at_the_root_and_leaves_single_nodes():
    random.seed(4)
    a, b = cross_h1(tree_from_text("AV", H1_SET), tree_from_text("add(TA, DUR)", H1_SET))
    assert (str(a), str(b)) == ("AV", "add(TA, DUR)")
    for _ in range(50):
        a, b = cross_h1(tree_from_text("add(AV, TA)", H1_SET),
                        tree_from_text("mul(DUR, CR)", H1_SET))
        assert str(a).startswith("add(") and str(b).startswith("mul(")  # the roots stay


def test_fixed_crossover_acts_on_h1_and_swaps_the_constants():
    random.seed(5)
    p1 = from_text("GP-fixed | h1: add(AV, TA) | c: 0.5")
    p2 = from_text("GP-fixed | h1: mul(DUR, CR) | c: 2.0")
    a, b = crossover(p1, p2, counts())
    assert (a[1], b[1]) == (2.0, 0.5) and str(a[0]).startswith("add(") and str(b[0]).startswith("mul(")


def test_adaptive_crossover_swaps_the_other_tree_whole():
    random.seed(6)
    p1 = from_text("GP-adaptive | h1: add(AV, TA) | h2: mul(GDAY, 2.0)")
    p2 = from_text("GP-adaptive | h1: mul(DUR, CR) | h2: add(ARR, 0.5)")
    for _ in range(20):
        a, b = crossover(p1, p2, counts())
        swapped_h2 = (str(a[1]), str(b[1])) == ("add(ARR, 0.5)", "mul(GDAY, 2.0)")
        swapped_h1 = (str(a[0]), str(b[0])) == ("mul(DUR, CR)", "add(AV, TA)")
        assert swapped_h1 or swapped_h2  # one tree crossed, the other swapped whole
        assert names(a[0]) <= H1_NAMES and names(a[1]) <= H2_NAMES | {"constant"}


def full(method, height):
    return make_rule(method, [gp.PrimitiveTree(gp.genFull(p, height, height)) if p else 1.0
                              for p in tree_sets(method)])


def test_depth_limit_and_reverts_leave_the_parents_unchanged():
    random.seed(7)
    c = counts()
    for method in ("GP-adaptive", "GP-fixed", "GP-single"):
        p1, p2 = full(method, 7), full(method, 7)
        before = (to_text(p1), to_text(p2))
        for _ in range(30):
            children = crossover(p1, p2, c) + [mutation(p1, c)]
            for child in children:
                assert all(part.height <= 7 for part in child if isinstance(part, gp.PrimitiveTree))
                assert child is not p1 and child is not p2 and child.method == method
                assert [type(part) for part in child] == [type(part) for part in p1]
        assert (to_text(p1), to_text(p2)) == before
    assert c["reverts"] > 0


def test_constants_stay_in_range_and_cuts_are_counted(monkeypatch):
    c = counts()
    monkeypatch.setattr(random, "gauss", lambda mu, sigma: 10.0)
    assert step_constant(500.0, c) == 1e3 and c["constant cuts"] == 1
    monkeypatch.setattr(random, "gauss", lambda mu, sigma: -30.0)
    assert step_constant(0.01, c) == 1e-3 and c["constant cuts"] == 2
    monkeypatch.undo()
    random.seed(8)
    rule = from_text("GP-adaptive | h1: AV | h2: add(GDAY, 999.0)")
    for _ in range(200):
        rule = mutation(rule, c)
        assert all(1e-3 <= rule[1][i].value <= 1e3 for i in constants(rule[1]))


def test_tournament_prefers_the_lower_rank_then_the_larger_crowding():
    random.seed(10)
    a, b, c = (from_text(f"GP-fixed | h1: AV | c: {x!r}") for x in (0.1, 0.2, 0.3))
    a.rank, b.rank, c.rank = 0, 1, 0
    a.fitness.values = b.fitness.values = c.fitness.values = (1.0, 1.0)
    a.fitness.crowding_dist, b.fitness.crowding_dist, c.fitness.crowding_dist = 0.5, 9.0, 2.0
    assert all(search.tournament([a, b]) is a for _ in range(20))   # rank first
    assert all(search.tournament([a, c]) is c for _ in range(20))   # then crowding distance


def test_breeding_fills_an_odd_number_of_slots():
    random.seed(9)
    population = [from_text(f"GP-fixed | h1: add(AV, TA) | c: {c!r}") for c in (0.1, 0.2, 0.3, 0.4, 0.5)]
    for i, rule in enumerate(population):
        rule.fitness.values = (float(i), float(i))
    rank(population)
    before = [to_text(r) for r in population]
    children = breed(population, {"population": 5, "crossover": 0.8, "mutation": 0.15}, counts())
    assert len(children) == 5 and [to_text(r) for r in population] == before
    assert all(child is not parent for child in children for parent in population)


# ---------------------------------------------------------------------- #
# 2. Schedule and seeds
# ---------------------------------------------------------------------- #

@needs_data
def test_the_schedule_depends_on_the_run_number_only():
    dates = training_dates()
    one = schedule(3, 300)
    assert one == schedule(3, 300) and one != schedule(4, 300)
    first, second = [d for d, _ in one[:147]], [d for d, _ in one[147:294]]
    assert sorted(first) == sorted(second) == dates  # each pass is a permutation of the training days
    assert [d for d, _ in one[:147]] != dates  # shuffled, not in date order
    assert [d for d, _ in schedule(4, 147)] != [d for d, _ in schedule(3, 147)]
    # Never a validation or test replicate (those are 0..K-1): over 30 runs, 9,000 draws.
    assert all(r >= 1000 for run in range(30) for _, r in schedule(run, 300))
    runs = [Run(method, 3, SMALL, monitor=False) for method in search.METHODS]
    assert all(run.schedule == runs[0].schedule for run in runs)


def test_search_seeds_differ():
    seeds = {search_seed(m, r) for m in search.METHODS for r in range(30)}
    assert len(seeds) == 4 * 30


# ---------------------------------------------------------------------- #
# 3. A generation
# ---------------------------------------------------------------------- #

@needs_data
def test_parents_are_chosen_on_scores_from_the_new_environment(monkeypatch):
    """Two environments reverse the order of the rules; breeding must see the new order."""
    run = Run("GP-fixed", 0, SMALL, monitor=False)
    first_date = run.schedule[0][0]

    def fake_score(population, date, replicate, days, fallbacks, options=None, cache=None):
        for i, rule in enumerate(population):
            better = i if date == first_date else -i
            rule.fitness.values = (float(better), 0.0)

    seen = []
    real_breed = search.breed

    def watching_breed(population, settings, c):
        seen.append([(rule.fitness.values[0], rule.rank) for rule in population])
        return real_breed(population, settings, c)

    monkeypatch.setattr(search, "score", fake_score)
    monkeypatch.setattr(search, "breed", watching_breed)
    run.step()
    run.step()
    for scores in seen:  # in every generation the best current score has rank 0
        best = max(scores)[0]
        assert all(r == 0 for s, r in scores if s == best) and all(r > 0 for s, r in scores if s < best)


SCRIPT = """
import json, sys
sys.path.insert(0, {src!r})
from search import Run
settings = json.loads({settings!r})
run = Run({method!r}, {number}, settings, monitor={monitor})
out = []
for _ in range(settings["generations"]):
    run.step()
    out.append([__import__("gp_rule").to_text(r) for r in run.population])
print(json.dumps(out))
"""


def run_in_process(method, number, monitor=False, settings=SMALL):
    code = SCRIPT.format(src=str(ROOT / "src"), settings=json.dumps(settings), method=method, number=number,
                         monitor=monitor)
    return subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, stdout=subprocess.PIPE, text=True)


@needs_data
@pytest.mark.slow
def test_a_run_is_the_same_alone_alongside_another_and_with_monitoring():
    alone = json.loads(run_in_process("GP-adaptive", 2).communicate()[0])
    together = [run_in_process("GP-adaptive", 2), run_in_process("GP-fixed", 2, monitor=True)]
    outputs = [json.loads(p.communicate()[0]) for p in together]
    monitored = json.loads(run_in_process("GP-adaptive", 2, monitor=True).communicate()[0])
    assert alone == outputs[0] == monitored and len(alone) == SMALL["generations"]


@needs_data
@pytest.mark.slow
def test_a_run_resumes_from_a_checkpoint_with_the_same_result(tmp_path):
    whole = Run("GP-prop", 5, SMALL, folder=tmp_path / "whole", monitor=False)
    texts = []
    for _ in range(3):
        whole.step()
        texts.append([to_text(r) for r in whole.population])
    first = Run("GP-prop", 5, SMALL, folder=tmp_path / "part", monitor=False)
    first.step()
    random.seed(12345)  # anything else drawing in between must not matter
    resumed = Run.resume("GP-prop", 5, tmp_path / "part", monitor=False)
    later = []
    for _ in range(2):
        resumed.step()
        later.append([to_text(r) for r in resumed.population])
    assert later == texts[1:]
    assert [row["date"] for row in resumed.log] == [d for d, _ in whole.schedule[:3]]


# ---------------------------------------------------------------------- #
# 4. Checkpoints: identity and state (third review)
# ---------------------------------------------------------------------- #

import pickle  # noqa: E402


def saved(tmp_path, method="GP-fixed", number=5):
    run = Run(method, number, SMALL, folder=tmp_path, monitor=False)
    run.checkpoint()
    return run


def tamper(tmp_path, change):
    path = tmp_path / "checkpoint.pkl"
    state = pickle.loads(path.read_bytes())
    change(state)
    path.write_bytes(pickle.dumps(state))


@needs_data
def test_a_checkpoint_is_never_relabelled(tmp_path):
    saved(tmp_path)
    Run.resume("GP-fixed", 5, tmp_path, monitor=False)  # the right identity resumes
    for method, number in (("GP-adaptive", 5), ("GP-fixed", 6)):
        with pytest.raises(ValueError):
            Run.resume(method, number, tmp_path, monitor=False)
    changes = {"data version": lambda s: s.update({"data version": "other"}),
               "code version": lambda s: s.update({"code version": "other"}),
               "settings": lambda s: s["settings"].update({"crossover": 0.7}),
               "schedule": lambda s: s.update({"schedule": schedule(6, SMALL["generations"])}),
               "another method's rule": lambda s: s["population"].__setitem__(0, "GP-adaptive | h1: AV | h2: 0.5"),
               "population size": lambda s: s.update({"population": s["population"][:-1],
                                                      "fitness": s["fitness"][:-1]})}
    for name, change in changes.items():
        saved(tmp_path)
        tamper(tmp_path, change)
        with pytest.raises(ValueError):
            Run.resume("GP-fixed", 5, tmp_path, monitor=False)


@needs_data
@pytest.mark.slow
def test_a_resumed_run_has_the_same_state_also_when_finished(tmp_path):
    whole = Run("GP-adaptive", 7, SMALL, folder=tmp_path, monitor=False)
    whole.step()
    resumed = Run.resume("GP-adaptive", 7, tmp_path, monitor=False)
    state = lambda run: ([to_text(r) for r in run.population], [r.fitness.values for r in run.population])  # noqa
    assert state(resumed) == state(whole) and resumed.generation == whole.generation == 1
    whole.evolve()
    finished = Run.resume("GP-adaptive", 7, tmp_path, monitor=False)
    returned = finished.evolve()  # nothing left to run: the saved population, with its scores, comes back
    assert returned is finished.population and state(finished) == state(whole)
    assert finished.generation == SMALL["generations"]
    assert all(r.fitness.valid for r in finished.population)


def test_invalid_settings_are_refused_before_the_run():
    bad = [{"population": 1}, {"population": 2.5}, {"generations": 0}, {"monitor every": 0},
           {"crossover": 0.9, "mutation": 0.2}, {"crossover": float("nan")}, {"mutation": -0.1},
           {"validation replicates": 1001}, {"population": True}]
    for change in bad:
        with pytest.raises(ValueError):
            search.check_settings({**SMALL, **change})
    with pytest.raises(ValueError):
        Run("GP-other", 0, SMALL, monitor=False)


@needs_data
def test_settings_are_fixed_and_until_stays_in_the_schedule():
    run = Run("GP-fixed", 0, SMALL, monitor=False)
    with pytest.raises(TypeError):
        run.settings["generations"] = 50
    with pytest.raises(ValueError):
        run.evolve(until=SMALL["generations"] + 1)


def test_non_finite_objectives_are_refused(monkeypatch):
    class Result:
        mean_r, g_hat = float("nan"), 0.4
        run = type("R", (), {"fallbacks": {}})()
    monkeypatch.setattr(search, "evaluate", lambda *args, **kwargs: Result())
    rule = from_text("GP-fixed | h1: AV | c: 1.0")
    with pytest.raises(ValueError):
        search.score([rule], "2019-01-01", 1000, None, {})


SAVE = """
import sys
sys.path.insert(0, "src")
from search import Run
Run("GP-fixed", 5, {settings}, folder="saved", monitor=False).checkpoint()
"""
RESUME = """
import sys
sys.path.insert(0, "src")
from search import Run
try:
    Run.resume("GP-fixed", 5, "saved", monitor=False)
    print("resumed")
except ValueError as error:
    print("refused:", error)
"""


@needs_data
def test_a_change_to_a_dependency_of_the_search_refuses_the_resume(tmp_path):
    """Only single_gphh.py changes (it holds grow(), used by mutation); the resume must be refused.
    An untouched copy resumes."""
    results = {}
    for name, change in (("changed", True), ("untouched", False)):
        copy_ = tmp_path / name
        shutil.copytree(ROOT / "src", copy_ / "src", ignore=shutil.ignore_patterns("__pycache__"))
        (copy_ / "data").symlink_to(ROOT / "data")
        subprocess.run([sys.executable, "-c", SAVE.format(settings=repr(SMALL))], cwd=copy_, check=True)
        if change:
            path = copy_ / "src" / "single_gphh.py"
            path.write_text(path.read_text() + "\n# a change after the checkpoint\n")
        out = subprocess.run([sys.executable, "-c", RESUME], cwd=copy_, capture_output=True, text=True, check=True)
        results[name] = out.stdout.strip()
    assert results["changed"].startswith("refused: the checkpoint's code version")
    assert results["untouched"] == "resumed"


@needs_data
def test_scores_from_another_environment_are_refused(tmp_path):
    saved(tmp_path)
    tamper(tmp_path, lambda s: s.update({"scored on": ("2019-01-01", 1000)}))
    with pytest.raises(ValueError):
        Run.resume("GP-fixed", 5, tmp_path, monitor=False)


# ---------------------------------------------------------------------- #
# 5. The end of a run (4d, section 5)
# ---------------------------------------------------------------------- #

from search import finish, hypervolume, knee, non_dominated, normalise, validation_dates  # noqa: E402


def test_non_dominated_points_on_the_raw_objectives():
    means = [(1.0, 0.5), (1.0, 0.5), (0.8, 0.4), (0.9, 0.6), (1.2, 0.7), (0.7, 0.45)]
    assert non_dominated(means) == [0, 1, 2, 4]  # equal points are both kept; (0.9, 0.6) and (0.7, 0.45) lose


def test_normalisation_is_not_clipped_and_refuses_bad_input():
    out = normalise([(1.5, 0.0), (1.8, -0.1), (0.0, 1.0)], {"R": (0.0, 1.5), "G": (0.0, 1.0)})
    assert out == [(1.0, 1.0), (1.2, 1.1), (0.0, 0.0)]
    for bad_range in ({"R": (1.0, 1.0), "G": (0.0, 1.0)}, {"R": (0.0, 1.0), "G": (1.0, 0.0)},
                      {"R": (0.0, float("inf")), "G": (0.0, 1.0)}):
        with pytest.raises(ValueError):
            normalise([(1.0, 0.5)], bad_range)
    with pytest.raises(ValueError):
        normalise([(float("nan"), 0.5)])


def test_hypervolume_by_hand():
    assert hypervolume([(0.5, 0.5)]) == (0.25, 0)
    assert hypervolume([(1.0, 0.2), (0.4, 0.8)])[0] == pytest.approx(0.2 + 0.4 * 0.6)
    assert hypervolume([(1.0, 0.2), (0.4, 0.8), (1.0, 0.2)])[0] == pytest.approx(0.44)  # a duplicate adds nothing
    assert hypervolume([(1.0, 0.2), (0.4, 0.8), (0.3, 0.1)])[0] == pytest.approx(0.44)  # nor a dominated point
    assert hypervolume([(-0.1, 0.9), (0.5, 0.0), (0.6, 0.5)]) == (0.3, 2)  # two not better than (0, 0)
    assert hypervolume([]) == (0.0, 0)
    assert hypervolume([(1.2, 1.1)])[0] == pytest.approx(1.32)  # beyond the range counts in full


def test_knee_in_every_case():
    same = lambda n: ([1] * n, [chr(97 + i) for i in range(n)])  # noqa: E731
    assert knee([(0.4, 0.6)], *same(1)) == 0                                       # one point
    assert knee([(1, 0), (0.8, 0.8), (0.5, 0.9), (0, 1)], *same(4)) == 1            # farthest on the ideal side
    assert knee([(1, 0), (0.5, 0.5), (0, 1)], *same(3)) == 1                        # collinear: nearest the ideal
    assert knee([(1, 0), (0.3, 0.3), (0, 1)], *same(3)) == 1                        # only the wrong side: nearest
    assert knee([(1, 0), (0, 1)], [3, 1], ["a", "b"]) == 1                          # two points, a tie: smaller rule
    assert knee([(1, 0), (0.8, 0.8), (0.8, 0.8), (0, 1)], [1, 5, 2, 1], list("abcd")) == 2  # duplicates: smaller
    assert knee([(1, 0), (0.8, 0.6), (0.6, 0.8), (0, 1)], [1, 4, 4, 1], ["a", "z", "y", "b"]) == 2  # tie: text


@needs_data
def test_finishing_is_refused_before_the_last_generation():
    run = Run("GP-fixed", 0, SMALL, monitor=False)
    with pytest.raises(ValueError):
        finish(run)


FEW_DATES = ["2019-01-28", "2019-06-19", "2019-11-08"]  # three validation days; search_check.py runs all 35


@needs_data
@pytest.mark.slow
def test_the_end_of_a_run(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "validation_dates", lambda: FEW_DATES)
    tiny = {**SMALL, "population": 4, "generations": 1, "validation replicates": 1}
    run = Run("GP-adaptive", 3, tiny, monitor=False)
    run.evolve()
    out = finish(run, tmp_path)
    means, front = out["means"], out["front"]
    members = non_dominated(means)
    distinct = {to_text(run.population[i]) for i in members}
    assert len(means) == 4 and len(front) == len(distinct) and front["knee"].sum() == 1
    assert front["copies"].sum() == len(members) == out["summary"]["front members"]
    dates = FEW_DATES
    assert set(FEW_DATES) <= set(validation_dates())
    for name in ("groups", "peaks", "knee visitors", "knee decisions", "knee hourly"):
        table = out[name]
        assert {"method", "run", "rule", "date", "replicate", "settings hash", "code version",
                "data version"} <= set(table.columns)
        assert set(table["date"]) <= set(dates)
    groups = out["groups"]
    assert set(groups["date"]) == set(dates) and set(groups["rule"]) == set(front["rule"])
    # Mean R_v rebuilt from the grouped outputs equals the front's validation mean.
    for rule_id, g in groups.groupby("rule"):
        per_day = g.groupby(["date", "replicate"])[["sum R_v", "visitors"]].sum()
        rebuilt = (per_day["sum R_v"] / per_day["visitors"]).mean()
        assert rebuilt == pytest.approx(front.loc[front["rule"] == rule_id, "mean R_v"].iloc[0], rel=1e-12)
    knee_id = front.loc[front["knee"], "rule"].iloc[0]
    assert set(out["knee decisions"]["rule"]) == {knee_id} and len(out["knee decisions"]) > 0
    assert len(out["knee hourly"]) == len(FEW_DATES) * 15
    for name in ("summary.json", "front.csv.gz", "groups.csv.gz", "knee_decisions.csv.gz"):
        assert (tmp_path / name).exists()



class Stub:
    """A stand-in evaluation result."""
    def __init__(self, mean_r, g_hat):
        self.mean_r, self.g_hat = mean_r, g_hat


def stub_evaluate(values):
    """evaluate() replaced: (rule text, date, replicate) -> (mean R_v, G_hat)."""
    def fake(rule, date, r, days, **kwargs):
        return Stub(*values[(to_text(rule), date, r)])
    return fake


def test_the_front_uses_validation_means_of_daily_objectives(monkeypatch):
    a = from_text("GP-fixed | h1: AV | c: 0.5")
    b = from_text("GP-fixed | h1: TA | c: 0.5")
    c = from_text("GP-fixed | h1: DUR | c: 0.5")
    a.fitness.values, b.fitness.values = (2.0, 0.1), (0.1, 0.9)  # on the last training day A was best
    dates, values = ["d1", "d2"], {}
    for r in range(2):
        values[(to_text(a), "d1", r)] = values[(to_text(a), "d2", r)] = (0.5, 0.6)
        values[(to_text(b), "d1", r)] = values[(to_text(b), "d2", r)] = (0.9, 0.4)
        values[(to_text(c), "d1", r)], values[(to_text(c), "d2", r)] = (1.0, 0.3), (0.0, 0.5)
    monkeypatch.setattr(search, "evaluate", stub_evaluate(values))
    means = search.validation_means([a, b, c], dates, 2, None)
    # C's days count equally (a quiet and a busy day weigh the same), not by visitors pooled over days.
    assert means == [(0.5, 0.6), (0.9, 0.4), (0.5, 0.4)]
    assert non_dominated(means) == [1]  # A, best on the last training day, is dominated on validation


def test_non_finite_validation_results_stop_the_run_before_the_front(monkeypatch):
    good, bad = from_text("GP-fixed | h1: AV | c: 0.5"), from_text("GP-fixed | h1: TA | c: 0.5")
    for bad_value in ((float("-inf"), 0.9), (float("inf"), 0.1), (0.5, float("nan"))):
        values = {(to_text(good), "d1", 0): (1.0, 0.2), (to_text(bad), "d1", 0): bad_value}
        monkeypatch.setattr(search, "evaluate", stub_evaluate(values))
        with pytest.raises(ValueError, match="on d1, replicate 0: non-finite"):  # stopped at the evaluation
            search.validation_means([good, bad], ["d1"], 1, None)


@needs_data
@pytest.mark.slow
def test_copies_of_a_rule_are_exported_once(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "validation_dates", lambda: FEW_DATES)
    tiny = {**SMALL, "population": 4, "generations": 1, "validation replicates": 1}
    run = Run("GP-fixed", 0, tiny, monitor=False)
    a, b, c = ("GP-fixed | h1: AV | c: 0.5", "GP-fixed | h1: max(AV, AV) | c: 0.5", "GP-fixed | h1: TA | c: 0.5")
    run.population = [from_text(t) for t in (a, a, b, c)]  # B differs from A in text only
    run.generation = 1
    out = finish(run, tmp_path)
    assert len(pd.read_csv(tmp_path / "final_population.csv")) == 4
    front = out["front"].set_index("text")
    assert out["means"][0] == out["means"][1] == out["means"][2]  # A, its copy and B score the same
    assert front.loc[a, "copies"] == 2 and front.loc[b, "copies"] == 1  # A once; B kept as its own rule
    keys = ["rule", "date", "replicate", "follows", "arrival group", "budget group", "stop group"]
    assert not out["groups"].duplicated(keys).any()
    assert not out["peaks"].duplicated(["rule", "date", "replicate", "site"]).any()
    knee_id = out["front"].loc[out["front"]["knee"], "rule"].iloc[0]
    knee_groups = out["groups"][out["groups"]["rule"] == knee_id]
    assert knee_groups["visitors"].sum() == len(out["knee visitors"])
    assert out["summary"]["front members"] == out["front"]["copies"].sum()


# ---------------------------------------------------------------------- #
# 6. Settings, definitions, the cache, pooling and the tuned rule (F2, 4e)
# ---------------------------------------------------------------------- #

import json as json_module  # noqa: E402

from search import DEFINITIONS, collect, complete, load_settings, settings_hash, tuned_front  # noqa: E402


class FakeResult:
    def __init__(self, mean_r, g_hat):
        self.mean_r, self.g_hat = mean_r, g_hat
        self.run = type("R", (), {"fallbacks": {"ARR: usual value not positive": 1}})()


def test_copies_are_simulated_once_with_the_same_scores(monkeypatch):
    calls = []

    def fake(rule, date, r, days, **kwargs):
        calls.append(to_text(rule))
        return FakeResult(len(to_text(rule)) / 100, 0.3)
    monkeypatch.setattr(search, "evaluate", fake)
    a, b = from_text("GP-fixed | h1: AV | c: 0.5"), from_text("GP-fixed | h1: TA | c: 0.5")
    population, fallbacks, cache = [a, copy.deepcopy(a), b], {}, {}
    search.score(population, "d", 1000, None, fallbacks, {}, cache)
    search.score([copy.deepcopy(b)], "d", 1000, None, fallbacks, {}, cache)  # a child equal to a parent
    assert sorted(calls) == sorted({to_text(a), to_text(b)})
    assert population[0].fitness.values == population[1].fitness.values
    assert fallbacks == {"ARR: usual value not positive": 4}  # counted for every member, as before
    calls.clear()
    means = search.validation_means([a, copy.deepcopy(a), b], ["d1", "d2"], 2, None)
    assert len(calls) == 2 * 2 * 2 and means[0] == means[1]


def test_settings_hold_every_definition():
    full = complete(SMALL)
    assert set(DEFINITIONS) <= set(full) and full["R range"] == (0.0, 1.5)
    assert settings_hash(complete({**SMALL, **DEFINITIONS})) == settings_hash(full)
    assert settings_hash(complete({**SMALL, "budget": "span"})) != settings_hash(full)  # never pooled together
    bad = [{"budget": "observed"}, {"same day": 1}, {"stop limit": "yes"}, {"peak floor": 0},
           {"peak floor": float("nan")}, {"alpha": -0.1}, {"beta": float("inf")}, {"R range": (1.0, 1.0)},
           {"G range": (0.0, float("nan"))}]
    for change in bad:
        with pytest.raises(ValueError):
            search.check_settings(complete({**SMALL, **change}))


@needs_data
def test_the_definitions_reach_every_evaluation(monkeypatch):
    days_made, scored = [], []

    class FakeDays:
        def __init__(self, **kwargs):
            days_made.append(kwargs)

    def fake(rule, date, r, days, **kwargs):
        scored.append(kwargs)
        return FakeResult(0.5, 0.3)
    monkeypatch.setattr(search, "Days", FakeDays)
    monkeypatch.setattr(search, "evaluate", fake)
    settings = {**SMALL, "budget": "span", "same day": True, "peak floor": 10.0, "stop limit": False,
                "alpha": 0.5, "beta": 0.0}
    run = Run("GP-fixed", 0, settings, monitor=True)
    for _ in range(SMALL["generations"]):
        run.step()
    assert days_made and all(d == {"budget": "span", "same_day": True, "peak_floor": 10.0} for d in days_made)
    assert scored and all(k == {"alpha": 0.5, "beta": 0.0, "stop_limit": False} for k in scored)


@needs_data
def test_the_settings_file_must_match_the_data(tmp_path):
    from evaluate import budget_cuts
    good = {"settings": SMALL, "budget cuts": list(budget_cuts())}
    (tmp_path / "good.json").write_text(json_module.dumps(good))
    assert load_settings(tmp_path / "good.json") == complete(SMALL)
    (tmp_path / "bad.json").write_text(json_module.dumps({**good, "budget cuts": [1.0, 2.0]}))
    with pytest.raises(ValueError):
        load_settings(tmp_path / "bad.json")


def test_runs_under_other_settings_are_never_pooled(tmp_path):
    for name, settings in (("a", "x1"), ("b", "x1"), ("c", "x2")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text(json_module.dumps(
            {"settings hash": settings, "code version": "c", "data version": "d"}))
    assert len(collect([tmp_path / "a", tmp_path / "b"])) == 2
    with pytest.raises(ValueError):
        collect([tmp_path / "a", tmp_path / "c"])


def test_tuned_front_by_hand(monkeypatch):
    values = {0.0: (1.1, 0.6), 0.1: (1.0, 0.5), 1.0: (0.9, 0.55), 10.0: (0.6, 0.2)}

    def fake(rule, date, r, days, **kwargs):
        return FakeResult(*values[rule.weight])
    monkeypatch.setattr(search, "evaluate", fake)
    monkeypatch.setattr(search, "validation_dates", lambda: ["d1"])
    monkeypatch.setattr(search, "days_for", lambda settings: None)
    table, summary = tuned_front(SMALL, weights=list(values))
    assert table["front"].tolist() == [True, True, False, True]  # 1.0 is dominated by 0.1
    points = [((1.1 - 0) / 1.5, 1 - 0.6), ((1.0) / 1.5, 1 - 0.5), (0.6 / 1.5, 1 - 0.2)]
    assert summary["hypervolume"] == pytest.approx(search.hypervolume(points)[0])
