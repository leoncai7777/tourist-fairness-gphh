"""
Checks of the GP rules and their evaluation on real days (Step 4, stage D)
==========================================================================

  1. The time of a whole evaluation (state inputs, tree, simulation, scoring)
     for random rules of every method, on every 10th training day, and what
     it means for a run and for the 90 formal runs (search_design.md,
     section 3). Random trees of DEAP height 1 to 5 are what a run starts
     with; full trees of height 7, the limit, bound the cost from above.
  2. GP-adaptive with a constant h2 against GP-fixed with that c, on the
     pilot days: the same objectives and visitor scores.
  3. Grouped RQ3 outputs against the per-visitor rows, on the pilot days.
  4. A short search of every method with the same run number (Step 4c):
     the same schedule, training days only, what the operators changed,
     and the time per generation. Too short to say anything about results.

The checks are assertions; times are only reported.

Run:  python3 src/search_check.py   (from the repository root)
Writes outputs/search/check/record.md.
"""

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from deap import gp

from day import PROCESSED
from engine import simulate
from evaluate import Days, evaluate, grouped, hourly_peaks, share_at_least, visitor_rows
from gp_rule import METHODS, GPRule, from_text, make_rule, new_individual, size, tree_sets
from scoring import score_day
from single_gphh import fingerprint

OUT = Path("outputs/search/check")
PILOT_DAYS = ["2019-12-18", "2019-10-09", "2019-03-17", "2019-04-26", "2019-06-03"]
RULES_PER_METHOD, FULL_RULES_PER_METHOD = 5, 3
N, G, K, F = 200, 100, 5, 20  # planning values of search_design.md, section 3

lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


class Timed(GPRule):
    """GPRule, adding up the time spent in state inputs and the tree."""
    seconds = 0.0

    def choose(self, engine, k, candidates):
        started = time.perf_counter()
        j = super().choose(engine, k, candidates)
        Timed.seconds += time.perf_counter() - started
        return j


def full_individual(method):
    """A rule whose trees are full trees of DEAP height 7, the depth limit."""
    parts = [gp.PrimitiveTree(gp.genFull(pset, 7, 7)) if pset else 1.0 for pset in tree_sets(method)]
    return make_rule(method, parts)


# ---------------------------------------------------------------------- #
# 1. Time of a whole evaluation
# ---------------------------------------------------------------------- #

def timing():
    say("## 1. Time of a whole evaluation")
    say()
    table = pd.read_csv(PROCESSED / "days_2019.csv")
    dates = table["date"][table["split"] == "train"].tolist()[::10]
    days = Days()
    for date in dates:
        days.get(date)  # loaded before timing: a run loads each day once
    random.seed(0)
    rows = []
    for kind, make, count in (("random, height 1-5", new_individual, RULES_PER_METHOD),
                              ("full, height 7", full_individual, FULL_RULES_PER_METHOD)):
        for method in METHODS:
            for _ in range(count):
                rule = make(method)
                for date in dates:
                    Timed.seconds = 0.0
                    day, observed = days.get(date)
                    started = time.perf_counter()
                    run = simulate(day, Timed(rule), replicate=0)
                    simulated = time.perf_counter()
                    score_day(day, run, observed, 0.25, 0.5)
                    done = time.perf_counter()
                    rows.append((kind, method, sum(n for n, _ in size(rule)), simulated - started, Timed.seconds,
                                 done - simulated, done - started))
    t = pd.DataFrame(rows, columns=["kind", "method", "nodes", "simulation", "inputs and tree", "scoring", "total"])
    say(f"{len(dates)} training days (every 10th by date, {', '.join(dates[:3])}, ...), replicate 0, days loaded "
        f"beforehand. Seconds per evaluation; the simulation includes the state inputs and the tree.")
    say()
    say("| Rules | Method | Nodes (mean) | Simulation | of which inputs and tree | Scoring | Total (mean) | Total (max) |")
    say("|---|---|---|---|---|---|---|---|")
    for (kind, method), g in t.groupby(["kind", "method"], sort=False):
        say(f"| {kind} | {method} | {g['nodes'].mean():.0f} | {g['simulation'].mean():.3f} | "
            f"{g['inputs and tree'].mean():.3f} | {g['scoring'].mean():.3f} | {g['total'].mean():.3f} | "
            f"{g['total'].max():.3f} |")
    say()
    per_run = 2 * N * G + 35 * N * K + (G // 10) * F * 7 + 35 * K
    test = 181 * F * K
    say(f"With N = {N}, G = {G}, K = {K} and a front of about F = {F} rules, a run is about {per_run:,} evaluations "
        f"and the test evaluation of one run about {test:,} (section 3).")
    say()
    say("| Seconds per evaluation | One run (hours) | 90 runs (core-hours) | 90 runs on 9 cores (hours) | "
        "Test evaluation, 90 runs (core-hours) |")
    say("|---|---|---|---|---|")
    for label, seconds in (("random rules, mean", t["total"][t["kind"].str.startswith("random")].mean()),
                           ("full rules, mean", t["total"][t["kind"].str.startswith("full")].mean())):
        say(f"| {label}: {seconds:.3f} | {per_run * seconds / 3600:.1f} | {90 * per_run * seconds / 3600:.0f} | "
            f"{10 * per_run * seconds / 3600:.0f} | {90 * test * seconds / 3600:.0f} |")
    say()
    say("The two rows are budget references, not bounds: a full tree fixes the size of a rule, not how many "
        "decisions its routes lead to, and these times leave out grouped outputs, files and the search's own work. "
        "Parallel runs also share memory bandwidth. The pilot (F1) measures whole runs.")
    say()


# ---------------------------------------------------------------------- #
# 2. Adaptive with a constant h2 is fixed
# ---------------------------------------------------------------------- #

def adaptive_is_fixed():
    say("## 2. GP-adaptive with a constant h2 against GP-fixed")
    say()
    days = Days()
    h1 = "sub(add(AV, min(AV, HEAD)), div(mul(TA, CR), RemT))"
    checked = 0
    for date in PILOT_DAYS:
        for c in (0.01, 0.5, 40.0):
            fixed = evaluate(from_text(f"GP-fixed | h1: {h1} | c: {c!r}"), date, 1, days)
            adaptive = evaluate(from_text(f"GP-adaptive | h1: {h1} | h2: {c!r}"), date, 1, days)
            assert (fixed.mean_r, fixed.g_hat) == (adaptive.mean_r, adaptive.g_hat)
            assert fixed.visitors.equals(adaptive.visitors) and fixed.run.routes == adaptive.run.routes
            checked += 1
    say(f"- {checked} cases (5 pilot days, c = 0.01, 0.5, 40, replicate 1), h1 = `{h1}`: the same objectives, "
        f"routes and visitor scores (assertions).")
    say()


# ---------------------------------------------------------------------- #
# 3. Grouped outputs against per-visitor rows
# ---------------------------------------------------------------------- #

def grouped_outputs():
    say("## 3. Grouped RQ3 outputs against per-visitor rows")
    say()
    days = Days()
    random.seed(1)
    rules = [new_individual(method) for method in METHODS]
    checked = 0
    for date in PILOT_DAYS:
        day, _ = days.get(date)
        for rule in rules:
            result = evaluate(rule, date, 0, days)
            rows = visitor_rows(day, result)
            table = grouped(rows)
            assert table["visitors"].sum() == len(rows)
            assert np.isclose(table["sum R_v"].sum() / len(rows), result.mean_r)
            for column in ("change A", "change C", "change T", "stops", "observed stops"):
                assert np.isclose(table[column].sum(), rows[column].sum())
            assert table["ended early"].sum() == (rows["stops"] < rows["observed stops"]).sum()
            assert table["overrun records"].sum() == len(result.run.overruns)
            for theta in (0.8, 0.9, 1.0):
                assert share_at_least(table, theta) == (rows["R_v"] >= theta).mean()
            for follows, g in rows.groupby("follows"):
                assert table.loc[table["follows"] == follows, "visitors"].sum() == len(g)
            assert (hourly_peaks(day, result.crowding)["peak C"].to_numpy() == result.crowding.max(axis=1)).all()
            checked += 1
    say(f"- {checked} evaluations (one random rule per method on each pilot day): visitors, mean R_v, the sums of "
        f"the changes and stops, early endings, overrun records, the shares at R_v >= 0.8, 0.9 and 1 (exact), the "
        f"follows split and the hourly peaks all equal those computed from the per-visitor rows (assertions).")
    say()


# ---------------------------------------------------------------------- #
# 4. Short searches
# ---------------------------------------------------------------------- #

def short_searches():
    from search import Run
    settings = {"population": 20, "generations": 5, "crossover": 0.8, "mutation": 0.15, "monitor every": 5,
                "validation replicates": 1}
    say("## 4. A short search of every method (run number 1)")
    say()
    say(f"Population {settings['population']}, {settings['generations']} generations, monitoring after the last. "
        f"Too short to compare methods; this checks the machinery.")
    say()
    say("| Method | Dates of the schedule | Crossover changed | Mutation changed | Reverts | Constant cuts "
        "| Duplicates (last) | Mean nodes (last) | Front size (last) | Seconds per generation |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    table = pd.read_csv(PROCESSED / "days_2019.csv")
    train = set(table["date"][table["split"] == "train"])
    schedules = []
    for method in METHODS:
        run = Run(method, 1, settings)
        run.evolve()
        dates = [row["date"] for row in run.log]
        schedules.append(dates)
        assert set(dates) <= train and len(set(dates)) == len(dates)
        total = {k: sum(row[k] for row in run.log) for k in
                 ("crossover", "crossover changed", "mutation", "mutation changed", "reverts", "constant cuts")}
        last = run.log[-1]
        say(f"| {method} | {', '.join(dates)} | {total['crossover changed']}/{2 * total['crossover']} children | "
            f"{total['mutation changed']}/{total['mutation']} | {total['reverts']} | {total['constant cuts']} | "
            f"{last['duplicates']} | {last['mean nodes']:.1f} | {len(last['front'])} | "
            f"{np.mean([row['seconds'] for row in run.log]):.1f} |")
    assert all(s == schedules[0] for s in schedules)
    say()
    say("- Every method read the same dates in the same order, all training days, none twice (assertions).")
    say()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    code = fingerprint("src/gp_rule.py", "src/evaluate.py", "src/search.py", "src/features.py", "src/engine.py",
                       "src/scoring.py")
    say("# Step 4: GP rules, their evaluation and the search on real days")
    say()
    say(f"Data version {Days().get(PILOT_DAYS[0])[0].data_version}; code {code}. Generated by `src/search_check.py`. "
        f"The checks are assertions; times are only reported.")
    say()
    timing()
    adaptive_is_fixed()
    grouped_outputs()
    short_searches()
    (OUT / "record.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
