"""
Checks of the simulator on real days (Steps 3a to 3f)
=====================================================

Assertions on the five pilot days and on every training day of 2019, one
section per sub-step:

  3a, 3b  load_day() and replay() against the processed tables and against
          the EDA's own way of computing the observed route value;
  3c, 3d  the engine with its debug checks, and every route against its own
          constraints, under Nearest (a test driver only) and the
          observed-order control;
  3e      an independent ledger of the counts at every decision on the pilot
          days, and the range of every terminal over the training days;
  3f      day scoring: with no followers every R_v is 1 and G_hat is the
          observed one; the change of V splits into A, C and T.

Counts, shares and timings are reported; the checks are assertions.

Run:  python3 src/simulator_check.py   (from the repository root)
Writes outputs/sim/check/record.md.
"""

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from day import PROCESSED, load_day
from engine import Nearest, ObservedOrder, simulate
from features import H1_TERMINALS, H2_TERMINALS, h1_inputs, h2_inputs, visit_share
from preprocess import gini_hat, observed_occupancy_intervals
from replay import baseline_value, replay
from scoring import score_day, summary
from single_gphh import fingerprint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_features import Ledger  # noqa: E402

OUT = Path("outputs/sim/check")
ALPHA, BETA = 0.25, 0.5
PILOT_DAYS = ["2019-12-18", "2019-10-09", "2019-03-17", "2019-04-26", "2019-06-03"]

lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


# ====================================================================== #
# 3a, 3b: day input and observed replay
# ====================================================================== #

# ---------------------------------------------------------------------- #
# 1. The pilot days against the tables
# ---------------------------------------------------------------------- #

def visits_of_day(date):
    visits = pd.read_csv(PROCESSED / "visits_2019.csv.gz", parse_dates=["date"])
    return visits[visits.date == pd.Timestamp(date)]


def check_pilot_days(days, visitors):
    say("### Pilot days")
    say()
    say("| Date | Visitors | Evaluated | Entries | Available sites | Person-minutes conserved | Seconds |")
    say("|---|---|---|---|---|---|---|")
    for date in PILOT_DAYS:
        started = time.time()
        day = load_day(date)
        result = replay(day)
        seconds = time.time() - started
        row = days.loc[pd.Timestamp(date)]
        table = visitors[visitors.date == pd.Timestamp(date)]
        assert row.split == "train"
        assert len(day.visitors) == len(table) and day.visitors.evaluated.sum() == (table.stops >= 2).sum()
        assert len(day.fixed) == row.visits == result.entries_per_site.sum()
        observed = visits_of_day(date).site.value_counts().reindex(day.site_ids, fill_value=0)
        assert (result.entries_per_site == observed.to_numpy()).all(), "entries per site, by site id"
        assert day.available.sum() == row.open_sites
        assert np.allclose(day.visitors.sort_values("visitor").budget_limit, table.sort_values("visitor").budget)
        conserved = np.isclose(result.person_minutes.sum(), (day.fixed.exit - day.fixed.entry).sum())
        assert conserved
        say(f"| {date} ({pd.Timestamp(date):%a}) | {len(day.visitors):,} | {day.visitors.evaluated.sum():,} | "
            f"{len(day.fixed):,} | {day.available.sum()} | yes | {seconds:.2f} |")
    say()


# ---------------------------------------------------------------------- #
# 2. Every training day against the EDA's computation
# ---------------------------------------------------------------------- #

def eda_route_values(visits, sites, walk):
    """The observed route value as src/eda.py computes it: occupancy from the shared intervals,
    crowding by (date, site, hour), summed along each route of a multi-stop visitor."""
    train = visits.copy()
    train["end"] = observed_occupancy_intervals(train, sites.d).exit.values
    rows = []
    for hour in range(8, 22):
        overlap = (train.end.clip(upper=60 * (hour + 1)) - train.t.clip(lower=60 * hour)).clip(lower=0)
        inside = (overlap.groupby([train.date, train.site]).sum() / 60).rename("inside").reset_index()
        inside["hour"] = hour
        rows.append(inside)
    occupancy = pd.concat(rows)
    occupancy["C"] = occupancy.inside / occupancy.site.map(sites.usual_peak)
    stops = train.groupby("visitor").stop.transform("max")
    keep = train[stops >= 2].copy()
    keep["hour"] = (keep.t // 60).astype(int)
    keep = keep.merge(occupancy[["date", "site", "hour", "C"]], on=["date", "site", "hour"], how="left")
    keep["A"] = keep.site.map(sites.A)
    keep = keep.sort_values(["visitor", "stop"])
    keep["next_site"] = keep.groupby("visitor").site.shift(-1)
    has_next = keep.next_site.notna()
    keep["walk"] = 0.0
    keep.loc[has_next, "walk"] = [walk[(a, int(b))] / 60 for a, b in zip(keep.site[has_next], keep.next_site[has_next])]
    return keep.groupby("visitor").agg(A=("A", "sum"), C=("C", "sum"), T=("walk", "sum"))


def check_training_days(days):
    say("### Every training day of 2019")
    say()
    sites = pd.read_csv(PROCESSED / "sites.csv", index_col=0)
    visits = pd.read_csv(PROCESSED / "visits_2019.csv.gz", parse_dates=["date"])
    walk_table = pd.read_csv("data/verona/poi_time_travel.csv")
    walk = {(a, b): t for a, b, t in zip(walk_table.poi_start, walk_table.poi_dest, walk_table.time_travel)}

    started = time.time()
    parts = []
    for date in days.index[days.split == "train"]:
        day = load_day(date)
        result = replay(day)
        b = result.baseline.copy()
        by_hand = b["A"] - ALPHA * b["C"] - BETA * b["T"]
        assert np.allclose(baseline_value(result, ALPHA, BETA), by_hand)
        b.index = day.visitors.visitor[b.index].to_numpy()
        parts.append(b)
    ours = pd.concat(parts).sort_index()
    seconds = time.time() - started

    # The EDA computes crowding for 08:00 to 22:00 only, so every interval must end before 22:00.
    train_visits = visits[visits.split == "train"]
    assert observed_occupancy_intervals(train_visits, sites.d).exit.max() < 22 * 60
    eda = eda_route_values(train_visits, sites, walk).sort_index()
    assert ours.index.equals(eda.index), "the same evaluated visitors"
    gap = (ours - eda).abs().max()
    assert (gap < 1e-9).all(), gap
    value = ours.A - ALPHA * ours.C - BETA * ours["T"]
    eda_value = eda.A - ALPHA * eda.C - BETA * eda["T"]
    assert (value <= 0).sum() == (eda_value <= 0).sum()

    say(f"- Training days {int((days.split == 'train').sum())}, evaluated visitors {len(ours):,}; "
        f"load_day and replay took {seconds:.1f} s in all.")
    say(f"- Largest difference from the EDA computation, per visitor: A {gap.A:.1e}, C {gap.C:.1e}, T {gap['T']:.1e} "
        f"(tolerance 1e-9).")
    say(f"- Means: sum of A {ours.A.mean():.4f}, alpha x C {ALPHA * ours.C.mean():.4f}, "
        f"beta x T {BETA * ours['T'].mean():.4f}, V(r^O) {value.mean():.4f}")
    say(f"- V(r^O) <= 0: {(value <= 0).sum():,} routes ({(value <= 0).mean():.2%}), "
        f"the same count as the EDA computation; "
        f"smallest sum of A {ours.A.min():.4f}")
    say()


# ====================================================================== #
# 3c, 3d: event engine and observed-order control
# ====================================================================== #

def check_routes(day, run):
    """Each later stop met closing and budget under expected stays when chosen; one place at a time;
    the stop limit holds; x counts every entry."""
    start = np.array([e[0] for e in day.entries])
    budget_end = start + day.visitors.budget_limit.to_numpy()
    for k, route in run.routes.items():
        assert len(route) <= day.visitors.stop_limit[k]
        for (a, t_a, stay_a), (b, t_b, _) in zip(route, route[1:]):
            assert t_b >= t_a + stay_a + day.walk[a, b] - 1e-9
            assert t_b + day.stay[b] <= day.closing[b] + 1e-9 and t_b + day.stay[b] <= budget_end[k] + 1e-9
    fixed_entries = (~day.visitors.follows.to_numpy()[day.fixed.visitor]).sum()
    simulated_entries = sum(len(route) for route in run.routes.values())
    assert run.visits.sum() == len(run.entries) == fixed_entries + simulated_entries


def completion(day, run):
    """Stops on the simulated routes over stops on the observed routes, over the followers."""
    followers = day.visitors.index[day.visitors.follows]
    return sum(len(run.routes[k]) for k in followers) / sum(len(day.routes[k]) for k in followers)


# ---------------------------------------------------------------------- #
# 1. Pilot days
# ---------------------------------------------------------------------- #

def engine_pilot_days():
    say("### Pilot days")
    say()
    say("| Date | Followers | Policy | Budget | Events | Completion | Ended early | Overrun records "
        "| Skips by reason | Seconds |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for date in PILOT_DAYS:
        for budget in ("corrected", "span"):
            day = load_day(date, budget=budget)
            policies = [("observed order", ObservedOrder())]
            if budget == "corrected":
                policies.insert(0, ("nearest (test driver)", Nearest()))
            for name, policy in policies:
                started = time.time()
                run = simulate(day, policy, check=True)
                seconds = time.time() - started
                check_routes(day, run)
                early = sum(reason != "limit" for reason in run.end_reason.values())
                skips = Counter(reason for _, _, reason in run.skips)
                skip_text = ", ".join(f"{reason} {count}" for reason, count in sorted(skips.items())) or "none"
                say(f"| {date} | {day.visitors.follows.sum()} | {name} | {budget} | {run.events:,} | "
                    f"{completion(day, run):.1%} | {early} | {len(run.overruns)} | {skip_text} | {seconds:.2f} |")
    say()


# ---------------------------------------------------------------------- #
# 2. Every training day
# ---------------------------------------------------------------------- #

def engine_training_days():
    say("### Every training day of 2019")
    say()
    days = pd.read_csv(PROCESSED / "days_2019.csv", parse_dates=["date"])
    dates = days.date[days.split == "train"]
    totals = {"nearest": Counter(), "observed order": Counter()}
    seconds = {"nearest": 0.0, "observed order": 0.0}
    stops = {"nearest": [0, 0], "observed order": [0, 0]}
    for date in dates:
        day = load_day(date)
        for name, policy in (("nearest", Nearest()), ("observed order", ObservedOrder())):
            started = time.time()
            run = simulate(day, policy, check=True)
            seconds[name] += time.time() - started
            check_routes(day, run)
            totals[name].update(run.end_reason.values())
            totals[name].update(f"skip: {reason}" for _, _, reason in run.skips)
            totals[name]["overrun records"] += len(run.overruns)
            followers = day.visitors.index[day.visitors.follows]
            stops[name][0] += sum(len(run.routes[k]) for k in followers)
            stops[name][1] += sum(len(day.routes[k]) for k in followers)
    say(f"- {len(dates)} training days, both policies. Debug checks in the engine, after every event: no negative "
        f"count; at the end of the day: everyone left and every follower finished. Route checks in this script: "
        f"every later stop met closing and budget when chosen, visits do not overlap and leave time to walk, "
        f"the stop limit holds.")
    say("- A lost stop is counted under the first reason it failed (calendar, closing, budget). An overrun record "
        "is one visit past closing or past the budget; one visit can give both.")
    for name in totals:
        counts = ", ".join(f"{key} {value:,}" for key, value in sorted(totals[name].items()))
        say(f"- {name}: completion {stops[name][0] / stops[name][1]:.1%}; {counts}; "
            f"{seconds[name] / len(dates) * 1000:.0f} ms per day")
    say()

# ====================================================================== #
# 3e: state inputs
# ====================================================================== #

class Collector(Nearest):
    """Nearest, keeping every terminal value and the time spent computing them."""
    def __init__(self):
        self.values = {name: [] for name in H1_TERMINALS + H2_TERMINALS + ["VS"]}
        self.seconds, self.decisions, self.candidates = 0.0, 0, 0

    def choose(self, engine, k, candidates):
        started = time.perf_counter()
        h1 = h1_inputs(engine, k, candidates)
        h2 = h2_inputs(engine, k)
        vs = visit_share(engine, candidates)
        self.seconds += time.perf_counter() - started
        self.decisions += 1
        self.candidates += len(candidates)
        for name, value in zip(H1_TERMINALS, h1):
            self.values[name].append(value)
        for name, value in zip(H2_TERMINALS, h2):
            self.values[name].append(np.array([value]))
        self.values["VS"].append(vs)
        return super().choose(engine, k, candidates)


def pilot_ledger():
    say("### Ledger on the pilot days")
    say()
    say("| Date | Policy | Decisions checked |")
    say("|---|---|---|")
    for date in PILOT_DAYS:
        day = load_day(date)
        for name, inner in (("nearest (test driver)", Nearest), ("observed order", ObservedOrder)):
            policy = Ledger(inner())
            simulate(day, policy, check=True)
            say(f"| {date} | {name} | {policy.checked:,} |")
    say()
    say("At every decision the ledger's people inside, on the way and visits so far equal the engine's, and CR, "
        "HEAD and ARR equal the values recomputed from the ledger (assertions).")
    say()


def terminals_on_training_days():
    say("### Terminals over every training day of 2019")
    say()
    days = pd.read_csv(PROCESSED / "days_2019.csv", parse_dates=["date"])
    dates = days["date"][days["split"] == "train"]
    policy, fallbacks = Collector(), Counter()
    started = time.time()
    for date in dates:
        run = simulate(load_day(date), policy)
        fallbacks.update(run.fallbacks)
    seconds = time.time() - started
    say(f"- {len(dates)} days, {policy.decisions:,} decisions, {policy.candidates / policy.decisions:.1f} candidates "
        f"per decision on average; state inputs {policy.seconds / policy.decisions * 1e6:.0f} microseconds per "
        f"decision, {policy.seconds / seconds:.0%} of the run time.")
    shown = ", ".join(f"{reason} {n:,} ({n / policy.decisions:.1%})" for reason, n in sorted(fallbacks.items()))
    say(f"- Fallbacks to 1, by reason, over decisions: {shown or 'none'}.")
    say()
    say("| Terminal | Min | 5% | Median | 95% | Max |")
    say("|---|---|---|---|---|---|")
    for name, chunks in policy.values.items():
        values = np.concatenate(chunks)
        assert np.isfinite(values).all(), f"{name} has a non-finite value"
        q = np.quantile(values, [0, 0.05, 0.5, 0.95, 1])
        say(f"| {name} | " + " | ".join(f"{v:.3g}" for v in q) + " |")
    say()
    say("Every value is finite (assertion). h2 terminals are counted once per decision, h1 terminals and VS once per "
        "candidate.")
    say()

# ====================================================================== #
# 3f: scoring a day
# ====================================================================== #

def scoring_on_training_days():
    say("### Every training day of 2019")
    say()
    days = pd.read_csv(PROCESSED / "days_2019.csv", parse_dates=["date"])
    dates = days["date"][days["split"] == "train"]
    worst, results = 0.0, {"observed order": [], "nearest": []}
    for date in dates:
        observed = replay(load_day(date))
        background = load_day(date, follows=[])
        v, g_hat, c = score_day(background, simulate(background, Nearest()), observed, ALPHA, BETA)
        worst = max(worst, (v["R_v"] - 1).abs().max())
        assert np.allclose(c, observed.crowding) and g_hat == gini_hat(observed.entries_per_site[background.available])
        day = load_day(date)
        for name, policy in (("observed order", ObservedOrder()), ("nearest", Nearest())):
            v, g_hat, _ = score_day(day, simulate(day, policy), observed, ALPHA, BETA)
            out = summary(v, ALPHA, BETA)["all"]
            assert out["split adds up"] and len(v) == day.visitors["evaluated"].sum()
            results[name].append((out["mean R_v"], g_hat, out["completion"]))
    assert worst < 1e-12
    say(f"- With no followers, on every day: largest |R_v - 1| {worst:.1e}, hourly crowding equal to the replay's, "
        f"G_hat equal to the observed one (assertions).")
    say("- On every day the change of V equals change of A - alpha x change of C - beta x change of T (assertion).")
    for name, rows in results.items():
        r, g, done = np.array(rows).T
        say(f"- {name}: mean R_v {r.mean():.3f} (days from {r.min():.3f} to {r.max():.3f}), "
            f"G_hat {g.mean():.3f}, completion {done.mean():.1%}")
    say()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    days = pd.read_csv(PROCESSED / "days_2019.csv", parse_dates=["date"]).set_index("date")
    visitors = pd.read_csv(PROCESSED / "visitors_2019.csv.gz", parse_dates=["date"])
    code = fingerprint("src/day.py", "src/replay.py", "src/scoring.py", "src/engine.py", "src/features.py")
    say("# Step 3: checks of the simulator on real days")
    say()
    say(f"Data version {load_day(PILOT_DAYS[0]).data_version}; code {code}; alpha {ALPHA}, beta {BETA}. "
        f"Generated by `src/simulator_check.py`. The checks are assertions; counts, shares and timings are "
        f"only reported. Nearest is only a driver for testing the engine, not a rule of the study.")
    say()
    say("## 3a, 3b: day input and observed replay")
    say()
    check_pilot_days(days, visitors)
    check_training_days(days)
    say("## 3c, 3d: event engine and observed-order control")
    say()
    engine_pilot_days()
    engine_training_days()
    say("## 3e: state inputs")
    say()
    pilot_ledger()
    terminals_on_training_days()
    say("## 3f: scoring a day")
    say()
    scoring_on_training_days()
    (OUT / "record.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
