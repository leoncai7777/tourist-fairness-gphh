"""
Checks of the event engine and the observed-order control (Steps 3c and 3d)
===========================================================================

Runs the engine on the five pilot days and on every training day of 2019,
with the debug checks on (no negative counts, everyone leaves, every
follower finishes), and checks each route against its own constraints.
Two policies drive the engine: Nearest, only a test driver, and the
observed-order control, under the corrected and the uncorrected budget.

Run:  python3 src/engine_check.py   (from the repository root)
Writes outputs/sim/3cd/record.md.
"""

import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from day import PROCESSED, load_day
from engine import Nearest, ObservedOrder, simulate
from single_gphh import fingerprint

OUT = Path("outputs/sim/3cd")
PILOT_DAYS = ["2019-12-18", "2019-10-09", "2019-03-17", "2019-04-26", "2019-06-03"]

lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


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

def pilot_days():
    say("## Pilot days")
    say()
    say("| Date | Followers | Policy | Budget | Events | Completion | Ended early | Overruns "
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

def training_days():
    say("## Every training day of 2019")
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
            totals[name]["overruns"] += len(run.overruns)
            followers = day.visitors.index[day.visitors.follows]
            stops[name][0] += sum(len(run.routes[k]) for k in followers)
            stops[name][1] += sum(len(day.routes[k]) for k in followers)
    say(f"- {len(dates)} training days; on every day and for both policies: no negative count, everyone left, "
        f"every follower finished, every later stop met closing and budget when chosen, one place at a time.")
    for name in totals:
        counts = ", ".join(f"{key} {value:,}" for key, value in sorted(totals[name].items()))
        say(f"- {name}: completion {stops[name][0] / stops[name][1]:.1%}; {counts}; "
            f"{seconds[name] / len(dates) * 1000:.0f} ms per day")
    say()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    code = fingerprint("src/day.py", "src/engine.py", "src/scoring.py")
    say("# Steps 3c and 3d: event engine and observed-order control")
    say()
    say(f"Data version {load_day(PILOT_DAYS[0]).data_version}; code {code}. Generated by `src/engine_check.py`. "
        f"The checks named below are assertions in the script; counts, shares and timings are descriptive. "
        f"Nearest is only a driver for testing the engine, not a rule of the study.")
    say()
    pilot_days()
    training_days()
    (OUT / "record.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
