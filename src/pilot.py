"""
Pilot on the five fixed days (Step 3g)
======================================

Section 9 of simulator_design.md. On each pilot day:

  1. the three scenarios: observed replay, the observed-order control and the
     four hand rules, under the corrected and the uncorrected budget;
  2. five replicates of the stays, to see how much a day's result moves with
     the random stays alone (the protocol leaves the number of replicates to
     the pilot);
  3. where the time goes: the engine (events, heap, feasibility) against the
     policy (state inputs and the rule);
  4. a common lower bound on the usual peak (a sensitivity option, off by
     default); no bound is chosen here;
  5. where the change of route value comes from: attractiveness, crowding
     and walking, for all visitors and for the worst tenth.

Scores use alpha = 0.25 and beta = 0.5, over all evaluated visitors.

Run:  python3 src/pilot.py   (from the repository root)
Writes outputs/sim/3g/record.md.
"""

import time
from pathlib import Path

import numpy as np

from day import load_day
from engine import ObservedOrder, simulate
from preprocess import gini_hat
from replay import replay
from rules import RULES
from scoring import score_day, summary
from single_gphh import fingerprint

OUT = Path("outputs/sim/3g")
ALPHA, BETA = 0.25, 0.5
PILOT_DAYS = ["2019-12-18", "2019-10-09", "2019-03-17", "2019-04-26", "2019-06-03"]
POLICIES = {"observed order": ObservedOrder, **RULES}
REPLICATES = 5
PEAK_FLOORS = [None, 5.0, 10.0]

lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


class Timed:
    """Wraps a policy and adds up the time spent choosing."""
    def __init__(self, inner):
        self.inner, self.seconds = inner, 0.0
        self.may_stop = getattr(inner, "may_stop", False)
        self.name = getattr(inner, "name", type(inner).__name__)

    def reset(self):
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    def finish(self, engine, k):
        if hasattr(self.inner, "finish"):
            self.inner.finish(engine, k)

    def choose(self, engine, k, candidates):
        started = time.perf_counter()
        j = self.inner.choose(engine, k, candidates)
        self.seconds += time.perf_counter() - started
        return j


def run_once(day, observed, name, replicate=0):
    policy = Timed(POLICIES[name]())
    started = time.perf_counter()
    run = simulate(day, policy, replicate=replicate)
    seconds = time.perf_counter() - started
    v, g_hat, _ = score_day(day, run, observed, ALPHA, BETA)
    return v, g_hat, seconds, policy.seconds


# ---------------------------------------------------------------------- #
# 1. Three scenarios
# ---------------------------------------------------------------------- #

def scenarios():
    say("## 1. Three scenarios")
    say()
    say("Replay is the observed day. Every other row is one run (replicate 0). R_v is over all evaluated "
        "visitors; p10 is its 10th percentile.")
    say()
    say("| Date | Budget | Policy | Mean R_v | p10 R_v | R_v < 0 | G_hat | Completion | Ended early | Seconds |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    timing = {name: [0.0, 0.0, 0, 0.0] for name in POLICIES}
    for date in PILOT_DAYS:
        for budget in ("corrected", "span"):
            day = load_day(date, budget=budget)
            observed = replay(day)
            if budget == "corrected":
                g = gini_hat(observed.entries_per_site[day.available])
                say(f"| {date} | | replay | 1 | 1 | 0 | {g:.3f} | 100% | 0 | |")
            for name in POLICIES:
                v, g_hat, seconds, policy_seconds = run_once(day, observed, name)
                out = summary(v, ALPHA, BETA)["all"]
                assert out["split adds up"]
                if budget == "corrected":
                    timing[name][0] += seconds
                    timing[name][1] += policy_seconds
                    timing[name][2] += 1
                    timing[name][3] = max(timing[name][3], seconds)
                say(f"| {date} | {budget} | {name} | {out['mean R_v']:.3f} | {v['R_v'].quantile(0.1):.3f} | "
                    f"{(v['R_v'] < 0).mean():.1%} | {g_hat:.3f} | {out['completion']:.1%} | "
                    f"{out['ended early']:,} | {seconds:.2f} |")
    say()
    return timing


# ---------------------------------------------------------------------- #
# 2. Replicates
# ---------------------------------------------------------------------- #

def replicates():
    say(f"## 2. {REPLICATES} replicates of the stays")
    say()
    say("Spread of the day's result across replicates: the range (max - min) of mean R_v and of G_hat, the largest "
        "over the five pilot days.")
    say()
    say("| Policy | Range of mean R_v | Range of G_hat | Day with the largest range of mean R_v |")
    say("|---|---|---|---|")
    days = {date: load_day(date) for date in PILOT_DAYS}
    observed = {date: replay(day) for date, day in days.items()}
    for name in POLICIES:
        worst_r, worst_g, worst_day = 0.0, 0.0, ""
        for date, day in days.items():
            results = [run_once(day, observed[date], name, replicate) for replicate in range(REPLICATES)]
            means = [v["R_v"].mean() for v, _, _, _ in results]
            gs = [g for _, g, _, _ in results]
            if np.ptp(means) > worst_r:
                worst_r, worst_day = np.ptp(means), date
            worst_g = max(worst_g, np.ptp(gs))
        say(f"| {name} | {worst_r:.4f} | {worst_g:.4f} | {worst_day} |")
    say()


# ---------------------------------------------------------------------- #
# 3. Time
# ---------------------------------------------------------------------- #

def time_per_day(timing):
    say("## 3. Time per day")
    say()
    say("The simulation of one day with a hand rule, corrected budget, five pilot days. Not included: scoring "
        "(score_day), h2 and a GP tree; the Step 4 pilot times a whole evaluation. The policy share is the state "
        "inputs and the rule; the rest is the engine (events, heap, feasibility).")
    say()
    say("| Policy | Mean seconds per day | Most seconds in a day | Policy share |")
    say("|---|---|---|---|")
    for name, (seconds, policy_seconds, n, most) in timing.items():
        say(f"| {name} | {seconds / n:.3f} | {most:.3f} | {policy_seconds / seconds:.0%} |")
    say()


# ---------------------------------------------------------------------- #
# 4. A lower bound on the usual peak
# ---------------------------------------------------------------------- #

def peak_floor():
    say("## 4. A common lower bound on the usual peak")
    say()
    day = load_day(PILOT_DAYS[0])
    low = ", ".join(f"{site} {peak:.1f}" for site, peak in zip(day.site_ids, day.usual_peak) if peak < 10)
    say(f"Usual peaks below 10 people: {low}. The bound applies to CR and to C alike, so the replay baseline moves "
        "with it. Means over the five pilot days.")
    say()
    say("| Bound | Policy | Mean R_v | Lowest p10 R_v | Mean G_hat |")
    say("|---|---|---|---|---|")
    for floor in PEAK_FLOORS:
        for name in ("observed order", "POP", "LEAST"):
            means, p10s, gs = [], [], []
            for date in PILOT_DAYS:
                day = load_day(date, peak_floor=floor)
                v, g_hat, _, _ = run_once(day, replay(day), name)
                means.append(v["R_v"].mean())
                p10s.append(v["R_v"].quantile(0.1))
                gs.append(g_hat)
            say(f"| {floor or 'none'} | {name} | {np.mean(means):.3f} | {min(p10s):.3f} | {np.mean(gs):.3f} |")
    say()


# ---------------------------------------------------------------------- #
# 5. Where the change of value comes from
# ---------------------------------------------------------------------- #

def parts(g):
    return (f"{(g['V'] - g['V observed']).mean():.3f} | {g['change A'].mean():.3f} | "
            f"{-ALPHA * g['change C'].mean():.3f} | {-BETA * g['change T'].mean():.3f}")


def where_the_change_comes_from():
    say("## 5. Where the change of route value comes from")
    say()
    say("Change of V = change of A - alpha x change of C - beta x change of T, as means over the evaluated visitors "
        "and over the worst tenth by R_v; lost stops show in A. Highest C: the site and hour with the highest hourly "
        "crowding of the run, with its usual peak.")
    say()
    say("| Date | Policy | All: change V | A | -alpha C | -beta T | Worst tenth: change V | A | -alpha C | -beta T "
        "| Highest C |")
    say("|---|---|---|---|---|---|---|---|---|---|---|")
    for date in ("2019-10-09", "2019-04-26"):
        day = load_day(date)
        observed = replay(day)
        for name in POLICIES:
            run = simulate(day, POLICIES[name]())
            v, _, c = score_day(day, run, observed, ALPHA, BETA)
            worst = v[v["R_v"] <= v["R_v"].quantile(0.1)]
            i, hour = np.unravel_index(np.argmax(c), c.shape)
            say(f"| {date} | {name} | {parts(v)} | {parts(worst)} | {c[i, hour]:.1f} at site {day.site_ids[i]}, "
                f"{hour}:00 (usual peak {day.usual_peak[i]:.1f}) |")
    say()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    code = fingerprint("src/day.py", "src/engine.py", "src/features.py", "src/rules.py", "src/scoring.py")
    say("# Step 3g: hand rules and the pilot")
    say()
    say(f"Data version {load_day(PILOT_DAYS[0]).data_version}; code {code}; alpha {ALPHA}, beta {BETA}. "
        f"Generated by `src/pilot.py`. Results are descriptive; the pilot days were fixed before any result "
        f"(simulator_design.md, section 9).")
    say()
    timing = scenarios()
    replicates()
    time_per_day(timing)
    peak_floor()
    where_the_change_comes_from()
    (OUT / "record.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
