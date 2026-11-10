"""
Checks of the day input and the observed replay (Steps 3a and 3b)
=================================================================

Runs load_day() and replay() on the five pilot days and on every training day
of 2019, and checks them against the processed tables and against the EDA's
own way of computing the observed route value (src/eda.py, section 7).

Run:  python3 src/replay_check.py   (from the repository root)
Writes outputs/sim/3ab/record.md.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from day import PROCESSED, load_day
from preprocess import observed_occupancy_intervals
from replay import baseline_value, replay
from single_gphh import fingerprint

OUT = Path("outputs/sim/3ab")
ALPHA, BETA = 0.25, 0.5
PILOT_DAYS = ["2019-12-18", "2019-10-09", "2019-03-17", "2019-04-26", "2019-06-03"]

lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


# ---------------------------------------------------------------------- #
# 1. The pilot days against the tables
# ---------------------------------------------------------------------- #

def check_pilot_days(days, visitors):
    say("## Pilot days")
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
    say("## Every training day of 2019")
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


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    days = pd.read_csv(PROCESSED / "days_2019.csv", parse_dates=["date"]).set_index("date")
    visitors = pd.read_csv(PROCESSED / "visitors_2019.csv.gz", parse_dates=["date"])
    data_version = load_day(PILOT_DAYS[0]).data_version
    code = fingerprint("src/day.py", "src/replay.py", "src/scoring.py")

    say("# Steps 3a and 3b: day input and observed replay")
    say()
    say(f"Data version {data_version}; code {code}; alpha {ALPHA}, beta {BETA}. "
        f"Generated by `src/replay_check.py`; "
        f"every line below is an assertion that passed.")
    say()
    check_pilot_days(days, visitors)
    check_training_days(days)
    (OUT / "record.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
