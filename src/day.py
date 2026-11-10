"""
One simulated day (Step 3a)
===========================

Builds the input of one day from the tables of src/preprocess.py: the sites
with their hours and availability, the walking times, the usual curves behind
GDAY and ARR, and every visitor of the day with the observed route and fixed
occupancy intervals. Section 2 of simulator_design.md.

Sites are numbered 0 to 14 in ascending order of site id. Visitors are
numbered in order of visitor id, before anyone is marked as following, so the
numbers never depend on the rule or on who follows.

Usage:  from day import load_day;  day = load_day("2019-10-09")
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import FIRST_MINUTE, LAST_MINUTE, LOW_COVERAGE, observed_occupancy_intervals

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "verona" / "processed"
WALKING = ROOT / "data" / "verona" / "poi_time_travel.csv"


@dataclass
class Day:
    date: pd.Timestamp
    day_type: str            # "monday" or "other"
    site_ids: np.ndarray     # the 15 site ids, ascending; position = site index
    available: np.ndarray    # bool per site: in the candidates and in N today
    attractiveness: np.ndarray
    stay: np.ndarray         # expected stay d, minutes
    usual_peak: np.ndarray
    opening: np.ndarray      # o for this month, minutes after midnight
    closing: np.ndarray      # c for this month
    walk: np.ndarray         # sites x sites, minutes
    usual_gday: np.ndarray   # one value per minute from FIRST_MINUTE to LAST_MINUTE
    usual_arrivals: np.ndarray
    visitors: pd.DataFrame   # one row per visitor index
    routes: list             # observed route of each visitor, as site indices
    entries: list            # observed entry times of each visitor
    fixed: pd.DataFrame      # observed occupancy intervals: visitor, entry_index, site, entry, exit
    data_version: str


# ---------------------------------------------------------------------- #
# 1. Tables, read once per year
# ---------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def load_tables(year):
    tables = {
        "sites": pd.read_csv(PROCESSED / "sites.csv", index_col=0),
        "hours": pd.read_csv(PROCESSED / "site_hours.csv"),
        "open_sites": pd.read_csv(PROCESSED / f"open_sites_{year}.csv", parse_dates=["date"]),
        "usual_gday": pd.read_csv(PROCESSED / "usual_gday.csv"),
        "usual_arrivals": pd.read_csv(PROCESSED / "usual_arrivals.csv"),
        "visits": pd.read_csv(PROCESSED / f"visits_{year}.csv.gz", parse_dates=["date"]),
        "visitors": pd.read_csv(PROCESSED / f"visitors_{year}.csv.gz", parse_dates=["date"]),
        "walk": pd.read_csv(WALKING),
        "version": json.loads((PROCESSED / "manifest.json").read_text())["version"],
    }
    return tables


# ---------------------------------------------------------------------- #
# 2. Sites
# ---------------------------------------------------------------------- #

def month_hours(hours, sites, month):
    """Opening and closing of each site in a month. A site with no training entries
    in that month uses its yearly hours, as window() in preprocess.py does."""
    rows = hours[hours.month == month].set_index("site")
    opening = [rows.o.get(i, sites.o[i]) for i in sites.index]
    closing = [rows.c.get(i, sites.c[i]) for i in sites.index]
    return np.array(opening, float), np.array(closing, float)


def availability(tables, date, site_ids, same_day, visits):
    """Inferred calendar by default; with same_day, the sites with entries that day.
    Low-coverage sites are never available."""
    if same_day:
        open_ids = set(visits.site)
    else:
        open_sites = tables["open_sites"]
        open_ids = set(open_sites.site[open_sites.date == date])
    return np.array([i in open_ids and i not in LOW_COVERAGE for i in site_ids])


def usual_curve(table, column, day_type):
    part = table[table.day_type == day_type].sort_values("minute")
    assert part.minute.tolist() == list(range(FIRST_MINUTE, LAST_MINUTE + 1))
    return part[column].to_numpy(float)


def walking_matrix(walk_table, site_ids):
    walk = {(a, b): t for a, b, t in zip(walk_table.poi_start, walk_table.poi_dest, walk_table.time_travel)}
    return np.array([[walk[(a, b)] for b in site_ids] for a in site_ids], float)


# ---------------------------------------------------------------------- #
# 3. Visitors
# ---------------------------------------------------------------------- #

def visitor_frame(visitors, follows):
    """Visitors in order of visitor id, with the two flags. `evaluated` is fixed (two or more
    observed stops); `follows` defaults to every evaluated visitor and can only be a subset."""
    v = visitors.sort_values("visitor").reset_index(drop=True)
    v["evaluated"] = v.stops >= 2
    if follows is None:
        v["follows"] = v.evaluated
    else:
        v["follows"] = v.visitor.isin(set(follows))
        assert not (v.follows & ~v.evaluated).any(), "only evaluated visitors can follow"
    return v


def load_day(date, year=None, follows=None, budget="corrected", same_day=False):
    """One day. budget is "corrected" (main) or "span" (the uncorrected sensitivity run);
    same_day switches availability to the sites with entries that day."""
    date = pd.Timestamp(date)
    year = year or date.year
    tables = load_tables(year)
    sites = tables["sites"]
    site_ids = sites.index.to_numpy()
    assert (np.diff(site_ids) > 0).all(), "sites must be in ascending order of id"
    index_of = {site: k for k, site in enumerate(site_ids)}

    visits = tables["visits"][tables["visits"].date == date]
    visitors = visitor_frame(tables["visitors"][tables["visitors"].date == date], follows)
    visitors["budget_used"] = visitors.budget if budget == "corrected" else visitors.span
    visitors["stop_limit"] = visitors.stops

    # Routes, entries and fixed intervals, by visitor index.
    number = dict(zip(visitors.visitor, visitors.index))
    visits = visits.sort_values(["visitor", "stop"])
    routes, entries = [None] * len(visitors), [None] * len(visitors)
    for visitor, g in visits.groupby("visitor"):
        k = number[visitor]
        routes[k] = np.array([index_of[s] for s in g.site])
        entries[k] = g.t.to_numpy(float)
    intervals = observed_occupancy_intervals(visits, sites.d)
    fixed = pd.DataFrame({"visitor": intervals.visitor.map(number).to_numpy(),
                          "entry_index": (visits.stop - 1).to_numpy(),
                          "site": intervals.site.map(index_of).to_numpy(),
                          "entry": intervals.entry.to_numpy(float),
                          "exit": intervals.exit.to_numpy(float)})

    day_type = "monday" if date.dayofweek == 0 else "other"
    opening, closing = month_hours(tables["hours"], sites, date.month)
    return Day(date=date, day_type=day_type, site_ids=site_ids,
               available=availability(tables, date, site_ids, same_day, visits),
               attractiveness=sites.A.to_numpy(float), stay=sites.d.to_numpy(float),
               usual_peak=sites.usual_peak.to_numpy(float), opening=opening, closing=closing,
               walk=walking_matrix(tables["walk"], site_ids),
               usual_gday=usual_curve(tables["usual_gday"], "usual_gday", day_type),
               usual_arrivals=usual_curve(tables["usual_arrivals"], "usual_arrivals", day_type),
               visitors=visitors, routes=routes, entries=entries, fixed=fixed,
               data_version=tables["version"])
