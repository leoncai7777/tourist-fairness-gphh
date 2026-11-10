"""
Evaluating a rule on a day (Step 4b)
====================================

One evaluation: a rule, a day and a replicate give the two objectives, mean
R_v over the evaluated visitors and G_hat, with the scores of every visitor.
From these come the outputs kept for RQ3: scores by groups fixed in advance,
the busiest hour of each site, and per-visitor rows. Section 3 of
search_design.md.

Days and their replays are cached in a Days object, one per setting, so that
days built with different settings are never mixed. A test day is refused
unless the caller says it is the test evaluation of Step 6.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

import numpy as np
import pandas as pd

from day import PROCESSED, load_day
from engine import simulate
from gp_rule import GPRule
from replay import replay
from scoring import score_day, summary

ALPHA, BETA = 0.25, 0.5
ARRIVAL_EDGES = [600, 720, 840]   # minutes: before 10:00, 10:00-12:00, 12:00-14:00, 14:00 or later
STOP_GROUPS = [2, 3, 4, 5]        # 5 means 5 or more
BINS_PER_UNIT = 20                # histogram bins of 0.05
LOWEST_BIN, HIGHEST_BIN = -2 * BINS_PER_UNIT, 2 * BINS_PER_UNIT  # open bins below -2 and from 2 up
END_REASONS = ["limit", "none feasible", "no observed stop left", "not a follower"]


@lru_cache(maxsize=None)
def split_of(year):
    """Split of each date; 2022 and 2023 are test years and their tables have no split column.
    Without a table there is nothing to read, and load_day() refuses the date."""
    path = PROCESSED / f"days_{year}.csv"
    if not path.exists():
        return {}
    days = pd.read_csv(path)
    splits = days["split"] if "split" in days.columns else ["test"] * len(days)
    return dict(zip(days["date"], splits))


@lru_cache(maxsize=None)
def budget_cuts():
    """Tertiles of the corrected budget over the evaluated visitors of the 2019 training days,
    computed once and used for every year."""
    visitors = pd.read_csv(PROCESSED / "visitors_2019.csv.gz")
    budgets = visitors["budget"][(visitors["split"] == "train") & (visitors["stops"] >= 2)]
    return tuple(float(q) for q in np.quantile(budgets, [1 / 3, 2 / 3]))


class Days:
    """Days and their replays for one setting, loaded once each. The setting is fixed when the
    object is made (a copy of follows is kept), so a cached day always matches it."""
    def __init__(self, budget="corrected", same_day=False, peak_floor=None, follows=None, test=False):
        self._options = MappingProxyType({"budget": budget, "same_day": same_day, "peak_floor": peak_floor})
        self._follows = None if follows is None else frozenset(follows)
        self._test, self._cache = bool(test), {}

    options = property(lambda self: self._options)
    follows = property(lambda self: self._follows)
    test = property(lambda self: self._test)

    def get(self, date):
        """A day and its replay. Outside the test evaluation of Step 6, only dates known to be training
        or validation days are read; a test day or a date of unknown split is refused before loading."""
        date = pd.Timestamp(date)
        if date not in self._cache:
            split = split_of(date.year).get(f"{date:%Y-%m-%d}")
            if split not in ("train", "validation", "test"):  # None or any other label
                raise ValueError(f"{date:%Y-%m-%d} has no known split; it is not read")
            if split == "test" and not self._test:
                raise ValueError(f"{date:%Y-%m-%d} is a test day; only the test evaluation of Step 6 reads it")
            day = load_day(date, follows=self._follows, **self._options)
            self._cache[date] = (day, replay(day))
        return self._cache[date]


@dataclass
class Evaluation:
    mean_r: float         # first objective, maximised
    g_hat: float          # second objective, minimised
    visitors: pd.DataFrame
    crowding: np.ndarray  # sites x 24 hours
    run: object


def evaluate(rule, date, replicate, days, log=False, alpha=ALPHA, beta=BETA, stop_limit=True):
    """One rule (an individual of gp_rule, or any policy) on one day and replicate. stop_limit off
    is the sensitivity run without the stop limit."""
    day, observed = days.get(date)
    policy = GPRule(rule, log=log) if isinstance(rule, list) else rule
    run = simulate(day, policy, replicate=replicate, stop_limit=stop_limit)
    v, g_hat, c = score_day(day, run, observed, alpha, beta)
    return Evaluation(float(summary(v, alpha, beta)["all"]["mean R_v"]), float(g_hat), v, c, run)


# ---------------------------------------------------------------------- #
# Outputs for RQ3
# ---------------------------------------------------------------------- #

def visitor_rows(day, evaluation):
    """One row per evaluated visitor: who they are and how their day went (saved for knee rules)."""
    v = evaluation.visitors
    k = v.index.to_numpy()
    out = pd.DataFrame({"visitor": day.visitors["visitor"].to_numpy()[k], "follows": v["follows"].to_numpy(),
                        "arrival": [day.entries[i][0] for i in k],
                        "budget": day.visitors["budget_limit"].to_numpy()[k],
                        "observed stops": v["observed stops"].to_numpy(), "stops": v["stops"].to_numpy(),
                        "R_v": v["R_v"].to_numpy(), "change A": v["change A"].to_numpy(),
                        "change C": v["change C"].to_numpy(), "change T": v["change T"].to_numpy(),
                        "end reason": v["end reason"].to_numpy(), "overruns": v["overruns"].to_numpy()})
    return out


def histogram_bin(r):
    """Bin of each R_v: [k/20, (k+1)/20) for k from -40 to 39, below -2 in bin -41, from 2 up in bin 40.
    The edges are the floats k/20, the same numbers as 0.8 or 0.9 written in code, so that a bin
    edge splits exactly as R_v >= theta does; r * 20 alone can round across an edge."""
    r = np.asarray(r, float)
    k = np.floor(r * BINS_PER_UNIT)
    k = np.where(r < k / BINS_PER_UNIT, k - 1, k)
    k = np.where(r >= (k + 1) / BINS_PER_UNIT, k + 1, k)
    return np.clip(k, LOWEST_BIN - 1, HIGHEST_BIN).astype(int)


def grouped(rows):
    """Scores by the RQ3 groups fixed in advance, split by follows."""
    low, high = budget_cuts()
    g = pd.DataFrame({
        "follows": rows["follows"],
        "arrival group": np.searchsorted(ARRIVAL_EDGES, rows["arrival"], side="right"),
        "budget group": np.searchsorted([low, high], rows["budget"], side="right"),
        "stop group": np.minimum(rows["observed stops"], STOP_GROUPS[-1])})
    g["visitors"] = 1
    for column in ("R_v", "change A", "change C", "change T", "stops", "observed stops", "overruns"):
        g[column] = rows[column]
    g["ended early"] = (rows["stops"] < rows["observed stops"]).astype(int)
    for reason in END_REASONS:
        g[f"end: {reason}"] = (rows["end reason"] == reason).astype(int)
    bins = histogram_bin(rows["R_v"]) - (LOWEST_BIN - 1)
    for b in range(HIGHEST_BIN - LOWEST_BIN + 2):
        g[f"bin {b + LOWEST_BIN - 1}"] = (bins == b).astype(int)
    keys = ["follows", "arrival group", "budget group", "stop group"]
    return g.groupby(keys).sum().reset_index().rename(columns={"R_v": "sum R_v", "overruns": "overrun records"})


def share_at_least(table, theta):
    """Share of visitors with R_v >= theta from grouped histograms. Only the bin edges from -2 to 2
    can be answered exactly: beyond them the open end bins hold everything, so other values are refused."""
    if not math.isfinite(theta) or not -2 <= theta <= 2:
        raise ValueError(f"theta {theta} is outside -2 to 2, where the histogram cannot answer exactly")
    edge = round(theta * BINS_PER_UNIT)
    if abs(edge - theta * BINS_PER_UNIT) > 1e-9:
        raise ValueError(f"theta {theta} is not a bin edge (a multiple of 0.05)")
    columns = [c for c in table.columns if c.startswith("bin ") and int(c.split()[1]) >= edge]
    return table[columns].to_numpy().sum() / table["visitors"].sum()


def hourly_peaks(day, crowding):
    """Each site's highest hourly C and its hour."""
    return pd.DataFrame({"site": day.site_ids, "peak C": crowding.max(axis=1), "hour": crowding.argmax(axis=1)})
