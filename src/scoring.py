"""
Scoring shared by the replay and the simulation (Step 3)
========================================================

Crowding by hour from exact person-minutes, and the parts of a route's value:
attractiveness A, crowding C and walking T. Section 7 of simulator_design.md.

Times are minutes after midnight. Walking is kept in minutes until a route is
scored, where it is turned into hours once.

score_day() scores one simulated day (Step 3f): R_v of every evaluated
visitor, follower or not, against the observed baseline, and G_hat of the
visits at the sites available that day. Section 7.2 of simulator_design.md.
"""

import numpy as np
import pandas as pd

from preprocess import gini_hat

HOURS = 24
DAY_END = 60 * HOURS  # the hourly arrays cover 0 to 24 h; anything outside is refused, not cut


def hourly_person_minutes(site, start, end, n_sites):
    """Person-minutes at each site in each clock hour, from occupancy intervals [start, end].
    Intervals are not cut at any closing time; a zero-length interval adds nothing."""
    site, start, end = np.asarray(site), np.asarray(start, float), np.asarray(end, float)
    if not (np.isfinite(start).all() and np.isfinite(end).all()) or (end < start).any():
        raise ValueError("occupancy intervals must be finite and end after they start")
    if (start < 0).any() or (end > DAY_END).any():
        raise ValueError("an occupancy interval falls outside 0 to 24 h; it would be cut silently")
    minutes = np.zeros((n_sites, HOURS))
    for hour in range(HOURS):
        overlap = np.minimum(end, 60 * (hour + 1)) - np.maximum(start, 60 * hour)
        np.add.at(minutes[:, hour], site, np.clip(overlap, 0, None))
    return minutes


def crowding(person_minutes, usual_peak):
    """C for each site and hour: people inside on average in that hour, over the usual peak."""
    return person_minutes / 60 / np.asarray(usual_peak)[:, None]


def route_parts(sites, starts, c, attractiveness, walk):
    """A, C and T of one route. C is read for the hour in which each visit starts; T is in hours."""
    sites = np.asarray(sites)
    starts = np.asarray(starts, float)
    if (starts < 0).any() or (starts >= DAY_END).any():
        raise ValueError("a visit starts outside the day")
    hours = (starts // 60).astype(int)
    a = attractiveness[sites].sum()
    c_sum = c[sites, hours].sum()
    t = walk[sites[:-1], sites[1:]].sum() / 60
    return a, c_sum, t


def route_value(a, c, t, alpha, beta):
    return a - alpha * c - beta * t


def relative_value(value_new, value_observed, a_observed):
    """R_v = 1 + (V(r^N) - V(r^O)) / sum of A on the observed route."""
    return 1 + (value_new - value_observed) / a_observed


# ---------------------------------------------------------------------- #
# A simulated day (Step 3f)
# ---------------------------------------------------------------------- #

def new_route(day, run, k):
    """Sites and start times of visitor k's route in the run: the simulated one for a follower,
    the observed one for anyone else."""
    if day.visitors["follows"][k]:
        route = run.routes[k]
        return [j for j, _, _ in route], [start for _, start, _ in route]
    return day.routes[k], day.entries[k]


def score_day(day, run, observed, alpha, beta):
    """One row per evaluated visitor: the parts of the new and the observed route, V, R_v,
    stops and how the route ended; and G_hat of the day. observed is replay() of the same date,
    data and usual peaks; who follows and the budget may differ, as they do not change the baseline."""
    same = (observed.date == day.date and observed.data_version == day.data_version
            and np.array_equal(observed.usual_peak, day.usual_peak)
            and np.array_equal(observed.visitor_ids, day.visitors["visitor"].to_numpy()))
    if not same:
        raise ValueError("the observed baseline belongs to another day, data version or usual peak")
    site, start, end = (np.array(column, float) for column in zip(*run.intervals))
    c = crowding(hourly_person_minutes(site.astype(int), start, end, len(day.site_ids)), day.usual_peak)
    overruns = pd.Series([k for k, _, _ in run.overruns]).value_counts()
    rows = []
    for k in day.visitors.index[day.visitors["evaluated"]]:
        sites, starts = new_route(day, run, k)
        a, c_sum, t = route_parts(sites, starts, c, day.attractiveness, day.walk)
        rows.append((k, bool(day.visitors["follows"][k]), a, c_sum, t, len(sites), len(day.routes[k]),
                     run.end_reason.get(k, "not a follower"), int(overruns.get(k, 0))))
    v = pd.DataFrame(rows, columns=["visitor", "follows", "A", "C", "T", "stops", "observed stops",
                                    "end reason", "overruns"]).set_index("visitor")
    b = observed.baseline.loc[v.index]
    v["V"] = route_value(v["A"], v["C"], v["T"], alpha, beta)
    v["V observed"] = route_value(b["A"], b["C"], b["T"], alpha, beta)
    v["R_v"] = relative_value(v["V"], v["V observed"], b["A"])
    for part in ("A", "C", "T"):
        v[f"change {part}"] = v[part] - b[part]
    g_hat = gini_hat(run.visits[day.available])
    return v, g_hat, c


def summary(v, alpha, beta):
    """The day for all evaluated visitors together, then for followers and for non-followers.
    Means and completion are pooled over the visitors of each group, not averaged over groups."""
    groups = {"all": v, "followers": v[v["follows"]], "non-followers": v[~v["follows"]]}
    out = {}
    for name, g in groups.items():
        if g.empty:
            continue
        change = g["change A"] - alpha * g["change C"] - beta * g["change T"]
        out[name] = {"visitors": len(g), "mean R_v": g["R_v"].mean(),
                     "completion": g["stops"].sum() / g["observed stops"].sum(),
                     "ended early": int((g["stops"] < g["observed stops"]).sum()),
                     "stops lost": int((g["observed stops"] - g["stops"]).sum()),
                     "overruns": int(g["overruns"].sum()),
                     "mean change V": (g["V"] - g["V observed"]).mean(), "mean change A": g["change A"].mean(),
                     "mean change C": g["change C"].mean(), "mean change T": g["change T"].mean(),
                     "split adds up": bool(np.allclose(change, g["V"] - g["V observed"])),
                     "end reasons": g["end reason"].value_counts().to_dict()}
    return out
