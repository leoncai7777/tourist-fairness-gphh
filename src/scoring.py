"""
Scoring shared by the replay and the simulation (Step 3)
========================================================

Crowding by hour from exact person-minutes, and the parts of a route's value:
attractiveness A, crowding C and walking T. Section 7 of simulator_design.md.

Times are minutes after midnight. Walking is kept in minutes until a route is
scored, where it is turned into hours once.
"""

import numpy as np

HOURS = 24
DAY_END = 60 * HOURS  # a one-day model: nothing may start or end after midnight


def hourly_person_minutes(site, start, end, n_sites):
    """Person-minutes at each site in each clock hour, from occupancy intervals [start, end].
    Intervals are not cut at any closing time; a zero-length interval adds nothing."""
    site, start, end = np.asarray(site), np.asarray(start, float), np.asarray(end, float)
    if (start < 0).any() or (end > DAY_END).any():
        raise ValueError("an occupancy interval falls outside the day; it would be cut silently")
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
