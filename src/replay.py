"""
Observed replay (Step 3b)
=========================

Replays a day as it was recorded: every visitor keeps the cleaned entries,
each with its fixed occupancy interval. Nothing is sampled, reordered or
filtered. Section 3 of simulator_design.md.

The result is the baseline of every evaluated visitor, fixed for the day and
independent of any rule: the parts A, C and T of the observed route, from
which V(r^O) follows for any alpha and beta.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from day import validate
from scoring import crowding, hourly_person_minutes, route_parts, route_value


@dataclass
class Replay:
    entries_per_site: np.ndarray
    person_minutes: np.ndarray   # sites x 24 hours
    crowding: np.ndarray         # sites x 24 hours
    baseline: pd.DataFrame       # one row per evaluated visitor: A, C, T


def replay(day):
    validate(day)
    fixed = day.fixed
    n_sites = len(day.site_ids)
    person_minutes = hourly_person_minutes(fixed["site"], fixed["entry"], fixed["exit"], n_sites)
    c = crowding(person_minutes, day.usual_peak)
    entries_per_site = np.bincount(fixed["site"], minlength=n_sites)

    rows = []
    for k in day.visitors.index[day.visitors["evaluated"]]:
        a, c_sum, t = route_parts(day.routes[k], day.entries[k], c, day.attractiveness, day.walk)
        rows.append((k, a, c_sum, t))
    baseline = pd.DataFrame(rows, columns=["visitor", "A", "C", "T"]).set_index("visitor")
    assert (baseline["A"] > 0).all(), "R_v divides by the sum of A on the observed route"
    return Replay(entries_per_site, person_minutes, c, baseline)


def baseline_value(result, alpha, beta):
    """V(r^O) of every evaluated visitor for these weights."""
    b = result.baseline
    return route_value(b["A"], b["C"], b["T"], alpha, beta)  # b.T would be the transpose
