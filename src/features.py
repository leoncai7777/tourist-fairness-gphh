"""
State inputs of the rules and the decision log (Step 3e)
========================================================

The terminals of h1 (one value per candidate site) and of h2 (one value per
decision), VS for the fairness term, and the tie rule. Section 6 of
simulator_design.md and section 5 of implementation_protocol.md.

Everything here reads only what the policy contract in engine.py allows:
the clock, the counts, the entries already processed, the visitor's own
place, stops, start and budget, and the day's fixed statistics.

GDAY and ARR fall back to 1 ("as usual") when they are not defined, and
every fallback is counted by reason in run.fallbacks.
"""

import bisect

import numpy as np

from preprocess import FIRST_MINUTE, LAST_MINUTE, gini_hat

H1_TERMINALS = ["AV", "DUR", "TA", "TO", "TC", "TSV", "TFV", "SL", "RemT", "CR", "HEAD"]
H2_TERMINALS = ["TLD", "ARR", "MCR", "GDAY", "RemT", "STOPS"]
MINUTES = np.arange(FIRST_MINUTE, LAST_MINUTE + 1)


def count(engine, reason, n=1):
    engine.run.fallbacks[reason] = engine.run.fallbacks.get(reason, 0) + n


def usual_value(engine, curve, name):
    """The usual curve at the current time, linear between neighbouring minutes; None if not usable."""
    t = engine.now
    if t < FIRST_MINUTE or t > LAST_MINUTE:
        count(engine, f"{name}: outside 08:00-21:00")
        return None
    value = np.interp(t, MINUTES, curve)
    if not np.isfinite(value) or value <= 0:
        count(engine, f"{name}: usual value not positive")
        return None
    return value


def h1_inputs(engine, k, candidates):
    """Terminals of h1 for each candidate, in the order of H1_TERMINALS."""
    day, t, here = engine.day, engine.now, engine.place[k]
    to = day.opening[candidates] - t
    tc = day.closing[candidates] - t
    ta = day.walk[here, candidates]
    tsv = np.maximum(to, ta)
    dur = day.stay[candidates]
    remaining = np.full(len(candidates), engine.budget_end[k] - t)
    peak = day.usual_peak[candidates]
    return [day.attractiveness[candidates], dur, ta, to, tc, tsv, tsv + dur, tc - ta, remaining,
            engine.inside[candidates] / peak, engine.on_the_way[candidates] / peak]


def arrivals(engine):
    """ARR: entries at all sites in (t - 60, t], over the usual number at t."""
    entries = engine.run.entries  # in time order, and only those already processed
    recent = len(entries) - bisect.bisect_right(entries, engine.now - 60, key=lambda entry: entry[1])
    usual = usual_value(engine, engine.day.usual_arrivals, "ARR")
    return 1.0 if usual is None else recent / usual


def gday(engine):
    """GDAY: G_hat of the visits so far at the available sites, over the usual value at t."""
    g = gini_hat(engine.visits[engine.day.available])
    if not np.isfinite(g):
        count(engine, "GDAY: no visits yet at the available sites")
        return 1.0
    usual = usual_value(engine, engine.day.usual_gday, "GDAY")
    return 1.0 if usual is None else g / usual


def h2_inputs(engine, k):
    """Terminals of h2, in the order of H2_TERMINALS; the same for every candidate."""
    day, t = engine.day, engine.now
    available = day.available
    return [day.closing[available].max() - t, arrivals(engine),
            (engine.inside[available] / day.usual_peak[available]).mean(), gday(engine),
            engine.budget_end[k] - t, float(engine.stops_used[k])]


def visit_share(engine, candidates):
    """VS: visits so far over their mean at the available sites; 1 before any visit there."""
    mean = engine.visits[engine.day.available].mean()
    if mean == 0:
        return np.ones(len(candidates))
    return engine.visits[candidates] / mean


def finite(engine, values, name):
    """A rule's output with every non-finite value set to 0 and counted (protocol section 5)."""
    values = np.asarray(values, float)
    bad = ~np.isfinite(values)
    if bad.any():
        count(engine, f"{name}: not finite", int(bad.sum()))
        values = np.where(bad, 0.0, values)
    return values


def best(engine, candidates, scores):
    """The candidate with the highest score; equal scores go to the lowest site id."""
    top = candidates[scores == scores.max()]
    return top[np.argmin(engine.day.site_ids[top])]


def log_decision(engine, k, candidates, h1, w, vs, h2):
    """One row of the RQ2 decision log: the choice with and without the fairness term. j0 is a
    shadow choice: it is never acted on and changes nothing."""
    j0 = best(engine, candidates, h1)
    jw = best(engine, candidates, h1 - w * vs)
    position = {j: i for i, j in enumerate(candidates)}
    row = {"visitor": k, "time": engine.now, "candidates": len(candidates), "j0": int(j0), "jw": int(jw),
           "changed": bool(j0 != jw), "VS change": vs[position[jw]] - vs[position[j0]],
           "h1 given up": h1[position[j0]] - h1[position[jw]], "w": w}
    row.update(zip(H2_TERMINALS, h2) if h2 is not None else [])
    engine.run.decisions.append(row)
    return jw
