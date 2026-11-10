"""
Stochastic TOPTW for one tourist (Step 1)
=========================================

The simulation of Mei and Zhang (2018), Algorithm 2: after each visit, the
policy scores every feasible next point and the tourist goes to the highest.
Feasibility and the policy's inputs use the expected visit length; the clock
moves on with the sampled one.

Instances: Righini and Salani (2009) and Montemanni and Gambardella (2009),
in the KU Leuven format, kept in data/toptw/.

A policy takes one numpy array per terminal, with one value per candidate,
and returns their priorities. The terminals, in order: SCORE, DUR, TO, TC,
TA, TR, TSV, TFV, SL, RemT (Table I of the paper).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TEST_SEED = 20180101
counters = {"non_finite": 0}  # priorities that were not finite and were set to 0


# ---------------------------------------------------------------------- #
# 1. Instances
# ---------------------------------------------------------------------- #

@dataclass
class Instance:
    name: str
    score: np.ndarray
    duration: np.ndarray
    opening: np.ndarray
    closing: np.ndarray
    travel: np.ndarray


def load_instance(path, max_points=None):
    """Point 0 is both the start and the end. Distances are Euclidean, rounded down to one
    decimal for Solomon-type instances and two for Cordeau-type (pr) instances.
    max_points keeps only the first points after the start: Table III of Mei and Zhang (2018)
    matches the first 100 points of the larger pr instances, not the full files."""
    path = Path(path)
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    rows = lines[2:]  # the first two lines are a header
    if max_points is not None:
        rows = rows[:max_points + 1]
    # Each row: number, x, y, visit length, score, ..., opening, closing.
    points = np.array([[float(a[1]), float(a[2]), float(a[3]), float(a[4]), float(a[-2]), float(a[-1])]
                       for a in rows])
    x, y = points[:, 0], points[:, 1]
    distance = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    factor = 10 ** (2 if path.stem.startswith("pr") else 1)
    travel = np.floor(distance * factor + 1e-9) / factor
    return Instance(path.stem, points[:, 3], points[:, 2], points[:, 4], points[:, 5], travel)


def sample_durations(instance, rng, sigma=0.2):
    """Visit lengths drawn from N(d, 0.2 d); a negative draw is set to 0."""
    return np.maximum(0.0, rng.normal(instance.duration, sigma * instance.duration))


# ---------------------------------------------------------------------- #
# 2. Simulation
# ---------------------------------------------------------------------- #

def solve(instance, policy, durations, days=1):
    """Run one policy on one sampled instance. Returns the total score and the list of stops."""
    start_time, end_time = instance.opening[0], instance.closing[0]
    expected = instance.duration
    back = instance.travel[:, 0]
    unvisited = np.ones(len(expected), dtype=bool)
    unvisited[0] = False
    day, here, now, total, stops = 0, 0, start_time, 0.0, []

    while day < days:
        walk = instance.travel[here]
        arrive = now + walk
        # Reachable before closing, and back at the end point in time after an expected visit.
        feasible = unvisited & (arrive <= instance.closing)
        feasible &= np.maximum(arrive, instance.opening) + expected + back <= end_time
        candidates = np.flatnonzero(feasible)
        if candidates.size == 0:
            day, here, now = day + 1, 0, start_time
            continue

        to_open = instance.opening[candidates] - now
        to_close = instance.closing[candidates] - now
        to_arrive = walk[candidates]
        to_start = np.maximum(to_open, to_arrive)
        to_finish = to_start + expected[candidates]
        slack = to_close - to_arrive
        remaining = (days - day - 1) * (end_time - start_time) + (end_time - now)
        terminals = (instance.score[candidates], expected[candidates], to_open, to_close, to_arrive,
                     back[candidates], to_start, to_finish, slack, np.full(candidates.size, remaining))
        priority = np.asarray(policy(*terminals), dtype=float)

        bad = ~np.isfinite(priority)
        if bad.any():
            counters["non_finite"] += int(bad.sum())
            priority[bad] = 0.0

        chosen = candidates[int(np.argmax(priority))]  # ties go to the earliest point
        unvisited[chosen] = False
        total += instance.score[chosen]
        stops.append(int(chosen))
        now = max(arrive[chosen], instance.opening[chosen]) + durations[chosen]
        here = int(chosen)
    return total, stops


def test_score(instance, policy, days=1, samples=500, seed=TEST_SEED):
    """Mean score over the same sampled instances for every policy."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(samples):
        scores.append(solve(instance, policy, sample_durations(instance, rng), days)[0])
    return float(np.mean(scores))


# ---------------------------------------------------------------------- #
# 3. Protected division and the hand-written rules
# ---------------------------------------------------------------------- #

def protected_div(a, b):
    """Division that returns 1 when dividing by zero (Mei and Zhang 2018)."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.ones_like(a), where=b != 0)


def sos(score, dur, to, tc, ta, tr, tsv, tfv, sl, rem):
    """Score over slack."""
    return protected_div(score, sl)


def sot(score, dur, to, tc, ta, tr, tsv, tfv, sl, rem):
    """Score over time to arrive."""
    return protected_div(score, ta)
