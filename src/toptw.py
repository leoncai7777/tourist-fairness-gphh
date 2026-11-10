"""Stochastic TOPTW for one tourist, following Mei and Zhang (2018), Algorithm 2.

Instances: Righini and Salani (2009) and Montemanni and Gambardella (2009), KU Leuven format.
A policy gets one numpy array per terminal, one value per candidate, and returns the priorities.
"""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Instance:
    name: str
    score: np.ndarray
    duration: np.ndarray
    opening: np.ndarray
    closing: np.ndarray
    travel: np.ndarray


def load_instance(path, max_points=None):
    """Point 0 is both the start and the end. Distances are Euclidean, rounded down
    to one decimal for Solomon-type instances and two for Cordeau-type (pr) instances.
    max_points keeps only the first points after the start: Table III of Mei and Zhang (2018)
    matches the first 100 points of the larger pr instances, not the full files."""
    path = Path(path)
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    points = np.array([[float(a[1]), float(a[2]), float(a[3]), float(a[4]), float(a[-2]), float(a[-1])]
                       for a in lines[2:(None if max_points is None else 3 + max_points)]])
    decimals = 2 if path.stem.startswith("pr") else 1
    x, y = points[:, 0], points[:, 1]
    distance = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    factor = 10 ** decimals
    travel = np.floor(distance * factor + 1e-9) / factor
    return Instance(path.stem, points[:, 3], points[:, 2], points[:, 4], points[:, 5], travel)


def sample_durations(instance, rng, sigma=0.2):
    return np.maximum(0.0, rng.normal(instance.duration, sigma * instance.duration))


def solve(instance, policy, durations, days=1):
    """Run one policy on one sampled instance. Feasibility and terminals use the expected duration.
    Returns the total score and the list of stops."""
    start_time, end_time = instance.opening[0], instance.closing[0]
    expected = instance.duration
    unvisited = np.ones(len(expected), dtype=bool)
    unvisited[0] = False
    day, here, now, total, stops = 0, 0, start_time, 0.0, []
    while day < days:
        walk = instance.travel[here]
        arrive = now + walk
        back = instance.travel[:, 0]
        feasible = unvisited & (arrive <= instance.closing) & (
            np.maximum(arrive, instance.opening) + expected + back <= end_time)
        candidates = np.flatnonzero(feasible)
        if candidates.size == 0:
            day, here, now = day + 1, 0, start_time
            continue
        to_open = instance.opening[candidates] - now
        to_close = instance.closing[candidates] - now
        ta = walk[candidates]
        start_visit = np.maximum(to_open, ta)
        remaining = (days - day - 1) * (end_time - start_time) + (end_time - now)
        priority = np.asarray(policy(instance.score[candidates], expected[candidates], to_open, to_close, ta,
                                     back[candidates], start_visit, start_visit + expected[candidates], to_close - ta,
                                     np.full(candidates.size, remaining)), dtype=float)
        priority[~np.isfinite(priority)] = 0.0
        chosen = candidates[int(np.argmax(priority))]
        unvisited[chosen] = False
        total += instance.score[chosen]
        stops.append(int(chosen))
        now = max(arrive[chosen], instance.opening[chosen]) + durations[chosen]
        here = int(chosen)
    return total, stops


def protected_div(a, b):
    """Division that returns 1 when dividing by zero (Mei and Zhang 2018)."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.ones_like(a), where=b != 0)


def sos(score, dur, to, tc, ta, tr, tsv, tfv, sl, rem):
    return protected_div(score, sl)


def sot(score, dur, to, tc, ta, tr, tsv, tfv, sl, rem):
    return protected_div(score, ta)


def test_score(instance, policy, days=1, samples=500, seed=20180101):
    rng = np.random.default_rng(seed)
    return float(np.mean([solve(instance, policy, sample_durations(instance, rng), days)[0] for _ in range(samples)]))
