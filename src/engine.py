"""
Event engine for one simulated day (Steps 3c and 3d)
====================================================

Runs one day of the city: followers choose their next stops by a policy;
everyone else enters and leaves at the observed times. Section 4 of
simulator_design.md and section 3 of implementation_protocol.md.

One heap of events keyed by (time, type, visitor, entry index). At the same
time, visit ends come first, then arrivals, starts of visits and decisions:
a deciding visitor has already left its site and is not yet on the way.

A policy has one method, choose(engine, visitor, candidates), and returns one
of the candidate sites. Candidates are site indices in ascending order, which
is ascending site id, so taking the first of equal scores breaks ties by the
lowest site id. Only a policy with may_stop = True (the observed-order
control) may return None to end a visitor's day; for any other policy that is
an error, as the model has no stop action. A policy with per-run state has a
reset() method, called at the start of every run.

What a policy may read, so that it never sees the future: engine.now,
engine.inside, engine.on_the_way, engine.visits, engine.run.entries (entries
already processed), the visitor's place, stops_used, visited, start and
budget_end, and the day's site statistics, walking times and usual curves.
Never engine.stays, engine.heap or engine.run.intervals: they hold stays,
events and fixed intervals that have not happened yet. Read the counts, not
engine.state, which can lag by one event at the same time.

Usage:  run = simulate(load_day("2019-10-09"), Nearest())
"""

import heapq
from dataclasses import dataclass, field

import numpy as np

from day import Day, validate

STAYS = 1  # named random stream for stays (section 5 of the design)
VISIT_END, ARRIVAL, START, DECISION = 0, 1, 2, 3
MIN_STAY = 1.0  # minutes


def stay_table(day, replicate):
    """Stays for every visitor and site, drawn before the day starts. The rows follow the
    visitor numbers of load_day(), so a rule or a visiting order never changes a draw."""
    rng = np.random.default_rng([STAYS, day.date.toordinal(), replicate])
    z = rng.standard_normal((len(day.visitors), len(day.site_ids)))
    return np.maximum(MIN_STAY, day.stay * (1 + 0.2 * z))


@dataclass
class DayRun:
    """What happened on the day. Sites and visitors are the indices of the Day."""
    routes: dict[int, list[tuple[int, float, float]]] = field(default_factory=dict)  # follower -> (site, start, stay)
    end_reason: dict[int, str] = field(default_factory=dict)  # "limit", "none feasible", "no observed stop left"
    skips: list[tuple[int, int, str]] = field(default_factory=list)      # (follower, site, reason) passed over
    overruns: list[tuple[int, int, str]] = field(default_factory=list)   # (follower, site, "closing" or "budget")
    entries: list[tuple[int, float]] = field(default_factory=list)       # (site, time) of every entry, in order
    intervals: list[tuple[int, float, float]] = field(default_factory=list)  # (site, start, end), for scoring
    visits: np.ndarray | None = None                                     # x: entries per site at the end
    events: int = 0
    trace: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)  # what produced this run, saved with any result
    fallbacks: dict = field(default_factory=dict)  # state inputs that fell back, by reason (features.py)
    decisions: list = field(default_factory=list)  # RQ2 decision log, when a rule keeps one


class Engine:
    def __init__(self, day, policy, stays, stop_limit=True, trace=False, check=False):
        self.day, self.policy, self.stays = day, policy, stays
        self.use_stop_limit, self.tracing, self.checking = stop_limit, trace, check
        n_sites, n_visitors = len(day.site_ids), len(day.visitors)
        # Running state of this run. The day's numeric arrays are read-only; its tables and routes
        # are shared and must not be changed.
        self.now = 0.0
        self.inside = np.zeros(n_sites, int)
        self.on_the_way = np.zeros(n_sites, int)
        self.visits = np.zeros(n_sites, int)          # x: entries so far, everyone
        self.place = np.full(n_visitors, -1)
        self.stops_used = np.zeros(n_visitors, int)
        self.visited = np.zeros((n_visitors, n_sites), bool)
        self.state = {}                                # follower -> not started, walking, waiting, visiting, finished
        self.start = np.array([e[0] for e in day.entries], float)
        self.budget_end = self.start + day.visitors["budget_limit"].to_numpy(float)
        self.stop_limit = day.visitors["stop_limit"].to_numpy()
        self.heap = []
        self.run = DayRun()

    # ------------------------------------------------------------------ #
    # Queue
    # ------------------------------------------------------------------ #

    def push(self, time, kind, visitor, entry_index, site):
        heapq.heappush(self.heap, (time, kind, visitor, entry_index, site))

    def queue_the_day(self):
        visitors = self.day.visitors
        for k in visitors.index[visitors["follows"]]:
            # The first stop is exogenous: a start of visit at the observed time, no walk before it.
            self.push(self.day.entries[k][0], START, k, 0, self.day.routes[k][0])
            self.state[k] = "not started"
        # Visitors who do not follow keep their observed intervals; followers' intervals come from the run.
        follows = visitors["follows"].to_numpy()
        fixed_visitor_indices = self.day.fixed["visitor"].to_numpy()
        fixed = self.day.fixed.loc[~follows[fixed_visitor_indices]]
        rows = zip(fixed["visitor"], fixed["entry_index"], fixed["site"], fixed["entry"], fixed["exit"])
        for k, i, j, entry, exit_ in rows:
            self.push(entry, START, k, i, j)
            if exit_ > entry:  # a zero-length interval adds a visit but no one inside
                self.push(exit_, VISIT_END, k, i, j)
            self.run.intervals.append((j, entry, exit_))

    # ------------------------------------------------------------------ #
    # Feasibility (section 4.3 of the design)
    # ------------------------------------------------------------------ #

    def reasons(self, k):
        """For every site, why it cannot be the next stop of follower k now: "" if feasible."""
        day, here = self.day, self.place[k]
        start = np.maximum(self.now + day.walk[here], day.opening)
        finish = start + day.stay  # expected stays
        reason = np.full(len(day.site_ids), "", dtype=object)
        reason[finish > self.budget_end[k]] = "budget"
        reason[finish > day.closing] = "closing"
        reason[self.visited[k]] = "visited"
        reason[~day.available] = "calendar"
        return reason

    def feasible(self, k):
        return np.flatnonzero(self.reasons(k) == "")

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #

    def start_visit(self, k, entry_index, j):
        self.visits[j] += 1
        self.run.entries.append((j, self.now))
        if k not in self.state:  # a fixed entry; its exit is already queued
            if self.fixed_has_length(k, entry_index):
                self.inside[j] += 1
            return
        stay = self.stays[k, j]
        self.inside[j] += 1
        self.stops_used[k] += 1
        self.visited[k, j] = True
        self.place[k] = j
        self.state[k] = "visiting"
        self.run.routes.setdefault(k, []).append((j, self.now, stay))
        self.run.intervals.append((j, self.now, self.now + stay))
        if self.stops_used[k] > 1:
            if self.now + stay > self.day.closing[j]:
                self.run.overruns.append((k, j, "closing"))
            if self.now + stay > self.budget_end[k]:
                self.run.overruns.append((k, j, "budget"))
        self.push(self.now + stay, VISIT_END, k, self.stops_used[k], j)

    def fixed_has_length(self, k, entry_index):
        row = self.fixed_lookup[(k, entry_index)]
        return row[1] > row[0]

    def end_visit(self, k, entry_index, j):
        self.inside[j] -= 1
        if k in self.state:
            self.push(self.now, DECISION, k, self.stops_used[k], j)

    def arrive(self, k, entry_index, j):
        self.on_the_way[j] -= 1
        if self.now >= self.day.opening[j]:
            self.push(self.now, START, k, entry_index, j)
        else:
            self.state[k] = "waiting"
            self.push(self.day.opening[j], START, k, entry_index, j)

    def decide(self, k):
        if self.use_stop_limit and self.stops_used[k] >= self.stop_limit[k]:
            return self.finish(k, "limit")
        candidates = self.feasible(k)
        if candidates.size == 0:
            return self.finish(k, "none feasible")
        j = self.policy.choose(self, k, candidates)
        if j is None:
            if not getattr(self.policy, "may_stop", False):
                raise ValueError("a rule must choose one of the feasible sites; it cannot stop a visitor")
            return self.finish(k, "no observed stop left")
        assert j in candidates, "a policy must choose a feasible site"
        self.on_the_way[j] += 1
        self.state[k] = "walking"
        self.push(self.now + self.day.walk[self.place[k], j], ARRIVAL, k, self.stops_used[k] + 1, j)

    def finish(self, k, reason):
        self.state[k] = "finished"
        self.run.end_reason[k] = reason
        if hasattr(self.policy, "finish"):
            self.policy.finish(self, k)

    # ------------------------------------------------------------------ #
    # The day
    # ------------------------------------------------------------------ #

    def simulate(self):
        fixed = self.day.fixed
        rows = zip(fixed["visitor"], fixed["entry_index"], fixed["entry"], fixed["exit"])
        self.fixed_lookup = {(k, i): (entry, exit_) for k, i, entry, exit_ in rows}
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.queue_the_day()
        handlers = {VISIT_END: self.end_visit, ARRIVAL: self.arrive, START: self.start_visit}
        while self.heap:
            time, kind, k, entry_index, j = heapq.heappop(self.heap)
            assert time >= self.now, "events never go back in time"
            self.now = time
            if kind == DECISION:
                self.decide(k)
            else:
                handlers[kind](k, entry_index, j)
            self.run.events += 1
            if self.tracing:
                self.run.trace.append(self.describe(kind, k, j))
            if self.checking:
                assert (self.inside >= 0).all() and (self.on_the_way >= 0).all()
        if self.checking:
            assert not self.inside.any() and not self.on_the_way.any(), "everyone has left"
            assert all(state == "finished" for state in self.state.values())
        self.run.visits = self.visits
        return self.run

    def describe(self, kind, k, j):
        names = {VISIT_END: "visit end", ARRIVAL: "arrival", START: "start", DECISION: "decision"}
        who = f"v{k}" + ("" if k in self.state else " (fixed)")
        return (f"{self.now:8.2f}  {names[kind]:9}  {who:12} site {j:2}  "
                f"inside {self.inside.tolist()}  on the way {self.on_the_way.tolist()}")


def simulate(day: Day, policy, replicate: int = 0, stays: np.ndarray | None = None, stop_limit: bool = True,
             trace: bool = False, check: bool = False) -> DayRun:
    """One day. stays overrides the drawn table (for hand-built tests). policy follows the contract
    in the module docstring."""
    validate(day)
    given = stays is not None
    stays = np.asarray(stays, float) if given else stay_table(day, replicate)
    if stays.shape != (len(day.visitors), len(day.site_ids)):
        raise ValueError("the stay table needs one row per visitor and one column per site")
    if not np.isfinite(stays).all() or (stays < MIN_STAY).any():
        raise ValueError(f"stays must be finite and at least {MIN_STAY:g} minute")
    run = Engine(day, policy, stays, stop_limit, trace, check).simulate()
    run.settings = {"date": f"{day.date:%Y-%m-%d}", "policy": type(policy).__name__, "replicate": replicate,
                    "stays given": given, "stop_limit": stop_limit, **day.options,
                    "data_version": day.data_version}
    return run


# ---------------------------------------------------------------------- #
# Policies used to test the engine (rules come in Step 4)
# ---------------------------------------------------------------------- #

class Nearest:
    """The feasible site with the shortest walk; ties to the lowest site id."""
    def choose(self, engine, k, candidates):
        walk = engine.day.walk[engine.place[k], candidates]
        return candidates[int(np.argmin(walk))]


class ObservedOrder:
    """The observed-order control (3d): the first feasible stop further along the observed route.
    A stop passed over is dropped for good, with its reason saved when it is first passed over."""
    may_stop = True

    def __init__(self):
        self.position = {}

    def reset(self):
        """A new run starts at the second observed stop of every visitor."""
        self.position = {}

    def choose(self, engine, k, candidates):
        route = engine.day.routes[k]
        reasons = engine.reasons(k)
        position = self.position.get(k, 1)
        while position < len(route):
            j = route[position]
            position += 1
            if reasons[j] == "":
                self.position[k] = position
                return j
            engine.run.skips.append((k, j, reasons[j]))
        self.position[k] = position
        return None

    def finish(self, engine, k):
        """Stops never scanned get a reason now, so every lost stop is counted once."""
        route = engine.day.routes[k]
        reasons = engine.reasons(k)
        position = self.position.get(k, 1)
        for j in route[position:]:
            if not engine.visited[k, j]:
                engine.run.skips.append((k, j, reasons[j] or "no time left"))
        self.position[k] = len(route)
