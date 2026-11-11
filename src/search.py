"""
The search: NSGA-II over GP rules (Steps 4c and 4d)
===================================================

Section 4 of search_design.md (version 2.1, frozen).

Two random streams. The environment schedule (a training date and a replicate
for every generation) is made from the run number alone, with NumPy, and saved;
every method with the same run number reads the same schedule. The search
(initialisation, selection, crossover, mutation) draws only from Python's
random module, seeded once per run from the method and run numbers; nothing
else in a run draws from it.

One generation: score the whole population on the generation's date and
replicate; rank it; choose parents by binary tournament on rank and crowding
distance; breed N children; score them on the same date and replicate; keep N
of parents and children by NSGA-II. Parents are always chosen on scores from
the current environment.

Usage:  run = Run("GP-adaptive", run_number=1, settings=SETTINGS);  run.evolve()
"""

import copy
import hashlib
import json
import math
import pickle
import random
import time
from functools import partial
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
from deap import gp, tools

from day import PROCESSED
from evaluate import Days, evaluate, grouped, hourly_peaks, visitor_rows
from gp_rule import (H2_SET, HIGH, LOW, MAX_HEIGHT, METHODS, constants, from_text, identity, new_individual, size,
                     to_text, tree_sets)
from single_gphh import fingerprint, grow

SCHEDULE = 3  # named random stream for the environment schedule (STAYS is 1, COMPLIANCE 2)

# Planning values (search_design.md, sections 4 and 6); the pilot fixes them and F2 freezes them in a file.
SETTINGS = {"population": 200, "generations": 100, "crossover": 0.8, "mutation": 0.15,
            "monitor every": 10, "validation replicates": 5}

# The definitions a run is scored under, with their provisional values. A run's settings always hold all of
# them (complete() fills the missing ones), so they are part of its settings hash: results under other
# definitions can never be pooled with these, and a checkpoint under other definitions never resumes.
DEFINITIONS = {"budget": "corrected", "same day": False, "peak floor": None, "stop limit": True,
               "alpha": 0.25, "beta": 0.5, "R range": (0.0, 1.5), "G range": (0.0, 1.0)}


def complete(settings):
    """The settings with every definition present; ranges as tuples, so that nothing can be changed in place."""
    out = {**DEFINITIONS, **dict(settings)}
    for name in ("R range", "G range"):
        out[name] = tuple(float(v) for v in out[name])
    return out


def load_settings(path="configs/search.json"):
    """The frozen settings of the formal runs, from their file. Unlike runs made in code, nothing is filled
    in: the file must name every setting and every definition, and nothing else, so that what was run is
    read from the file alone. The budget cut points written there must be the ones the data gives, so that
    the RQ3 groups are the ones the file says."""
    from evaluate import budget_cuts
    data = json.loads(Path(path).read_text())
    required = set(SETTINGS) | set(DEFINITIONS)
    missing, unknown = required - set(data["settings"]), set(data["settings"]) - required
    if missing or unknown:
        raise ValueError(f"{path}: missing {sorted(missing)}, unknown {sorted(unknown)}")
    settings = complete(data["settings"])
    check_settings(settings)
    if tuple(data["budget cuts"]) != budget_cuts():
        raise ValueError(f"the budget cut points of {path} are not those of the data: {budget_cuts()}")
    return settings


def days_for(settings):
    """Days loaded under the run's definitions."""
    return Days(budget=settings["budget"], same_day=settings["same day"], peak_floor=settings["peak floor"])


def scoring(settings):
    """How every evaluation of the run is scored."""
    return {"alpha": settings["alpha"], "beta": settings["beta"], "stop_limit": settings["stop limit"]}


FIRST_TRAINING_REPLICATE = 1000  # validation and test use replicates 0..K-1, so K must be at most 1000


def settings_hash(settings):
    return hashlib.sha1(json.dumps(dict(settings), sort_keys=True).encode()).hexdigest()[:10]


def code_version():
    """Hash of every Python file in src/. A run depends on more than the files it imports directly
    (grow() in single_gphh.py, protected division in toptw.py, gini_hat() in preprocess.py), so all
    of them are taken; an unrelated edit also makes old checkpoints refuse to resume, which is safe."""
    return fingerprint(*sorted(Path(__file__).parent.glob("*.py")))


def data_version():
    return json.loads((PROCESSED / "manifest.json").read_text())["version"]


def check_settings(settings):
    """Refuse settings that would quietly change the algorithm, before anything runs."""
    def whole(name, low):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise ValueError(f"{name} must be a whole number of at least {low}, not {value!r}")
    whole("population", 2)
    whole("generations", 1)
    whole("monitor every", 1)
    whole("validation replicates", 1)
    if settings["validation replicates"] > FIRST_TRAINING_REPLICATE:
        raise ValueError("validation and test replicates would reach the training replicates")
    for name in ("crossover", "mutation"):
        if not (isinstance(settings[name], (int, float)) and math.isfinite(settings[name]) and settings[name] >= 0):
            raise ValueError(f"{name} must be a finite number of at least 0")
    if settings["crossover"] + settings["mutation"] > 1:
        raise ValueError("crossover and mutation add up to more than 1")
    if settings["budget"] not in ("corrected", "span"):
        raise ValueError(f"budget must be 'corrected' or 'span', not {settings['budget']!r}")
    for name in ("same day", "stop limit"):
        if not isinstance(settings[name], bool):
            raise ValueError(f"{name} must be true or false")
    floor = settings["peak floor"]
    if floor is not None and not (isinstance(floor, (int, float)) and math.isfinite(floor) and floor > 0):
        raise ValueError("peak floor must be none or a finite positive number")
    for name in ("alpha", "beta"):
        if not (isinstance(settings[name], (int, float)) and math.isfinite(settings[name]) and settings[name] >= 0):
            raise ValueError(f"{name} must be a finite number of at least 0")
    for name in ("R range", "G range"):
        low, high = settings[name]
        if not (math.isfinite(low) and math.isfinite(high) and high > low):
            raise ValueError(f"{name} must be two finite numbers, the second larger")


def search_seed(method, run_number):
    """One integer per method and run, with no collisions for run numbers below 1000."""
    assert 0 <= run_number < 1000
    return 1_000_000 + 1000 * METHODS[method] + run_number


def training_dates():
    days = pd.read_csv(PROCESSED / "days_2019.csv")
    return sorted(days["date"][days["split"] == "train"])


def schedule(run_number, generations):
    """A training date and a replicate for every generation, from the run number alone. The dates
    are a permutation of the training days; after a full pass, a new permutation follows."""
    rng = np.random.default_rng([SCHEDULE, run_number])
    dates = training_dates()
    order = []
    while len(order) < generations:
        order += [dates[i] for i in rng.permutation(len(dates))]
    replicates = rng.integers(FIRST_TRAINING_REPLICATE, 1_000_000, generations)  # never validation or test ones
    return [(order[g], int(replicates[g])) for g in range(generations)]


# ---------------------------------------------------------------------- #
# Variation (sections 4.3 and 4.4)
# ---------------------------------------------------------------------- #

def h2_point(tree):
    """A crossover point of h2: internal nodes, the root included, with probability 0.9, else a leaf;
    if the drawn class is empty, any node. A single-node tree's only point is its root."""
    internal = [i for i, node in enumerate(tree) if isinstance(node, gp.Primitive)]
    leaves = [i for i, node in enumerate(tree) if not isinstance(node, gp.Primitive)]
    chosen = internal if random.random() < 0.9 else leaves
    return random.choice(chosen or list(range(len(tree))))


def cross_h2(a, b):
    """Subtree exchange in two h2 trees, at points that may be the roots."""
    i, j = h2_point(a), h2_point(b)
    slice_a, slice_b = a.searchSubtree(i), b.searchSubtree(j)
    a[slice_a], b[slice_b] = b[slice_b], a[slice_a]
    return a, b


def cross_h1(a, b):
    """Step 1's crossover: DEAP's leaf-biased one-point crossover, never at the root."""
    return gp.cxOnePointLeafBiased(a, b, termpb=0.1)


def mutate_tree(tree, pset):
    """Subtree mutation at any node, the root included (DEAP's mutUniform, as in Step 1). h1 and
    GP-single grow Step 1's subtree of height 4; h2 grows one of height 0 to 4."""
    if pset is H2_SET:
        return gp.mutUniform(tree, expr=partial(grow, min_=0, max_=4), pset=pset)[0]
    return gp.mutUniform(tree, expr=partial(grow, min_=4, max_=4), pset=pset)[0]


def step_constant(value, counts):
    """c * e^eps, eps ~ N(0, 1), cut to [LOW, HIGH]."""
    new = value * math.exp(random.gauss(0.0, 1.0))
    if not LOW <= new <= HIGH:
        counts["constant cuts"] += 1
        new = min(max(new, LOW), HIGH)
    return new


def too_high(rule):
    return any(isinstance(part, gp.PrimitiveTree) and part.height > MAX_HEIGHT for part in rule)


def crossover(p1, p2, counts):
    """Two children from deep copies of the parents. A child with a tree over the limit is replaced,
    whole, by a copy of one of its parents chosen at random."""
    a, b = copy.deepcopy(p1), copy.deepcopy(p2)
    method = p1.method
    if method == "GP-single":
        a[0], b[0] = cross_h1(a[0], b[0])
    elif method == "GP-adaptive":  # tree swapping: the chosen trees exchange subtrees, the others swap whole
        if random.random() < 0.5:
            a[0], b[0] = cross_h1(a[0], b[0])
            a[1], b[1] = b[1], a[1]
        else:
            a[1], b[1] = cross_h2(a[1], b[1])
            a[0], b[0] = b[0], a[0]
    else:  # GP-fixed, GP-prop: crossover on h1, the constants swapped
        a[0], b[0] = cross_h1(a[0], b[0])
        a[1], b[1] = b[1], a[1]
    children = []
    for child in (a, b):
        if too_high(child):
            counts["reverts"] += 1
            child = copy.deepcopy(random.choice((p1, p2)))
        children.append(child)
    return children


def mutation(parent, counts):
    """One child from a deep copy: one part chosen at random; a tree gets subtree mutation (h2 with
    constants: half of the time one constant's step instead); c gets the step."""
    child = copy.deepcopy(parent)
    psets = tree_sets(child.method)
    i = random.randrange(len(child))
    if psets[i] is None:
        child[i] = step_constant(child[i], counts)
    elif psets[i] is H2_SET and constants(child[i]) and random.random() < 0.5:
        position = random.choice(constants(child[i]))
        value = step_constant(child[i][position].value, counts)
        child[i][position] = gp.Terminal(value, False, psets[i].ret)
    else:
        child[i] = mutate_tree(child[i], psets[i])
    if too_high(child):
        counts["reverts"] += 1
        child = copy.deepcopy(parent)
    return child


# ---------------------------------------------------------------------- #
# Selection and breeding (section 4.2)
# ---------------------------------------------------------------------- #

def rank(population):
    """Non-dominated rank and crowding distance of every member, from its current scores."""
    for number, front in enumerate(tools.sortNondominated(population, len(population))):
        tools.emo.assignCrowdingDist(front)
        for rule in front:
            rule.rank = number


def tournament(population):
    """Binary tournament: lower rank wins, then larger crowding distance, then a coin."""
    a, b = random.sample(population, 2)
    if a.rank != b.rank:
        return a if a.rank < b.rank else b
    if a.fitness.crowding_dist != b.fitness.crowding_dist:
        return a if a.fitness.crowding_dist > b.fitness.crowding_dist else b
    return a if random.random() < 0.5 else b


def breed(population, settings, counts):
    """N children, drawn per breeding operation as in Step 1: crossover gives two, mutation and
    reproduction one; the list is cut to N in the order made."""
    n = settings["population"]
    children, kinds = [], []  # kinds: (operator, whether the child's text is new, not a parent's)
    while len(children) < n:
        r = random.random()
        if r < settings["crossover"]:
            p1, p2 = tournament(population), tournament(population)
            made = crossover(p1, p2, counts)
            kinds += [("crossover", to_text(c) not in (to_text(p1), to_text(p2))) for c in made]
        elif r < settings["crossover"] + settings["mutation"]:
            parent = tournament(population)
            made = [mutation(parent, counts)]
            kinds.append(("mutation", to_text(made[0]) != to_text(parent)))
        else:
            made = [copy.deepcopy(tournament(population))]
            kinds.append(("reproduction", False))
        children.extend(made)
    for kind, new in kinds[:n]:  # only the children kept after the cut to N
        counts[f"{kind} children"] += 1
        counts[f"{kind} new texts"] += new
    return children[:n]


# ---------------------------------------------------------------------- #
# A run
# ---------------------------------------------------------------------- #

def score(population, date, replicate, days, fallbacks, options={}, cache=None):
    """Both objectives of every rule on one day and replicate. An evaluation is deterministic, so a rule
    whose text was already scored on this day and replicate (a copy, or a child equal to its parent)
    takes the stored result: the same numbers, one simulation fewer. Fallbacks count per member."""
    cache = {} if cache is None else cache
    for rule in population:
        text = to_text(rule)
        if text not in cache:
            result = evaluate(rule, date, replicate, days, **options)
            if not (math.isfinite(result.mean_r) and math.isfinite(result.g_hat)):
                raise ValueError(f"non-finite objectives {result.mean_r}, {result.g_hat} on {date}")
            cache[text] = (result.mean_r, result.g_hat, dict(result.run.fallbacks))
        mean_r, g_hat, rule_fallbacks = cache[text]
        rule.fitness.values = (mean_r, g_hat)
        for reason, n in rule_fallbacks.items():
            fallbacks[reason] = fallbacks.get(reason, 0) + n


def monitoring_dates():
    """Seven validation days fixed in advance: weekday d (Monday 0) of validation week number d mod 5
    in the list 5, 15, 25, 35, 45 (search_design.md, section 4.5)."""
    days = pd.read_csv(PROCESSED / "days_2019.csv", parse_dates=["date"])
    validation = days[days["split"] == "validation"]
    weeks = [5, 15, 25, 35, 45]
    out = []
    for d in range(7):
        match = validation[(validation["date"].dt.isocalendar().week == weeks[d % 5]) & (validation["date"].dt.dayofweek == d)]
        out.append(f"{match['date'].iloc[0]:%Y-%m-%d}")
    return out


class Run:
    """One run of one method. The schedule is made from the run number; the search seed from the
    method and run numbers. A checkpoint after every generation lets a stopped run resume with the
    same result."""
    def __init__(self, method, run_number, settings=SETTINGS, folder=None, monitor=True):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
        settings = complete(settings)
        check_settings(settings)
        self.method, self.run_number = method, run_number
        self.settings = MappingProxyType(settings)  # fixed: the schedule below depends on it
        self.schedule = schedule(run_number, self.settings["generations"])
        self.folder = Path(folder) if folder else None
        self.monitor = monitor
        self.log, self.monitor_log, self.generation = [], [], 0
        random.seed(search_seed(method, run_number))
        self.population = [new_individual(method) for _ in range(self.settings["population"])]
        self.validation = days_for(self.settings)

    # checkpoints

    def checkpoint(self):
        if self.folder is None:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        scored_on = self.schedule[self.generation - 1] if self.generation else None
        state = {"method": self.method, "run": self.run_number, "settings": dict(self.settings),
                 "settings hash": settings_hash(self.settings), "data version": data_version(),
                 "code version": code_version(), "generation": self.generation,
                 "population": [to_text(r) for r in self.population],
                 "fitness": [r.fitness.values for r in self.population], "scored on": scored_on,
                 "random": random.getstate(), "log": self.log, "monitor log": self.monitor_log,
                 "schedule": self.schedule}
        temporary = self.folder / "checkpoint.tmp"
        temporary.write_bytes(pickle.dumps(state))
        temporary.replace(self.folder / "checkpoint.pkl")

    @classmethod
    def resume(cls, method, run_number, folder, monitor=True):
        """The run saved in folder. Its identity comes from the file; the method and run number given
        must match it, as must the data and code versions now in use. Nothing is relabelled."""
        state = pickle.loads((Path(folder) / "checkpoint.pkl").read_bytes())
        settings = state["settings"]
        check_settings(settings)
        expected = {"method": method, "run": run_number, "settings hash": settings_hash(settings),
                    "data version": data_version(), "code version": code_version()}
        for key, value in expected.items():
            if state[key] != value:
                raise ValueError(f"the checkpoint's {key} is {state[key]!r}, not {value!r}")
        if state["schedule"] != schedule(run_number, settings["generations"]):
            raise ValueError("the checkpoint's schedule is not this run's")
        if not 0 <= state["generation"] <= settings["generations"]:
            raise ValueError("the checkpoint's generation is outside the run")
        if len(state["population"]) != settings["population"] or len(state["fitness"]) != settings["population"]:
            raise ValueError("the checkpoint's population has the wrong size")
        generation = state["generation"]
        scored_on = state["schedule"][generation - 1] if generation else None
        if state["scored on"] != scored_on:
            raise ValueError(f"the checkpoint's scores come from {state['scored on']}, not {scored_on}")
        if generation and not all(state["fitness"]):
            raise ValueError("a population after generation 0 must have its scores")
        population = [from_text(text) for text in state["population"]]
        if any(rule.method != method for rule in population):
            raise ValueError("the checkpoint holds rules of another method")
        for rule, values in zip(population, state["fitness"]):
            if values:
                rule.fitness.values = values
        run = cls.__new__(cls)
        run.method, run.run_number, run.settings = method, run_number, MappingProxyType(dict(settings))
        run.schedule, run.folder, run.monitor = state["schedule"], Path(folder), monitor
        run.log, run.monitor_log, run.generation = state["log"], state["monitor log"], state["generation"]
        run.population, run.validation = population, days_for(settings)
        random.setstate(state["random"])  # set after the population is rebuilt (section 13, item 7)
        return run

    # generations

    def step(self):
        """One generation (section 4.2)."""
        started = time.time()
        date, replicate = self.schedule[self.generation]
        days = days_for(self.settings)  # this generation's day only; a run reads each training day once per pass
        counts = {"crossover children": 0, "crossover new texts": 0, "mutation children": 0,
                  "mutation new texts": 0, "reproduction children": 0, "reproduction new texts": 0,
                  "reverts": 0, "constant cuts": 0}
        fallbacks = {}
        cache = {}  # this generation's day and replicate only
        score(self.population, date, replicate, days, fallbacks, scoring(self.settings), cache)
        rank(self.population)
        children = breed(self.population, self.settings, counts)
        score(children, date, replicate, days, fallbacks, scoring(self.settings), cache)
        self.population = tools.selNSGA2(self.population + children, self.settings["population"])
        front = tools.sortNondominated(self.population, len(self.population), first_front_only=True)[0]
        texts = [to_text(r) for r in self.population]
        nodes = [sum(n for n, _ in size(r)) for r in self.population]
        self.log.append({"generation": self.generation, "date": date, "replicate": replicate,
                         "front": sorted(r.fitness.values for r in front), "mean nodes": float(np.mean(nodes)),
                         "max nodes": max(nodes), "duplicates": len(texts) - len(set(texts)),
                         "unique share": len(set(texts)) / len(texts),
                         "largest share": max(texts.count(t) for t in set(texts)) / len(texts), **counts,
                         "fallbacks": fallbacks, "evaluations": 2 * len(self.population),
                         "evaluations run": len(cache), "seconds": time.time() - started})
        if self.monitor and (self.generation + 1) % self.settings["monitor every"] == 0:
            self.watch(front)
        self.generation += 1
        self.checkpoint()

    def watch(self, front):
        """Score the generation's non-dominated rules on the seven monitoring days, replicate 0.
        Only watched: nothing here changes the run or draws a random number."""
        scores, done = [], {}
        for rule in front:
            text = to_text(rule)
            if text not in done:  # copies of a rule are scored once
                results = [evaluate(rule, date, 0, self.validation, **scoring(self.settings))
                           for date in monitoring_dates()]
                done[text] = (float(np.mean([r.mean_r for r in results])), float(np.mean([r.g_hat for r in results])))
            scores.append(done[text])
        self.monitor_log.append({"generation": self.generation, "front": sorted(scores)})

    def evolve(self, until=None):
        until = self.settings["generations"] if until is None else until
        if not 0 <= until <= self.settings["generations"]:
            raise ValueError(f"until must be within the {self.settings['generations']} generations of the schedule")
        while self.generation < until:
            self.step()
        return self.population


# ---------------------------------------------------------------------- #
# The end of a run (4d, section 5)
# ---------------------------------------------------------------------- #

# Provisional normalisation, (low, high) of mean R_v and of G_hat; the pilot (F1) sets it from training and
# validation results only, and F2 freezes it in the settings file. It is never changed after a test result.
RANGE = {"R": (0.0, 1.5), "G": (0.0, 1.0)}


def validation_dates():
    days = pd.read_csv(PROCESSED / "days_2019.csv")
    return sorted(days["date"][days["split"] == "validation"])


def non_dominated(means):
    """Indices of the points not dominated on the raw objectives: mean R_v up, G_hat down.
    Two equal points do not dominate each other."""
    keep = []
    for i, (r, g) in enumerate(means):
        dominated = any(r2 >= r and g2 <= g and (r2 > r or g2 < g) for r2, g2 in means)
        if not dominated:
            keep.append(i)
    return keep


def normalise(means, value_range=RANGE):
    """(f1, f2), both maximised, not clipped: a point better than the range has f > 1."""
    (r_lo, r_hi), (g_lo, g_hi) = value_range["R"], value_range["G"]
    if not all(math.isfinite(v) for v in (r_lo, r_hi, g_lo, g_hi)) or r_hi <= r_lo or g_hi <= g_lo:
        raise ValueError(f"a bad normalisation range: {value_range}")
    out = []
    for r, g in means:
        if not (math.isfinite(r) and math.isfinite(g)):
            raise ValueError(f"a non-finite objective ({r}, {g})")
        out.append(((r - r_lo) / (r_hi - r_lo), (g_hi - g) / (g_hi - g_lo)))
    return out


def hypervolume(points):
    """Area dominated by points (f1, f2), both maximised, above the reference point (0, 0); a point not
    strictly better than the reference in both objectives adds nothing. Returns the area and the number
    of such points."""
    inside = sorted(((f1, f2) for f1, f2 in points if f1 > 0 and f2 > 0), key=lambda p: (-p[0], -p[1]))
    area, height = 0.0, 0.0
    for f1, f2 in inside:  # by f1 falling: each point adds the strip above the ones before it
        if f2 > height:
            area += f1 * (f2 - height)
            height = f2
    return area, len(points) - len(inside)


def knee(points, sizes, texts):
    """Index of the knee rule (section 5, item 5). points are the normalised front, sizes the rules'
    node counts, texts their texts; ties go to the smaller rule, then the first text."""
    order = lambda i: (sizes[i], texts[i])  # noqa: E731
    by_point = {}
    for i, p in enumerate(points):  # merge equal points: keep the smallest rule
        if p not in by_point or order(i) < order(by_point[p]):
            by_point[p] = i
    candidates = list(by_point.values())
    if len(candidates) == 1:
        return candidates[0]
    a = max(candidates, key=lambda i: points[i][0])  # best f1
    b = max(candidates, key=lambda i: points[i][1])  # best f2
    (a1, a2), (b1, b2) = points[a], points[b]
    ideal = (a1, b2)
    cross = lambda p: (b1 - a1) * (p[1] - a2) - (b2 - a2) * (p[0] - a1)  # noqa: E731
    side = 1 if cross(ideal) > 0 else -1
    length = math.hypot(b1 - a1, b2 - a2)
    beyond = [i for i in candidates if side * cross(points[i]) > 0]
    if len(candidates) == 2 or not beyond:
        return min(candidates, key=lambda i: (math.dist(points[i], ideal), *order(i)))
    return min(beyond, key=lambda i: (-side * cross(points[i]) / length, *order(i)))


def label(rule):
    """The identity of a GP rule, or the name of another policy."""
    return identity(rule) if isinstance(rule, list) else rule.name


def validation_means(population, dates, k, days, options={}):
    """Each rule's validation mean of both objectives: the mean over days and replicates of the day's
    objectives (not visitors pooled over days). Every evaluation and every mean must be finite; a bad
    one stops the run here, before any front is formed, so it can never drop out as a dominated point."""
    means, done = [], {}
    for rule in population:
        key = to_text(rule) if isinstance(rule, list) else rule.name
        if key in done:  # a copy of a rule already scored: the same days and replicates give the same means
            means.append(done[key])
            continue
        results = []
        for date in dates:
            for r in range(k):
                result = evaluate(rule, date, r, days, **options)
                if not (math.isfinite(result.mean_r) and math.isfinite(result.g_hat)):
                    raise ValueError(f"rule {label(rule)} on {date}, replicate {r}: non-finite objectives "
                                     f"({result.mean_r}, {result.g_hat})")
                results.append(result)
        mean = (float(np.mean([x.mean_r for x in results])), float(np.mean([x.g_hat for x in results])))
        if not all(math.isfinite(v) for v in mean):
            raise ValueError(f"rule {label(rule)}: non-finite validation mean {mean}")
        done[key] = mean
        means.append(mean)
    return means


def finish(run, folder=None):
    """Section 5: the final population re-evaluated on all validation days with replicates 0..K-1; the
    front of the validation means; its hypervolume and knee; and the outputs for RQ2 and RQ3.

    The population may hold copies of one rule (duplicates are not removed in the search). The front,
    hypervolume and knee use every member, as designed; the research outputs are made once per distinct
    rule text, with the number of copies, so no rule's visitors are counted twice. Two different texts
    with equal objectives stay two rules. The front rules are run again on the same days and replicates
    to make their outputs (the same results, as an evaluation is deterministic); the knee rule's run
    keeps the decision log. Test days are not read."""
    if run.generation != run.settings["generations"]:
        raise ValueError("the run has not finished its generations")
    k = run.settings["validation replicates"]
    dates, days, options = validation_dates(), days_for(run.settings), scoring(run.settings)
    value_range = {"R": run.settings["R range"], "G": run.settings["G range"]}
    population = run.population
    means = validation_means(population, dates, k, days, options)
    front = non_dominated(means)
    points = normalise([means[i] for i in front], value_range)
    area, outside = hypervolume(points)
    texts = [to_text(population[i]) for i in front]
    sizes = [sum(n for n, _ in size(population[i])) for i in front]
    knee_text = texts[knee(points, sizes, texts)]

    tag_run = {"method": run.method, "run": run.run_number, "settings hash": settings_hash(run.settings),
               "code version": code_version(), "data version": data_version()}
    copies = {text: texts.count(text) for text in texts}
    distinct = list(dict.fromkeys(texts))  # each rule once, in front order
    first = {text: texts.index(text) for text in distinct}
    front_rows, groups, peaks, visitors, decisions, hourly = [], [], [], [], [], []
    for text in distinct:
        i = front[first[text]]
        rule, rule_id, is_knee = population[i], identity(population[i]), text == knee_text
        point = points[first[text]]
        front_rows.append({**tag_run, "rule": rule_id, "text": text, "copies": copies[text],
                           "mean R_v": means[i][0], "G_hat": means[i][1], "f1": point[0], "f2": point[1],
                           "nodes": sum(n for n, _ in size(rule)), "knee": is_knee})
        for date in dates:
            day = days.get(date)[0]
            for r in range(k):
                result = evaluate(rule, date, r, days, log=is_knee, **options)
                tag = {**tag_run, "rule": rule_id, "date": date, "replicate": r}
                rows = visitor_rows(day, result)
                groups.append(grouped(rows).assign(**tag))
                peaks.append(hourly_peaks(day, result.crowding).assign(**tag))
                if is_knee:
                    visitors.append(rows.assign(**tag))
                    decisions.append(pd.DataFrame(result.run.decisions).assign(**tag))
                    table = pd.DataFrame(result.crowding, columns=[f"hour {h}" for h in range(24)])
                    hourly.append(table.assign(site=day.site_ids, **tag))
    summary = {**tag_run, "validation days": len(dates), "replicates": k, "range": value_range,
               "hypervolume": area, "front members": len(front), "front rules": len(distinct),
               "front points not better than the reference": outside, "knee rule": identity_of_text(knee_text)}
    out = {"summary": summary, "means": means, "front": pd.DataFrame(front_rows),
           "groups": pd.concat(groups, ignore_index=True), "peaks": pd.concat(peaks, ignore_index=True),
           "knee visitors": pd.concat(visitors, ignore_index=True),
           "knee decisions": pd.concat(decisions, ignore_index=True),
           "knee hourly": pd.concat(hourly, ignore_index=True)}
    if folder is not None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "summary.json").write_text(json.dumps(summary, indent=1))
        pd.DataFrame({"text": [to_text(r) for r in population], "mean R_v": [m[0] for m in means],
                      "G_hat": [m[1] for m in means]}).assign(**tag_run).to_csv(
            folder / "final_population.csv", index=False)
        for name in ("front", "groups", "peaks", "knee visitors", "knee decisions", "knee hourly"):
            out[name].to_csv(folder / f"{name.replace(' ', '_')}.csv.gz", index=False)
    return out


def identity_of_text(text):
    return identity(from_text(text))


# ---------------------------------------------------------------------- #
# The tuned weighted rule (4e) and pooling runs
# ---------------------------------------------------------------------- #

def tuned_front(settings, folder=None, weights=None):
    """The tuned weighted rule A - weight * VS for every weight of the fixed grid, scored like a run's
    final population: validation means over the same days and replicates, the front of those means and
    its hypervolume under the same normalisation. One deterministic front, not a set of runs."""
    from rules import TUNED_WEIGHTS, Tuned
    settings = complete(settings)
    check_settings(settings)
    weights = TUNED_WEIGHTS if weights is None else weights
    policies = [Tuned(w) for w in weights]
    means = validation_means(policies, validation_dates(), settings["validation replicates"], days_for(settings),
                             scoring(settings))
    front = non_dominated(means)
    points = normalise([means[i] for i in front], {"R": settings["R range"], "G": settings["G range"]})
    area, outside = hypervolume(points)
    table = pd.DataFrame({"weight": weights, "mean R_v": [m[0] for m in means], "G_hat": [m[1] for m in means]})
    table["front"] = table.index.isin(front)
    summary = {"rule": "tuned weighted", "settings hash": settings_hash(settings), "code version": code_version(),
               "data version": data_version(), "hypervolume": area, "front rules": len(front),
               "front points not better than the reference": outside}
    if folder is not None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        table.assign(**{k: summary[k] for k in ("settings hash", "code version", "data version")}).to_csv(
            folder / "tuned_front.csv", index=False)
        (folder / "summary.json").write_text(json.dumps(summary, indent=1))
    return table, summary


def collect(folders):
    """The summaries of several finished runs, for pooling. Runs made under other settings, other code or
    other data are refused, so no table ever mixes them."""
    summaries = [json.loads((Path(f) / "summary.json").read_text()) for f in folders]
    for key in ("settings hash", "code version", "data version"):
        values = {s[key] for s in summaries}
        if len(values) > 1:
            raise ValueError(f"the runs differ in {key}: {sorted(values)}; they cannot be pooled")
    return summaries
