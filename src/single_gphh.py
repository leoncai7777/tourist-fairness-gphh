"""
BasicGP of Mei and Zhang (2018) for the stochastic TOPTW (Step 1)
=================================================================

Evolves one priority tree for a single tourist, with a new sample of the visit
lengths in every generation, and scores the best tree on 500 test samples.

Settings come from Table II of the paper. Where the paper says nothing, ECJ's
Koza defaults are used, as the paper used ECJ: initial depth 2 to 6, 90%
internal crossover points, subtree mutation with grown trees of depth 5.
ECJ counts a lone terminal as depth 1, so its depths are one more than DEAP's
heights. ECJ's grow may stop at any node, while DEAP's genGrow may not stop
before min_, so grow() below replaces genGrow. These operators are a declared
variant, not ECJ's own code.

Run:  python3 src/single_gphh.py --instances c103 r101 --runs 30 --max-points 100 --out outputs/toptw/all40.csv
(from the repository root). Each row of the output records one run.
"""

import argparse
import csv
import hashlib
import operator
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from deap import base, creator, gp, tools

from toptw import TEST_SEED, counters, load_instance, protected_div, sample_durations, solve, test_score

TERMINALS = ["SCORE", "DUR", "TO", "TC", "TA", "TR", "TSV", "TFV", "SL", "RemT"]
POPULATION, ELITES, TOURNAMENT, GENERATIONS = 1024, 10, 7, 51
CROSSOVER, MUTATION = 0.80, 0.15  # the remaining 5% is reproduction
MAX_HEIGHT = 7  # maximal depth 8 in ECJ's count
FIELDS = ["instance", "seed", "train", "test", "height", "nodes", "seconds", "points", "max_points",
          "input_sha1", "code_sha1", "test_seed", "test_samples", "non_finite", "mutation_unchanged",
          "crossover_unchanged", "rule"]

operators = {"crossover": 0, "crossover_unchanged": 0, "mutation": 0, "mutation_unchanged": 0}


# ---------------------------------------------------------------------- #
# 1. Functions, terminals and tree builders
# ---------------------------------------------------------------------- #

pset = gp.PrimitiveSet("MAIN", len(TERMINALS))
pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(TERMINALS)})
for function, name in ((np.add, "add"), (np.subtract, "sub"), (np.multiply, "mul"),
                       (protected_div, "div"), (np.minimum, "min"), (np.maximum, "max")):
    pset.addPrimitive(function, 2, name=name)


def grow(pset, min_, max_, type_=None):
    """A tree of height at most a value drawn from [min_, max_]; any node above that height may be a terminal."""
    def stop(height, depth):
        return depth == height or random.random() < pset.terminalRatio
    return gp.generate(pset, min_, max_, stop, type_)


def ramped_half_and_half(pset, min_, max_, type_=None):
    return random.choice((gp.genFull, grow))(pset, min_, max_, type_)


# ---------------------------------------------------------------------- #
# 2. Toolbox
# ---------------------------------------------------------------------- #

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", ramped_half_and_half, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT)
toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
toolbox.register("subtree", grow, min_=4, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.subtree, pset=pset)
# A child over the depth limit is replaced by its parent.
limit = gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT)
toolbox.decorate("mate", limit)
toolbox.decorate("mutate", limit)


# ---------------------------------------------------------------------- #
# 3. Evolution
# ---------------------------------------------------------------------- #

def evaluate(population, instance, durations):
    for individual in population:
        policy = gp.compile(individual, pset)
        individual.fitness.values = (solve(instance, policy, durations)[0],)


def breed(population):
    """Algorithm 3 of the paper: elites, then crossover, mutation or reproduction.
    Also counts how often an operator leaves the trees unchanged."""
    offspring = [toolbox.clone(x) for x in tools.selBest(population, ELITES)]
    while len(offspring) < len(population):
        r = random.random()
        if r < CROSSOVER:
            a, b = (toolbox.clone(x) for x in toolbox.select(population, 2))
            before = (str(a), str(b))
            children = toolbox.mate(a, b)
            operators["crossover"] += 1
            operators["crossover_unchanged"] += (str(children[0]), str(children[1])) == before
            offspring.extend(children)
        elif r < CROSSOVER + MUTATION:
            parent = toolbox.clone(toolbox.select(population, 1)[0])
            before = str(parent)
            child = toolbox.mutate(parent)
            operators["mutation"] += 1
            operators["mutation_unchanged"] += str(child[0]) == before
            offspring.extend(child)
        else:
            offspring.append(toolbox.clone(toolbox.select(population, 1)[0]))
    return offspring[:len(population)]


# ---------------------------------------------------------------------- #
# 4. One run
# ---------------------------------------------------------------------- #

def fingerprint(*paths):
    """Short hash of files, to record which code and input made a result."""
    digest = hashlib.sha1()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()[:10]


def run(instance_name, seed, max_points=None, population_size=POPULATION, generations=GENERATIONS):
    """One run on one instance. Returns one row of the results file."""
    np.seterr(all="ignore")
    random.seed(seed)
    counters["non_finite"] = 0
    operators.update(crossover=0, crossover_unchanged=0, mutation=0, mutation_unchanged=0)
    path = Path("data/toptw") / f"{instance_name}.txt"
    instance = load_instance(path, max_points)

    started = time.time()
    population = [toolbox.individual() for _ in range(population_size)]
    for generation in range(generations):
        rng = np.random.default_rng([seed, generation])
        evaluate(population, instance, sample_durations(instance, rng))
        if generation < generations - 1:
            population = breed(population)
    best = tools.selBest(population, 1)[0]
    seconds = time.time() - started
    test = test_score(instance, gp.compile(best, pset))

    mutation_unchanged = operators["mutation_unchanged"] / max(operators["mutation"], 1)
    crossover_unchanged = operators["crossover_unchanged"] / max(operators["crossover"], 1)
    return {"instance": instance_name, "seed": seed, "train": best.fitness.values[0], "test": test,
            "height": best.height, "nodes": len(best), "seconds": round(seconds, 1),
            "points": len(instance.score) - 1, "max_points": max_points,
            "input_sha1": fingerprint(path), "code_sha1": fingerprint("src/toptw.py", "src/single_gphh.py"),
            "test_seed": TEST_SEED, "test_samples": 500, "non_finite": counters["non_finite"],
            "mutation_unchanged": round(mutation_unchanged, 4),
            "crossover_unchanged": round(crossover_unchanged, 4), "rule": str(best)}


def run_task(task):
    return run(*task)


# ---------------------------------------------------------------------- #
# 5. Main
# ---------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", required=True)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-points", type=int, default=None)
    args = parser.parse_args()

    tasks = [(name, seed, args.max_points) for name in args.instances for seed in range(args.runs)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f, Pool(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        # Rows are written as runs finish, so a stopped job keeps its results.
        for row in pool.imap_unordered(run_task, tasks):
            writer.writerow(row)
            f.flush()
            print(f"{row['instance']} seed {row['seed']}: test {row['test']:.2f}, {row['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
