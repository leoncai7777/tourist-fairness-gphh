"""BasicGP of Mei and Zhang (2018) for the stochastic TOPTW, in DEAP.

Settings are from their Table II. Where the paper does not say, ECJ's Koza defaults are used, as the paper used ECJ:
initial depth 2-6, 90% internal crossover points, subtree mutation with grown trees of depth 5.
ECJ counts a lone terminal as depth 1, so its depths are one more than DEAP's heights.

Run from the repository root:
python3 src/single_gphh.py --instances c103 r101 --runs 5 --workers 10 --out outputs/toptw/pilot.csv
"""
import argparse
import csv
import operator
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from deap import base, creator, gp, tools

from toptw import load_instance, protected_div, sample_durations, solve, test_score

TERMINALS = ["SCORE", "DUR", "TO", "TC", "TA", "TR", "TSV", "TFV", "SL", "RemT"]
POPULATION, ELITES, TOURNAMENT, GENERATIONS = 1024, 10, 7, 51
MAX_HEIGHT = 7  # maximal depth 8 in ECJ's count
CROSSOVER, MUTATION = 0.80, 0.15

pset = gp.PrimitiveSet("MAIN", len(TERMINALS))
pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(TERMINALS)})
for function, name in ((np.add, "add"), (np.subtract, "sub"), (np.multiply, "mul"), (protected_div, "div"),
                       (np.minimum, "min"), (np.maximum, "max")):
    pset.addPrimitive(function, 2, name=name)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT)
toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
toolbox.register("subtree", gp.genGrow, min_=4, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.subtree, pset=pset)
limit = gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT)
toolbox.decorate("mate", limit)
toolbox.decorate("mutate", limit)


def evaluate(population, instance, durations):
    for individual in population:
        policy = gp.compile(individual, pset)
        individual.fitness.values = (solve(instance, policy, durations)[0],)


def breed(population):
    offspring = [toolbox.clone(x) for x in tools.selBest(population, ELITES)]
    while len(offspring) < len(population):
        r = random.random()
        if r < CROSSOVER:
            a, b = (toolbox.clone(x) for x in toolbox.select(population, 2))
            offspring.extend(toolbox.mate(a, b))
        elif r < CROSSOVER + MUTATION:
            offspring.extend(toolbox.mutate(toolbox.clone(toolbox.select(population, 1)[0])))
        else:
            offspring.append(toolbox.clone(toolbox.select(population, 1)[0]))
    return offspring[:len(population)]


def run(instance_name, seed, max_points=None, population_size=POPULATION, generations=GENERATIONS):
    """One run on one instance: a new sample of the visit lengths in every generation."""
    np.seterr(all="ignore")
    random.seed(seed)
    instance = load_instance(Path("data/toptw") / f"{instance_name}.txt", max_points)
    started = time.time()
    population = [toolbox.individual() for _ in range(population_size)]
    for generation in range(generations):
        rng = np.random.default_rng([seed, generation])
        evaluate(population, instance, sample_durations(instance, rng))
        if generation < generations - 1:
            population = breed(population)
    best = tools.selBest(population, 1)[0]
    seconds = time.time() - started
    return {"instance": instance_name, "seed": seed, "train": best.fitness.values[0],
            "test": test_score(instance, gp.compile(best, pset)), "height": best.height, "nodes": len(best),
            "seconds": round(seconds, 1), "rule": str(best)}


def run_task(task):
    return run(*task)


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
        writer = csv.DictWriter(f, fieldnames=["instance", "seed", "train", "test", "height", "nodes", "seconds", "rule"])
        writer.writeheader()
        for row in pool.imap_unordered(run_task, tasks):
            writer.writerow(row)
            f.flush()
            print(f"{row['instance']} seed {row['seed']}: test {row['test']:.2f}, {row['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
