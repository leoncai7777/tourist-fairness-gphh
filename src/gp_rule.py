"""
GP rules for many visitors (Step 4a)
====================================

The individuals of the four GP methods, how their trees are compiled, and the
policy that turns them into choices. Section 2 of search_design.md.

  GP-fixed     [h1, c]    w = c
  GP-prop      [h1, c]    w = c * GDAY
  GP-adaptive  [h1, h2]   w = max(0, h2)
  GP-single    [tree]     the tree's value is the priority (no fairness term)

h1 reads one value per candidate (arrays), h2 one value per decision. Every
method reaches a site through features.choose_site, which cleans non-finite
values, cuts w at 0 and applies the tie rule. A rule is saved as one line of
text and rebuilt from it exactly (constants are written with repr).

The search (Step 4c) draws all its random numbers from Python's random module,
as DEAP and Step 1's grow() do; so does new_individual() here.
"""

import hashlib
import math
import random

import numpy as np
from deap import base, creator, gp

from features import H1_TERMINALS, H2_TERMINALS, choose_site, h1_inputs, h2_inputs, visit_share
from single_gphh import ramped_half_and_half
from toptw import protected_div

METHODS = {"GP-single": 0, "GP-fixed": 1, "GP-prop": 2, "GP-adaptive": 3}  # fixed numbers, used in seeds
LOW, HIGH = 1e-3, 1e3  # range of every constant, for the whole search
SINGLE_TERMINALS = H1_TERMINALS + [t for t in H2_TERMINALS if t not in H1_TERMINALS] + ["VS"]
GDAY = H2_TERMINALS.index("GDAY")


def draw_constant():
    """A new constant, log-uniform on [LOW, HIGH]. A Python float, so that its text is a plain number."""
    return float(10 ** random.uniform(math.log10(LOW), math.log10(HIGH)))


def primitive_set(name, terminals, constants=False):
    pset = gp.PrimitiveSet(name, len(terminals))
    pset.renameArguments(**{f"ARG{i}": t for i, t in enumerate(terminals)})
    for function, label in ((np.add, "add"), (np.subtract, "sub"), (np.multiply, "mul"),
                            (protected_div, "div"), (np.minimum, "min"), (np.maximum, "max")):
        pset.addPrimitive(function, 2, name=label)
    if constants:
        pset.addEphemeralConstant("K", draw_constant)
    return pset


H1_SET = primitive_set("H1", H1_TERMINALS)
H2_SET = primitive_set("H2", H2_TERMINALS, constants=True)
SINGLE_SET = primitive_set("SINGLE", SINGLE_TERMINALS)

# mean R_v is maximised, G_hat minimised
creator.create("FitnessRule", base.Fitness, weights=(1.0, -1.0))
creator.create("Rule", list, fitness=creator.FitnessRule, method=None)


def tree_sets(method):
    """The primitive set of each tree of an individual; None for the constant c."""
    return {"GP-single": [SINGLE_SET], "GP-fixed": [H1_SET, None], "GP-prop": [H1_SET, None],
            "GP-adaptive": [H1_SET, H2_SET]}[method]


def make_rule(method, parts):
    rule = creator.Rule(parts)
    rule.method = method
    return rule


def new_individual(method):
    """A random individual: trees by Step 1's ramped half-and-half (DEAP height 1 to 5), c log-uniform."""
    parts = [gp.PrimitiveTree(ramped_half_and_half(pset, 1, 5)) if pset else draw_constant()
             for pset in tree_sets(method)]
    return make_rule(method, parts)


# ---------------------------------------------------------------------- #
# Text and identity
# ---------------------------------------------------------------------- #

LABELS = {"GP-single": ["tree"], "GP-fixed": ["h1", "c"], "GP-prop": ["h1", "c"], "GP-adaptive": ["h1", "h2"]}


def to_text(rule):
    parts = [f"{label}: {part!r}" if isinstance(part, float) else f"{label}: {part}"
             for label, part in zip(LABELS[rule.method], rule)]
    return " | ".join([rule.method] + parts)


def from_text(text):
    method, *parts = text.split(" | ")
    built = []
    for pset, part in zip(tree_sets(method), parts):
        value = part.split(": ", 1)[1]
        built.append(gp.PrimitiveTree.from_string(value, pset) if pset else float(value))
    return make_rule(method, built)


def identity(rule):
    """Short hash of the rule's text, saved with every result."""
    return hashlib.sha1(to_text(rule).encode()).hexdigest()[:10]


def size(rule):
    """Nodes and DEAP height of each tree; a constant counts as one node of height 0."""
    return [(len(part), part.height) if isinstance(part, gp.PrimitiveTree) else (1, 0) for part in rule]


def constants(tree):
    """Positions of the constant nodes of a tree."""
    return [i for i, node in enumerate(tree) if isinstance(node, gp.Terminal) and not isinstance(node.value, str)]


# ---------------------------------------------------------------------- #
# The policy
# ---------------------------------------------------------------------- #

class GPRule:
    """Runs one individual in the simulator. With log on, every decision is kept in the RQ2 log,
    with the same fields for every method."""
    def __init__(self, rule, log=False):
        self.method, self.log = rule.method, log
        self.name = f"{rule.method}:{identity(rule)}"
        self.functions = [gp.compile(part, pset) if pset else part for part, pset in zip(rule, tree_sets(rule.method))]

    def parts(self, engine, k, candidates):
        """h1 (one value per candidate), w and VS, before choose_site cleans them; and the h2 inputs."""
        h1_values = h1_inputs(engine, k, candidates)
        h2_values = h2_inputs(engine, k)
        vs = visit_share(engine, candidates)
        with np.errstate(all="ignore"):  # overflow gives inf or nan, which choose_site sets to 0 and counts
            if self.method == "GP-single":
                extra = [h2_values[H2_TERMINALS.index(t)] for t in SINGLE_TERMINALS[len(H1_TERMINALS):-1]]
                return self.functions[0](*h1_values, *extra, vs), 0.0, vs, h2_values
            h1 = self.functions[0](*h1_values)
            if self.method == "GP-fixed":
                w = self.functions[1]
            elif self.method == "GP-prop":
                w = self.functions[1] * h2_values[GDAY]
            else:
                w = self.functions[1](*h2_values)
        return h1, w, vs, h2_values

    def choose(self, engine, k, candidates):
        h1, w, vs, h2_values = self.parts(engine, k, candidates)
        return choose_site(engine, k, candidates, h1, w, vs, h2_values, log=self.log)
