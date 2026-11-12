"""
Reading a knee rule (Step 6 plan, section 7)
============================================

The full rule is used in every evaluation and log; nothing here changes how a
rule runs.

  - tidy(rule): every subtree made only of constants whose value is finite is
    replaced by that value, computed with the GP's own operators (protected
    division included). Nothing else is removed, reordered or cancelled. A
    constant subtree whose value is not finite is kept as it is. The tidy form
    computes the same numbers as the full rule, so it makes the same choices.
  - display(rule): the tidy form with every constant (and c) rounded to 3
    significant figures. Approximate; never replaces the full rule. If a
    rounded constant is not finite, there is no display form and the reason is
    returned.
  - Shadow: a policy that acts as the full rule and, at every decision, also
    computes another rule's choice without acting on it, drawing no random
    number and leaving the fallback counts as they were.

Both forms are built directly as trees, not through the strict rule parser, so
their constants are not bound by the search's range [1e-3, 1e3].
"""

import copy
import math
import sys
from pathlib import Path

import numpy as np
from deap import gp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import best  # noqa: E402
from gp_rule import GPRule, LABELS, make_rule, size, tree_sets  # noqa: E402

SIGNIFICANT = 3


def is_constant(node):
    return isinstance(node, gp.Terminal) and not isinstance(node.value, str)


def constant_subtree(tree, i):
    """The slice of the subtree at i, if it is made only of primitives and constants and has a primitive."""
    span = tree.searchSubtree(i)
    nodes = tree[span]
    if len(nodes) > 1 and all(is_constant(n) or isinstance(n, gp.Primitive) for n in nodes):
        return span
    return None


def value_of(nodes, pset):
    """The value of a subtree without arguments, computed by the GP's own compiled operators."""
    function = gp.compile(gp.PrimitiveTree(nodes), pset)
    with np.errstate(all="ignore"):  # an overflow gives inf, which fold() keeps as it is
        return float(np.asarray(function(*([0.0] * len(pset.arguments)))))


def fold(tree, pset):
    """The tree with each outermost constant subtree of finite value replaced by its value; a subtree of
    non-finite value is kept, and the subtrees inside it are tried in turn."""
    tree = gp.PrimitiveTree(list(tree))
    i = 0
    while i < len(tree):
        span = constant_subtree(tree, i)
        if span is not None:
            value = value_of(tree[span], pset)
            if math.isfinite(value):
                tree[span] = [gp.Terminal(value, False, pset.ret)]
        i += 1
    return tree


def tidy(rule):
    parts = [fold(part, pset) if pset else part for part, pset in zip(rule, tree_sets(rule.method))]
    return make_rule(rule.method, parts)


def rounded(value):
    return float(f"{value:.{SIGNIFICANT}g}")


def display(rule):
    """(display rule, None), or (None, reason) when a rounded constant is not finite."""
    parts = []
    for part, pset in zip(tidy(rule), tree_sets(rule.method)):
        if pset is None:
            value = rounded(part)
            if not math.isfinite(value):
                return None, f"c rounds to {value}"
            parts.append(value)
            continue
        tree = gp.PrimitiveTree(list(part))
        for i, node in enumerate(tree):
            if is_constant(node):
                value = rounded(node.value)
                if not math.isfinite(value):
                    return None, f"a constant {node.value!r} rounds to {value}"
                tree[i] = gp.Terminal(value, False, pset.ret)
        parts.append(tree)
    return make_rule(rule.method, parts), None


def readable(rule):
    """The rule written for the report: protected division shown as pdiv."""
    parts = [f"{label}: {part!r}" if isinstance(part, float) else f"{label}: {str(part).replace('div(', 'pdiv(')}"
             for label, part in zip(LABELS[rule.method], rule)]
    return " | ".join([rule.method] + parts)


def nodes(rule):
    return sum(n for n, _ in size(rule))


def quiet_choice(engine, candidates, h1, w, vs):
    """choose_site's cleaning, w cut and tie rule, without counting anything or logging."""
    h1 = np.broadcast_to(np.asarray(h1, float), candidates.shape)
    h1 = np.where(np.isfinite(h1), h1, 0.0)
    w = float(w)
    w = max(0.0, w if math.isfinite(w) else 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        priority = h1 - w * vs
    priority = np.where(np.isfinite(priority), priority, 0.0)
    return best(engine, candidates, priority)


class Shadow:
    """Acts as the full rule (with its decision log); at every decision also computes the choice of each other
    rule, which is never acted on. others maps a name to a rule; agrees[name][i] is True when that rule's choice
    at decision i is the full rule's. The fallback counts are put back as they were, whatever happens."""
    def __init__(self, full, others):
        self.full = GPRule(full, log=True)
        self.others = {name: GPRule(rule, log=False) for name, rule in others.items()}
        self.name = self.full.name
        self.agrees = {name: [] for name in others}

    def choose(self, engine, k, candidates):
        j = self.full.choose(engine, k, candidates)
        for name, other in self.others.items():
            kept = dict(engine.run.fallbacks)
            try:
                with np.errstate(all="ignore"):
                    h1, w, vs, _ = other.parts(engine, k, candidates)
                choice = quiet_choice(engine, candidates, h1, w, vs)
            finally:
                engine.run.fallbacks.clear()
                engine.run.fallbacks.update(kept)
            self.agrees[name].append(bool(choice == j))
        return j


def copy_rule(rule):
    return make_rule(rule.method, copy.deepcopy(list(rule)))
