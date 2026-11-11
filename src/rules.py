"""
Hand-written rules (Step 3g)
============================

Four fixed rules, run as baselines and in the pilot. Each scores the feasible
candidates and takes the best, with the tie rule of features.py (lowest site
id). Section 8, item 6, of simulator_design.md.

  POP    highest attractiveness A
  LEAST  fewest visits so far today (x)
  SOT    A / walking time (TA), protected division as in Step 1
  SOS    A / slack (SL = TC - TA), protected division

Tuned(weight) is the tuned weighted rule of search_design.md, section 8:
A - weight * VS through choose_site, over a grid of weights fixed in advance.
"""

import numpy as np

from features import H1_TERMINALS, best, choose_site, h1_inputs, visit_share
from toptw import protected_div

TA, SL = H1_TERMINALS.index("TA"), H1_TERMINALS.index("SL")


class Pop:
    def choose(self, engine, k, candidates):
        return best(engine, candidates, engine.day.attractiveness[candidates])


class Least:
    def choose(self, engine, k, candidates):
        return best(engine, candidates, -engine.visits[candidates].astype(float))


class Sot:
    def choose(self, engine, k, candidates):
        inputs = h1_inputs(engine, k, candidates)
        return best(engine, candidates, protected_div(engine.day.attractiveness[candidates], inputs[TA]))


class Sos:
    def choose(self, engine, k, candidates):
        inputs = h1_inputs(engine, k, candidates)
        return best(engine, candidates, protected_div(engine.day.attractiveness[candidates], inputs[SL]))


class Tuned:
    def __init__(self, weight):
        self.weight = float(weight)
        self.name = f"tuned:{self.weight!r}"

    def choose(self, engine, k, candidates):
        return choose_site(engine, k, candidates, engine.day.attractiveness[candidates], self.weight,
                           visit_share(engine, candidates))


# 0 and 15 values log-spaced from 0.01 to 100, fixed before any result (search_design.md, section 8)
TUNED_WEIGHTS = [0.0] + [float(w) for w in np.logspace(-2, 2, 15)]

RULES = {"POP": Pop, "LEAST": Least, "SOT": Sot, "SOS": Sos}
