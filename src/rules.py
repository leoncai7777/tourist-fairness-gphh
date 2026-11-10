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
"""

from features import H1_TERMINALS, best, h1_inputs
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


RULES = {"POP": Pop, "LEAST": Least, "SOT": Sot, "SOS": Sos}
