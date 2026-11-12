"""
Reading the test outputs (Step 6 plan, sections 5 and 6)
========================================================

Every analysis starts here. Before anything is read, the whole batch is
checked: its task.json is the identity of the code, settings, rule list and
packages running now (for the formal batch, the full formal identity with the
reproduction pass record); every block is there, done under that identity, with
every file of its recorded hash. Then tables are read exactly (round-trip
floats) and put together over the blocks.

A rule's test mean of an objective is the mean over every test day and
replicate of that day's objective (the daily objectives averaged, as in the
search); each rule must have exactly one row per day and replicate.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate6  # noqa: E402
from run_test import (OBSERVED, SETTINGS_FILE, TEST_SETTINGS, blocks_of, dates_of, derived_identity,  # noqa: E402
                      identity_of, load_test_settings, rules_of, verify)
from step5 import read_table  # noqa: E402

from search import load_settings  # noqa: E402

ALPHAS = evaluate6.ALPHAS
MAIN = evaluate6.MAIN_ALPHA


def r_column(alpha=MAIN):
    return f"mean R_v (alpha {evaluate6.label(alpha)})"


class Results:
    """The checked outputs of one test evaluation."""

    def __init__(self, out, rules, dates, data, identity):
        self.out, self.rules, self.dates, self.data, self.identity = Path(out), rules, dates, data, identity
        self._tables = {}
        # the blocks the batch must have, and only those: nothing else in the folder is ever read
        self.blocks = {group: [] for group in dict.fromkeys(rules["group"])}
        for folder, members, _ in blocks_of(rules, dates, data, out):
            self.blocks[members["group"].iloc[0]].append(Path(folder))

    def files(self, name, group="*"):
        """The files of one kind in the checked blocks (of one group, or of every group); a kind that a block
        does not write (the knee tables of a group without knees) is simply absent."""
        groups = self.blocks if group == "*" else {group: self.blocks[group]}
        return [f / f"{name}.csv.gz" for folders in groups.values() for f in folders
                if (f / f"{name}.csv.gz").exists()]

    def table(self, name, group="*"):
        """One kind of table over the blocks of one group (or every group, for the small tables: objectives,
        observed, observed_peaks, observed_hourly). The large ones (groups, peaks, visitors, decisions) are read
        a group at a time or summed block by block (sums), so the formal batch fits in memory."""
        key = (name, group)
        if key not in self._tables:
            parts = [read_table(p) for p in self.files(name, group)]
            if not parts:
                raise ValueError(f"no {name} table for {group} in {self.out}")
            table = pd.concat(parts, ignore_index=True)
            if group != "*":
                return table
            self._tables[key] = table
        return self._tables[key]

    def groups(self):
        return list(dict.fromkeys(self.rules["group"]))

    def sums(self, name, by, columns, keep=None):
        """Sums of columns by the keys in by, block by block, then over blocks; keep filters rows first."""
        total = None
        for path in self.files(name):
            t = read_table(path)
            if keep is not None:
                t = t[keep(t)]
            part = t.groupby(by)[columns].sum()
            total = part if total is None else total.add(part, fill_value=0)
        return total

    def rule_means(self):
        """Each rule's test mean of every objective, indexed by rule id."""
        if "means" not in self._tables:  # computed once and kept
            values = [r_column(a) for a in ALPHAS] + ["G_hat"]
            objectives = self.table("objectives")
            per_rule = objectives.groupby("rule id").size()
            expected = len(self.dates) * self.data["replicates"]
            if set(per_rule.index) != set(self.rules["rule id"]) or (per_rule != expected).any():
                raise ValueError("not every rule has one row per day and replicate")
            self._tables["means"] = objectives.groupby("rule id")[values].mean()
        return self._tables["means"]

    def completion(self):
        """Each rule's visit completion (stops on the new routes over stops on the observed ones), the share
        of visitors ending early, the stops lost and the share of visitors by end reason, pooled over every
        test day and replicate. Computed once and kept."""
        if "completion" not in self._tables:
            from evaluate import END_REASONS
            reasons = [f"end: {r}" for r in END_REASONS]
            s = self.sums("groups", "rule id", ["visitors", "stops", "observed stops", "ended early"] + reasons)
            out = pd.DataFrame({"completion": s["stops"] / s["observed stops"],
                                "ended early": s["ended early"] / s["visitors"],
                                "stops lost": s["observed stops"] - s["stops"]})
            for r in reasons:
                out[r] = s[r] / s["visitors"]
            self._tables["completion"] = out
        return self._tables["completion"]

    def observed(self):
        """The observed days: one G_hat per date."""
        seen = self.table("observed")
        if sorted(seen["date"]) != sorted(self.dates):
            raise ValueError("the observed days are not one per date")
        return seen


def load(out, rules_path, test_settings=TEST_SETTINGS, settings_path=SETTINGS_FILE, formal=True):
    """The outputs in out, after checking them as a whole (module docstring): never while a launcher may still
    be running, never without every block's hashes."""
    out = Path(out)
    if (out / "launcher.lock").exists():
        raise SystemExit(f"{out / 'launcher.lock'} is there: the batch may still be running; nothing is read")
    rules = read_table(rules_path)
    data = load_test_settings(test_settings)
    settings = load_settings(settings_path)
    identity = identity_of(settings, test_settings, rules_path, rules, formal=formal)
    rules = rules_of(rules, data)  # a further analysis ran a subset of the list, or rules built from it
    identity.update(derived_identity(rules))
    task = json.loads((out / "task.json").read_text())
    if not formal:  # outputs made before these analysis files were written (development on pilot rules)
        identity["analysis version"] = task.get("analysis version")
    if task != json.loads(json.dumps(identity)):
        raise SystemExit(f"{out / 'task.json'} is not the identity of the code, settings and rule list here")
    dates = dates_of(data)
    missing, bad = verify(out, rules, dates, data, task)
    if missing or bad:
        raise SystemExit(f"{len(missing)} blocks missing, {len(bad)} changed or of another identity")
    return Results(out, rules, dates, data, identity)


def quartiles(values):
    """Median and quartiles, numpy's default (linear interpolation), of the values that are not NA."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return {"n": 0, "median": None, "q1": None, "q3": None}
    q1, median, q3 = np.percentile(v, [25, 50, 75])
    return {"n": len(v), "median": float(median), "q1": float(q1), "q3": float(q3)}


__all__ = ["Results", "load", "quartiles", "r_column", "ALPHAS", "MAIN", "OBSERVED"]
