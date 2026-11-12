"""
Time per suggestion (Step 6 plan, section 9)
============================================

Each knee rule (the GP knees, the tuned rule's knee, the hand rules) runs alone,
in this one process, on the first 10 test days by date with replicate 0. The
time spent in the rule's choice is summed over its decisions and divided by
their number. Run once, after the formal test evaluation, from the same frozen
snapshot (it reads test days); --not-formal uses validation days instead.

  python3 analysis/timing.py
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results as rs  # noqa: E402
from rq1 import METHODS  # noqa: E402
from run_test import OUT, RULE_LIST, SETTINGS_FILE, TEST_SETTINGS, policy  # noqa: E402

from evaluate import Days, evaluate  # noqa: E402
from gp_rule import GPRule  # noqa: E402
from search import load_settings  # noqa: E402

DAYS = 10


class Timed:
    """Acts as the policy; adds up the time spent in its choices."""
    def __init__(self, policy):
        self.policy = GPRule(policy) if isinstance(policy, list) else policy
        self.name = getattr(self.policy, "name", "rule")
        self.seconds, self.decisions = 0.0, 0

    def choose(self, engine, k, candidates):
        start = time.perf_counter()
        j = self.policy.choose(engine, k, candidates)
        self.seconds += time.perf_counter() - start
        self.decisions += 1
        return j


def time_rules(rules, dates, settings):
    days = Days(budget=settings["budget"], same_day=settings["same day"], peak_floor=settings["peak floor"],
                test=True)
    rows = []
    for _, row in rules.iterrows():
        timed = Timed(policy(row))
        for date in dates:
            evaluate(timed, date, 0, days, alpha=settings["alpha"], beta=settings["beta"],
                     stop_limit=settings["stop limit"])
        rows.append({"rule id": row["rule id"], "method": row["method"], "decisions": timed.decisions,
                     "microseconds per decision": 1e6 * timed.seconds / max(timed.decisions, 1)})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--rules", default=str(RULE_LIST))
    parser.add_argument("--test-settings", default=str(TEST_SETTINGS))
    parser.add_argument("--not-formal", action="store_true")
    args = parser.parse_args(argv)
    results = rs.load(args.out, args.rules, args.test_settings, formal=not args.not_formal, deep=False)
    rules = results.rules
    knees = rules[rules["knee"].astype(bool) & (rules["kind"] != "control")]
    table = time_rules(knees, results.dates[:DAYS], load_settings(SETTINGS_FILE))
    folder = Path(args.out) / "analysis" / "others"
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / "time_per_suggestion.csv", index=False)
    summary = table.groupby("method")["microseconds per decision"].median().reindex(
        [m for m in METHODS + ("tuned", "POP", "LEAST", "SOT", "SOS") if m in set(table["method"])])
    summary.to_csv(folder / "time_per_suggestion_summary.csv")
    print(summary.round(1).to_string())


if __name__ == "__main__":
    main()
