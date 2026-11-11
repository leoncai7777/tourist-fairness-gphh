"""
The reproduction check (Step 6 plan, section 4)
===============================================

Before any test day is read, Step 6 shows that it scores rules as Step 5 did.

  1. Static: each run's front, hypervolume and knee are recomputed from the
     saved validation means of its whole final population, with the frozen
     functions of src/search.py, and must match front.csv.gz and summary.json.
  2. Re-evaluation: every distinct front rule is evaluated again on the
     validation days with the Step 5 replicates, and so is every tuned weight;
     each mean, and each run's hypervolume, must match within 1e-12.

The rule set and the knees taken to the test days are those Step 5 saved; this
check never replaces them. It reports only counts and the largest differences,
never the values.

Run, from the frozen Step 6 snapshot:  python3 analysis/reproduce.py [--workers 12]
"""

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step5 import FORMAL, SETTINGS_FILE, check_complete, check_here, run_folder  # noqa: E402

from gp_rule import from_text, size  # noqa: E402
from search import (days_for, hypervolume, identity_of_text, knee, load_settings, non_dominated,  # noqa: E402
                    normalise, scoring, tuned_front, validation_dates, validation_means)

TOLERANCE = 1e-12


def value_range(settings):
    return {"R": settings["R range"], "G": settings["G range"]}


def nodes(text):
    return sum(n for n, _ in size(from_text(text)))


def front_of(texts, means, ranges):
    """The front, its hypervolume and the knee text, as finish() makes them from a population's means."""
    front = non_dominated(means)
    points = normalise([means[i] for i in front], ranges)
    area, _ = hypervolume(points)
    front_texts = [texts[i] for i in front]
    knee_text = front_texts[knee(points, [nodes(t) for t in front_texts], front_texts)]
    return front_texts, area, knee_text


def static_check(folder, ranges):
    """Step 1 of the check for one run folder (holding results/)."""
    results = Path(folder) / "results"
    population = pd.read_csv(results / "final_population.csv")
    saved_front = pd.read_csv(results / "front.csv.gz")
    summary = json.loads((results / "summary.json").read_text())
    means = list(zip(population["mean R_v"].astype(float), population["G_hat"].astype(float)))
    texts, area, knee_text = front_of(list(population["text"]), means, ranges)
    copies = Counter(texts)
    saved_copies = dict(zip(saved_front["text"], saved_front["copies"]))
    saved_knee = saved_front.loc[saved_front["knee"], "text"].tolist()
    return {"same rules and copies": dict(copies) == saved_copies,
            "hypervolume difference": abs(area - summary["hypervolume"]),
            "same knee": saved_knee == [knee_text] and identity_of_text(knee_text) == summary["knee rule"]}


def reevaluate(folder, settings, dates=None, k=None):
    """Step 2 of the check for one run folder: the largest difference over its front rules' validation means,
    and the difference of the hypervolume recomputed from the re-evaluated means."""
    results = Path(folder) / "results"
    saved = pd.read_csv(results / "front.csv.gz")
    summary = json.loads((results / "summary.json").read_text())
    dates = validation_dates() if dates is None else dates
    k = settings["validation replicates"] if k is None else k
    rules = [from_text(t) for t in saved["text"]]
    means = validation_means(rules, dates, k, days_for(settings), scoring(settings))
    differences = [max(abs(m[0] - r), abs(m[1] - g)) for m, r, g in zip(means, saved["mean R_v"], saved["G_hat"])]
    area, _ = hypervolume(normalise(means, value_range(settings)))  # copies add nothing to the area
    return {"rules": len(rules), "largest mean difference": max(differences),
            "hypervolume difference": abs(area - summary["hypervolume"])}


def tuned_check(settings, root=FORMAL):
    saved = pd.read_csv(Path(root) / "tuned" / "tuned_front.csv")
    table, _ = tuned_front(settings)
    same_front = list(table["front"]) == list(saved["front"])
    largest = max(max(abs(a - b) for a, b in zip(table[c], saved[c])) for c in ("mean R_v", "G_hat"))
    return {"weights": len(table), "same front": same_front, "largest mean difference": largest}


def check_one(key):
    method, run = key
    settings = load_settings(SETTINGS_FILE)
    folder = run_folder(method, run)
    return key, static_check(folder, value_range(settings)), reevaluate(folder, settings)


def passed(static, again):
    return (static["same rules and copies"] and static["same knee"]
            and static["hypervolume difference"] <= TOLERANCE and again["largest mean difference"] <= TOLERANCE
            and again["hypervolume difference"] <= TOLERANCE)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    check_here()
    folders = check_complete()
    settings = load_settings(SETTINGS_FILE)
    failed, largest = [], 0.0
    with ProcessPoolExecutor(args.workers) as pool:
        for (method, run), static, again in pool.map(check_one, sorted(folders)):
            largest = max(largest, again["largest mean difference"], again["hypervolume difference"],
                          static["hypervolume difference"])
            if not passed(static, again):
                failed.append(f"{method}_{run:02d}")
    tuned = tuned_check(settings)
    tuned_ok = tuned["same front"] and tuned["largest mean difference"] <= TOLERANCE
    print(f"{len(folders) - len(failed)} of {len(folders)} runs reproduced; tuned rule "
          f"{'reproduced' if tuned_ok else 'NOT reproduced'}; largest difference {largest:.1e} "
          f"(tuned {tuned['largest mean difference']:.1e}); failed: {failed or 'none'}")
    if failed or not tuned_ok or not math.isfinite(largest):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
