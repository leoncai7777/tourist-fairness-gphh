"""
The reproduction check (Step 6 plan, sections 4 and 7)
======================================================

Before any test day is read, Step 6 shows that its own evaluation path scores
rules as Step 5 did.

  1. Inputs: each run's saved tables are what they must be: the final
     population has N rows with finite means; the front's texts are unique,
     with positive copies and exactly one knee; every table carries the run's
     method and number and the Step 5 identity.
  2. Static: the front, its hypervolume and the knee are recomputed from the
     saved means of the whole final population, with the frozen functions of
     src/search.py, and must match front.csv.gz and summary.json.
  3. Re-evaluation through evaluate6.score(), the path of the test evaluation:
     every distinct front rule on the validation days with the Step 5
     replicates, the daily objectives at alpha 0.25 averaged as the search
     averages them; each mean and each run's hypervolume must match within
     1e-12. Each knee runs with its tidy and display forms as shadows: the tidy
     form must make the same choice at every decision; the display form's
     agreement is recorded. Every tuned weight likewise, with its front.

The rule set and the knees taken to the test days are those Step 5 saved; this
check never replaces them. It prints only counts and the largest differences.
Only when everything passes does it write the rule list, once, and then the
pass record reproduced.json, which binds the list's hash to the code, tables,
settings and packages it passed under; the formal test evaluation starts only
when that record matches.

Run, from the frozen Step 6 snapshot:  python3 analysis/reproduce.py [--workers 12]
"""

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step5 import (read_table, FORMAL, SETTINGS_FILE, STEP5, analysis_version, check_complete, check_here,  # noqa: E402
                   package_versions, run_folder)
import evaluate6  # noqa: E402
import forms  # noqa: E402

from gp_rule import from_text, size  # noqa: E402
from rules import TUNED_WEIGHTS, Tuned  # noqa: E402
from search import (days_for, hypervolume, identity_of_text, knee, load_settings, non_dominated,  # noqa: E402
                    normalise, settings_hash, validation_dates)

TOLERANCE = 1e-12
MEAN = f"mean R_v (alpha {evaluate6.label(evaluate6.MAIN_ALPHA)})"


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


def check_inputs(folder, method, run, population_size, expected=STEP5):
    """Step 1 for one run folder: a list of problems, empty if none."""
    results = Path(folder) / "results"
    population = read_table(results / "final_population.csv")
    front = read_table(results / "front.csv.gz")
    problems = []
    if len(population) != population_size:
        problems.append(f"final population has {len(population)} rows")
    if not np.isfinite(population[["mean R_v", "G_hat"]].to_numpy(float)).all():
        problems.append("a non-finite saved mean")
    if front["text"].duplicated().any() or (front["copies"] < 1).any() or int(front["knee"].sum()) != 1:
        problems.append("front texts not unique, a copy count below 1, or not exactly one knee")
    for name, table in (("final population", population), ("front", front)):
        tags = {"method": method, "run": run, **{k: expected[k] for k in ("settings hash", "code version",
                                                                          "data version")}}
        for column, value in tags.items():
            if not (table[column] == value).all():
                problems.append(f"{name}: {column} is not {value}")
    return problems


def static_check(folder, ranges):
    """Step 2 for one run folder (holding results/)."""
    results = Path(folder) / "results"
    population = read_table(results / "final_population.csv")
    saved_front = read_table(results / "front.csv.gz")
    summary = json.loads((results / "summary.json").read_text())
    means = list(zip(population["mean R_v"].astype(float), population["G_hat"].astype(float)))
    texts, area, knee_text = front_of(list(population["text"]), means, ranges)
    copies = Counter(texts)
    saved_copies = dict(zip(saved_front["text"], saved_front["copies"]))
    saved_knee = saved_front.loc[saved_front["knee"], "text"].tolist()
    return {"same rules and copies": dict(copies) == saved_copies,
            "hypervolume difference": abs(area - summary["hypervolume"]),
            "same knee": saved_knee == [knee_text] and identity_of_text(knee_text) == summary["knee rule"]}


def mean_objectives(policy, dates, k, days, settings, knee=False, watch=None):
    """A rule's validation means through evaluate6.score(): the daily objectives at the main alpha, averaged
    over days and replicates in the order and by the function the search uses (np.mean of the list).
    With watch, the agreements of each shadow over every decision."""
    r_values, g_values, agrees = [], [], {name: [] for name in (watch or {})}
    for date in dates:
        for r in range(k):
            out = evaluate6.score(policy, date, r, days, beta=settings["beta"], stop_limit=settings["stop limit"],
                                  knee=knee, watch=watch)
            r_values.append(out["objectives"][MEAN])
            g_values.append(out["objectives"]["G_hat"])
            for name in agrees:
                agrees[name] += list(out["decisions"][f"{name} agrees"])
    return (float(np.mean(r_values)), float(np.mean(g_values))), agrees


def reevaluate(folder, settings, dates=None, k=None):
    """Step 3 for one run folder: the largest difference over its front rules' validation means, the
    difference of the hypervolume recomputed from them, and the knee's shadow agreements."""
    results = Path(folder) / "results"
    saved = read_table(results / "front.csv.gz")
    summary = json.loads((results / "summary.json").read_text())
    dates = validation_dates() if dates is None else dates
    k = settings["validation replicates"] if k is None else k
    days = days_for(settings)
    means, tidy_agrees, display_agreement = [], None, None
    for text, is_knee in zip(saved["text"], saved["knee"]):
        rule = from_text(text)
        watch = None
        if is_knee:
            watch = {"tidy": forms.tidy(rule)}
            shown, _ = forms.display(rule)
            if shown is not None:
                watch["display"] = shown
        mean, agrees = mean_objectives(rule, dates, k, days, settings, knee=bool(is_knee), watch=watch)
        means.append(mean)
        if is_knee:
            tidy_agrees = agrees["tidy"]
            if "display" in agrees and agrees["display"]:
                display_agreement = float(np.mean(agrees["display"]))
    differences = [max(abs(m[0] - r), abs(m[1] - g)) for m, r, g in zip(means, saved["mean R_v"], saved["G_hat"])]
    area, _ = hypervolume(normalise(means, value_range(settings)))  # copies add nothing to the area
    return {"rules": len(means), "largest mean difference": max(differences),
            "hypervolume difference": abs(area - summary["hypervolume"]),
            "tidy always agrees": bool(tidy_agrees is not None and all(tidy_agrees)),
            "knee decisions": 0 if tidy_agrees is None else len(tidy_agrees),
            "display agreement": display_agreement}


def tuned_check(settings, root=FORMAL, dates=None, k=None):
    """Every tuned weight through evaluate6.score(); its means and its validation front against the saved
    table, the weights matched to the frozen grid by position."""
    from run_test import check_tuned_table
    saved = read_table(Path(root) / "tuned" / "tuned_front.csv")
    check_tuned_table(saved)
    dates = validation_dates() if dates is None else dates
    k = settings["validation replicates"] if k is None else k
    days = days_for(settings)
    means = [mean_objectives(Tuned(w), dates, k, days, settings)[0] for w in TUNED_WEIGHTS]
    front = set(non_dominated(means))
    largest = max(max(abs(m[0] - r), abs(m[1] - g)) for m, r, g in zip(means, saved["mean R_v"], saved["G_hat"]))
    return {"weights": len(means), "same front": [i in front for i in range(len(means))] == list(saved["front"]),
            "largest mean difference": largest}


def check_one(key):
    method, run = key
    settings = load_settings(SETTINGS_FILE)
    folder = run_folder(method, run)
    return (key, check_inputs(folder, method, run, settings["population"]),
            static_check(folder, value_range(settings)), reevaluate(folder, settings))


def passed(problems, static, again):
    return (not problems and static["same rules and copies"] and static["same knee"]
            and static["hypervolume difference"] <= TOLERANCE and again["largest mean difference"] <= TOLERANCE
            and again["hypervolume difference"] <= TOLERANCE and again["tidy always agrees"])


def main():
    from run_test import PASS_RECORD, RULE_LIST, rule_list_hash, write_atomic, write_rule_list
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if PASS_RECORD.exists():
        raise SystemExit(f"{PASS_RECORD} is there already; the check has passed once and is not run again")
    here = check_here()
    folders = check_complete()
    settings = load_settings(SETTINGS_FILE)
    failed, largest, agreement = [], 0.0, {}
    with ProcessPoolExecutor(args.workers) as pool:
        for (method, run), problems, static, again in pool.map(check_one, sorted(folders)):
            largest = max(largest, again["largest mean difference"], again["hypervolume difference"],
                          static["hypervolume difference"])
            agreement[f"{method}_{run:02d}"] = again["display agreement"]
            if not passed(problems, static, again):
                failed.append(f"{method}_{run:02d}")
    tuned = tuned_check(settings)
    tuned_ok = tuned["same front"] and tuned["largest mean difference"] <= TOLERANCE
    print(f"{len(folders) - len(failed)} of {len(folders)} runs reproduced; tuned rule "
          f"{'reproduced' if tuned_ok else 'NOT reproduced'}; largest difference {largest:.1e} "
          f"(tuned {tuned['largest mean difference']:.1e}); failed: {failed or 'none'}")
    if failed or not tuned_ok or not math.isfinite(largest):
        raise SystemExit(1)
    table = write_rule_list()
    record = {"rule list hash": rule_list_hash(), "rules": len(table), "analysis version": analysis_version(),
              "search settings hash": settings_hash(settings), "packages": package_versions(), "step 5": here,
              "runs reproduced": len(folders), "largest difference": largest,
              "tuned largest difference": tuned["largest mean difference"],
              "display agreement on validation days": agreement}
    write_atomic(PASS_RECORD, json.loads(json.dumps(record)))
    print(f"rule list written: {RULE_LIST} ({len(table)} rules, hash {record['rule list hash']}); "
          f"pass record: {PASS_RECORD}")


if __name__ == "__main__":
    main()
