"""
The formal test evaluation (Step 6 plan, sections 3 to 5)
=========================================================

Every rule of the rule list on every test day and replicate, one process per
block, several blocks at once. A block is one group of rules (the front rules
of one run, or the baselines) on a few consecutive dates.

The rule list is written once, by the reproduction check (write_rule_list),
which then writes its pass record, reproduced.json, binding the list's hash to
the code, tables, settings and packages it passed under. It holds every distinct
front rule of the 90 runs with its knee flag, the display form of each GP knee,
every weight of the tuned rule (its knee chosen on the validation days by the
frozen knee rule, ties to the smaller weight), the four hand rules and the
observed-order control.

Identity. The formal batch starts only when the code and tables are Step 5's,
the search settings actually loaded have the Step 5 settings hash, every Step 5
run is complete, the rule list covers what it must, and the pass record
matches the rule list, the code, the tables, the settings and the packages of
this start. The identity (the settings by their content hash, not their path)
goes to task.json; a folder holding another task.json is refused before
anything runs, and every block carries the same identity in its done.json.
Runs that are not formal (--not-formal) may only use validation days and may
not write under outputs/test/; they are marked as such.

Completion. A block is written to a temporary folder and checked before it
counts: every file reads back as written; the objectives are exactly one finite
row per rule, date and replicate; the groups, peaks, per-visitor rows, hourly
tables and decision logs have unique keys, cover what they must and hold
finite route parts and scores. Then it is renamed into place and done.json,
with the rows, size and hash of every file, is written last. On a restart a
block without done.json is computed again; one of another identity, or with a
file missing or of another size, stops the batch. --verify checks every block's
hashes after the batch. One launcher per output folder (launcher.lock). Ctrl-C
or a termination signal stops the workers before the lock is released.

  python3 analysis/run_test.py --workers 12          # the formal batch, from the frozen Step 6 snapshot
  python3 analysis/run_test.py --verify              # after it: every block complete and unchanged
  python3 analysis/run_test.py --not-formal --rules R --test-settings S --out O   # e.g. the measured run
"""

import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import signal
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step5 import (read_table, FORMAL, GP_METHODS, ROOT, RUNS, SETTINGS_FILE, STEP5, analysis_version,  # noqa: E402
                   check_complete, check_here, package_versions, run_folder)
import evaluate6  # noqa: E402
import forms  # noqa: E402

from day import PROCESSED  # noqa: E402
from engine import ObservedOrder  # noqa: E402
from evaluate import Days  # noqa: E402
from gp_rule import from_text, to_text  # noqa: E402
from rules import RULES, TUNED_WEIGHTS, Tuned  # noqa: E402
from search import knee, load_settings, normalise, settings_hash  # noqa: E402

TEST_SETTINGS = ROOT / "configs" / "test.json"
OUTPUTS = ROOT / "outputs" / "test"
OUT = OUTPUTS / "2019"
RULE_LIST = OUT / "rules.csv"
PASS_RECORD = OUT / "reproduced.json"
KEYS = ["rule id", "date", "replicate"]
CONTROL = "observed order"
SITES = 15


# ---------------------------------------------------------------------- #
# Settings, days and the rule list
# ---------------------------------------------------------------------- #

SCENARIO_KEYS = {"rules", "stop limit", "compliance", "peak floor", "same day", "budget"}
SCENARIO_RULES = ("knees", "hand and control", "diagnostic")


def load_test_settings(path=TEST_SETTINGS):
    """The settings of a test evaluation. A further analysis (plan, section 12) adds a scenario: which rules
    (the knees; the hand rules and the control; the h2 diagnostic) and what it changes (no stop limit, a
    compliance level, the peak floor, same-day availability, the uncorrected budget)."""
    data = json.loads(Path(path).read_text())
    required = {"comment", "year", "split", "replicates", "alphas", "dates per block"}
    if not required <= set(data) <= required | {"scenario"}:
        raise ValueError(f"{path}: needs exactly {sorted(required)}, and may have a scenario")
    if [float(a) for a in data["alphas"]] != list(evaluate6.ALPHAS):
        raise ValueError(f"{path}: the alpha grid is fixed by the plan: {evaluate6.ALPHAS}")
    scenario = data.get("scenario", {})
    if not set(scenario) <= SCENARIO_KEYS or ("rules" in scenario and scenario["rules"] not in SCENARIO_RULES):
        raise ValueError(f"{path}: a scenario may set only {sorted(SCENARIO_KEYS)}; rules one of {SCENARIO_RULES}")
    if "compliance" in scenario and not 0 < scenario["compliance"] < 1:
        raise ValueError(f"{path}: compliance is a share between 0 and 1")
    if scenario.get("peak floor", "median of the candidates") != "median of the candidates":
        raise ValueError(f"{path}: the peak floor is the median usual peak of the candidate sites (plan, section 12)")
    if scenario.get("budget", "span") != "span" or scenario.get("stop limit", False) is not False or \
            scenario.get("same day", True) is not True:
        raise ValueError(f"{path}: a scenario changes a setting away from the main one only")
    return data


def peak_floor():
    """The common lower bound on the usual peak: the median usual peak of the candidate sites (every site but
    the low-coverage one, which is never a candidate), from the frozen training table (plan, section 12)."""
    from preprocess import LOW_COVERAGE
    sites = read_table(PROCESSED / "sites.csv")
    return float(np.median(sites.loc[~sites["site"].isin(LOW_COVERAGE), "usual_peak"]))


def scenario_settings(settings, data):
    """The search settings with the scenario's changes."""
    scenario, out = data.get("scenario", {}), dict(settings)
    if "stop limit" in scenario:
        out["stop limit"] = False
    if "same day" in scenario:
        out["same day"] = True
    if "budget" in scenario:
        out["budget"] = "span"
    if "peak floor" in scenario:
        out["peak floor"] = peak_floor()
    return out


def diagnostic_rules(rules, root=FORMAL):
    """Item 6 of section 12: each GP-adaptive knee with h2 replaced by the constant median of its logged w on
    the validation days (Step 5's knee decision log). The constant may be 0 and is not bound by the search's
    constant range; the rule is built directly."""
    from deap import gp as deap_gp
    from gp_rule import H2_SET, make_rule
    rows = []
    knees = rules[(rules["kind"] == "gp") & (rules["method"] == "GP-adaptive") & rules["knee"].astype(bool)]
    for _, row in knees.iterrows():
        log = read_table(run_folder(row["method"], int(row["run"]), root) / "results" / "knee_decisions.csv.gz",
                         usecols=["w"])
        constant = float(np.median(log["w"]))
        full = from_text(row["text"])
        rule = make_rule("GP-adaptive", [full[0], deap_gp.PrimitiveTree([deap_gp.Terminal(constant, False,
                                                                                          H2_SET.ret)])])
        rows.append({"rule id": f"{row['rule id']}_h2_constant", "group": row["group"], "kind": "diagnostic",
                     "method": "GP-adaptive", "run": row["run"], "text": to_text(rule), "knee": True,
                     "parent": row["rule id"], "constant": constant})
    return pd.DataFrame(rows)


def rules_of(rules, data):
    """The rules a scenario runs (plan, section 12); all of them without a scenario."""
    which = data.get("scenario", {}).get("rules")
    if which is None:
        return rules
    if which == "diagnostic":
        return diagnostic_rules(rules)
    hand = rules["kind"].isin(["hand", "control"])
    if which == "hand and control":
        return rules[hand]
    keep = hand | ((rules["kind"] == "gp") & rules["knee"].astype(bool)) | \
        ((rules["kind"] == "tuned") & rules["validation front"].astype(bool))
    return rules[keep]


def test_settings_hash(data):
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:10]


def dates_of(data):
    """The dates of the year and split, without any day that has no evaluated visitor (plan, section 11):
    such a day has no mean R_v. Decided from the processed visitor table, before any rule runs."""
    days = read_table(PROCESSED / f"days_{data['year']}.csv")
    if data["split"] != "all":
        days = days[days["split"] == data["split"]]
    visitors = read_table(PROCESSED / f"visitors_{data['year']}.csv.gz", usecols=["date", "stops"])
    with_evaluated = set(visitors.loc[visitors["stops"] >= 2, "date"])
    return sorted(d for d in days["date"] if d in with_evaluated)


def tuned_knee_weight(table, settings):
    """Row position (in the table) of the tuned rule's knee: the frozen knee rule over its validation-non-
    dominated weights. Every weight is the same size, and the texts given to knee() are ranks by weight,
    so ties go to the smaller weight."""
    front = table[table["front"]].sort_values("weight")
    points = normalise(list(zip(front["mean R_v"], front["G_hat"])),
                       {"R": settings["R range"], "G": settings["G range"]})
    i = knee(points, [1] * len(points), [f"{n:03d}" for n in range(len(points))])
    return table.index.get_loc(front.index[i])


def check_tuned_table(tuned):
    """The tuned table went through a CSV file: its weights are matched to the frozen grid by position, and
    the rules are built from the frozen values, never from the numbers read back."""
    if len(tuned) != len(TUNED_WEIGHTS) or not np.allclose(tuned["weight"], TUNED_WEIGHTS, rtol=1e-12, atol=0):
        raise SystemExit("the tuned rule's weights are not the frozen grid")


def write_rule_list(path=RULE_LIST, root=FORMAL, methods=GP_METHODS, runs=RUNS):
    """Written once, by the reproduction check. Refuses to overwrite a list that is there."""
    path = Path(path)
    if path.exists():
        raise SystemExit(f"{path} is there already; the rule list is written once")
    settings = load_settings(SETTINGS_FILE)
    rows = []
    for method in methods:
        for run in runs:
            front = read_table(run_folder(method, run, root) / "results" / "front.csv.gz")
            group = f"{method}_{run:02d}"
            for n, (text, is_knee) in enumerate(zip(front["text"], front["knee"])):
                rule_id = f"{group}_{n:03d}"
                rows.append({"rule id": rule_id, "group": group, "kind": "gp", "method": method, "run": run,
                             "text": text, "knee": bool(is_knee)})
                if is_knee:  # its display form: run once as a rule of its own, and watched in the knee's run
                    shown, reason = forms.display(from_text(text))
                    if shown is None:
                        rows[-1]["display"] = f"none: {reason}"
                        continue
                    rows[-1]["display"] = f"{rule_id}_display"
                    rows.append({"rule id": f"{rule_id}_display", "group": group, "kind": "display",
                                 "method": method, "run": run, "text": to_text(shown), "knee": False,
                                 "parent": rule_id, "readable": forms.readable(shown),
                                 "tidy nodes": forms.nodes(forms.tidy(from_text(text)))})
    tuned = read_table(Path(root) / "tuned" / "tuned_front.csv")
    check_tuned_table(tuned)
    knee_row = tuned_knee_weight(tuned, settings)
    for n, (w, on_front) in enumerate(zip(TUNED_WEIGHTS, tuned["front"])):
        rows.append({"rule id": f"tuned_{n:02d}", "group": "baselines", "kind": "tuned", "method": "tuned",
                     "run": -1, "text": repr(float(w)), "knee": n == knee_row, "validation front": bool(on_front)})
    for name in list(RULES) + [CONTROL]:
        rows.append({"rule id": name, "group": "baselines", "kind": "hand" if name in RULES else "control",
                     "method": name, "run": -1, "text": name, "knee": True})
    table = pd.DataFrame(rows)
    check_rule_list(table, methods, runs)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def check_rule_list(rules, methods=GP_METHODS, runs=RUNS):
    """What the list must cover: every method and run as a group with exactly one GP knee, whose display form
    (if it has one) points back to it; the 16 tuned weights with one knee; the four hand rules and the
    control; unique rule ids."""
    problems = []
    if rules["rule id"].duplicated().any():
        problems.append("a repeated rule id")
    gp = rules[rules["kind"] == "gp"]
    expected = {f"{m}_{r:02d}" for m in methods for r in runs}
    if set(gp["group"]) != expected:
        problems.append(f"GP groups {len(set(gp['group']))}, expected {len(expected)}")
    knees = gp.groupby("group")["knee"].sum()
    if not (knees == 1).all():
        problems.append(f"groups without exactly one knee: {list(knees[knees != 1].index)[:5]}")
    shown = rules[rules["kind"] == "display"]
    knee_ids = set(gp.loc[gp["knee"].astype(bool), "rule id"])
    if not set(shown["parent"]) <= knee_ids:
        problems.append("a display form whose parent is not a GP knee")
    tuned = rules[rules["kind"] == "tuned"]
    if len(tuned) != len(TUNED_WEIGHTS) or tuned["knee"].astype(bool).sum() != 1:
        problems.append("the tuned weights are not 16 with one knee")
    if sorted(rules.loc[rules["kind"].isin(["hand", "control"]), "text"]) != sorted(list(RULES) + [CONTROL]):
        problems.append("the hand rules and the control are not all there")
    if problems:
        raise SystemExit("the rule list does not cover what it must: " + "; ".join(problems))


def derived_identity(rules):
    """Rules built at the start rather than read from the list (the h2 diagnostic): their texts and constants
    are part of the identity, so a change in the Step 5 logs they come from cannot pass unseen."""
    built = rules[rules["kind"] == "diagnostic"]
    if built.empty:
        return {}
    text = built[["rule id", "parent", "text"]].to_csv(index=False)
    return {"derived rules hash": hashlib.sha256(text.encode()).hexdigest()[:12]}


def rule_list_hash(path=RULE_LIST):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def policy(row):
    if row["kind"] == "gp":
        return from_text(row["text"])
    if row["kind"] in ("display", "diagnostic"):  # built directly; their constants may leave the search range
        return from_text(row["text"], strict=False)
    if row["kind"] == "tuned":
        return Tuned(float(row["text"]))
    if row["kind"] == "hand":
        return RULES[row["text"]]()
    return ObservedOrder()


# ---------------------------------------------------------------------- #
# One block
# ---------------------------------------------------------------------- #

FILES = ("objectives", "groups", "peaks", "visitors", "hourly", "decisions")
OBSERVED = ("observed", "observed_peaks", "observed_hourly")  # with the baselines: once per date
VISITOR_VALUES = ["A", "C", "T", "A observed", "C observed", "T observed", "change A", "change C", "change T",
                  "stops", "observed stops"] + \
    [f"R_v (alpha {evaluate6.label(a)})" for a in evaluate6.ALPHAS]


COMPLIANCE = 2  # named random stream for who follows (simulator design, section 5; STAYS is 1, SCHEDULE 3)


def followers(day, date, replicate, share):
    """Who follows at a compliance share: one uniform number per visitor index (card id order) from the stream
    [COMPLIANCE, date, replicate]; an evaluated visitor follows when it is below the share, so the followers at
    50% are among those at 70%, and they are the same for every rule."""
    u = np.random.default_rng([COMPLIANCE, pd.Timestamp(date).toordinal(), replicate]).random(len(day.visitors))
    chosen = day.visitors["evaluated"].to_numpy() & (u < share)
    return list(day.visitors["visitor"].to_numpy()[chosen])


def compute_block(rules, dates, replicates, settings, data=None):
    """Every rule of the block on every date and replicate; the tables of the block. The days are loaded for
    this block only, so a worker's memory does not grow with the dates it has met."""
    data = data or {}
    settings = scenario_settings(settings, data)
    share = data.get("scenario", {}).get("compliance")
    options = {"budget": settings["budget"], "same_day": settings["same day"], "peak_floor": settings["peak floor"]}
    days = Days(**options, test=True)
    by_replicate = {}

    def days_of(date, r):
        if share is None:
            return days
        if (date, r) not in by_replicate:
            by_replicate[date, r] = Days(**options, follows=followers(days.get(date)[0], date, r, share), test=True)
        return by_replicate[date, r]
    tables = {name: [] for name in FILES + OBSERVED}
    if (rules["kind"] == "control").any():
        for date in dates:
            for name, table in evaluate6.observed(*days.get(date)).items():
                tables[name].append(table.assign(date=date))
    shown = {}
    if "parent" in rules:
        shown = {p: t for p, t in zip(rules["parent"], rules["text"]) if isinstance(p, str)}
    for _, row in rules.iterrows():
        rule = policy(row)
        watch = None
        if row["kind"] == "gp" and bool(row["knee"]) and row["rule id"] in shown:
            watch = {"display": from_text(shown[row["rule id"]], strict=False)}
        for date in dates:
            for r in range(replicates):
                out = evaluate6.score(rule, date, r, days_of(date, r), beta=settings["beta"],
                                      stop_limit=settings["stop limit"], knee=bool(row["knee"]), watch=watch)
                tag = {"rule id": row["rule id"], "date": date, "replicate": r}
                tables["objectives"].append({**tag, **out["objectives"]})
                for name in FILES[1:]:
                    if name in out:
                        tables[name].append(out[name].assign(**tag))
    return {name: (pd.DataFrame(rows) if name == "objectives" else pd.concat(rows, ignore_index=True))
            for name, rows in tables.items() if len(rows)}


def check_block(tables, rules, dates, replicates):
    """The content checks of a block (module docstring); raises on the first problem."""
    def keyset(table):
        return set(map(tuple, table[KEYS].astype(str).to_numpy()))

    every = {(i, d, str(r)) for i, d, r in product(rules["rule id"], dates, range(replicates))}
    knees = rules[rules["knee"].astype(bool)]
    knee_keys = {(i, d, str(r)) for i, d, r in product(knees["rule id"], dates, range(replicates))}
    gp_knees = set(knees.loc[knees["kind"].isin(["gp", "diagnostic"]), "rule id"])

    objectives = tables["objectives"]
    if len(objectives) != len(every) or keyset(objectives) != every:
        raise ValueError("objectives: not exactly one row per rule, date and replicate")
    if not np.isfinite(objectives.drop(columns=KEYS).to_numpy(float)).all():
        raise ValueError("objectives: a non-finite value")

    groups = tables["groups"]
    if keyset(groups) != every or groups.duplicated(KEYS + evaluate6.GROUP_KEYS).any():
        raise ValueError("groups: a missing rule, date or replicate, or a repeated group")
    visitors_per_key = groups.groupby(KEYS)["visitors"].sum()
    if (visitors_per_key.groupby(level=["date", "replicate"]).nunique() != 1).any():
        raise ValueError("groups: rules of one date and replicate do not count the same visitors")

    if not np.isfinite(groups.drop(columns=KEYS).to_numpy(float)).all():
        raise ValueError("groups: a non-finite value")

    site_ids = set(read_table(PROCESSED / "sites.csv")["site"])
    hours = [f"hour {h}" for h in range(24)]

    def every_site(t, keys):
        return t.duplicated(keys + ["site"]).any() or (t.groupby(keys).size() != SITES).any() or \
            set(t["site"]) != site_ids

    peaks = tables["peaks"]
    if keyset(peaks) != every or every_site(peaks, KEYS):
        raise ValueError("peaks: not every site once per rule, date and replicate")
    if not np.isfinite(peaks[["peak C", "hour"]].to_numpy(float)).all():
        raise ValueError("peaks: a non-finite value")

    if (rules["kind"] == "control").any():
        seen = tables.get("observed")
        if seen is None or sorted(seen["date"]) != sorted(dates) or not np.isfinite(seen["G_hat"]).all():
            raise ValueError("observed: not one finite G_hat per date")
        for name in ("observed_peaks", "observed_hourly"):
            t = tables.get(name)
            if t is None or set(t["date"]) != set(dates) or every_site(t, ["date"]):
                raise ValueError(f"{name}: not every site once per date")
        if not np.isfinite(tables["observed_hourly"][hours].to_numpy(float)).all() or \
                not np.isfinite(tables["observed_peaks"][["peak C", "hour"]].to_numpy(float)).all():
            raise ValueError("observed: a non-finite crowding value")

    if knee_keys:
        visitors, hourly = tables.get("visitors"), tables.get("hourly")
        if visitors is None or keyset(visitors) != knee_keys or visitors.duplicated(KEYS + ["visitor"]).any():
            raise ValueError("visitors: not every knee's visitors once per date and replicate")
        if not np.isfinite(visitors[VISITOR_VALUES].to_numpy(float)).all():
            raise ValueError("visitors: a non-finite route part or score")
        counted = visitors.groupby(KEYS).size()
        if not counted.equals(visitors_per_key.loc[counted.index]):
            raise ValueError("visitors: the knee rows are not the visitors counted in the groups")
        if hourly is None or keyset(hourly) != knee_keys or every_site(hourly, KEYS):
            raise ValueError("hourly: not every site once per knee, date and replicate")
        if not np.isfinite(hourly[hours].to_numpy(float)).all():
            raise ValueError("hourly: a non-finite crowding value")
    if gp_knees:  # a GP knee may make no decision on a day: an empty log is legal, a missing one is not
        decisions = tables.get("decisions")
        needed = ["decision", "visitor", "time", "candidates", "changed", "VS change", "h1 given up", "w",
                  "j0 site", "jw site"]
        if decisions is None or not set(needed) <= set(decisions.columns):
            raise ValueError("decisions: the log or one of its fields is missing")
        if not set(decisions["rule id"]) <= gp_knees or decisions.duplicated(KEYS + ["decision"]).any():
            raise ValueError("decisions: rows of a rule that is not a GP knee, or a repeated decision")
        if len(decisions):
            if not keyset(decisions) <= knee_keys:
                raise ValueError("decisions: a date or replicate outside the block")
            known = set(map(tuple, visitors[KEYS + ["visitor"]].astype(str).to_numpy()))
            if not set(map(tuple, decisions[KEYS + ["visitor"]].astype(str).to_numpy())) <= known:
                raise ValueError("decisions: a visitor who is not among the knee's evaluated visitors")
            if not (set(decisions["j0 site"]) | set(decisions["jw site"])) <= site_ids:
                raise ValueError("decisions: a site id that is not a site")
            values = decisions[["time", "candidates", "VS change", "h1 given up", "w"]].to_numpy(float)
            if not np.isfinite(values).all() or not decisions["changed"].isin([True, False]).all():
                raise ValueError("decisions: a non-finite value or a changed flag that is not true or false")


def manifest(folder):
    out = {}
    for path in sorted(Path(folder).glob("*.csv.gz")):
        data = path.read_bytes()
        out[path.name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return out


def run_block(job):
    """Compute one block, write it to a temporary folder, read it back and check it, move it into place,
    mark it done with the manifest of its files."""
    folder, rules, dates, replicates, settings, identity, data = job
    folder = Path(folder)
    rules = pd.DataFrame(rules)
    temporary = folder.with_name(folder.name + ".partial")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    started = time.time()
    tables = compute_block(rules, dates, replicates, settings, data)
    for name, table in tables.items():
        table.to_csv(temporary / f"{name}.csv.gz", index=False)
    again = {}
    for name, table in tables.items():
        again[name] = read_table(temporary / f"{name}.csv.gz")
        if len(again[name]) != len(table) or list(again[name].columns) != list(table.columns):
            raise ValueError(f"{name} does not read back as written")
    check_block(again, rules, dates, replicates)
    files = manifest(temporary)
    for name, table in tables.items():
        files[f"{name}.csv.gz"]["rows"] = len(table)
    shutil.rmtree(folder, ignore_errors=True)
    temporary.rename(folder)
    done = {**identity, "rules": list(rules["rule id"]), "dates": dates, "replicates": replicates, "files": files,
            "seconds": round(time.time() - started, 1),
            "worker peak memory MB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)}
    write_atomic(folder / "done.json", done)
    return folder.name


def write_atomic(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=1))
    os.replace(temporary, path)


def block_files_ok(folder, done, deep=False):
    """The files listed in done.json are there with their sizes (and, deep, their hashes), and no other."""
    listed = done.get("files", {})
    there = {p.name for p in Path(folder).glob("*.csv.gz")}
    if set(listed) != there:
        return False
    for name, info in listed.items():
        path = Path(folder) / name
        if path.stat().st_size != info["bytes"]:
            return False
        if deep and hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            return False
    return True


# ---------------------------------------------------------------------- #
# The batch
# ---------------------------------------------------------------------- #

def identity_of(settings, test_path, rules_path, rules, formal=True):
    """The identity of this start. Formal (always, for the formal batch): the code and tables must be
    Step 5's, the loaded settings must have the Step 5 settings hash, every Step 5 run must be complete,
    the rule list must cover what it must, and the pass record must match all of this."""
    identity = {"mode": "formal" if formal else "not formal",
                "search settings hash": settings_hash(settings),
                "step 6 settings hash": test_settings_hash(load_test_settings(test_path)),
                "analysis version": analysis_version(), "rule list hash": rule_list_hash(rules_path),
                "packages": package_versions()}
    if not formal:
        return identity
    identity["step 5"] = check_here()
    identity["step 5 results"] = str(FORMAL)
    if identity["search settings hash"] != STEP5["settings hash"]:
        raise SystemExit(f"the settings loaded have hash {identity['search settings hash']}, not Step 5's")
    check_complete()
    check_rule_list(rules)
    if not PASS_RECORD.exists():
        raise SystemExit(f"no {PASS_RECORD}: the reproduction check has not passed")
    record = json.loads(PASS_RECORD.read_text())
    keys = ("rule list hash", "analysis version", "search settings hash", "packages", "step 5", "step 5 results")
    wrong = [k for k in keys if record.get(k) != json.loads(json.dumps(identity[k]))]
    if wrong or Path(rules_path).resolve() != RULE_LIST.resolve():
        raise SystemExit(f"the reproduction pass record does not match this start: {wrong or 'rule list path'}")
    return identity


def blocks_of(rules, dates, data, out):
    per = data["dates per block"]
    for group, members in rules.groupby("group", sort=False):
        for start in range(0, len(dates), per):
            chunk = dates[start:start + per]
            yield Path(out) / "blocks" / group / chunk[0], members, chunk


def jobs_of(out, rules, dates, data, settings, identity):
    """The blocks still to do; a block marked done under another identity, or with a file missing, extra or
    of another size, stops the batch."""
    jobs, wrong = [], []
    for folder, members, chunk in blocks_of(rules, dates, data, out):
        mark = folder / "done.json"
        if mark.exists():
            done = json.loads(mark.read_text())
            same = {k: done.get(k) for k in identity} == json.loads(json.dumps(identity)) and done.get("dates") == chunk
            if not same or not block_files_ok(folder, done):
                wrong.append(str(folder))
            continue
        jobs.append((str(folder), members.to_dict("list"), chunk, data["replicates"], settings, identity, data))
    if wrong:
        raise SystemExit("blocks of another identity, or with changed files, are there; nothing is run:\n  "
                         + "\n  ".join(wrong[:10]))
    return jobs


def verify(out, rules, dates, data, identity):
    """After the batch: every block is done under this identity and every file has its recorded hash."""
    missing, bad = [], []
    for folder, _, chunk in blocks_of(rules, dates, data, out):
        mark = folder / "done.json"
        if not mark.exists():
            missing.append(str(folder))
            continue
        done = json.loads(mark.read_text())
        same = {k: done.get(k) for k in identity} == json.loads(json.dumps(identity)) and done.get("dates") == chunk
        if not same or not block_files_ok(folder, done, deep=True):
            bad.append(str(folder))
    return missing, bad


def take_lock(out):
    lock = Path(out) / "launcher.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{lock} is there: another launcher may be running")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return lock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--settings", default=str(SETTINGS_FILE))
    parser.add_argument("--test-settings", default=str(TEST_SETTINGS))
    parser.add_argument("--rules", default=str(RULE_LIST))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--verify", action="store_true", help="check every block after the batch; run nothing")
    parser.add_argument("--not-formal", action="store_true",
                        help="only validation days, only outside outputs/test/: e.g. the measured run")
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    out = Path(args.out)
    settings = load_settings(args.settings)
    data = load_test_settings(args.test_settings)
    if args.not_formal:
        if data["split"] != "validation":
            raise SystemExit("a run that is not formal may only use validation days")
        if out.resolve().is_relative_to(OUTPUTS.resolve()):
            raise SystemExit(f"a run that is not formal may not write under {OUTPUTS}")
    rules = read_table(args.rules)
    if rules["rule id"].duplicated().any():
        raise SystemExit("the rule list has a repeated rule id")
    dates = dates_of(data)
    identity = identity_of(settings, args.test_settings, args.rules, rules, formal=not args.not_formal)
    rules = rules_of(rules, data)  # a further analysis runs a subset of the list, or rules built from it
    identity.update(derived_identity(rules))
    if args.verify:
        missing, bad = verify(out, rules, dates, data, identity)
        print(f"{len(missing)} blocks missing, {len(bad)} changed or of another identity", flush=True)
        if missing or bad:
            raise SystemExit(1)
        return
    out.mkdir(parents=True, exist_ok=True)
    task = out / "task.json"
    if task.exists() and json.loads(task.read_text()) != json.loads(json.dumps(identity)):
        raise SystemExit(f"{task} names another identity; nothing is run")
    lock = take_lock(out)
    pool = None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        write_atomic(task, identity)
        jobs = jobs_of(out, rules, dates, data, settings, identity)
        total = math.ceil(len(dates) / data["dates per block"]) * rules["group"].nunique()
        print(f"{len(jobs)} of {total} blocks to do in {out}, {args.workers} at a time; "
              f"{len(rules)} rules, {len(dates)} dates, {data['replicates']} replicates", flush=True)
        failed = []
        pool = Pool(args.workers)
        results = [pool.apply_async(run_block, (job,)) for job in jobs]
        for n, (job, result) in enumerate(zip(jobs, results), 1):
            try:
                result.get()
            except Exception as error:  # noqa: BLE001 - a failed block is reported and left undone
                failed.append(f"{Path(job[0]).parent.name}/{Path(job[0]).name}: {error!r}")
            if n % 50 == 0 or n == len(jobs):
                print(f"{n} of {len(jobs)} blocks finished, {len(failed)} failed", flush=True)
        pool.close()
        pool.join()
        pool = None
        print(f"done; failed: {failed or 'none'}", flush=True)
        if failed:
            raise SystemExit(1)
    except KeyboardInterrupt:
        print("stopped; the workers are being stopped", flush=True)
        raise SystemExit(130)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
