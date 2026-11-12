"""
GP-single and the walking weight (addendum to the Step 6 plan, frozen version 3)
================================================================================

Everything here reuses the frozen src/ and analysis/ and changes neither, so
the analysis version of the main batch stays what it was. Steps, in order:

  check      the 30 GP-single runs are complete with the Step 5 identity, and
             each run's saved environment schedule equals the one saved by
             GP-fixed, GP-prop and GP-adaptive with the same run number
  reproduce  inputs, static check and re-evaluation of every distinct front rule
             on the validation days (tolerance 1e-12), as analysis/reproduce.py,
             without the display and tidy shadows (RQ2 is not run for
             GP-single); prints only counts, the largest difference and the
             failed runs; then writes the rule list and its pass record, once
  run        the test evaluation, with the frozen compute_block, check_block and
             run_block, under this batch's own identity
  verify     every block done under that identity, every file of its hash
  tables     Tables S1-S3, Figure S1, the dominated counts and the other ranges,
             after checking both batches as a whole and against each other
  beta       Table S4: the fixed knees re-scored at other walking weights

The main batch is read through the frozen results.load(), from
outputs/test/2019 of the worktree this runs in (a link to the verified copy).

  python3 addendum/single.py check
  python3 addendum/single.py reproduce --workers 12
  python3 addendum/single.py run --workers 12
  python3 addendum/single.py verify
  python3 addendum/single.py tables
  python3 addendum/single.py beta [--with-single]
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import signal
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import evaluate6  # noqa: E402
import reproduce  # noqa: E402
import results as rs  # noqa: E402
import rq1  # noqa: E402
import run_test as rt  # noqa: E402
import stats  # noqa: E402
from step5 import (FORMAL, RUNS, SETTINGS_FILE, STEP5, analysis_version, check_complete, check_here,  # noqa: E402
                   package_versions, read_table, run_folder)

from scoring import relative_value, route_value  # noqa: E402
from search import days_for, load_settings, settings_hash, validation_dates  # noqa: E402

METHOD = "GP-single"
MAIN_METHODS = rq1.METHODS
ALL_METHODS = (METHOD,) + MAIN_METHODS
SINGLE = Path(os.environ.get("SINGLE_RESULTS", ROOT / "outputs" / "search" / "single" / STEP5["settings hash"]))
OUT = rt.OUTPUTS / "2019_single"
RULE_LIST = OUT / "rules.csv"
PASS_RECORD = OUT / "reproduced.json"
BETAS = (0.0, 0.25, 0.5, 1.0)
MAIN_BETA = 0.5
H2_ONLY = ("TLD", "ARR", "MCR", "GDAY", "STOPS")  # terminals only h2 has; RemT is also an h1 terminal
COLOURS = {**rq1.COLOURS, METHOD: "#7b5ea7"}
MARKERS = {**rq1.MARKERS, METHOD: "D"}


def addendum_version():
    """Hash of every Python file in addendum/."""
    digest = hashlib.sha1()
    for path in sorted((ROOT / "addendum").glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


# ---------------------------------------------------------------------- #
# The search batch
# ---------------------------------------------------------------------- #

def check_single(root=SINGLE, runs=RUNS):
    """Every GP-single run (and the copied tuned rule) complete with the Step 5 identity."""
    return check_complete(root, methods=(METHOD,), runs=runs)


def search_identity(root=SINGLE, runs=RUNS):
    """The task identity of the GP-single search batch and a hash of what its runs saved."""
    digest = hashlib.sha256()
    for run in runs:
        folder = run_folder(METHOD, run, root)
        for name in ("complete.json", "results/front.csv.gz", "results/summary.json", "results/final_population.csv"):
            digest.update(f"{folder.name}/{name}".encode())
            digest.update((folder / name).read_bytes())
    return {"task": json.loads((Path(root) / "task.json").read_text()), "results hash": digest.hexdigest()[:12],
            "path": str(root)}


def saved_schedule(folder, method, run):
    """The environment schedule a finished run saved in its checkpoint, after checking that the checkpoint is
    that run's, at its last generation. Only the project's own checkpoints are read this way: pickle is not a
    safe reader for files from elsewhere. Texts and numbers only; no class is unpickled."""
    state = pickle.loads((Path(folder) / "checkpoint" / "checkpoint.pkl").read_bytes())
    if state["method"] != method or state["run"] != run or state["generation"] != state["settings"]["generations"]:
        raise SystemExit(f"{folder}: the checkpoint is not the finished run {method} {run}")
    return state["schedule"]


def schedules_match(single_root=SINGLE, main_root=FORMAL, runs=RUNS, methods=MAIN_METHODS, method=METHOD):
    """The run numbers whose saved GP-single schedule differs from any main method's with the same number."""
    wrong = []
    for run in runs:
        mine = saved_schedule(run_folder(method, run, single_root), method, run)
        if any(saved_schedule(run_folder(m, run, main_root), m, run) != mine for m in methods):
            wrong.append(run)
    return wrong


def schedules_hash(root=SINGLE, runs=RUNS, method=METHOD):
    """A short hash of the compared schedules, kept in the pass record as the evidence of pairing."""
    text = json.dumps([saved_schedule(run_folder(method, r, root), method, r) for r in runs])
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------- #
# Reproduction and the rule list
# ---------------------------------------------------------------------- #

def reevaluate(folder, settings, dates=None, k=None):
    """Every distinct front rule of one run on the validation days, through evaluate6.score() as
    analysis/reproduce.py does, without shadows; the largest difference of the means and of the
    hypervolume against what Step 5 saved."""
    results = Path(folder) / "results"
    saved = read_table(results / "front.csv.gz")
    summary = json.loads((results / "summary.json").read_text())
    dates = validation_dates() if dates is None else dates
    k = settings["validation replicates"] if k is None else k
    days = days_for(settings)
    from gp_rule import from_text
    means = [reproduce.mean_objectives(from_text(t), dates, k, days, settings, knee=bool(knee))[0]
             for t, knee in zip(saved["text"], saved["knee"])]
    differences = [max(abs(m[0] - r), abs(m[1] - g)) for m, r, g in zip(means, saved["mean R_v"], saved["G_hat"])]
    from search import hypervolume, normalise
    area, _ = hypervolume(normalise(means, reproduce.value_range(settings)))
    return {"rules": len(means), "largest mean difference": max(differences),
            "hypervolume difference": abs(area - summary["hypervolume"])}


def check_run(job):
    folder, run, settings, dates, k, expected = job
    problems = reproduce.check_inputs(folder, METHOD, run, settings["population"], expected)
    static = reproduce.static_check(folder, reproduce.value_range(settings))
    again = reevaluate(folder, settings, dates, k)
    ok = (not problems and static["same rules and copies"] and static["same knee"]
          and static["hypervolume difference"] <= reproduce.TOLERANCE
          and again["largest mean difference"] <= reproduce.TOLERANCE
          and again["hypervolume difference"] <= reproduce.TOLERANCE)
    largest = max(static["hypervolume difference"], again["largest mean difference"], again["hypervolume difference"])
    return run, ok, largest


def make_rule_list(root=SINGLE, runs=RUNS):
    """The distinct front rules of the GP-single runs, kind gp, with each run's knee mark; no display forms."""
    rows = []
    for run in runs:
        front = read_table(run_folder(METHOD, run, root) / "results" / "front.csv.gz")
        group = f"{METHOD}_{run:02d}"
        for n, (text, is_knee) in enumerate(zip(front["text"], front["knee"])):
            rows.append({"rule id": f"{group}_{n:03d}", "group": group, "kind": "gp", "method": METHOD, "run": run,
                         "text": text, "knee": bool(is_knee), "nodes": int(front["nodes"].iloc[n])})
    table = pd.DataFrame(rows)
    check_rule_list(table, runs)
    return table


def check_rule_list(rules, runs=RUNS):
    problems = []
    if rules["rule id"].duplicated().any():
        problems.append("a repeated rule id")
    if set(rules["kind"]) != {"gp"} or set(rules["method"]) != {METHOD}:
        problems.append("a rule that is not a GP-single front rule")
    if set(rules["group"]) != {f"{METHOD}_{r:02d}" for r in runs}:
        problems.append("not every run is a group")
    knees = rules.groupby("group")["knee"].sum()
    if not (knees == 1).all():
        problems.append("a group without exactly one knee")
    if problems:
        raise SystemExit("the GP-single rule list does not cover what it must: " + "; ".join(problems))


def reproduce_main(workers, root=SINGLE, runs=RUNS, out=OUT, dates=None, k=None, settings=None, expected=STEP5,
                   main_root=FORMAL):
    """The reproduction check, then the rule list and its pass record, once. The keyword arguments other than
    out are for the tests on a pilot run (its settings and identity); the formal check uses the defaults."""
    rule_list, record_path = Path(out) / "rules.csv", Path(out) / "reproduced.json"
    if record_path.exists() or rule_list.exists():
        raise SystemExit(f"{out} already has a rule list or a pass record; the check is not run again")
    here = check_here()
    check_single(root, runs)
    wrong = schedules_match(root, main_root, runs)
    if wrong:
        raise SystemExit(f"saved schedules differ from the main methods' for runs {wrong}")
    settings = load_settings(SETTINGS_FILE) if settings is None else settings
    jobs = [(run_folder(METHOD, r, root), r, settings, dates, k, expected) for r in runs]
    failed, largest = [], 0.0
    with ProcessPoolExecutor(workers) as pool:
        for run, ok, diff in pool.map(check_run, jobs):
            largest = max(largest, diff)
            if not ok:
                failed.append(run)
    print(f"{len(jobs) - len(failed)} of {len(jobs)} GP-single runs reproduced; largest difference {largest:.1e}; "
          f"failed: {failed or 'none'}", flush=True)
    if failed or not math.isfinite(largest):
        raise SystemExit(1)
    table = make_rule_list(root, runs)
    Path(out).mkdir(parents=True, exist_ok=True)
    table.to_csv(rule_list, index=False)
    record = {"rule list hash": rt.rule_list_hash(rule_list), "rules": len(table),
              "analysis version": analysis_version(), "addendum version": addendum_version(),
              "search settings hash": settings_hash(settings), "packages": package_versions(), "step 5": here,
              "gp-single search": search_identity(root, runs), "schedules hash": schedules_hash(root, runs),
              "runs reproduced": len(jobs),
              "largest difference": largest}
    rt.write_atomic(record_path, json.loads(json.dumps(record)))
    print(f"rule list written: {rule_list} ({len(table)} rules); pass record: {record_path}", flush=True)


# ---------------------------------------------------------------------- #
# The test evaluation
# ---------------------------------------------------------------------- #

def identity_of(settings, test_path, rules_path, rules, formal=True, root=SINGLE, out=OUT, runs=RUNS,
                expected=STEP5):
    """This batch's identity: the main batch's terms, with the addendum version and the GP-single search batch
    in place of the Step 5 results; formal starts also need the pass record to match."""
    identity = {"mode": "formal" if formal else "not formal", "batch": "GP-single",
                "search settings hash": settings_hash(settings),
                "step 6 settings hash": rt.test_settings_hash(rt.load_test_settings(test_path)),
                "analysis version": analysis_version(), "addendum version": addendum_version(),
                "rule list hash": rt.rule_list_hash(rules_path), "packages": package_versions()}
    if not formal:
        return identity
    identity["step 5"] = check_here()
    identity["gp-single search"] = search_identity(root, runs)
    if identity["search settings hash"] != expected["settings hash"]:
        raise SystemExit("the settings loaded do not have the Step 5 settings hash")
    check_single(root, runs)
    check_rule_list(rules, runs)
    record_path = Path(out) / "reproduced.json"
    if not record_path.exists():
        raise SystemExit(f"no {record_path}: the GP-single reproduction check has not passed")
    record = json.loads(record_path.read_text())
    keys = ("rule list hash", "analysis version", "addendum version", "search settings hash", "packages", "step 5",
            "gp-single search")
    wrong = [k for k in keys if record.get(k) != json.loads(json.dumps(identity[k]))]
    if wrong or Path(rules_path).resolve() != (Path(out) / "rules.csv").resolve():
        raise SystemExit(f"the GP-single pass record does not match this start: {wrong or 'rule list path'}")
    return identity


def run_main(workers, out, rules_path, test_path, formal=True):
    settings = load_settings(SETTINGS_FILE)
    data = rt.load_test_settings(test_path)
    if "scenario" in data or data["year"] != 2019:
        raise SystemExit("GP-single is evaluated on the 2019 main setting only (addendum, section 2a)")
    if not formal:
        if data["split"] != "validation" or Path(out).resolve().is_relative_to(rt.OUTPUTS.resolve()):
            raise SystemExit("a run that is not formal uses validation days and writes outside outputs/test/")
    elif data["split"] != "test":
        raise SystemExit("the formal GP-single batch uses the test days")
    rules = read_table(rules_path)
    dates = rt.dates_of(data)
    identity = identity_of(settings, test_path, rules_path, rules, formal=formal)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    task = out / "task.json"
    if task.exists() and json.loads(task.read_text()) != json.loads(json.dumps(identity)):
        raise SystemExit(f"{task} names another identity; nothing is run")
    lock = rt.take_lock(out)
    pool = None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        rt.write_atomic(task, identity)
        jobs = rt.jobs_of(out, rules, dates, data, settings, identity)
        total = math.ceil(len(dates) / data["dates per block"]) * rules["group"].nunique()
        print(f"{len(jobs)} of {total} blocks to do in {out}, {workers} at a time; {len(rules)} rules, "
              f"{len(dates)} dates, {data['replicates']} replicates", flush=True)
        failed = []
        pool = Pool(workers)
        handles = [pool.apply_async(rt.run_block, (job,)) for job in jobs]
        for n, (job, handle) in enumerate(zip(jobs, handles), 1):
            try:
                handle.get()
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


def load_single(out=OUT, rules_path=RULE_LIST, test_path=rt.TEST_SETTINGS, formal=True):
    """The GP-single outputs, checked as a whole as results.load() checks the main batch."""
    out = Path(out)
    if (out / "launcher.lock").exists():
        raise SystemExit(f"{out / 'launcher.lock'} is there: the batch may still be running; nothing is read")
    rules = read_table(rules_path)
    data = rt.load_test_settings(test_path)
    identity = identity_of(load_settings(SETTINGS_FILE), test_path, rules_path, rules, formal=formal)
    task = json.loads((out / "task.json").read_text())
    if task != json.loads(json.dumps(identity)):
        raise SystemExit(f"{out / 'task.json'} is not the identity of the code, settings and rule list here")
    dates = rt.dates_of(data)
    missing, bad = rt.verify(out, rules, dates, data, task)
    if missing or bad:
        raise SystemExit(f"{len(missing)} blocks missing, {len(bad)} changed or of another identity")
    return rs.Results(out, rules, dates, data, identity)


def check_batches(single, main):
    """Before any table: the two batches must share the data, settings, dates and replicates, and the main
    batch must be the 2019 main setting; each keeps its own rule list and code identity."""
    a, b = single.identity, main.identity
    problems = []
    for key in ("search settings hash", "step 6 settings hash", "analysis version", "packages"):
        if a.get(key) != b.get(key):
            problems.append(key)
    if a.get("step 5") != b.get("step 5"):
        problems.append("step 5 (code, data version and content)")
    if single.dates != main.dates or single.data["replicates"] != main.data["replicates"]:
        problems.append("dates or replicates")
    if not rq1.is_main_test(main.data) or not rq1.is_main_test(single.data):
        problems.append("not the 2019 main setting")
    if problems:
        raise SystemExit("the GP-single and main batches cannot be read together: " + ", ".join(problems))
    return {"GP-single": {k: a.get(k) for k in ("rule list hash", "addendum version", "gp-single search")},
            "main": {k: b.get(k) for k in ("rule list hash", "step 5 results")},
            "shared": {k: a.get(k) for k in ("search settings hash", "step 6 settings hash", "analysis version",
                                             "packages", "step 5")},
            "main path": str(Path(main.out).resolve()), "GP-single path": str(Path(single.out).resolve())}


# ---------------------------------------------------------------------- #
# Tables S1-S3 and Figure S1
# ---------------------------------------------------------------------- #

def single_fronts(single, alpha, value_range):
    rules, means = single.rules, single.rule_means()
    rows = []
    for run, members in rules.groupby("run"):
        out = rq1.score_front(means, list(members["rule id"]), alpha, value_range)
        rows.append({"method": METHOD, "run": int(run), **{k: v for k, v in out.items() if k != "front"}})
    return pd.DataFrame(rows)


def all_fronts(single, main, alpha, value_range):
    return pd.concat([single_fronts(single, alpha, value_range), rq1.run_fronts(main, alpha, value_range)],
                     ignore_index=True)


def table_s1(fronts, validation):
    rows = []
    for method in ALL_METHODS:
        m = fronts[fronts["method"] == method]
        test, size = rs.quartiles(m["hypervolume"]), rs.quartiles(m["front rules"])
        val = rs.quartiles([validation.get((method, r)) for r in m["run"]])
        rows.append({"method": method, "runs": test["n"], "test HV median": test["median"], "test HV q1": test["q1"],
                     "test HV q3": test["q3"], "validation HV median": val["median"], "validation HV q1": val["q1"],
                     "validation HV q3": val["q3"], "test front rules median": size["median"],
                     "test front rules q1": size["q1"], "test front rules q3": size["q3"]})
    tuned = fronts[fronts["method"] == "tuned"].iloc[0]
    rows.append({"method": "tuned (one value, not tested)", "runs": 1, "test HV median": tuned["hypervolume"],
                 "test front rules median": tuned["front rules"]})
    return pd.DataFrame(rows)


def table_s2(fronts):
    """GP-single minus each two-tree method, paired by run number, by the frozen stats functions: n, zeros,
    median difference with its per-comparison 95% interval, rank-biserial r. No p-value is kept."""
    rows = []
    mine = rq1.by_run(fronts, METHOD)
    for other in MAIN_METHODS:
        out = stats.compare(mine, rq1.by_run(fronts, other))
        rows.append({"pair": f"{METHOD} - {other}", "n": out["n"], "zeros": out["zeros"], "m": out["m"],
                     "median difference": out["median median"], "95% low": out["median low"],
                     "95% high": out["median high"], "coverage": out["median coverage"],
                     "rank-biserial r": out["r"], "note": out["note"],
                     "interval": "per comparison, not adjusted for the three"})
    return pd.DataFrame(rows)


def other_ranges(single, main, settings):
    """The wider and narrower ranges: ranking of the four methods by median hypervolume and the GP-single paired
    median differences with their intervals; no tests."""
    rows = []
    for name, value_range in (("main range", rq1.main_range(settings)), ("wider range", rq1.WIDER),
                              ("narrower range", rq1.NARROWER)):
        fronts = all_fronts(single, main, rs.MAIN, value_range)
        medians = {m: rs.quartiles(fronts.loc[fronts["method"] == m, "hypervolume"])["median"] for m in ALL_METHODS}
        row = {"setting": name, "ranking by median HV": rq1.ranking_of(medians),
               **{f"{m} median HV": medians[m] for m in ALL_METHODS}}
        for other in MAIN_METHODS:
            _, d = stats.paired_differences(rq1.by_run(fronts, METHOD), rq1.by_run(fronts, other))
            iv = stats.median_interval(d)
            row[f"{METHOD} - {other}"] = iv["median"]
            row[f"{METHOD} - {other} 95%"] = (iv["low"], iv["high"])
        rows.append(row)
    return pd.DataFrame(rows)


def dominated_by_single(single, main, settings):
    """For each hand rule and the control: how many GP-single runs have a test front that dominates it."""
    alpha, value_range = rs.MAIN, rq1.main_range(settings)
    means, hand_means = single.rule_means(), main.rule_means()
    r, g = rs.r_column(alpha), "G_hat"
    fronts = []
    for _, members in single.rules.groupby("run"):
        ids = rq1.score_front(means, list(members["rule id"]), alpha, value_range)["front"]
        fronts.append([(means.loc[i, r], means.loc[i, g]) for i in ids])
    rows = []
    for _, row in main.rules[main.rules["kind"].isin(["hand", "control"])].iterrows():
        p = (hand_means.loc[row["rule id"], r], hand_means.loc[row["rule id"], g])
        count = sum(any(q[0] >= p[0] and q[1] <= p[1] and (q[0] > p[0] or q[1] < p[1]) for q in pts) for pts in fronts)
        rows.append({"rule": row["text"], "mean R_v": p[0], "G_hat": p[1], f"runs of {METHOD} dominating it": count,
                     "runs": len(fronts)})
    return pd.DataFrame(rows)


def uses(text, names):
    """True if the rule text names any of the terminals (as whole words)."""
    import re
    return any(re.search(rf"\b{n}\b", text) for n in names)


def table_s3(single, main):
    """The knees: mean R_v, G_hat and completion (median and quartiles over each method's knees, with the
    completion mark), knee size (nodes of the full rule), and the GP-single knees' terminal shares."""
    rows = []
    for name, res, methods in ((METHOD, single, (METHOD,)), ("main", main, MAIN_METHODS)):
        rules = res.rules
        means, done = res.rule_means(), res.completion()
        for method in methods:
            k = rules[(rules["kind"] == "gp") & (rules["method"] == method) & rules["knee"].astype(bool)]
            ids = list(k["rule id"])
            if method == METHOD:
                nodes = list(k["nodes"])
            else:
                from gp_rule import from_text, size
                nodes = [sum(n for n, _ in size(from_text(t))) for t in k["text"]]
            row = {"method": method, "knees": len(ids)}
            for label, values in (("mean R_v", means.loc[ids, rs.r_column()]), ("G_hat", means.loc[ids, "G_hat"]),
                                  ("completion", done.loc[ids, "completion"]), ("knee nodes", nodes)):
                q = rs.quartiles(list(values))
                row.update({label: q["median"], f"{label} q1": q["q1"], f"{label} q3": q["q3"]})
            row["knees with completion below 0.95"] = int((done.loc[ids, "completion"] < 0.95).sum())
            if method == METHOD:
                row["share of knees with VS"] = float(np.mean([uses(t, ["VS"]) for t in k["text"]]))
                row["share of knees with an h2-only terminal"] = float(np.mean([uses(t, H2_ONLY) for t in k["text"]]))
            rows.append(row)
    return pd.DataFrame(rows)


def figure_s1(fronts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    rq1.style(ax)
    data = [fronts.loc[fronts["method"] == m, "hypervolume"].to_numpy() for m in ALL_METHODS]
    ax.boxplot(data, positions=range(len(ALL_METHODS)), widths=0.5, showfliers=False,
               medianprops={"color": rq1.INK}, boxprops={"color": rq1.MUTED}, whiskerprops={"color": rq1.MUTED},
               capprops={"color": rq1.MUTED})
    rng = np.random.default_rng(0)
    for i, (m, values) in enumerate(zip(ALL_METHODS, data)):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(values)), values, s=18, color=COLOURS[m], marker=MARKERS[m],
                   zorder=3, label=f"{m} ({len(values)} runs)")
    tuned = fronts.loc[fronts["method"] == "tuned", "hypervolume"].iloc[0]
    ax.axhline(tuned, color=COLOURS["tuned"], linestyle="--", linewidth=1.5, label="tuned weighted rule (one value)")
    ax.set_xticks(range(len(ALL_METHODS)), ALL_METHODS)
    ax.set_ylabel("test hypervolume", color=rq1.INK)
    ax.set_title("Test hypervolume of each run, with GP-single (exploratory)", color=rq1.INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def tables_main(single, main, settings, folder, single_root=SINGLE, step5_root=FORMAL):
    folder = Path(folder)
    if (folder / "sources.json").exists():  # written last: the tables are made once
        raise SystemExit(f"{folder} already holds the GP-single tables; they are made once")
    folder.mkdir(parents=True, exist_ok=True)
    sources = check_batches(single, main)
    fronts = all_fronts(single, main, rs.MAIN, rq1.main_range(settings))
    gp = fronts[fronts["method"].isin(MAIN_METHODS)]
    validation = rq1.validation_hypervolumes(step5_root, list(zip(gp["method"], gp["run"])))
    validation.update({(METHOD, r): json.loads((run_folder(METHOD, r, single_root) / "results" / "summary.json")
                                                .read_text())["hypervolume"]
                       for r in fronts.loc[fronts["method"] == METHOD, "run"]})
    fronts.to_csv(folder / "test_fronts.csv", index=False)
    table_s1(fronts, validation).to_csv(folder / "table_s1.csv", index=False)
    table_s2(fronts).to_csv(folder / "table_s2.csv", index=False)
    table_s3(single, main).to_csv(folder / "table_s3.csv", index=False)
    dominated_by_single(single, main, settings).to_csv(folder / "hand_rules_dominated.csv", index=False)
    other_ranges(single, main, settings).to_csv(folder / "other_ranges.csv", index=False)
    figure_s1(fronts, folder / "figure_s1.png")
    rt.write_atomic(folder / "sources.json", json.loads(json.dumps(sources)))
    return folder


# ---------------------------------------------------------------------- #
# Table S4: the walking weight
# ---------------------------------------------------------------------- #

PARTS = ["A", "C", "T", "A observed", "C observed", "T observed"]


def r_at_beta(v, beta, alpha=rs.MAIN):
    """Each visitor's R_v at (alpha, beta) from the saved parts, by the frozen route_value and relative_value:
    the new and the observed route at the same beta, the denominator the observed A."""
    new = tuple(v[c].to_numpy(float) for c in ("A", "C", "T"))
    old = tuple(v[c].to_numpy(float) for c in ("A observed", "C observed", "T observed"))
    return relative_value(route_value(*new, alpha, beta), route_value(*old, alpha, beta), old[0])


def check_knee_rows(v, ids, dates, replicates):
    """The inputs of the re-scoring: unique keys, every date and replicate of every knee, finite parts, A_O > 0."""
    if v.duplicated(rt.KEYS + ["visitor"]).any():
        raise ValueError("knee rows: a repeated (rule, date, replicate, visitor)")
    have = set(map(tuple, v[rt.KEYS].astype(str).drop_duplicates().to_numpy()))
    need = {(i, d, str(r)) for i in ids for d in dates for r in range(replicates)}
    if have != need:
        raise ValueError("knee rows: not every knee on every date and replicate")
    if not np.isfinite(v[PARTS].to_numpy(float)).all() or not (v["A observed"] > 0).all():
        raise ValueError("knee rows: a non-finite part or an observed A that is not positive")


def check_saved_r(v):
    """At beta = 0.5 every visitor's re-scored R_v equals the R_v saved in the same row, exactly."""
    saved = v[f"R_v (alpha {evaluate6.label(rs.MAIN)})"].to_numpy(float)
    again = r_at_beta(v, MAIN_BETA)
    if not np.isfinite(saved).all():
        raise ValueError("knee rows: a non-finite saved R_v")
    if not np.array_equal(again, saved):
        bad = v.loc[again != saved, rt.KEYS].drop_duplicates().head(3).to_dict("records")
        raise AssertionError(f"at beta 0.5 a visitor's R_v differs from the saved one (largest difference "
                             f"{np.max(np.abs(again - saved)):.1e}; first at {bad})")


def knee_beta_measures(v, betas=BETAS):
    """One knee: at each beta, its mean R_v as the test means are made (the mean over visitors of each date and
    replicate, then over dates and replicates) and the share of visitors with R_v < 1 pooled over them."""
    out = {}
    for beta in betas:
        rv = pd.Series(r_at_beta(v, beta), index=v.index)
        # each day as evaluate() takes it (Series.mean of the day's rows, in their order), then the days as
        # results.rule_means() averages the objectives (a groupby mean over the rows, in block order)
        daily = [float(pd.Series(rv[idx].to_numpy()).mean())
                 for idx in v.groupby(["date", "replicate"], sort=False).groups.values()]
        mean = pd.DataFrame({"rule": 0, "m": daily}).groupby("rule")["m"].mean().iloc[0]
        out[beta] = {"mean R_v": float(mean), "share R_v < 1": float((rv < 1).mean())}
    return out


def beta_rows(res, label):
    rules = res.rules
    knees = rules[rules["knee"].astype(bool) & rules["kind"].isin(["gp", "tuned", "hand", "control"])]
    rows = []
    for group in res.groups():
        ids = list(knees.loc[knees["group"] == group, "rule id"])
        if not ids:
            continue
        v = res.table("visitors", group)
        v = v[v["rule id"].isin(ids)]
        check_knee_rows(v, ids, res.dates, res.data["replicates"])
        check_saved_r(v)
        for rule_id, k in v.groupby("rule id", sort=False):
            meta = knees[knees["rule id"] == rule_id].iloc[0]
            for beta, m in knee_beta_measures(k).items():
                rows.append({"batch": label, "rule id": rule_id, "method": meta["method"], "kind": meta["kind"],
                             "beta": beta, **m})
        del v
    return pd.DataFrame(rows)


def check_at_main_beta(rows, res):
    """At beta = 0.5 the re-scored knees must give, value for value, the main outputs: the mean R_v of the test
    means (results.rule_means) and the share below 1 of the main RQ3 knee rows."""
    import rq3
    at = rows[rows["beta"] == MAIN_BETA].set_index("rule id")
    means = res.rule_means()
    if not (at["mean R_v"] == means.loc[at.index, rs.r_column()]).all():
        raise AssertionError("at beta 0.5 the knee mean R_v differs from the test means")
    knees = rq3.knee_tables(res, MAIN_BETA)
    overall = knees[knees["dimension"] == "all"].set_index("rule id")
    if not (at["share R_v < 1"] == overall.loc[at.index, f"R_v < 1 (alpha {rs.MAIN:g})"]).all():
        raise AssertionError("at beta 0.5 the share below 1 differs from Table 5's knee rows")


def table_s4(rows, means_g):
    """Per beta and method: median and quartiles over the knees of mean R_v and the share below 1, the median
    knee G_hat beside them, and the order of the methods by median knee mean R_v against the order at 0.5."""
    out, orders = [], {}
    gp = rows[rows["kind"] == "gp"]
    for beta in BETAS:
        medians = {}
        for method, k in gp[gp["beta"] == beta].groupby("method", sort=False):
            r, s = rs.quartiles(k["mean R_v"]), rs.quartiles(k["share R_v < 1"])
            g = rs.quartiles(means_g.loc[k["rule id"]])
            medians[method] = r["median"]
            out.append({"beta": beta, "method": method, "knees": r["n"], "mean R_v": r["median"],
                        "mean R_v q1": r["q1"], "mean R_v q3": r["q3"], "share R_v < 1": s["median"],
                        "share R_v < 1 q1": s["q1"], "share R_v < 1 q3": s["q3"], "knee G_hat (median)": g["median"]})
        for _, row in rows[(rows["beta"] == beta) & (rows["kind"] != "gp")].iterrows():
            out.append({"beta": beta, "method": row["method"], "knees": 1, "mean R_v": row["mean R_v"],
                        "share R_v < 1": row["share R_v < 1"], "knee G_hat (median)": float(means_g.loc[row["rule id"]])})
        orders[beta] = rq1.ranking_of(medians)
    table = pd.DataFrame(out)
    table["order of methods by median knee mean R_v"] = table["beta"].map(orders)
    table["order differs from beta 0.5"] = table["beta"].map(lambda b: orders[b] != orders[MAIN_BETA])
    return table


def beta_main(main, folder, single=None, single_root=SINGLE):
    """Table S4 from the verified main batch; with single, the separately labelled version with GP-single,
    whose main-batch values must equal the first version's."""
    folder = Path(folder)
    name = "table_s4" if single is None else "table_s4_with_gp_single"
    if (folder / f"{name}_sources.json").exists():  # written last: each version is made once
        raise SystemExit(f"{folder / name} is there already; it is made once")
    folder.mkdir(parents=True, exist_ok=True)
    rows = beta_rows(main, "main")
    check_at_main_beta(rows, main)
    g = main.rule_means()["G_hat"]
    if single is not None:
        check_batches(single, main)
        first = folder / "knees_beta.csv"
        if not first.exists():
            raise SystemExit("the main-batch Table S4 must be made first")
        before = read_table(first)
        if not before.reset_index(drop=True).equals(read_table_roundtrip(rows)):
            raise AssertionError("the main-batch values differ from the first Table S4")
        extra = beta_rows(single, METHOD)
        check_at_main_beta(extra, single)
        rows = pd.concat([extra, rows], ignore_index=True)
        g = pd.concat([single.rule_means()["G_hat"], g])
    else:
        rows.to_csv(folder / "knees_beta.csv", index=False)
    table_s4(rows, g).to_csv(folder / f"{name}.csv", index=False)
    knees = rows[["batch", "rule id", "method", "kind"]].drop_duplicates()
    record = {"alpha": rs.MAIN, "betas": list(BETAS), "dates": list(main.dates), "replicates": main.data["replicates"],
              "knees": knees.to_dict("records"),
              "main batch": {k: main.identity.get(k) for k in ("rule list hash", "analysis version", "step 5 results")},
              "main path": str(Path(main.out).resolve()),
              "main visitor files": file_hashes(main),
              "addendum version": addendum_version(), "code version": main.identity.get("step 5", {}).get("code version"),
              "mean R_v": "mean over visitors of each date and replicate, then over dates and replicates",
              "share R_v < 1": "pooled over visitors, dates and replicates"}
    if single is not None:
        record["GP-single batch"] = {k: single.identity.get(k) for k in ("rule list hash", "gp-single search")}
        record["GP-single path"] = str(Path(single.out).resolve())
        record["GP-single visitor files"] = file_hashes(single)
    rt.write_atomic(folder / f"{name}_sources.json", json.loads(json.dumps(record)))
    return folder


def read_table_roundtrip(table):
    """A table as it reads back from CSV, to compare with a saved one."""
    import io
    buffer = io.StringIO()
    table.to_csv(buffer, index=False)
    buffer.seek(0)
    return pd.read_csv(buffer, float_precision="round_trip")


def file_hashes(res):
    return {str(p.relative_to(res.out)): hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in res.files("visitors")}


# ---------------------------------------------------------------------- #

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["check", "reproduce", "run", "verify", "tables", "beta"])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--with-single", action="store_true", help="beta: add the GP-single knees")
    args = parser.parse_args(argv)
    if args.step == "check":
        check_single()
        wrong = schedules_match()
        print(f"{len(RUNS)} GP-single runs complete; schedules differing from the main methods': {wrong or 'none'}")
        if wrong:
            raise SystemExit(1)
    elif args.step == "reproduce":
        reproduce_main(args.workers)
    elif args.step == "run":
        run_main(args.workers, OUT, RULE_LIST, rt.TEST_SETTINGS)
    elif args.step == "verify":
        load_single()  # the same checks as every reader: no lock, task.json, every block and hash
        print("0 blocks missing, 0 changed or of another identity", flush=True)
    elif args.step == "tables":
        main_batch = rs.load(rt.OUT, rt.RULE_LIST)
        print(tables_main(load_single(), main_batch, load_settings(SETTINGS_FILE), OUT / "analysis"))
    else:
        main_batch = rs.load(rt.OUT, rt.RULE_LIST)
        single = load_single() if args.with_single else None
        print(beta_main(main_batch, rt.OUTPUTS / "2019_beta", single))


if __name__ == "__main__":
    main()
