"""
The formal test evaluation (Step 6 plan, sections 3 to 5)
=========================================================

Every rule of the rule list on every test day and replicate, one process per
block, several blocks at once. A block is one group of rules (the front rules
of one run, or the baselines) on a few consecutive dates.

The rule list is written once, by the reproduction check (write_rule_list),
with its content hash; this file reads only that list. It holds every distinct
front rule of the 90 runs with its knee flag, every weight of the tuned rule
(its knee chosen on the validation days by the frozen knee rule, ties to the
smaller weight), the four hand rules and the observed-order control.

Identity. At the start the launcher fixes what the batch is: the Step 5
identity of the code and tables it runs on (checked), the Step 6 settings hash,
the analysis code version, the rule list hash and the package versions, written
to task.json. A folder holding another task.json is refused before anything
runs. Every block carries the same identity in its done.json.

Completion. A block is written to a temporary folder, read back (every file;
unique keys; finite objectives; one row per rule, date and replicate), then
renamed into place, and done.json is written last. A block without done.json is
computed again from the start; one with done.json of another identity stops the
batch. One launcher per output folder (launcher.lock). Ctrl-C or a termination
signal stops the workers before the lock is released.

  python3 analysis/run_test.py --workers 12                    # from the frozen Step 6 snapshot
  python3 analysis/run_test.py --rules R --settings S --out O  # other lists or days, e.g. the measured run
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step5 import (FORMAL, GP_METHODS, ROOT, RUNS, SETTINGS_FILE, analysis_version, check_here,  # noqa: E402
                   package_versions, run_folder)
import evaluate6  # noqa: E402
import forms  # noqa: E402

from day import PROCESSED  # noqa: E402
from engine import ObservedOrder  # noqa: E402
from evaluate import Days  # noqa: E402
from gp_rule import from_text, to_text  # noqa: E402
from rules import RULES, TUNED_WEIGHTS, Tuned  # noqa: E402
from search import days_for, knee, load_settings, non_dominated, normalise  # noqa: E402

TEST_SETTINGS = ROOT / "configs" / "test.json"
OUT = ROOT / "outputs" / "test" / "2019"
RULE_LIST = OUT / "rules.csv"
KEYS = ["rule id", "date", "replicate"]
CONTROL = "observed order"


# ---------------------------------------------------------------------- #
# Settings, days and the rule list
# ---------------------------------------------------------------------- #

def load_test_settings(path=TEST_SETTINGS):
    data = json.loads(Path(path).read_text())
    required = {"comment", "year", "split", "replicates", "alphas", "dates per block"}
    if set(data) != required:
        raise ValueError(f"{path}: needs exactly {sorted(required)}")
    if [float(a) for a in data["alphas"]] != list(evaluate6.ALPHAS):
        raise ValueError(f"{path}: the alpha grid is fixed by the plan: {evaluate6.ALPHAS}")
    return data


def test_settings_hash(data):
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:10]


def dates_of(data):
    """The dates of the year and split, without any day that has no evaluated visitor (plan, section 11):
    such a day has no mean R_v. Decided from the processed visitor table, before any rule runs."""
    days = pd.read_csv(PROCESSED / f"days_{data['year']}.csv")
    if data["split"] != "all":
        days = days[days["split"] == data["split"]]
    visitors = pd.read_csv(PROCESSED / f"visitors_{data['year']}.csv.gz", usecols=["date", "stops"])
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


def write_rule_list(path=RULE_LIST, root=FORMAL, methods=GP_METHODS, runs=RUNS):
    """Written once, by the reproduction check. Refuses to overwrite a list that is there."""
    path = Path(path)
    if path.exists():
        raise SystemExit(f"{path} is there already; the rule list is written once")
    settings = load_settings(SETTINGS_FILE)
    rows = []
    for method in methods:
        for run in runs:
            front = pd.read_csv(run_folder(method, run, root) / "results" / "front.csv.gz")
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
    tuned = pd.read_csv(Path(root) / "tuned" / "tuned_front.csv")
    # the table went through a CSV file: its weights are matched to the frozen grid by position, and the
    # rules are built from the frozen values, never from the numbers read back
    if len(tuned) != len(TUNED_WEIGHTS) or not np.allclose(tuned["weight"], TUNED_WEIGHTS, rtol=1e-12, atol=0):
        raise SystemExit("the tuned rule's weights are not the frozen grid")
    knee_row = tuned_knee_weight(tuned, settings)
    for n, (w, on_front) in enumerate(zip(TUNED_WEIGHTS, tuned["front"])):
        rows.append({"rule id": f"tuned_{n:02d}", "group": "baselines", "kind": "tuned", "method": "tuned",
                     "run": -1, "text": repr(float(w)), "knee": n == knee_row, "validation front": bool(on_front)})
    for name in list(RULES) + [CONTROL]:
        rows.append({"rule id": name, "group": "baselines", "kind": "hand" if name in RULES else "control",
                     "method": name, "run": -1, "text": name, "knee": True})
    table = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def rule_list_hash(path=RULE_LIST):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def policy(row):
    if row["kind"] == "gp":
        return from_text(row["text"])
    if row["kind"] == "display":  # built from the rounded tidy form; its constants may leave the search range
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
_days = {}


def test_days(settings):
    key = (settings["budget"], settings["same day"], settings["peak floor"])
    if key not in _days:
        _days[key] = Days(budget=settings["budget"], same_day=settings["same day"],
                          peak_floor=settings["peak floor"], test=True)
    return _days[key]


def compute_block(rules, dates, replicates, settings):
    """Every rule of the block on every date and replicate; the tables of the block."""
    days = test_days(settings)
    tables = {name: [] for name in FILES}
    shown = {}
    if "parent" in rules:
        shown = {p: t for p, t in zip(rules["parent"], rules["text"]) if isinstance(p, str)}
    for _, row in rules.iterrows():
        rule = policy(row)
        watch = None
        if row["kind"] == "gp" and bool(row["knee"]) and row["rule id"] in shown:
            watch = from_text(shown[row["rule id"]], strict=False)
        for date in dates:
            for r in range(replicates):
                out = evaluate6.score(rule, date, r, days, beta=settings["beta"], stop_limit=settings["stop limit"],
                                      knee=bool(row["knee"]), watch=watch)
                tag = {"rule id": row["rule id"], "date": date, "replicate": r}
                tables["objectives"].append({**tag, **out["objectives"]})
                for name in FILES[1:]:
                    if name in out:
                        tables[name].append(out[name].assign(**tag))
    return {name: (pd.DataFrame(rows) if name == "objectives" else pd.concat(rows, ignore_index=True))
            for name, rows in tables.items() if len(rows)}


def read_back(folder, tables, rules, dates, replicates):
    """Every file reads back with the rows written; the objectives have one finite row per rule, date and
    replicate of the block."""
    for name, table in tables.items():
        again = pd.read_csv(folder / f"{name}.csv.gz")
        if len(again) != len(table) or list(again.columns) != list(table.columns):
            raise ValueError(f"{name} does not read back as written")
    objectives = pd.read_csv(folder / "objectives.csv.gz")
    expected = len(rules) * len(dates) * replicates
    if len(objectives) != expected or objectives.duplicated(KEYS).any():
        raise ValueError(f"objectives: {len(objectives)} rows, expected {expected} unique")
    values = objectives.drop(columns=KEYS).to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("a non-finite objective")


def run_block(job):
    """Compute one block, write it to a temporary folder, read it back, move it into place, mark it done."""
    folder, rules, dates, replicates, settings, identity = job
    folder = Path(folder)
    rules = pd.DataFrame(rules)
    temporary = folder.with_name(folder.name + ".partial")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    started = time.time()
    tables = compute_block(rules, dates, replicates, settings)
    for name, table in tables.items():
        table.to_csv(temporary / f"{name}.csv.gz", index=False)
    read_back(temporary, tables, rules, dates, replicates)
    shutil.rmtree(folder, ignore_errors=True)
    temporary.rename(folder)
    done = {**identity, "rules": list(rules["rule id"]), "dates": dates, "replicates": replicates,
            "seconds": round(time.time() - started, 1)}
    write_atomic(folder / "done.json", done)
    return folder.name


def write_atomic(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=1))
    os.replace(temporary, path)


# ---------------------------------------------------------------------- #
# The batch
# ---------------------------------------------------------------------- #

def identity_of(settings_path, test_path, rules_path, check=True):
    here = check_here() if check else None
    return {"step 5": here, "step 6 settings hash": test_settings_hash(load_test_settings(test_path)),
            "analysis version": analysis_version(), "rule list hash": rule_list_hash(rules_path),
            "packages": package_versions(), "search settings": str(settings_path)}


def jobs_of(out, rules, dates, data, settings, identity):
    """The blocks still to do; a block marked done under another identity stops the batch."""
    jobs, wrong = [], []
    per = data["dates per block"]
    for group, members in rules.groupby("group", sort=False):
        for start in range(0, len(dates), per):
            chunk = dates[start:start + per]
            folder = Path(out) / "blocks" / group / chunk[0]
            mark = folder / "done.json"
            if mark.exists():
                done = json.loads(mark.read_text())
                if {k: done.get(k) for k in identity} != identity or done.get("dates") != chunk:
                    wrong.append(str(folder))
                continue
            jobs.append((str(folder), members.to_dict("list"), chunk, data["replicates"], settings, identity))
    if wrong:
        raise SystemExit("blocks of another identity are there; nothing is run:\n  " + "\n  ".join(wrong[:10]))
    return jobs


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
    parser.add_argument("--no-identity-check", action="store_true",
                        help="only for runs on pilot rules and validation days, never for the formal batch")
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = load_settings(args.settings)
    data = load_test_settings(args.test_settings)
    rules = pd.read_csv(args.rules)
    if rules["rule id"].duplicated().any():
        raise SystemExit("the rule list has a repeated rule id")
    dates = dates_of(data)
    identity = identity_of(args.settings, args.test_settings, args.rules, check=not args.no_identity_check)
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
