"""
Self-check after each sub-step
==============================

Run before handing a step over for review:

  1. All tests, with warnings treated as errors (two known, harmless numpy
     warnings from averaging empty slices are allowed).
  2. The check scripts: every assertion in them must pass.
  3. Deliberate bugs: each mutation below is put into a copy of the code, and
     the tests must fail. A mutation the tests miss means the tests are weak.
  4. A sweep over every day of 2019 and some days of the other years.
  5. Which committed outputs changed after the checks were rerun.

The sixth step, reading the code line by line against the design, is done by hand.

Run:  python3 tools/self_check.py   (from the repository root)
"""

import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
CHECK_SCRIPTS = ["src/replay_check.py"]
ALLOWED_WARNINGS = ["ignore:Mean of empty slice:RuntimeWarning",
                    "ignore:invalid value encountered in scalar divide:RuntimeWarning"]

# (file, text, deliberate bug, what it breaks). Add one or more for every new piece of logic.
MUTATIONS = [
    ("src/scoring.py", "t = walk[sites[:-1], sites[1:]].sum() / 60", "t = walk[sites[:-1], sites[1:]].sum()",
     "walking left in minutes"),
    ("src/scoring.py", "return person_minutes / 60 / np.asarray(usual_peak)[:, None]",
     "return person_minutes / np.asarray(usual_peak)[:, None]", "crowding not divided by 60"),
    ("src/scoring.py", "overlap = np.minimum(end, 60 * (hour + 1)) - np.maximum(start, 60 * hour)",
     "overlap = np.minimum(end, 60 * (hour + 1)) - start", "overlap not cut at the start of the hour"),
    ("src/replay.py", 'return route_value(b["A"], b["C"], b["T"], alpha, beta)',
     "return route_value(b.A, b.C, b.T, alpha, beta)", "the .T transpose bug"),
    ("src/day.py", "opening = [rows.o.get(i, sites.o[i]) for i in sites.index]",
     "opening = [rows.o.get(i, 0.0) for i in sites.index]", "missing month opens at midnight"),
    ("src/day.py", 'v = visitors.sort_values("visitor").reset_index(drop=True)',
     "v = visitors.reset_index(drop=True)", "visitors not numbered by id"),
    ("src/day.py", "return np.array([i in open_ids and i not in LOW_COVERAGE for i in site_ids])",
     "return np.array([i in open_ids for i in site_ids])", "low-coverage site available"),
    ("src/preprocess.py", "out[\"exit\"] = np.fmin(visits.t + visits.site.map(d_i), visits.next_t - visits.walk)",
     "out[\"exit\"] = np.fmax(visits.t + visits.site.map(d_i), visits.next_t - visits.walk)",
     "occupancy runs to the next entry"),
    ("src/preprocess.py", "nodes = np.searchsorted(minutes, part.t.values, side=\"left\")",
     "nodes = np.searchsorted(minutes, np.floor(part.t.values), side=\"left\")", "seconds rounded down"),
    ("src/toptw.py", "feasible &= np.maximum(arrive, instance.opening) + expected + back <= end_time",
     "feasible &= arrive + expected + back <= end_time", "Step 1: no wait for opening"),
]


def run(command, cwd=ROOT):
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True)


def pytest_command():
    command = [PYTHON, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "-W", "error"]
    for allowed in ALLOWED_WARNINGS:
        command += ["-W", allowed]
    return command


# ---------------------------------------------------------------------- #
# 1-2. Tests and check scripts
# ---------------------------------------------------------------------- #

def check_tests():
    result = run(pytest_command())
    last = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else result.stderr[-300:]
    return result.returncode == 0, last


def check_scripts():
    rows = []
    for script in CHECK_SCRIPTS:
        result = run([PYTHON, script])
        rows.append((script, result.returncode == 0, result.stderr.strip()[-300:]))
    return rows


# ---------------------------------------------------------------------- #
# 3. Deliberate bugs, each in a fresh copy of the code
# ---------------------------------------------------------------------- #

def check_mutations():
    rows = []
    for path, text, bug, meaning in MUTATIONS:
        original = (ROOT / path).read_text()
        if original.count(text) != 1:
            rows.append((path, meaning, "text not found: update this mutation"))
            continue
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder)
            shutil.copytree(ROOT / "src", copy / "src")
            shutil.copytree(ROOT / "tests", copy / "tests")
            (copy / "data").symlink_to(ROOT / "data")  # read-only use of the real data
            (copy / path).write_text(original.replace(text, bug))
            result = run(pytest_command(), cwd=copy)
        rows.append((path, meaning, "caught" if result.returncode != 0 else "NOT CAUGHT"))
    return rows


# ---------------------------------------------------------------------- #
# 4. Sweep over days
# ---------------------------------------------------------------------- #

def sweep_days():
    sys.path.insert(0, str(ROOT / "src"))
    import pandas as pd
    from day import PROCESSED, load_day
    from replay import replay

    warnings.simplefilter("error")
    dates = list(pd.read_csv(PROCESSED / "days_2019.csv").date) + ["2022-07-14", "2022-01-10", "2023-02-01"]
    problems = []
    for date in dates:
        try:
            day = load_day(date)
            result = replay(day)
            assert result.entries_per_site.sum() == len(day.fixed) == sum(len(r) for r in day.routes)
            assert (result.person_minutes >= 0).all()
        except Exception as error:  # report every failing day, not only the first
            problems.append(f"{date}: {error!r}")
    return len(dates), problems


# ---------------------------------------------------------------------- #
# 5. Outputs changed by rerunning the checks
# ---------------------------------------------------------------------- #

def changed_outputs():
    result = run(["git", "status", "--short", "--", "outputs"])
    return [line for line in result.stdout.splitlines() if line.strip()]


def main():
    print("1. Tests (warnings as errors)")
    passed, summary = check_tests()
    print(f"   {'pass' if passed else 'FAIL'}: {summary}")

    print("2. Check scripts")
    for script, ok, error in check_scripts():
        print(f"   {'pass' if ok else 'FAIL'}: {script}" + ("" if ok else f"\n      {error}"))

    print("3. Deliberate bugs")
    mutations = check_mutations()
    for path, meaning, status in mutations:
        print(f"   {status:10} {path}: {meaning}")

    print("4. Sweep over days")
    count, problems = sweep_days()
    print(f"   {count} days, {len(problems)} with problems")
    for problem in problems[:10]:
        print(f"   {problem}")

    print("5. Outputs changed after the checks (expect timings and code hashes only)")
    for line in changed_outputs() or ["   none"]:
        print(f"   {line}")

    print("6. Read the code line by line against the design: by hand.")
    failed = (not passed or problems or any(status != "caught" for _, _, status in mutations))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
