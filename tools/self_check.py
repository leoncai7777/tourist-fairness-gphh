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
CHECK_SCRIPTS = ["src/simulator_check.py", "src/pilot.py", "src/search_check.py"]
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
    ("src/day.py", 'opening = [rows["o"].get(i, sites["o"][i]) for i in sites.index]',
     'opening = [rows["o"].get(i, 0.0) for i in sites.index]', "missing month opens at midnight"),
    ("src/day.py", 'v = visitors.sort_values("visitor").reset_index(drop=True)',
     "v = visitors.reset_index(drop=True)", "visitors not numbered by id"),
    ("src/day.py", "return np.array([i in open_ids and i not in LOW_COVERAGE for i in site_ids])",
     "return np.array([i in open_ids for i in site_ids])", "low-coverage site available"),
    ("src/preprocess.py", "out[\"exit\"] = np.fmin(visits.t + visits.site.map(d_i), visits.next_t - visits.walk)",
     "out[\"exit\"] = np.fmax(visits.t + visits.site.map(d_i), visits.next_t - visits.walk)",
     "occupancy runs to the next entry"),
    ("src/preprocess.py", "nodes = np.searchsorted(minutes, part.t.values, side=\"left\")",
     "nodes = np.searchsorted(minutes, np.floor(part.t.values), side=\"left\")", "seconds rounded down"),
    ("src/engine.py", "VISIT_END, ARRIVAL, START, DECISION = 0, 1, 2, 3",
     "VISIT_END, ARRIVAL, START, DECISION = 0, 3, 2, 1", "decisions before arrivals at the same time"),
    ("src/engine.py", "        self.on_the_way[j] -= 1\n", "        pass\n", "arrival not taken off the way"),
    ("src/engine.py", "if self.now >= self.day.opening[j]:", "if True:", "no wait for opening"),
    ("src/engine.py", "self.stops_used[k] >= self.stop_limit[k]", "self.stops_used[k] > self.stop_limit[k]",
     "stop limit off by one"),
    ("src/engine.py", 'reason[finish > self.budget_end[k]] = "budget"',
     'reason[finish > self.budget_end[k] + 1e9] = "budget"', "budget not checked"),
    ("src/engine.py", 'reason[finish > day.closing] = "closing"', 'reason[start > day.closing] = "closing"',
     "closing checked on the start"),
    ("src/engine.py", "        return row[1] > row[0]\n", "        return True\n", "zero-length entry counted inside"),
    ("src/engine.py", "np.maximum(MIN_STAY, day.stay * (1 + 0.2 * z))", "np.maximum(0.0, day.stay * (1 + 0.2 * z))",
     "no 1-minute floor on stays"),
    ("src/engine.py", "            engine.run.skips.append((k, j, reasons[j]))\n", "            pass\n",
     "observed-order skip not recorded"),
    ("src/engine.py", "if self.stops_used[k] > 1:", "if self.stops_used[k] > 0:", "first stop treated as chosen"),
    ("src/engine.py", "            self.policy.reset()\n", "            pass\n", "policy state kept between runs"),
    ("src/engine.py", 'if not getattr(self.policy, "may_stop", False):', "if False:", "any rule may stop a visitor"),
    ("src/engine.py", "            self.run.intervals.append((j, entry, exit_))\n",
     "            self.run.intervals.append((j, entry, entry + self.day.stay[j]))\n",
     "background interval from the mean stay, not the observed one"),
    ("src/engine.py", "if stays.shape != (len(day.visitors), len(day.site_ids)):", "if False:",
     "stay table of the wrong shape accepted"),
    ("src/day.py", "        if unknown:\n", "        if False:\n", "unknown follower ids accepted"),
    ("src/day.py", "    if (day.walk < 0).any():", "    if False:", "negative walking times accepted"),
    ("src/day.py", "    if visits.empty:", "    if False:", "a day with no visits loads as an empty day"),
    ("src/features.py", "bisect.bisect_right(entries, engine.now - 60,", "bisect.bisect_right(entries, engine.now - 30,",
     "ARR over 30 minutes"),
    ("src/features.py", "engine.on_the_way[candidates] / peak]", "engine.inside[candidates] / peak]",
     "HEAD reads people inside"),
    ("src/features.py", "tsv = np.maximum(to, ta)", "tsv = ta", "TSV ignores opening"),
    ("src/features.py", "return 1.0 if usual is None else g / usual", "return 1.0 if usual is None else g",
     "GDAY not divided by its usual value"),
    ("src/features.py", "mean = engine.visits[engine.day.available].mean()", "mean = engine.visits.mean()",
     "VS mean over every site"),
    ("src/features.py", "return top[np.argmin(engine.day.site_ids[top])]", "return top[0]",
     "ties to the first position, not the lowest site id"),
    ("src/features.py", "value = np.interp(t, MINUTES, curve)", "value = curve[int(t) - FIRST_MINUTE]",
     "usual curves not interpolated"),
    ("src/day.py", 'np.maximum(sites["usual_peak"], peak_floor or 0)', 'sites["usual_peak"]', "peak floor ignored"),
    ("src/scoring.py", 'if day.visitors["follows"][k]:', "if False:", "followers scored on their observed routes"),
    ("src/scoring.py", 'relative_value(v["V"], v["V observed"], b["A"])', 'relative_value(v["V"], v["V observed"], v["A"])',
     "R_v divided by A of the new route"),
    ("src/scoring.py", "g_hat = gini_hat(run.visits[day.available])", "g_hat = gini_hat(run.visits)",
     "G_hat over every site, not the available ones"),
    ("src/scoring.py", 'v[f"change {part}"] = v[part] - b[part]', 'v[f"change {part}"] = b[part] - v[part]',
     "change of each part with the wrong sign"),
    ("src/scoring.py", '"completion": g["stops"].sum() / g["observed stops"].sum(),',
     '"completion": (g["stops"] / g["observed stops"]).mean(),', "completion averaged, not pooled"),
    ("src/rules.py", "engine.day.attractiveness[candidates])\n", "-engine.day.attractiveness[candidates])\n",
     "POP takes the least attractive site"),
    ("src/rules.py", "-engine.visits[candidates].astype(float)", "engine.visits[candidates].astype(float)",
     "LEAST takes the most visited site"),
    ("src/rules.py", "protected_div(engine.day.attractiveness[candidates], inputs[TA])",
     "protected_div(engine.day.attractiveness[candidates], inputs[SL])", "SOT divides by the slack"),
    ("src/features.py", 'priority = finite(engine, priority, "priority")', "priority = priority",
     "final priority not cleaned"),
    ("src/features.py", 'w = max(0.0, float(finite(engine, w, "w")))', 'w = float(finite(engine, w, "w"))',
     "negative w not cut at 0"),
    ("src/scoring.py", "    if not same:\n", "    if False:\n", "a baseline from another day accepted"),
    ("src/gp_rule.py", "w = self.functions[1] * h2_values[GDAY]", "w = self.functions[1]",
     "GP-prop weight ignores GDAY"),
    ("src/gp_rule.py", "return self.functions[0](*h1_values, *extra, vs), 0.0, vs, h2_values",
     "return self.functions[0](*h1_values, *extra, 1.0), 0.0, vs, h2_values", "GP-single does not read VS"),
    ("src/gp_rule.py", "math.log10(HIGH)))", "math.log10(HIGH) + 1))", "constants drawn above the range"),
    ("src/gp_rule.py", "else float(value))", "else round(float(value), 3))", "constants rebuilt with lost digits"),
    ("src/gp_rule.py", "w = self.functions[1](*h2_values)", "w = self.functions[1](*h2_values) + 1",
     "adaptive weight off by one"),
    ("src/evaluate.py", 'if split == "test" and not self._test:', "if False:", "a test day is read"),
    ("src/evaluate.py", "    k = np.where(r < k / BINS_PER_UNIT, k - 1, k)\n", "", "histogram edges left to r * 20"),
    ("src/evaluate.py", 'g["ended early"] = (rows["stops"] < rows["observed stops"])',
     'g["ended early"] = (rows["stops"] <= rows["observed stops"])', "everyone counted as ending early"),
    ("src/evaluate.py", '"same_day": same_day, "peak_floor": peak_floor}', '"same_day": same_day}',
     "the cache ignores the peak bound"),
    ("src/evaluate.py", "int(c.split()[1]) >= edge]", "int(c.split()[1]) > edge]", "share at theta misses the edge bin"),
    ("src/evaluate.py", "frozenset(follows)", "follows", "the caller's follows list is kept, not copied"),
    ("src/evaluate.py", "not -2 <= theta <= 2", "False", "thresholds beyond the open bins answered"),
    ("src/evaluate.py", "replicate=replicate, stop_limit=stop_limit)", "replicate=replicate)",
     "the stop-limit switch is dropped"),
    ("src/gp_rule.py", "if values and pset is not H2_SET:", "if False:", "a constant in h1 is accepted"),
    ("src/gp_rule.py", "        tree[i] = gp.Terminal(tree[i].value, False, pset.ret)\n", "        pass\n",
     "constants read from text keep the type float"),
    ("src/search.py", "        a[1], b[1] = b[1], a[1]\n    children = []", "    children = []",
     "GP-fixed crossover keeps the constants"),
    ("src/search.py", "        score(self.population, date, replicate, days, fallbacks)\n", "",
     "parents not re-scored in the new environment"),
    ("src/search.py", "return a if a.rank < b.rank else b", "return a if a.rank > b.rank else b",
     "tournament prefers the worse rank"),
    ("src/search.py", "        if too_high(child):\n            counts[\"reverts\"] += 1\n            child = copy.deepcopy(random.choice((p1, p2)))",
     "        if False:\n            pass", "crossover keeps children over the height limit"),
    ("src/search.py", "rng.permutation(len(dates))", "range(len(dates))", "the schedule is not shuffled"),
    ("src/search.py", "rng.integers(FIRST_TRAINING_REPLICATE, 1_000_000, generations)", "rng.integers(0, 1_000_000, generations)",
     "training may use validation replicates"),
    ("src/search.py", "        random.seed(search_seed(method, run_number))\n", "", "the search is not seeded"),
    ("src/search.py", '        random.setstate(state["random"])', "        pass", "a resumed run loses its random state"),
    ("src/search.py", "partial(grow, min_=0, max_=4)", "partial(grow, min_=1, max_=4)",
     "h2 mutation cannot make a one-node tree"),
    ("src/search.py", "        new = min(max(new, LOW), HIGH)\n", "", "constants not cut to the range"),
    ("src/search.py", "            if state[key] != value:\n", "            if False:\n",
     "a checkpoint resumes under another identity"),
    ("src/search.py", "    if state[\"schedule\"] != schedule(run_number, settings[\"generations\"]):",
     "    if False:", "a checkpoint with another run's schedule resumes"),
    ("src/search.py", "if any(rule.method != method for rule in population):", "if False:",
     "a checkpoint with rules of another method resumes"),
    ("src/search.py", "                rule.fitness.values = values\n", "                pass\n",
     "a resumed population loses its scores"),
    ("src/search.py", 'if settings["crossover"] + settings["mutation"] > 1:', 'if settings["crossover"] + settings["mutation"] > 2:',
     "operator rates over 1 accepted"),
    ("src/search.py", "        if not (math.isfinite(result.mean_r) and math.isfinite(result.g_hat)):\n",
     "        if False:\n", "non-finite objectives enter the sorting"),
    ("src/evaluate.py", 'if split not in ("train", "validation", "test"):', "if False:",
     "a date of unknown split, or an unknown label, is read"),
    ("src/search.py", 'fingerprint(*sorted(Path(__file__).parent.glob("*.py")))',
     'fingerprint(Path(__file__), Path(__file__).parent / "gp_rule.py")',
     "the code version leaves out dependencies of the search"),
    ("src/search.py", '        if state["scored on"] != scored_on:\n', "        if False:\n",
     "scores from another environment resume"),
    ("src/toptw.py", "feasible &= np.maximum(arrive, instance.opening) + expected + back <= end_time",
     "feasible &= arrive + expected + back <= end_time", "Step 1: no wait for opening"),
]


TIMEOUT = 900  # seconds; a mutation that makes the code loop must not hang the self-check


def run(command, cwd=ROOT):
    try:
        return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(command, 124, "", f"timed out after {TIMEOUT} s")


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

def run_tests_in_copy(path=None, text=None, bug=None):
    """Run the tests in a fresh copy of the code, with one deliberate bug if given."""
    with tempfile.TemporaryDirectory() as folder:
        copy = Path(folder)
        no_cache = shutil.ignore_patterns("__pycache__")
        shutil.copytree(ROOT / "src", copy / "src", ignore=no_cache)
        shutil.copytree(ROOT / "tests", copy / "tests", ignore=no_cache)
        shutil.copy(ROOT / "pytest.ini", copy / "pytest.ini")
        (copy / "data").symlink_to(ROOT / "data")  # read-only use of the real data
        if path is not None:
            (copy / path).write_text((ROOT / path).read_text().replace(text, bug))
        command = pytest_command()
        if path is not None and path != "src/search.py":
            command += ["-m", "not slow"]  # the short searches in subprocesses catch bugs of the search only
        return run(command, cwd=copy)


def classify(result):
    """pytest exits with 0 when all tests pass and 1 when some test failed. Anything else
    (a syntax error, a failed import, a usage error) is not a test catching the bug."""
    if result.returncode == 0:
        return "NOT CAUGHT"
    if result.returncode == 1 and " failed" in result.stdout:
        return "caught"
    return "ERROR"


def check_mutations():
    baseline = run_tests_in_copy()
    if baseline.returncode != 0:
        return [("(unchanged copy)", "the tests must pass before any bug is put in", "ERROR")]
    rows = []
    for path, text, bug, meaning in MUTATIONS:
        if (ROOT / path).read_text().count(text) != 1:
            rows.append((path, meaning, "text not found: update this mutation"))
            continue
        result = run_tests_in_copy(path, text, bug)
        status = classify(result)
        if status == "ERROR":
            meaning += "\n" + (result.stdout + result.stderr).strip()[-400:]
        rows.append((path, meaning, status))
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
    problems, skipped = [], []
    for date in dates:
        if not (PROCESSED / f"visits_{date[:4]}.csv.gz").exists():
            skipped.append(date)  # that year's tables are not here, e.g. in a review package
            continue
        try:
            day = load_day(date)
            result = replay(day)
            assert result.entries_per_site.sum() == len(day.fixed) == sum(len(r) for r in day.routes)
            assert (result.person_minutes >= 0).all()
        except Exception as error:  # report every failing day, not only the first
            problems.append(f"{date}: {error!r}")
    return len(dates) - len(skipped), problems, skipped


# ---------------------------------------------------------------------- #
# 5. Outputs changed by rerunning the checks
# ---------------------------------------------------------------------- #

def changed_outputs():
    result = run(["git", "status", "--short", "--", "outputs"])
    return [line for line in result.stdout.splitlines() if line.strip()]


def main():
    import numpy, pandas
    print(f"Python {sys.version.split()[0]}, numpy {numpy.__version__}, pandas {pandas.__version__}")
    print("1. Tests (warnings as errors)")
    passed, summary = check_tests()
    print(f"   {'pass' if passed else 'FAIL'}: {summary}")

    print("2. Check scripts")
    scripts = check_scripts()
    for script, ok, error in scripts:
        print(f"   {'pass' if ok else 'FAIL'}: {script}" + ("" if ok else f"\n      {error}"))

    print("3. Deliberate bugs")
    mutations = check_mutations()
    for path, meaning, status in mutations:
        print(f"   {status:10} {path}: {meaning}")

    print("4. Sweep over days (load_day and replay; not a GP evaluation of every day)")
    count, problems, skipped = sweep_days()
    print(f"   {count} days, {len(problems)} with problems" + (f"; skipped (no tables): {skipped}" if skipped else ""))
    for problem in problems[:10]:
        print(f"   {problem}")

    print("5. Outputs changed after the checks (expect timings and code hashes only)")
    for line in changed_outputs() or ["   none"]:
        print(f"   {line}")

    print("6. Read the code line by line against the design: by hand.")
    failed = (not passed or any(not ok for _, ok, _ in scripts) or problems
              or any(status != "caught" for _, _, status in mutations))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
