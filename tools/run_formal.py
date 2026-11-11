"""
The formal runs (Step 5)
========================

Reads the frozen settings file, then runs every method for every run number,
one process per run, several at once, and finishes each run on the validation
days. Kept in tools/, not src/, so that it is not part of the code version: it
only decides which runs start when, never what a run does.

  python3 tools/run_formal.py                      # runs 0-29 of GP-fixed, GP-prop, GP-adaptive, 12 at a time
  python3 tools/run_formal.py --workers 9 --runs 0-4 --methods GP-adaptive

The queue is interleaved by run number (run 0 of every method, then run 1, ...),
so a batch stopped early still holds comparable runs of every method. Results
go to outputs/search/formal/<settings hash>/. The tuned weighted rule (4e) is
run once under the same settings.

Identity. At the start the launcher fixes what this batch must be: the settings
hash of the file, the code version and the data version, written to task.json.
Every worker checks them again before it starts, a resumed checkpoint must hold
exactly the frozen settings, and every finished result, GP runs and the tuned
rule alike, must carry this identity (and its own method and run number) to be
skipped or pooled. A result of another identity stops the batch before anything
runs; it is never skipped, overwritten or pooled.

Completion. A run counts as done only when its complete.json is there: written
last, atomically, after finish() has returned, run.json is written and every
output file has been read back. A run stopped while writing its outputs has no
complete.json, so running the same command again resumes it from its last
checkpoint (the search is not repeated) and writes the outputs again.

Run from the repository root.
"""

import argparse
import gzip
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

METHODS = ["GP-fixed", "GP-prop", "GP-adaptive"]
RUN_FILES = ["summary.json", "final_population.csv", "front.csv.gz", "groups.csv.gz", "peaks.csv.gz",
             "knee_visitors.csv.gz", "knee_decisions.csv.gz", "knee_hourly.csv.gz"]
TUNED_FILES = ["summary.json", "tuned_front.csv"]
IDENTITY = ("settings hash", "code version", "data version")


def tasks(runs, methods):
    """(method, run) in the order they start: interleaved by run number."""
    return [(method, run) for run in runs for method in methods]


def folder_of(out, method, run):
    return out / f"{method}_{run:02d}"


def check_arguments(runs, methods, workers):
    """Refuse a command that would start nothing, start two processes on one run, or never advance."""
    if workers < 1:
        raise SystemExit(f"--workers must be at least 1, not {workers}")
    unknown = [m for m in methods if m not in METHODS + ["GP-single"]]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}")
    if len(set(methods)) != len(methods):
        raise SystemExit(f"a method is named twice: {methods}")
    first, sep, last = runs.partition("-")
    if not (sep and first.isdigit() and last.isdigit() and int(first) <= int(last) < 900):
        raise SystemExit(f"--runs must be first-last with first <= last < 900 (900 on are pilot runs), not {runs!r}")
    return int(first), int(last)


def lock(out):
    """One launcher per results folder: a second one stops at once instead of writing the same runs."""
    path = out / "launcher.lock"
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{path} exists: another launcher is using {out}; if none is running, delete the file")
    os.write(handle, str(os.getpid()).encode())
    os.close(handle)
    return path


def tools_version():
    """Hash of the launcher and the other tools, and the git commit the batch was started from."""
    digest = hashlib.sha1()
    for path in sorted((ROOT / "tools").glob("*.py")):
        digest.update(path.read_bytes())
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {"tools version": digest.hexdigest()[:10], "commit": commit.stdout.strip() or "unknown"}


def expected_identity(settings):
    from search import code_version, data_version, settings_hash
    return {"settings hash": settings_hash(settings), "code version": code_version(), "data version": data_version()}


# ---------------------------------------------------------------------- #
# Completion and identity
# ---------------------------------------------------------------------- #

def readable(path):
    """Read a whole output file back; a truncated or missing one raises."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            f.read()
    else:
        path.read_text()


def mark_complete(folder, results, files, identity):
    """Check every output can be read, then write complete.json last, through a temporary file."""
    for name in files:
        readable(results / name)
    temporary = folder / "complete.tmp"
    temporary.write_text(json.dumps(identity, indent=1))
    temporary.replace(folder / "complete.json")


def done(folder, identity):
    """True if the run (or the tuned rule) in folder is complete with this identity; False if it is not
    complete. A complete one of another identity raises: it must not be skipped, redone over or pooled."""
    marker = folder / "complete.json"
    if not marker.exists():
        return False
    found = json.loads(marker.read_text())
    if found != identity:
        raise SystemExit(f"{folder} holds a result of another identity: {found}, not {identity}")
    return True


def check_summary(results, identity):
    """The summary written by the run itself must carry the same identity as its marker."""
    summary = json.loads((results / "summary.json").read_text())
    wrong = {k: summary.get(k) for k in identity if summary.get(k) != identity[k]}
    if wrong:
        raise SystemExit(f"{results} has a summary of another identity: {wrong}")
    return summary


# ---------------------------------------------------------------------- #
# Workers
# ---------------------------------------------------------------------- #

def load_task(settings_path, out):
    """The frozen settings and this batch's identity; the worker refuses if either moved since the start."""
    from search import load_settings
    settings = load_settings(settings_path)
    identity = expected_identity(settings)
    task = json.loads((Path(out) / "task.json").read_text())
    if identity != task or identity["settings hash"] != Path(out).name:
        raise SystemExit(f"the settings, code or data are not those of {out}: {identity}, task {task}")
    return settings, identity


def worker(settings_path, out, method, run):
    """One run in this process: resume it if a checkpoint is there, else start it; then finish it."""
    from search import Run, finish
    settings, identity = load_task(settings_path, out)
    folder = folder_of(Path(out), method, run)
    checkpoint = folder / "checkpoint"
    started = time.time()
    if (checkpoint / "checkpoint.pkl").exists():
        job = Run.resume(method, run, checkpoint)
        if dict(job.settings) != dict(settings):  # a valid checkpoint, but of other settings
            raise SystemExit(f"{checkpoint} holds a run under other settings than {settings_path}")
    else:
        job = Run(method, run, settings, folder=checkpoint)
    job.evolve()
    searched = time.time()
    finish(job, folder / "results")
    log = job.log
    report = {"method": method, "run": run, **identity, "search seconds": searched - started,
              "end seconds": time.time() - searched,
              "peak memory MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
              "evaluations": sum(g["evaluations"] for g in log if "evaluations" in g),
              "evaluations run": sum(g["evaluations run"] for g in log if "evaluations run" in g)}
    (folder / "run.json").write_text(json.dumps(report, indent=1))
    check_summary(folder / "results", {**identity, "method": method, "run": run})
    mark_complete(folder, folder / "results", RUN_FILES + ["../run.json"],
                  {**identity, "method": method, "run": run})


def tuned(settings_path, out):
    from search import tuned_front
    settings, identity = load_task(settings_path, out)
    folder = Path(out) / "tuned"
    tuned_front(settings, folder)
    check_summary(folder, identity)
    mark_complete(folder, folder, TUNED_FILES, {**identity, "rule": "tuned weighted"})


# ---------------------------------------------------------------------- #
# The batch
# ---------------------------------------------------------------------- #

def run_all(commands, workers, start=None):
    """Run the commands, at most `workers` at once; returns the exit code of each, in order."""
    start = start or (lambda command: subprocess.Popen(command, cwd=ROOT))
    waiting, running, codes = list(enumerate(commands)), {}, {}
    while waiting or running:
        while waiting and len(running) < workers:
            i, command = waiting.pop(0)
            running[i] = start(command)
        for i, process in list(running.items()):
            code = process.poll()
            if code is not None:
                codes[i] = code
                del running[i]
        if running:
            time.sleep(1)
    return [codes[i] for i in range(len(commands))]


def pool(out, todo_tasks, identity):
    """Every finished run of this batch, each checked against this batch's identity and its own method and
    run, then pooled with search.collect(); and the tuned rule, checked the same way. Writes index.json."""
    from search import collect
    finished = []
    for method, run in todo_tasks:
        folder = folder_of(out, method, run)
        if done(folder, {**identity, "method": method, "run": run}):
            check_summary(folder / "results", {**identity, "method": method, "run": run})
            finished.append(folder / "results")
    summaries = collect(finished)  # also refuses runs that differ among themselves
    tuned_folder = out / "tuned"
    tuned_summary = None
    if done(tuned_folder, {**identity, "rule": "tuned weighted"}):
        tuned_summary = check_summary(tuned_folder, identity)
    index = {"identity": identity, "tools": tools_version(),
             "runs": [{"method": s["method"], "run": s["run"], "hypervolume": s["hypervolume"],
                       "front members": s["front members"], "front rules": s["front rules"],
                       "knee rule": s["knee rule"]} for s in summaries],
             "tuned weighted": tuned_summary}
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return summaries, tuned_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="configs/search.json")
    parser.add_argument("--runs", default="0-29", help="first-last run numbers, e.g. 0-29")
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out-root", default=str(ROOT / "outputs" / "search" / "formal"))
    parser.add_argument("--part", nargs=4, metavar=("ROLE", "SETTINGS", "OUT", "WHAT"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.part:  # inside one worker process
        role, settings_path, out, what = args.part
        if role == "run":
            method, run = what.split(":")
            worker(settings_path, out, method, int(run))
        else:
            tuned(settings_path, out)
        return

    from search import load_settings
    first, last = check_arguments(args.runs, args.methods, args.workers)
    settings = load_settings(args.settings)  # refuses an incomplete or inconsistent file before anything runs
    identity = expected_identity(settings)
    out = Path(args.out_root) / identity["settings hash"]
    out.mkdir(parents=True, exist_ok=True)
    held = lock(out)
    try:
        batch_main(args, settings, identity, out, first, last)
    finally:
        held.unlink()


def batch_main(args, settings, identity, out, first, last):
    task_file = out / "task.json"
    if task_file.exists() and json.loads(task_file.read_text()) != identity:
        raise SystemExit(f"{out} was started under another code or data version: {task_file.read_text()}")
    task_file.write_text(json.dumps(identity, indent=1))
    (out / "settings.json").write_text(json.dumps(settings, indent=1))
    started = {**tools_version(), "runs": f"{first}-{last}", "methods": args.methods, "workers": args.workers,
               "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(out / "launches.jsonl", "a") as f:  # one line per launch: which tools and commit ran this batch
        f.write(json.dumps(started) + "\n")

    batch = tasks(range(first, last + 1), args.methods)
    # Every complete result is checked before anything runs: one of another identity stops the batch here.
    todo = [(m, r) for m, r in batch if not done(folder_of(out, m, r), {**identity, "method": m, "run": r})]
    commands = [[sys.executable, __file__, "--part", "run", args.settings, str(out), f"{m}:{r}"] for m, r in todo]
    if not done(out / "tuned", {**identity, "rule": "tuned weighted"}):
        commands.append([sys.executable, __file__, "--part", "tuned", args.settings, str(out), "-"])
    print(f"{len(todo)} runs to do in {out}, {args.workers} at a time; identity {identity}")
    codes = run_all(commands, args.workers)
    failed = [command[-1] for command, code in zip(commands, codes) if code != 0]

    summaries, tuned_summary = pool(out, batch, identity)
    print(f"{len(summaries)} of {len(batch)} runs complete and pooled; tuned rule "
          f"{'complete' if tuned_summary else 'not complete'}; failed: {failed or 'none'}")
    if failed or len(summaries) < len(batch) or tuned_summary is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
