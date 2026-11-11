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
so a batch stopped early still holds comparable runs of every method. Running
the same command again skips finished runs and resumes the others from their
checkpoints. Results go to outputs/search/formal/<settings hash>/, so runs
under other settings can never land in the same folder. At the end every
finished run is pooled with search.collect(), which refuses mixed settings,
code or data, and an index of the runs is written. The tuned weighted rule
(4e) is run once under the same settings.

Run from the repository root.
"""

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

METHODS = ["GP-fixed", "GP-prop", "GP-adaptive"]


def tasks(runs, methods):
    """(method, run) in the order they start: interleaved by run number."""
    return [(method, run) for run in runs for method in methods]


def folder_of(out, method, run):
    return out / f"{method}_{run:02d}"


def done(out, method, run):
    return (folder_of(out, method, run) / "results" / "summary.json").exists()


def worker(settings_path, out, method, run):
    """One run in this process: resume it if a checkpoint is there, else start it; then finish it."""
    from search import Run, finish, load_settings, settings_hash
    settings = load_settings(settings_path)
    if settings_hash(settings) != Path(out).name:
        raise SystemExit(f"{out} is not the folder of these settings ({settings_hash(settings)})")
    folder = folder_of(Path(out), method, run)
    checkpoint = folder / "checkpoint"
    started = time.time()
    if (checkpoint / "checkpoint.pkl").exists():
        job = Run.resume(method, run, checkpoint)
    else:
        job = Run(method, run, settings, folder=checkpoint)
    job.evolve()
    searched = time.time()
    finish(job, folder / "results")
    log = job.log
    report = {"method": method, "run": run, "search seconds": searched - started,
              "end seconds": time.time() - searched,
              "peak memory MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
              "evaluations": sum(g["evaluations"] for g in log if "evaluations" in g),
              "evaluations run": sum(g["evaluations run"] for g in log if "evaluations run" in g)}
    (folder / "run.json").write_text(json.dumps(report, indent=1))


def tuned(settings_path, out):
    from rules import TUNED_WEIGHTS  # noqa: F401  (the grid is fixed in rules.py)
    from search import load_settings, tuned_front
    tuned_front(load_settings(settings_path), Path(out) / "tuned")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="configs/search.json")
    parser.add_argument("--runs", default="0-29", help="first-last run numbers, e.g. 0-29")
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--workers", type=int, default=12)
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

    from search import collect, load_settings, settings_hash
    settings = load_settings(args.settings)  # refuses an incomplete or inconsistent file before anything runs
    out = ROOT / "outputs" / "search" / "formal" / settings_hash(settings)
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps(settings, indent=1))
    first, last = (int(x) for x in args.runs.split("-"))
    todo = [(m, r) for m, r in tasks(range(first, last + 1), args.methods) if not done(out, m, r)]
    commands = [[sys.executable, __file__, "--part", "run", args.settings, str(out), f"{m}:{r}"] for m, r in todo]
    if not (out / "tuned" / "summary.json").exists():
        commands.append([sys.executable, __file__, "--part", "tuned", args.settings, str(out), "-"])
    print(f"{len(todo)} runs to do in {out}, {args.workers} at a time")
    codes = run_all(commands, args.workers)
    failed = [command[-1] for command, code in zip(commands, codes) if code != 0]

    finished = [folder_of(out, m, r) / "results" for m, r in tasks(range(first, last + 1), args.methods)
                if done(out, m, r)]
    summaries = collect(finished)  # refuses runs under other settings, code or data
    index = [{"method": s["method"], "run": s["run"], "hypervolume": s["hypervolume"],
              "front members": s["front members"], "front rules": s["front rules"], "knee rule": s["knee rule"]}
             for s in summaries]
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(finished)} runs finished and pooled; failed: {failed or 'none'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
