"""
The measured end-to-end run before the formal test evaluation (Step 6 plan, section 4, step 2)
=============================================================================================

The test-evaluation launcher, unchanged, on pilot rules and the 35 validation
days with replicates 0-4: the front rules of the three full pilot runs, the
display forms of their knees, every weight of the tuned rule, the hand rules
and the observed-order control. No test day and no formal result is read.

Twice into two folders: once without interruption, once interrupted by Ctrl-C
(SIGINT to the launcher's process group) after a few blocks and then resumed
with the same command. Then: no worker left and no lock after the stop; every
output file of the two folders has the same content; and the measured time,
memory and output size, scaled to the formal test evaluation.

  python3 analysis/measure_run.py --out <folder outside the repository>
Writes <out>/record.md.
"""

import argparse
import gzip
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_test as rt  # noqa: E402

from rules import TUNED_WEIGHTS  # noqa: E402

PILOT = rt.ROOT / "outputs" / "search" / "pilot"
METHODS = ("GP-fixed", "GP-prop", "GP-adaptive")


def pilot_rule_list(out):
    """The rule list of the pilot, written as write_rule_list() writes the formal one."""
    root = out / "pilot_root"
    for method in METHODS:
        shutil.copytree(PILOT / "full" / f"{method}_910" / "results", root / f"{method}_910" / "results")
    base = pd.read_csv(PILOT / "figures" / "baselines.csv")
    tuned = base[base["kind"] == "tuned"].reset_index(drop=True)
    assert len(tuned) == len(TUNED_WEIGHTS)
    (root / "tuned").mkdir()
    pd.DataFrame({"weight": TUNED_WEIGHTS, "mean R_v": tuned["mean R_v"], "G_hat": tuned["G_hat"],
                  "front": tuned["front"]}).to_csv(root / "tuned" / "tuned_front.csv", index=False)
    return rt.write_rule_list(out / "rules.csv", root, methods=METHODS, runs=[910])


def command(out, folder, workers):
    settings = out / "validation.json"
    return [sys.executable, "-u", str(Path(__file__).with_name("run_test.py")), "--workers", str(workers),
            "--not-formal", "--rules", str(out / "rules.csv"), "--test-settings", str(settings),
            "--out", str(folder)]


def workers_of(group):
    found = subprocess.run(["pgrep", "-g", str(group)], capture_output=True, text=True).stdout.split()
    return [p for p in found if p != str(group)]


def contents(folder):
    """Every output file of a folder, decompressed (gzip headers hold a time), with done.json without its
    time and memory."""
    out = {}
    for path in sorted(Path(folder).glob("blocks/*/*/*")):
        key = str(path.relative_to(folder))
        if path.name == "done.json":
            d = json.loads(path.read_text())
            d.pop("seconds"), d.pop("worker peak memory MB"), d.pop("files")  # gzip headers hold a time
            out[key] = json.dumps(d, sort_keys=True)
        else:
            out[key] = gzip.decompress(path.read_bytes())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} is there; give a new folder")
    out.mkdir(parents=True)
    rules = pilot_rule_list(out)
    data = {**rt.load_test_settings(), "split": "validation"}
    (out / "validation.json").write_text(json.dumps(data))

    started = time.time()
    a = subprocess.run(command(out, out / "A", args.workers), capture_output=True, text=True)
    seconds_a = time.time() - started
    if a.returncode:
        raise SystemExit(f"run A failed:\n{a.stdout}\n{a.stderr}")

    b = subprocess.Popen(command(out, out / "B", args.workers), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True)
    while len(list((out / "B").glob("blocks/*/*/done.json"))) < 4 and b.poll() is None:
        time.sleep(1)
    blocks_before = len(list((out / "B").glob("blocks/*/*/done.json")))
    os.killpg(b.pid, signal.SIGINT)
    stopped_log = b.communicate(timeout=120)[0]
    time.sleep(2)
    left, lock_left = workers_of(b.pid), (out / "B" / "launcher.lock").exists()
    resumed = subprocess.run(command(out, out / "B", args.workers), capture_output=True, text=True)
    if resumed.returncode:
        raise SystemExit(f"resumed run B failed:\n{resumed.stdout}\n{resumed.stderr}")

    ca, cb = contents(out / "A"), contents(out / "B")
    same = ca.keys() == cb.keys() and all(ca[k] == cb[k] for k in ca)
    done = [json.loads(p.read_text()) for p in (out / "A").glob("blocks/*/*/done.json")]
    sizes = {}
    for path in (out / "A").glob("blocks/*/*/*.csv.gz"):
        sizes[path.stem.replace(".csv", "")] = sizes.get(path.stem.replace(".csv", ""), 0) + path.stat().st_size
    evaluations = len(rules) * len(rt.dates_of(data)) * data["replicates"]
    formal_rules = 90 * 80 + 90 + 16 + 5  # pilot fronts: 65-96 distinct rules a run
    formal_evaluations = formal_rules * 181 * 5
    scale = formal_evaluations / evaluations
    core_seconds = sum(d["seconds"] for d in done)

    lines = [
        "# Measured end-to-end run of the test evaluation (pilot rules, validation days)", "",
        f"Rules: {len(rules)} ({(rules['kind'] == 'gp').sum()} pilot front rules, "
        f"{(rules['kind'] == 'display').sum()} display forms, 16 tuned weights, 4 hand rules, the control); "
        f"35 validation days x {data['replicates']} replicates = {evaluations} evaluations; {args.workers} workers.", "",
        "## Interruption and resume", "",
        f"- Run A, uninterrupted: {seconds_a:.0f} s, exit {a.returncode}.",
        f"- Run B: stopped by SIGINT to its process group after {blocks_before} blocks were done; "
        f"exit {b.returncode}; workers left after the stop: {len(left)}; lock left: {lock_left}.",
        f"- Run B resumed with the same command: exit {resumed.returncode}; "
        f"last line: {resumed.stdout.strip().splitlines()[-1]}",
        f"- Every output file of A and B has the same content (decompressed; done.json without time and "
        f"memory): **{same}** ({len(ca)} files).", "",
        "## Cost", "",
        f"- Block time summed over blocks: {core_seconds / 3600:.2f} core-hours, "
        f"{core_seconds / evaluations:.3f} s an evaluation (with every output written).",
        f"- Peak memory of a worker: {max(d['worker peak memory MB'] for d in done)} MB.",
        f"- Output size: {sum(sizes.values()) / 1e6:.0f} MB: "
        + ", ".join(f"{k} {v / 1e6:.0f} MB" for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])) + ".", "",
        "## Scaled to the formal test evaluation", "",
        f"About {formal_rules} rules x 181 days x 5 replicates = {formal_evaluations:,} evaluations, "
        f"{scale:.0f} times this run: about {core_seconds * scale / 3600:.0f} core-hours "
        f"({core_seconds * scale / 3600 / args.workers:.1f} h at {args.workers} at a time) and about "
        f"{sum(sizes.values()) * scale / 1e9:.0f} GB. The knee outputs scale with the knees (90, not 7,200), so "
        f"their share is overstated here and the size is an upper bound.", "",
        "## Launcher output of the stopped run", "", "```", stopped_log.strip(), "```", ""]
    (out / "record.md").write_text("\n".join(lines))
    print("\n".join(lines))
    if not same or left or lock_left:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
