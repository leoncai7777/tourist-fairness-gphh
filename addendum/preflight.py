"""
Before and right after the GP-single search (addendum, sections 2 and 3)
========================================================================

Run with the Step 5 environment's Python (gphh), from anywhere:

  .../envs/gphh/bin/python addendum/preflight.py env              # before: the environment is Step 5's
  .../envs/gphh/bin/python addendum/preflight.py copy-tuned       # before: the tuned rule's folder, copied once
  .../envs/gphh/bin/python addendum/preflight.py launched         # after the launch: tools and commit are Step 5's

  env         Python and package versions and the pip freeze hash must equal
              those recorded for Step 5 (step5_run_record.md), and the frozen
              step5 worktree must be at d2b4af4 with no change.
  copy-tuned  copies the complete tuned folder from the Step 5 results into the
              GP-single root, only if it is not there; records the SHA-256 of
              every file, which must equal its source; refuses while a Step 5
              launcher lock is there. An existing target is compared, never
              overwritten.
  launched    the last line of the GP-single root's launches.jsonl names the
              Step 5 tools version and commit.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

T3 = Path(__file__).resolve().parents[2]
WORKTREE = T3 / "tourist-fairness-gphh-step5"
SETTINGS = "84c41d36f6"
STEP5_ROOT = WORKTREE / "outputs" / "search" / "formal" / SETTINGS
SINGLE_ROOT = WORKTREE / "outputs" / "search" / "single" / SETTINGS
# step5_run_record.md, section 1
STEP5_ENV = {"python": "3.11.14", "numpy": "2.1.3", "pandas": "3.0.1", "deap": "1.4.4", "pip freeze": "6dd2427a94f9"}
STEP5_TOOLS = {"tools version": "469e9e5b75", "commit": "d2b4af4"}


def environment():
    out = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "deap"):
        out[name] = metadata.version(name)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    out["pip freeze"] = hashlib.sha1(freeze.encode()).hexdigest()[:12]
    return out


def check_env(worktree=WORKTREE):
    here = environment()
    wrong = {k: (here[k], v) for k, v in STEP5_ENV.items() if here[k] != v}
    head = subprocess.run(["git", "-C", str(worktree), "rev-parse", "--short=7", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(worktree), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if head != STEP5_TOOLS["commit"]:
        wrong["worktree commit"] = (head, STEP5_TOOLS["commit"])
    if dirty:
        wrong["worktree changes"] = dirty.splitlines()[:5]
    return here, wrong


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_tuned(source_root=STEP5_ROOT, target_root=SINGLE_ROOT):
    source, target = Path(source_root) / "tuned", Path(target_root) / "tuned"
    if (Path(source_root) / "launcher.lock").exists():
        raise SystemExit("a Step 5 launcher lock is there: the source may still be written")
    if not (source / "complete.json").exists():
        raise SystemExit(f"{source} is not complete")
    files = sorted(p.relative_to(source) for p in source.rglob("*") if p.is_file())
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
    record = {str(f): {"source": sha256(source / f), "copy": sha256(target / f)} for f in files}
    there = sorted(p.relative_to(target) for p in target.rglob("*") if p.is_file())
    differ = [f for f, h in record.items() if h["source"] != h["copy"]]
    if there != files or differ:
        raise SystemExit(f"the copy differs from its source: {differ or 'other files'}; it is not overwritten")
    (Path(target_root) / "tuned_copy.json").write_text(json.dumps({"source": str(source), "files": record}, indent=1))
    return record


def check_launched(target_root=SINGLE_ROOT):
    lines = (Path(target_root) / "launches.jsonl").read_text().splitlines()
    last = json.loads(lines[-1])
    wrong = {k: (last.get(k), v) for k, v in STEP5_TOOLS.items() if not str(last.get(k, "")).startswith(v)}
    return last, wrong


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["env", "copy-tuned", "launched"])
    args = parser.parse_args(argv)
    if args.step == "env":
        here, wrong = check_env()
        print(f"environment {here}; differences from Step 5: {wrong or 'none'}")
    elif args.step == "copy-tuned":
        record = copy_tuned()
        print(f"{len(record)} tuned files in {SINGLE_ROOT / 'tuned'}, each the same SHA-256 as its source")
        wrong = None
    else:
        last, wrong = check_launched()
        print(f"last launch {last}; differences from Step 5: {wrong or 'none'}")
    if wrong:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
