"""
What Step 6 reads from Step 5 (Step 6 plan, sections 2 and 3)
=============================================================

The identity Step 5 ran under, where its results are, and the checks made
before anything of them is used: every method and run complete, with the
Step 5 identity, and the code and the tables Step 6 runs on the same as
Step 5's. Reading a run's results is left to the callers; this file opens only
the completion marks.

Also the identity of Step 6 itself: a hash of analysis/*.py and the versions of
the packages, written into every Step 6 output.
"""

import hashlib
import json
import os
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

STEP5 = {"settings hash": "84c41d36f6", "code version": "a86c0dfd9f", "data version": "a6a88b365c11",
         "data content": "7db21a4331c5"}
# Where the Step 5 results are read: the Drive-synced copy in the development checkout (plan, section 2). A
# frozen Step 6 worktree has no outputs of its own, so it names that folder in STEP5_RESULTS; the path used
# is written into the pass record and the identity of the test evaluation.
FORMAL = Path(os.environ.get("STEP5_RESULTS", ROOT / "outputs" / "search" / "formal" / STEP5["settings hash"]))
SETTINGS_FILE = ROOT / "configs" / "search.json"
GP_METHODS = ("GP-fixed", "GP-prop", "GP-adaptive")
RUNS = range(30)
PACKAGES = ("numpy", "pandas", "deap", "scipy")


def read_table(path, **kwargs):
    """A saved table read back exactly: pandas' default float parser can change the last bit of a number
    (about a quarter of random floats move by up to 2.2e-16); round_trip reads every number as written."""
    import pandas as pd
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def analysis_version():
    """Hash of every Python file in analysis/."""
    digest = hashlib.sha1()
    for path in sorted((ROOT / "analysis").glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


def package_versions():
    out = {"python": sys.version.split()[0]}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def run_folder(method, run, root=FORMAL):
    return Path(root) / f"{method}_{run:02d}"


def current_identity():
    """The identity of the code and tables Step 6 is running on, in the terms of Step 5."""
    from run_formal import data_content
    from search import code_version, data_version, load_settings, settings_hash
    return {"settings hash": settings_hash(load_settings(SETTINGS_FILE)), "code version": code_version(),
            "data version": data_version(), "data content": data_content()}


def check_here(expected=STEP5):
    """Refuse to go on unless src/, the settings file and the tables are those of Step 5."""
    here = current_identity()
    wrong = {k: (here[k], v) for k, v in expected.items() if here[k] != v}
    if wrong:
        raise SystemExit(f"not the Step 5 identity (found, expected): {wrong}")
    return here


def check_complete(root=FORMAL, methods=GP_METHODS, runs=RUNS, expected=STEP5):
    """Every method and run, and the tuned rule, must be complete with the Step 5 identity. Returns the
    run folders. Only complete.json is read."""
    problems, folders = [], {}
    for method in methods:
        for run in runs:
            folder = run_folder(method, run, root)
            mark = folder / "complete.json"
            if not mark.exists():
                problems.append(f"{folder.name}: not complete")
                continue
            m = json.loads(mark.read_text())
            wrong = [k for k, v in expected.items() if m.get(k) != v]
            if wrong or m.get("method") != method or m.get("run") != run:
                problems.append(f"{folder.name}: another identity ({wrong or 'method or run'})")
            folders[method, run] = folder
    mark = Path(root) / "tuned" / "complete.json"
    if not mark.exists():
        problems.append("tuned: not complete")
    else:
        m = json.loads(mark.read_text())
        if any(m.get(k) != v for k, v in expected.items()) or m.get("rule") != "tuned weighted":
            problems.append("tuned: another identity")
    if problems:
        raise SystemExit("Step 5 results not usable:\n  " + "\n  ".join(problems))
    return folders
