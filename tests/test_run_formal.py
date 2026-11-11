"""
Tests of the launcher of the formal runs (tools/run_formal.py, Step 5). Skipped inside the self-check's
copies, which hold src and tests only.

Run:  python3 -m pytest tests -q   (from the repository root)
"""

import importlib.util
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "run_formal.py"
sys.path.insert(0, str(ROOT / "src"))
pytestmark = pytest.mark.skipif(not TOOL.exists(), reason="inside a copy made by the self-check")
from day import PROCESSED  # noqa: E402

needs_data = pytest.mark.skipif(not (PROCESSED / "manifest.json").exists(), reason="processed tables not built")
TINY = {"population": 4, "generations": 1, "crossover": 0.8, "mutation": 0.15, "monitor every": 1,
        "validation replicates": 1}


def launcher():
    spec = importlib.util.spec_from_file_location("run_formal", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IDENTITY = {"settings hash": "h", "code version": "c", "data version": "d"}


# ---------------------------------------------------------------------- #
# 1. Queue and parallel runs
# ---------------------------------------------------------------------- #

def test_the_queue_is_interleaved_by_run_number():
    order = launcher().tasks(range(3), ["GP-fixed", "GP-prop", "GP-adaptive"])
    assert order[:4] == [("GP-fixed", 0), ("GP-prop", 0), ("GP-adaptive", 0), ("GP-fixed", 1)]
    assert len(order) == 9 and len(set(order)) == 9


class FakeProcess:
    running, most = 0, 0

    def __init__(self, polls, code):
        self.polls, self.code = polls, code
        FakeProcess.running += 1
        FakeProcess.most = max(FakeProcess.most, FakeProcess.running)

    def poll(self):
        self.polls -= 1
        if self.polls > 0:
            return None
        FakeProcess.running -= 1
        return self.code


def test_at_most_the_workers_run_at_once_and_failures_are_reported(monkeypatch):
    module = launcher()
    monkeypatch.setattr(module.time, "sleep", lambda s: None)
    FakeProcess.running = FakeProcess.most = 0
    commands = [("job", i) for i in range(7)]
    codes = module.run_all(commands, 3, start=lambda c: FakeProcess(polls=1 + c[1] % 3, code=1 if c[1] == 4 else 0))
    assert codes == [0, 0, 0, 0, 1, 0, 0] and FakeProcess.most == 3


# ---------------------------------------------------------------------- #
# 2. Completion and identity, by hand
# ---------------------------------------------------------------------- #

def outputs(folder, identity, files):
    folder.mkdir(parents=True, exist_ok=True)
    for name in files:
        if name == "summary.json":
            (folder / name).write_text(json.dumps(identity))
        elif name.endswith(".gz"):
            import gzip
            with gzip.open(folder / name, "wt") as f:
                f.write("a,b\n1,2\n")
        else:
            (folder / name).write_text("a,b\n1,2\n")


def test_a_summary_alone_is_not_completion(tmp_path):
    module = launcher()
    run = {**IDENTITY, "method": "GP-prop", "run": 3}
    folder = tmp_path / "GP-prop_03"
    (folder / "results").mkdir(parents=True)
    (folder / "results" / "summary.json").write_text(json.dumps(run))
    assert not module.done(folder, run)  # stopped while writing the outputs: to be finished again
    outputs(folder / "results", run, module.RUN_FILES)
    (folder / "run.json").write_text("{}")
    module.mark_complete(folder, folder / "results", module.RUN_FILES + ["../run.json"], run)
    assert module.done(folder, run)


def test_a_truncated_output_stops_the_completion_mark(tmp_path):
    module = launcher()
    run = {**IDENTITY, "method": "GP-prop", "run": 3}
    folder = tmp_path / "GP-prop_03"
    outputs(folder / "results", run, module.RUN_FILES)
    (folder / "results" / "groups.csv.gz").write_bytes(b"\x1f\x8b\x08\x00broken")  # cut off while writing
    (folder / "run.json").write_text("{}")
    with pytest.raises(Exception):
        module.mark_complete(folder, folder / "results", module.RUN_FILES + ["../run.json"], run)
    assert not (folder / "complete.json").exists() and not module.done(folder, run)


def test_complete_results_of_another_identity_are_refused(tmp_path):
    module = launcher()
    run = {**IDENTITY, "method": "GP-prop", "run": 3}
    for change in ({"code version": "old"}, {"data version": "old"}, {"settings hash": "other"},
                   {"method": "GP-fixed"}, {"run": 4}):
        folder = tmp_path / f"case_{len(list(tmp_path.iterdir()))}"
        folder.mkdir()
        (folder / "complete.json").write_text(json.dumps({**run, **change}))
        with pytest.raises(SystemExit, match="another identity"):
            module.done(folder, run)


def test_a_summary_must_match_its_marker(tmp_path):
    module = launcher()
    run = {**IDENTITY, "method": "GP-prop", "run": 3}
    (tmp_path / "summary.json").write_text(json.dumps({**run, "run": 5}))
    with pytest.raises(SystemExit, match="another identity"):
        module.check_summary(tmp_path, run)


# ---------------------------------------------------------------------- #
# 3. Whole batches, small
# ---------------------------------------------------------------------- #

def settings_file(path, **changes):
    from evaluate import budget_cuts
    from search import complete
    path.write_text(json.dumps({"settings": complete({**TINY, **changes}), "budget cuts": list(budget_cuts())}))
    return path


def batch(settings, out_root, *extra):
    return subprocess.run([sys.executable, str(TOOL), "--settings", str(settings), "--out-root", str(out_root),
                           "--runs", "0-0", "--methods", "GP-fixed", "GP-adaptive", "--workers", "3", *extra],
                          cwd=ROOT, capture_output=True, text=True)


@needs_data
@pytest.mark.slow
def test_a_batch_restarts_skips_finishes_and_refuses(tmp_path):
    module = launcher()
    from search import settings_hash, load_settings
    settings = settings_file(tmp_path / "search.json")
    first = batch(settings, tmp_path / "formal")
    assert first.returncode == 0, first.stderr[-500:]
    out = tmp_path / "formal" / settings_hash(load_settings(settings))
    index = json.loads((out / "index.json").read_text())
    assert len(index["runs"]) == 2 and index["tuned weighted"]["settings hash"] == out.name
    assert index["tools"]["tools version"] and (out / "launches.jsonl").exists()
    assert not (out / "launcher.lock").exists()  # released when the batch ends

    # Everything complete: nothing runs again.
    again = batch(settings, tmp_path / "formal")
    assert again.returncode == 0 and "0 runs to do" in again.stdout

    # Stopped while writing outputs: no marker, a file missing. The run is finished again from its
    # checkpoint (the search is not repeated) and marked complete.
    run = module.folder_of(out, "GP-fixed", 0)
    population_before = pickle.loads((run / "checkpoint" / "checkpoint.pkl").read_bytes())["population"]
    (run / "complete.json").unlink()
    (run / "results" / "groups.csv.gz").unlink()
    redo = batch(settings, tmp_path / "formal")
    assert redo.returncode == 0 and "1 runs to do" in redo.stdout
    assert (run / "results" / "groups.csv.gz").exists() and (run / "complete.json").exists()
    assert pickle.loads((run / "checkpoint" / "checkpoint.pkl").read_bytes())["population"] == population_before

    # A complete GP result of an old code version: the batch stops before anything runs.
    marker = json.loads((run / "complete.json").read_text())
    (run / "complete.json").write_text(json.dumps({**marker, "code version": "old"}))
    old = batch(settings, tmp_path / "formal")
    assert old.returncode != 0 and "another identity" in old.stderr
    (run / "complete.json").write_text(json.dumps(marker))

    # The tuned rule of an old version: refused the same way.
    tuned_marker = json.loads((out / "tuned" / "complete.json").read_text())
    (out / "tuned" / "complete.json").write_text(json.dumps({**tuned_marker, "data version": "old"}))
    old = batch(settings, tmp_path / "formal")
    assert old.returncode != 0 and "another identity" in old.stderr
    (out / "tuned" / "complete.json").write_text(json.dumps(tuned_marker))
    assert batch(settings, tmp_path / "formal").returncode == 0


@needs_data
@pytest.mark.slow
def test_a_checkpoint_of_other_settings_is_refused_before_the_search(tmp_path):
    from search import Run, load_settings, settings_hash
    module = launcher()
    settings = settings_file(tmp_path / "search.json")
    out = tmp_path / "formal" / settings_hash(load_settings(settings))
    out.mkdir(parents=True)
    (out / "task.json").write_text(json.dumps(module.expected_identity(load_settings(settings))))
    other = Run("GP-fixed", 0, {**TINY, "crossover": 0.7}, folder=module.folder_of(out, "GP-fixed", 0) / "checkpoint",
                monitor=False)
    other.checkpoint()  # a valid checkpoint, but of other settings, put in this batch's folder
    with pytest.raises(SystemExit, match="under other settings"):
        module.worker(settings, out, "GP-fixed", 0)
    assert not (module.folder_of(out, "GP-fixed", 0) / "results").exists()


def test_a_worker_refuses_a_folder_of_other_settings(tmp_path):
    module = launcher()
    from evaluate import budget_cuts  # noqa: F401
    settings = settings_file(tmp_path / "search.json")
    folder = tmp_path / "another-hash"
    folder.mkdir()
    (folder / "task.json").write_text(json.dumps(IDENTITY))
    with pytest.raises(SystemExit, match="not those of"):
        module.worker(settings, folder, "GP-fixed", 0)


# ---------------------------------------------------------------------- #
# 4. Arguments, the folder lock and the tools version (after the release review)
# ---------------------------------------------------------------------- #

def test_bad_arguments_are_refused_before_anything_starts():
    module = launcher()
    assert module.check_arguments("0-29", ["GP-fixed", "GP-prop", "GP-adaptive"], 12) == (0, 29)
    for runs, methods, workers in (("0-29", ["GP-fixed"], 0),                 # nothing would ever start
                                   ("0-29", ["GP-fixed", "GP-fixed"], 4),     # two processes on one run
                                   ("0-29", ["GP-other"], 4),
                                   ("29-0", ["GP-fixed"], 4),                 # an empty batch
                                   ("0-900", ["GP-fixed"], 4),                # pilot run numbers
                                   ("0", ["GP-fixed"], 4), ("a-b", ["GP-fixed"], 4)):
        with pytest.raises(SystemExit):
            module.check_arguments(runs, methods, workers)


def test_a_second_launcher_on_the_same_folder_stops(tmp_path):
    module = launcher()
    held = module.lock(tmp_path)
    with pytest.raises(SystemExit, match="another launcher"):
        module.lock(tmp_path)
    held.unlink()
    module.lock(tmp_path).unlink()  # free again once the first one is done


def test_the_tools_version_names_the_commit():
    version = launcher().tools_version()
    assert len(version["tools version"]) == 10 and version["commit"] not in ("", None)


class Endless:
    """A worker that never finishes on its own; records being stopped."""
    def __init__(self):
        self.code, self.terminated = None, False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated, self.code = True, -15

    def kill(self):
        self.code = -9

    def wait(self):
        return self.code


def test_stopping_the_launcher_stops_its_workers(monkeypatch):
    module = launcher()
    started = []

    def start(command):
        started.append(Endless())
        return started[-1]

    def interrupted(seconds):
        raise KeyboardInterrupt  # Ctrl-C while the workers run
    monkeypatch.setattr(module.time, "sleep", interrupted)
    with pytest.raises(KeyboardInterrupt):
        module.run_all([("job", i) for i in range(5)], 3, start=start)
    assert len(started) == 3 and all(p.terminated for p in started)  # no worker left running
