"""
Tests of the launcher of the formal runs (tools/run_formal.py, Step 5). Skipped inside the self-check's
copies, which hold src and tests only.

Run:  python3 -m pytest tests -q   (from the repository root)
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "run_formal.py"
sys.path.insert(0, str(ROOT / "src"))
pytestmark = pytest.mark.skipif(not TOOL.exists(), reason="inside a copy made by the self-check")


def launcher():
    spec = importlib.util.spec_from_file_location("run_formal", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_queue_is_interleaved_by_run_number():
    order = launcher().tasks(range(3), ["GP-fixed", "GP-prop", "GP-adaptive"])
    assert order[:4] == [("GP-fixed", 0), ("GP-prop", 0), ("GP-adaptive", 0), ("GP-fixed", 1)]
    assert len(order) == 9 and len(set(order)) == 9


class FakeProcess:
    """Finishes after a number of polls with an exit code; records how many were running at once."""
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


def test_finished_runs_are_skipped(tmp_path):
    module = launcher()
    folder = module.folder_of(tmp_path, "GP-prop", 3) / "results"
    folder.mkdir(parents=True)
    assert not module.done(tmp_path, "GP-prop", 3)
    (folder / "summary.json").write_text(json.dumps({}))
    assert module.done(tmp_path, "GP-prop", 3) and not module.done(tmp_path, "GP-prop", 4)


def test_a_worker_refuses_a_folder_of_other_settings(tmp_path):
    from evaluate import budget_cuts
    from search import complete
    settings = complete({"population": 4, "generations": 1, "crossover": 0.8, "mutation": 0.15,
                         "monitor every": 1, "validation replicates": 1})
    path = tmp_path / "search.json"
    path.write_text(json.dumps({"settings": settings, "budget cuts": list(budget_cuts())}))
    with pytest.raises(SystemExit, match="not the folder of these settings"):
        launcher().worker(path, tmp_path / "another-hash", "GP-fixed", 0)
