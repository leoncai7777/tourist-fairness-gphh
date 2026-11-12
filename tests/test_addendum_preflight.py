"""Tests of addendum/preflight.py on temporary folders: the environment check, the copy of the tuned folder and
the check after the launch."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addendum"))
import preflight  # noqa: E402


def git_repo(folder):
    folder.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(folder), *a], check=True, capture_output=True, text=True)  # noqa
    run("init", "-q")
    (folder / "f.py").write_text("x = 1\n")
    run("add", "f.py")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")
    return run("rev-parse", "--short=7", "HEAD").stdout.strip()


def test_the_environment_check_reports_every_difference(tmp_path, monkeypatch):
    head = git_repo(tmp_path / "w")
    monkeypatch.setattr(preflight, "STEP5_TOOLS", {**preflight.STEP5_TOOLS, "commit": head})
    monkeypatch.setattr(preflight, "environment", lambda: dict(preflight.STEP5_ENV))
    assert preflight.check_env(tmp_path / "w")[1] == {}
    monkeypatch.setattr(preflight, "environment", lambda: {**preflight.STEP5_ENV, "numpy": "9.9", "pip freeze": "x"})
    assert set(preflight.check_env(tmp_path / "w")[1]) == {"numpy", "pip freeze"}
    (tmp_path / "w" / "f.py").write_text("x = 2\n")  # a change to a tracked file
    assert "worktree changes" in preflight.check_env(tmp_path / "w")[1]
    monkeypatch.setattr(preflight, "STEP5_TOOLS", {**preflight.STEP5_TOOLS, "commit": "0000000"})
    assert "worktree commit" in preflight.check_env(tmp_path / "w")[1]


def tuned_source(root):
    t = root / "tuned"
    t.mkdir(parents=True)
    (t / "complete.json").write_text("{}")
    (t / "tuned_front.csv").write_text("weight\n0\n")
    return root


def test_the_tuned_folder_is_copied_once_with_its_hashes(tmp_path):
    source, target = tuned_source(tmp_path / "s"), tmp_path / "t"
    record = preflight.copy_tuned(source, target)
    assert set(record) == {"complete.json", "tuned_front.csv"}
    assert all(h["source"] == h["copy"] for h in record.values())
    assert json.loads((target / "tuned_copy.json").read_text())["files"] == record
    assert preflight.copy_tuned(source, target) == record  # a second call compares and keeps the copy


def test_the_tuned_copy_is_refused_when_unsafe(tmp_path):
    source = tuned_source(tmp_path / "s")
    (source / "launcher.lock").write_text("1")
    with pytest.raises(SystemExit):  # the source may still be written
        preflight.copy_tuned(source, tmp_path / "t1")
    (source / "launcher.lock").unlink()
    (source / "tuned" / "complete.json").unlink()
    with pytest.raises(SystemExit):  # not complete
        preflight.copy_tuned(source, tmp_path / "t2")
    source = tuned_source(tmp_path / "s2")
    target = tmp_path / "t3"
    preflight.copy_tuned(source, target)
    (target / "tuned" / "tuned_front.csv").write_text("weight\n1\n")
    with pytest.raises(SystemExit):  # a target that differs is not overwritten
        preflight.copy_tuned(source, target)
    assert (target / "tuned" / "tuned_front.csv").read_text() == "weight\n1\n"
    (target / "tuned" / "tuned_front.csv").write_text("weight\n0\n")
    (target / "tuned" / "extra.txt").write_text("x")
    with pytest.raises(SystemExit):  # another file in the target
        preflight.copy_tuned(source, target)


def test_the_launch_check(tmp_path):
    good = {"tools version": "469e9e5b75", "commit": "d2b4af4", "methods": ["GP-single"]}
    (tmp_path / "launches.jsonl").write_text(json.dumps(good) + "\n")
    assert preflight.check_launched(tmp_path)[1] == {}
    (tmp_path / "launches.jsonl").write_text(json.dumps(good) + "\n" + json.dumps({**good, "tools version": "abc"}) + "\n")
    assert "tools version" in preflight.check_launched(tmp_path)[1]
    (tmp_path / "launches.jsonl").write_text(json.dumps({**good, "commit": "1234567"}) + "\n")
    assert "commit" in preflight.check_launched(tmp_path)[1]
