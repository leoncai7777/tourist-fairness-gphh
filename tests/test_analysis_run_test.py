"""Tests of analysis/run_test.py on pilot rules and validation days (no test day, no formal result)."""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import run_test as rt  # noqa: E402

from rules import TUNED_WEIGHTS  # noqa: E402
from search import complete  # noqa: E402

PILOT = ROOT / "outputs" / "search" / "pilot" / "full"
needs_pilot = pytest.mark.skipif(not (PILOT / "GP-fixed_910" / "results").exists(), reason="no pilot outputs")


def test_the_dates_leave_out_days_without_evaluated_visitors():
    assert len(rt.dates_of({"year": 2019, "split": "test"})) == 181
    assert len(rt.dates_of({"year": 2019, "split": "validation"})) == 35
    all_2022 = rt.dates_of({"year": 2022, "split": "all"})  # dates only, from the visitor table
    assert len(all_2022) == 362 and "2022-03-07" not in all_2022


def test_the_settings_file_is_strict(tmp_path):
    good = rt.load_test_settings()
    assert good["replicates"] == 5 and good["split"] == "test"
    bad = tmp_path / "t.json"
    bad.write_text(json.dumps({**good, "alphas": [0.25]}))
    with pytest.raises(ValueError):
        rt.load_test_settings(bad)
    bad.write_text(json.dumps({**good, "extra": 1}))
    with pytest.raises(ValueError):
        rt.load_test_settings(bad)


def test_the_tuned_knee_takes_the_smaller_weight_on_a_tie():
    settings = complete({"R range": (0.5, 1.25), "G range": (0.1, 0.75)})
    table = pd.DataFrame({"weight": [0.0, 0.1, 1.0, 10.0], "mean R_v": [1.1, 1.0, 1.0, 0.7],
                          "G_hat": [0.6, 0.4, 0.4, 0.2], "front": [True, True, True, True]})
    assert rt.tuned_knee_weight(table, settings) == 1  # 0.1 and 1.0 are the same point: the smaller


@needs_pilot
def test_the_rule_list_is_written_once(tmp_path):
    root = tmp_path / "formal"
    for method in ("GP-fixed", "GP-adaptive"):
        shutil.copytree(PILOT / f"{method}_910" / "results", root / f"{method}_910" / "results")
    (root / "tuned").mkdir()
    pd.DataFrame({"weight": TUNED_WEIGHTS, "mean R_v": [1.2 - 0.03 * i for i in range(16)],
                  "G_hat": [0.65 - 0.03 * i for i in range(16)], "front": [True] * 16}).to_csv(
        root / "tuned" / "tuned_front.csv", index=False)
    path = tmp_path / "rules.csv"
    table = rt.write_rule_list(path, root, methods=("GP-fixed", "GP-adaptive"), runs=[910])
    assert table["rule id"].is_unique
    assert table.groupby("group")["knee"].sum().loc[["GP-fixed_910", "GP-adaptive_910"]].tolist() == [1, 1]
    assert (table["kind"] == "tuned").sum() == 16 and table.loc[table["kind"] == "tuned", "knee"].sum() == 1
    assert set(table.loc[table["kind"].isin(["hand", "control"]), "text"]) == {"POP", "LEAST", "SOT", "SOS",
                                                                               "observed order"}
    with pytest.raises(SystemExit):
        rt.write_rule_list(path, root, methods=("GP-fixed",), runs=[910])


def small_batch(tmp_path):
    """Two groups (two pilot rules; POP and the control), two validation dates, one replicate."""
    texts = pd.read_csv(PILOT / "GP-adaptive_910" / "results" / "front.csv.gz")["text"].iloc[:2]
    rows = [{"rule id": f"g_{n}", "group": "GP-adaptive_910", "kind": "gp", "method": "GP-adaptive", "run": 910,
             "text": t, "knee": n == 0} for n, t in enumerate(texts)]
    rows += [{"rule id": "POP", "group": "baselines", "kind": "hand", "method": "POP", "run": -1, "text": "POP",
              "knee": True},
             {"rule id": "observed order", "group": "baselines", "kind": "control", "method": "observed order",
              "run": -1, "text": "observed order", "knee": True}]
    rules = tmp_path / "rules.csv"
    pd.DataFrame(rows).to_csv(rules, index=False)
    settings = tmp_path / "test.json"
    data = {**rt.load_test_settings(), "split": "validation", "replicates": 1, "dates per block": 1}
    settings.write_text(json.dumps(data))
    return ["--workers", "2", "--no-identity-check", "--rules", str(rules), "--test-settings", str(settings),
            "--out", str(tmp_path / "out")], data


@needs_pilot
def test_a_small_batch_runs_resumes_and_refuses(tmp_path, monkeypatch, capsys):
    args, data = small_batch(tmp_path)
    monkeypatch.setattr(rt, "dates_of", lambda d: ["2019-01-28", "2019-01-29"])  # two validation days
    rt.main(args)
    out = tmp_path / "out"
    blocks = sorted(out.glob("blocks/*/*/done.json"))
    assert len(blocks) == 4 and not list(out.glob("blocks/*/*.partial"))
    first = pd.read_csv(out / "blocks" / "GP-adaptive_910" / "2019-01-28" / "objectives.csv.gz")
    assert len(first) == 2 and set(first.columns) >= {"mean R_v (alpha 0)", "mean R_v (alpha 1)", "G_hat"}
    assert (out / "blocks" / "baselines" / "2019-01-29" / "visitors.csv.gz").exists()
    assert not (out / "launcher.lock").exists()

    rt.main(args)  # nothing left to do
    assert "0 of 4 blocks to do" in capsys.readouterr().out

    again = out / "blocks" / "GP-adaptive_910" / "2019-01-28"
    (again / "done.json").unlink()  # as if stopped before the mark: computed again, the same
    rt.main(args)
    assert pd.read_csv(again / "objectives.csv.gz").equals(first)

    done = json.loads((again / "done.json").read_text())
    (again / "done.json").write_text(json.dumps({**done, "rule list hash": "another"}))
    with pytest.raises(SystemExit, match="another identity"):
        rt.main(args)


@needs_pilot
def test_one_launcher_per_folder(tmp_path, monkeypatch):
    args, _ = small_batch(tmp_path)
    monkeypatch.setattr(rt, "dates_of", lambda d: ["2019-01-28"])
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "launcher.lock").write_text("1")
    with pytest.raises(SystemExit, match="another launcher"):
        rt.main(args)
