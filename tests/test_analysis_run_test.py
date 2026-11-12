"""Tests of analysis/run_test.py on pilot rules and validation days (no test day, no formal result)."""

import gzip
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import forms  # noqa: E402
import run_test as rt  # noqa: E402

from gp_rule import from_text, to_text  # noqa: E402
from rules import TUNED_WEIGHTS  # noqa: E402
from search import complete, load_settings  # noqa: E402

PILOT = ROOT / "outputs" / "search" / "pilot" / "full"
needs_pilot = pytest.mark.skipif(not (PILOT / "GP-fixed_910" / "results").exists(), reason="no pilot outputs")
TWO_DAYS = ["2019-01-28", "2019-01-29"]  # validation days


def test_the_dates_leave_out_days_without_evaluated_visitors():
    assert len(rt.dates_of({"year": 2019, "split": "test"})) == 181
    assert len(rt.dates_of({"year": 2019, "split": "validation"})) == 35
    if not (rt.PROCESSED / "visitors_2022.csv.gz").exists():
        pytest.skip("the 2022 tables are not here (they are not redistributed)")
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


def pilot_root(tmp_path, methods=("GP-fixed", "GP-adaptive")):
    root = tmp_path / "formal"
    for method in methods:
        shutil.copytree(PILOT / f"{method}_910" / "results", root / f"{method}_910" / "results")
    (root / "tuned").mkdir()
    pd.DataFrame({"weight": TUNED_WEIGHTS, "mean R_v": [1.2 - 0.03 * i for i in range(16)],
                  "G_hat": [0.65 - 0.03 * i for i in range(16)], "front": [True] * 16}).to_csv(
        root / "tuned" / "tuned_front.csv", index=False)
    return root


@needs_pilot
def test_the_rule_list_is_written_once_and_covers_what_it_must(tmp_path):
    root = pilot_root(tmp_path)
    path = tmp_path / "rules.csv"
    table = rt.write_rule_list(path, root, methods=("GP-fixed", "GP-adaptive"), runs=[910])
    assert table["rule id"].is_unique
    shown = table[table["kind"] == "display"]  # one display form per GP knee, pointing to it
    assert set(shown["parent"]) == set(table.loc[table["knee"] & (table["kind"] == "gp"), "rule id"])
    assert all(rt.policy(row) is not None for _, row in shown.iterrows())
    with pytest.raises(SystemExit):
        rt.write_rule_list(path, root, methods=("GP-fixed",), runs=[910])
    rt.check_rule_list(table, methods=("GP-fixed", "GP-adaptive"), runs=[910])
    for broken in (table[table["group"] != "GP-fixed_910"],                       # a run missing
                   table.assign(knee=table["knee"] | (table["kind"] == "gp")),    # every GP rule a knee
                   table[table["kind"] != "control"]):                            # the control missing
        with pytest.raises(SystemExit, match="does not cover"):
            rt.check_rule_list(broken, methods=("GP-fixed", "GP-adaptive"), runs=[910])


@needs_pilot
def test_the_formal_start_needs_full_coverage_and_the_pass_record(tmp_path, monkeypatch):
    table = rt.write_rule_list(tmp_path / "rules.csv", pilot_root(tmp_path), methods=("GP-fixed", "GP-adaptive"),
                               runs=[910])
    settings = load_settings(rt.SETTINGS_FILE)
    monkeypatch.setattr(rt, "check_here", lambda: {"stub": True})
    monkeypatch.setattr(rt, "check_complete", lambda: None)
    with pytest.raises(SystemExit, match="does not cover"):  # a pilot list is not the formal one
        rt.identity_of(settings, rt.TEST_SETTINGS, tmp_path / "rules.csv", table)
    monkeypatch.setattr(rt, "check_rule_list", lambda rules: None)
    monkeypatch.setattr(rt, "PASS_RECORD", tmp_path / "reproduced.json")
    with pytest.raises(SystemExit, match="has not passed"):
        rt.identity_of(settings, rt.TEST_SETTINGS, tmp_path / "rules.csv", table)
    other = dict(settings, beta=0.6)  # other definitions loaded
    with pytest.raises(SystemExit, match="not Step 5's"):
        rt.identity_of(other, rt.TEST_SETTINGS, tmp_path / "rules.csv", table)


def small_batch(tmp_path):
    """Two groups (two pilot rules and one display form; POP and the control), one replicate."""
    texts = pd.read_csv(PILOT / "GP-adaptive_910" / "results" / "front.csv.gz")["text"].iloc[:2]
    rows = [{"rule id": f"g_{n}", "group": "GP-adaptive_910", "kind": "gp", "method": "GP-adaptive", "run": 910,
             "text": t, "knee": n == 0} for n, t in enumerate(texts)]
    shown, _ = forms.display(from_text(texts.iloc[0]))
    rows.append({"rule id": "g_0_display", "group": "GP-adaptive_910", "kind": "display", "method": "GP-adaptive",
                 "run": 910, "text": to_text(shown), "knee": False, "parent": "g_0"})
    rows += [{"rule id": "POP", "group": "baselines", "kind": "hand", "method": "POP", "run": -1, "text": "POP",
              "knee": True},
             {"rule id": "observed order", "group": "baselines", "kind": "control", "method": "observed order",
              "run": -1, "text": "observed order", "knee": True}]
    rules = tmp_path / "rules.csv"
    pd.DataFrame(rows).to_csv(rules, index=False)
    settings = tmp_path / "test.json"
    data = {**rt.load_test_settings(), "split": "validation", "replicates": 1, "dates per block": 1}
    settings.write_text(json.dumps(data))
    return ["--workers", "2", "--not-formal", "--rules", str(rules), "--test-settings", str(settings),
            "--out", str(tmp_path / "out")], data


@needs_pilot
def test_a_small_batch_runs_resumes_verifies_and_refuses(tmp_path, monkeypatch, capsys):
    args, data = small_batch(tmp_path)
    monkeypatch.setattr(rt, "dates_of", lambda d: TWO_DAYS)
    rt.main(args)
    out = tmp_path / "out"
    assert len(list(out.glob("blocks/*/*/done.json"))) == 4 and not list(out.glob("blocks/*/*.partial"))
    block = out / "blocks" / "GP-adaptive_910" / TWO_DAYS[0]
    first = pd.read_csv(block / "objectives.csv.gz")
    assert len(first) == 3 and set(first.columns) >= {"mean R_v (alpha 0)", "mean R_v (alpha 1)", "G_hat"}
    log = pd.read_csv(block / "decisions.csv.gz")
    assert set(log["rule id"]) == {"g_0"} and log["display agrees"].isin([True, False]).all()
    assert {"visitor", "j0 site", "jw site"} <= set(log.columns)
    assert json.loads((out / "task.json").read_text())["mode"] == "not formal"
    assert not (out / "launcher.lock").exists()

    rt.main(args)  # nothing left to do
    assert "0 of 4 blocks to do" in capsys.readouterr().out
    rt.main(args + ["--verify"])
    assert "0 blocks missing, 0 changed" in capsys.readouterr().out

    (block / "done.json").unlink()  # as if stopped before the mark: computed again, the same
    rt.main(args)
    assert pd.read_csv(block / "objectives.csv.gz").equals(first)

    peaks = block / "peaks.csv.gz"  # a file changed after its block was done, same size
    data_bytes = bytearray(gzip.decompress(peaks.read_bytes()))
    data_bytes[-2:-1] = b"9" if data_bytes[-2:-1] != b"9" else b"8"
    changed = gzip.compress(bytes(data_bytes), mtime=0)
    done = json.loads((block / "done.json").read_text())
    done["files"]["peaks.csv.gz"]["bytes"] = len(changed)
    (block / "done.json").write_text(json.dumps(done))
    peaks.write_bytes(changed)
    rt.main(args)  # sizes agree: the quick check at the start lets it pass ...
    with pytest.raises(SystemExit):
        rt.main(args + ["--verify"])  # ... the hashes after the batch do not
    peaks.write_bytes(changed[:-10])  # truncated: caught at the next start
    with pytest.raises(SystemExit, match="changed files"):
        rt.main(args)

    done = json.loads((block / "done.json").read_text())
    (block / "done.json").write_text(json.dumps({**done, "rule list hash": "another"}))
    with pytest.raises(SystemExit, match="another identity"):
        rt.main(args)


@needs_pilot
def test_a_run_that_is_not_formal_stays_on_validation_days_and_out_of_the_formal_folders(tmp_path, monkeypatch):
    args, data = small_batch(tmp_path)
    test_days = tmp_path / "t.json"
    test_days.write_text(json.dumps({**data, "split": "test"}))
    common = ["--workers", "2", "--not-formal", "--rules", str(tmp_path / "rules.csv")]
    with pytest.raises(SystemExit, match="only use validation days"):
        rt.main(common + ["--test-settings", str(test_days), "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="may not write under"):
        rt.main(common + ["--test-settings", str(tmp_path / "test.json"), "--out", str(rt.OUTPUTS / "not_here")])
    assert not (rt.OUTPUTS / "not_here").exists()


@needs_pilot
def test_the_block_check_refuses_what_is_not_complete(tmp_path, monkeypatch):
    args, data = small_batch(tmp_path)
    rules = pd.read_csv(tmp_path / "rules.csv")
    settings = load_settings(rt.SETTINGS_FILE)
    baselines = rules[rules["group"] == "baselines"]
    tables = rt.compute_block(baselines, TWO_DAYS[:1], 1, settings)
    rt.check_block(tables, baselines, TWO_DAYS[:1], 1)  # complete as made

    def broken(name, change):
        t = {k: v.copy() for k, v in tables.items()}
        t[name] = change(t[name])
        with pytest.raises(ValueError):
            rt.check_block(t, baselines, TWO_DAYS[:1], 1)
    broken("objectives", lambda t: t.assign(date=TWO_DAYS[1]))                       # same rows, wrong date
    broken("objectives", lambda t: t.assign(**{"G_hat": float("nan")}))
    broken("visitors", lambda t: pd.concat([t, t.iloc[:1]], ignore_index=True))      # a visitor twice
    broken("visitors", lambda t: t.assign(**{"C observed": float("inf")}))           # a route part not finite
    broken("peaks", lambda t: t.iloc[1:])                                             # a site missing
    broken("hourly", lambda t: t[t["rule id"] != "POP"])                              # a knee's table missing
    broken("groups", lambda t: pd.concat([t, t.iloc[:1]], ignore_index=True))        # a group twice


@needs_pilot
def test_one_launcher_per_folder(tmp_path, monkeypatch):
    args, _ = small_batch(tmp_path)
    monkeypatch.setattr(rt, "dates_of", lambda d: TWO_DAYS[:1])
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "launcher.lock").write_text("1")
    with pytest.raises(SystemExit, match="another launcher"):
        rt.main(args)
