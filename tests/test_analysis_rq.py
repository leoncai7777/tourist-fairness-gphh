"""Tests of the analysis scripts (results, rq1, rq2, rq3, others, timing) on a small batch of pilot rules on one
validation day (no test day, no formal result), and of the further-analysis scenarios."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import forms  # noqa: E402
import results as rs  # noqa: E402
import rq1  # noqa: E402
import rq2  # noqa: E402
import rq3  # noqa: E402
import run_test as rt  # noqa: E402

from evaluate import Days, evaluate  # noqa: E402
from gp_rule import from_text, to_text  # noqa: E402
from rules import TUNED_WEIGHTS  # noqa: E402
from search import hypervolume, load_settings, non_dominated, normalise  # noqa: E402

PILOT = ROOT / "outputs" / "search" / "pilot" / "full"
needs_pilot = pytest.mark.skipif(not (PILOT / "GP-fixed_910" / "results").exists(), reason="no pilot outputs")
DAY = "2019-01-28"  # a validation day
SETTINGS = load_settings(rt.SETTINGS_FILE)


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    if not (PILOT / "GP-fixed_910" / "results").exists():
        pytest.skip("no pilot outputs")
    tmp = tmp_path_factory.mktemp("batch")
    rows = []
    for method in rq1.METHODS:
        front = pd.read_csv(PILOT / f"{method}_910" / "results" / "front.csv.gz")
        texts = [front.loc[front["knee"], "text"].iloc[0]] + list(front.loc[~front["knee"], "text"].iloc[:2])
        group = f"{method}_910"
        for n, text in enumerate(texts):
            rows.append({"rule id": f"{group}_{n:03d}", "group": group, "kind": "gp", "method": method, "run": 910,
                         "text": text, "knee": n == 0})
        shown, _ = forms.display(from_text(texts[0]))
        rows.append({"rule id": f"{group}_000_display", "group": group, "kind": "display", "method": method,
                     "run": 910, "text": to_text(shown), "knee": False, "parent": f"{group}_000",
                     "readable": forms.readable(shown), "tidy nodes": forms.nodes(forms.tidy(from_text(texts[0])))})
    for n, w in enumerate(TUNED_WEIGHTS[:3]):
        rows.append({"rule id": f"tuned_{n:02d}", "group": "baselines", "kind": "tuned", "method": "tuned", "run": -1,
                     "text": repr(w), "knee": n == 1, "validation front": True})
    for name in ["POP", "LEAST", "SOT", "SOS", "observed order"]:
        rows.append({"rule id": name, "group": "baselines", "kind": "control" if name == "observed order" else "hand",
                     "method": name, "run": -1, "text": name, "knee": True})
    pd.DataFrame(rows).to_csv(tmp / "rules.csv", index=False)
    data = {**rt.load_test_settings(), "split": "validation", "replicates": 1, "dates per block": 1}
    (tmp / "test.json").write_text(json.dumps(data))
    original = rt.dates_of
    rt.dates_of = lambda d: [DAY]
    rs.dates_of = rt.dates_of
    try:
        rt.main(["--workers", "2", "--not-formal", "--rules", str(tmp / "rules.csv"), "--test-settings",
                 str(tmp / "test.json"), "--out", str(tmp / "out")])
        results = rs.load(tmp / "out", tmp / "rules.csv", tmp / "test.json", formal=False)
    finally:
        rt.dates_of = original
        rs.dates_of = original
    return results


def test_the_rule_means_are_the_evaluations(batch):
    means = batch.rule_means()
    direct = evaluate(from_text(batch.rules["text"].iloc[0]), DAY, 0, Days())
    assert means.loc[batch.rules["rule id"].iloc[0], "mean R_v (alpha 0.25)"] == direct.mean_r
    assert means.loc[batch.rules["rule id"].iloc[0], "G_hat"] == direct.g_hat


def test_loading_refuses_a_changed_block(batch, tmp_path):
    import shutil
    copy = tmp_path / "out"
    shutil.copytree(batch.out, copy)
    block = next(copy.glob("blocks/baselines/*/objectives.csv.gz"))
    block.write_bytes(block.read_bytes()[:-5])
    original = rt.dates_of
    rt.dates_of = rs.dates_of = lambda d: [DAY]
    try:
        with pytest.raises(SystemExit):
            rs.load(copy, batch.out.parent / "rules.csv", batch.out.parent / "test.json", formal=False)
    finally:
        rt.dates_of = rs.dates_of = original


def test_rq1_fronts_and_tables(batch, tmp_path):
    folder = rq1.run_all(batch, SETTINGS, PILOT, tmp_path / "rq1")
    fronts = pd.read_csv(folder / "test_fronts.csv")
    means = batch.rule_means()
    for method in rq1.METHODS:  # the hypervolume by hand
        ids = list(batch.rules[(batch.rules["kind"] == "gp") & (batch.rules["method"] == method)]["rule id"])
        points = [(means.loc[i, "mean R_v (alpha 0.25)"], means.loc[i, "G_hat"]) for i in ids]
        front = [points[j] for j in non_dominated(points)]
        area, _ = hypervolume(normalise(front, rq1.main_range(SETTINGS)))
        assert fronts.loc[fronts["method"] == method, "hypervolume"].iloc[0] == area
    table2 = pd.read_csv(folder / "table2.csv")
    assert len(table2) == 3 and (table2["n"] == 1).all() and (table2["p"] == 1.0).all()
    assert len(pd.read_csv(folder / "table3.csv")) == 6
    assert (folder / "figure1.png").exists() and (folder / "figure2.png").exists()


def test_rq2_knees_and_display_forms(batch, tmp_path):
    folder = rq2.run_all(batch, PILOT, None, tmp_path / "rq2")
    knees = pd.read_csv(folder / "knees.csv")
    assert len(knees) == 3 and knees["changed share"].between(0, 1).all()
    display = pd.read_csv(folder / "display_forms.csv")
    assert len(display) == 3 and display["display - full, G_hat"].abs().max() < 1  # a difference, not a level
    assert len(pd.read_csv(folder / "table4.csv")) == 3 and (folder / "figure3.png").exists()


def test_rq2_measures_are_not_defined_without_a_change():
    d = pd.DataFrame({"rule id": ["r"] * 3, "changed": [False] * 3, "candidates": [1, 2, 3], "VS change": [0.0] * 3,
                      "h1 given up": [0.0] * 3})
    row = rq2.knee_measures(d).iloc[0]
    assert row["changed share"] == 0 and pd.isna(row["VS change when changed"])
    assert rq2.spearman([1, 1, 1], [1, 2, 3]) is None


def test_rq3_envelope_and_losses(batch, tmp_path):
    folder = rq3.run_all(batch, SETTINGS, tmp_path / "rq3")
    env = pd.read_csv(folder / "envelope_alpha_0.25.csv")
    for _, run in env[env["reached"]].groupby(["method", "run"]):
        assert np.all(np.diff(run["best mean R_v"].to_numpy()) >= 0)  # more rules allowed: never worse
    knees = pd.read_csv(folder / "knees.csv")
    overall = knees[knees["dimension"] == "all"]
    assert set(overall["rule id"]) == set(batch.rules.loc[batch.rules["knee"].astype(bool), "rule id"])
    assert (overall["R_v < 1 (alpha 0.25)"].between(0, 1)).all()
    assert (folder / "figure4.png").exists() and (folder / "figure5.png").exists()


def test_the_envelope_breaks_ties_by_rule_text():
    class Fake:
        rules = pd.DataFrame({"rule id": ["b", "a"], "kind": ["gp", "gp"], "method": ["GP-fixed"] * 2, "run": [0, 0],
                              "text": ["z rule", "a rule"], "knee": [True, False], "validation front": [None, None]})

        def rule_means(self):
            return pd.DataFrame({"mean R_v (alpha 0.25)": [1.0, 1.0], "G_hat": [0.3, 0.3]}, index=["b", "a"])

        def completion(self):
            return pd.DataFrame({"completion": [0.9, 0.99]}, index=["b", "a"])
    Fake.rules = pd.concat([Fake.rules, pd.DataFrame({"rule id": ["t"], "kind": ["tuned"], "method": ["tuned"],
                                                      "run": [-1], "text": ["0.1"], "knee": [True],
                                                      "validation front": [True]})], ignore_index=True)
    fake = Fake()
    fake.rule_means = lambda: pd.DataFrame({"mean R_v (alpha 0.25)": [1.0, 1.0, 0.5], "G_hat": [0.3, 0.3, 0.2]},
                                           index=["b", "a", "t"])
    fake.completion = lambda: pd.DataFrame({"completion": [0.9, 0.99, 0.8]}, index=["b", "a", "t"])
    env = rq3.envelope(fake)
    chosen = env[(env["method"] == "GP-fixed") & env["reached"]]["rule id"].unique()
    assert list(chosen) == ["a"]  # the same best R_v: "a rule" comes first by text, whatever its completion


@needs_pilot
def test_the_scenarios_choose_their_rules_and_settings(tmp_path, monkeypatch):
    rules = pd.read_csv(ROOT / "outputs" / "search" / "pilot" / "full" / "GP-adaptive_910" / "results" /
                        "front.csv.gz")
    table = pd.DataFrame([{"rule id": f"g{n}", "group": "GP-adaptive_910", "kind": "gp", "method": "GP-adaptive",
                           "run": 910, "text": t, "knee": bool(k)} for n, (t, k) in
                          enumerate(zip(rules["text"], rules["knee"]))]
                         + [{"rule id": "POP", "group": "baselines", "kind": "hand", "method": "POP", "run": -1,
                             "text": "POP", "knee": True},
                            {"rule id": "t0", "group": "baselines", "kind": "tuned", "method": "tuned", "run": -1,
                             "text": "0.0", "knee": True, "validation front": True}])
    knees = rt.rules_of(table, {"scenario": {"rules": "knees"}})
    assert set(knees["rule id"]) == {table.loc[table["knee"] & (table["kind"] == "gp"), "rule id"].iloc[0], "POP", "t0"}
    assert set(rt.rules_of(table, {"scenario": {"rules": "hand and control"}})["rule id"]) == {"POP"}
    diagnostic = rt.diagnostic_rules(table, root=PILOT)
    log = pd.read_csv(PILOT / "GP-adaptive_910" / "results" / "knee_decisions.csv.gz")
    assert len(diagnostic) == 1 and diagnostic["constant"].iloc[0] == float(np.median(log["w"]))
    rule = rt.policy(diagnostic.iloc[0])
    assert rule[1].height == 0 and rule[1][0].value == diagnostic["constant"].iloc[0]
    changed = rt.scenario_settings(SETTINGS, {"scenario": {"stop limit": False, "peak floor": "median of the candidates",
                                                            "same day": True, "budget": "span"}})
    assert (changed["stop limit"], changed["same day"], changed["budget"]) == (False, True, "span")
    assert changed["peak floor"] == pytest.approx(18.35)


def test_compliance_sets_are_nested_and_the_same_for_every_rule():
    day = Days().get(DAY)[0]
    seventy = set(rt.followers(day, DAY, 0, 0.7))
    fifty = set(rt.followers(day, DAY, 0, 0.5))
    assert fifty < seventy and seventy == set(rt.followers(day, DAY, 0, 0.7))
    evaluated = set(day.visitors.loc[day.visitors["evaluated"], "visitor"])
    assert seventy <= evaluated and 0.6 < len(seventy) / len(evaluated) < 0.8
    assert set(rt.followers(day, DAY, 1, 0.7)) != seventy  # another replicate, another draw


def test_a_scenario_cannot_change_other_things(tmp_path):
    good = rt.load_test_settings()
    for scenario in ({"rules": "all"}, {"alpha": 1}, {"compliance": 1.5}, {"peak floor": 10}, {"budget": "corrected"}):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({**good, "scenario": scenario}))
        with pytest.raises(ValueError):
            rt.load_test_settings(path)
