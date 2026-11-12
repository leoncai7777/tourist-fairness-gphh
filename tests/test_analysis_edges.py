"""Edge cases of the analysis and of the block checks, on small made-up tables (after the code review)."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import further  # noqa: E402
import results as rs  # noqa: E402
import rq1  # noqa: E402
import rq2  # noqa: E402
import rq3  # noqa: E402
import run_test as rt  # noqa: E402

from search import load_settings  # noqa: E402

SETTINGS = load_settings(rt.SETTINGS_FILE)


def bare_results(rules, dates, replicates, objectives):
    r = object.__new__(rs.Results)
    r.out, r.rules, r.dates, r.identity = Path("."), rules, dates, {}
    r.data = {"year": 2019, "split": "test", "replicates": replicates}
    r._tables = {("objectives", "*"): objectives}
    return r


def test_a_rules_test_mean_weights_days_equally_and_rq3_pools_visitors():
    rules = pd.DataFrame({"rule id": ["a"], "group": ["g"]})
    objectives = pd.DataFrame({"rule id": ["a"] * 4, "date": ["d1", "d1", "d2", "d2"], "replicate": [0, 1, 0, 1],
                               **{rs.r_column(x): [1.0, 1.0, 0.0, 0.0] for x in rs.ALPHAS}, "G_hat": [0.5] * 4})
    means = bare_results(rules, ["d1", "d2"], 2, objectives).rule_means()
    assert means.loc["a", rs.r_column()] == 0.5  # two days, equal weight, whatever their visitors
    # RQ3 pools visitors: 1 visitor at R = 1 on one day, 3 at R = 0 on the other gives 0.25, not 0.5
    v = pd.DataFrame({"follows": True, "arrival": 600, "budget": 300, "observed stops": 3, "stops": 3,
                      "change A": [0.0, -1.0, -1.0, -1.0], "change C": 0.0, "change T": 0.0,
                      "A observed": 1.0, **{f"R_v (alpha {x:g})": [1.0, 0.0, 0.0, 0.0] for x in rs.ALPHAS}})
    overall = [row for row in rq3.knee_visitors(v, 0.5) if row["dimension"] == "all"][0]
    assert overall["mean R_v (alpha 0.25)"] == 0.25 and overall["R_v < 1 (alpha 0.25)"] == 0.75


def test_other_years_are_not_tested():
    assert rq1.is_main_test({"year": 2019, "split": "test"})
    assert not rq1.is_main_test({"year": 2022, "split": "all"})
    assert not rq1.is_main_test({"year": 2019, "split": "test", "scenario": {"rules": "knees"}})


def test_a_tie_in_the_ranking_is_shown():
    assert rq1.ranking_of({"A": 0.6, "B": 0.6, "C": 0.5}) in ("A = B > C", "B = A > C")
    assert rq1.ranking_of({"A": 0.4, "B": 0.6, "C": None}) == "B > A"


def test_knees_without_decisions_or_changes_stay_and_are_na():
    d = pd.DataFrame({"rule id": ["k1"] * 3 + ["k2"] * 2, "changed": [False, False, False, True, False],
                      "candidates": [1, 1, 1, 2, 3], "VS change": [0.0, 0.0, 0.0, -1.0, 0.0],
                      "h1 given up": [0.0, 0.0, 0.0, 0.2, 0.0], "w": 1.0, "GDAY": 1.0, "ARR": [1, 2, 3, 4, 5]})
    m = rq2.knee_measures(d, ["k1", "k2", "k3"]).set_index("rule id")
    assert m.loc["k1", "changed share"] == 0 and pd.isna(m.loc["k1", "changed share, 2+ candidates"])
    assert pd.isna(m.loc["k1", "VS change when changed"])  # no change: NA, never 0
    assert m.loc["k3", "decisions"] == 0 and pd.isna(m.loc["k3", "changed share"])  # kept, with no denominator
    assert m.loc["k2", "changed share, 2+ candidates"] == 0.5


def test_correlations_say_why_they_are_not_defined():
    pytest.importorskip("scipy")
    assert rq2.spearman([1, 2], [1, 2]) == (None, "fewer than 3 decisions")
    assert rq2.spearman([1, 1, 1], [1, 2, 3]) == (None, "w constant")
    assert rq2.spearman([1, 2, 3], [5, 5, 5]) == (None, "state constant")
    assert rq2.spearman([1, 2, 3], [1, 2, 3])[0] == pytest.approx(1.0)


def test_an_envelope_that_no_run_reaches_still_has_its_columns():
    env = pd.DataFrame({"method": ["GP-fixed"] * 2, "run": [0, 0], "level": [0.15, 0.2], "reached": [False, False]})
    summary = rq3.envelope_summary(env)
    assert (summary["runs reaching it"] == 0).all() and summary["best mean R_v median"].isna().all()


def test_the_h2_diagnostic_is_set_beside_its_knee():
    main = pd.DataFrame({"method": "GP-adaptive", "kind": "gp", "knee": True, "mean R_v": [0.9, 0.8],
                         "G_hat": [0.3, 0.4], "completion": [0.95, 0.96]}, index=["k1", "k2"])
    scenario = pd.DataFrame({"method": "GP-adaptive", "kind": "diagnostic", "knee": True, "mean R_v": [0.7, 0.6],
                             "G_hat": [0.35, 0.45], "completion": [0.9, 0.9], "parent": ["k1", "k2"]},
                            index=["k1_h2_constant", "k2_h2_constant"])
    row = further.summary(scenario, main).iloc[0]
    assert row["mean R_v main median"] == pytest.approx(0.85) and row["mean R_v median"] == pytest.approx(0.65)
    with pytest.raises(SystemExit, match="without their match"):
        further.summary(scenario, main.drop(index="k2"))


def test_a_change_in_a_built_rule_changes_the_identity():
    built = pd.DataFrame({"rule id": ["k_h2_constant"], "parent": ["k"], "kind": ["diagnostic"],
                          "text": ["GP-adaptive | h1: AV | h2: 0.5"]})
    other = built.assign(text=["GP-adaptive | h1: AV | h2: 0.6"])
    assert rt.derived_identity(built) != rt.derived_identity(other)
    assert rt.derived_identity(built.assign(kind="gp")) == {}


@pytest.fixture(scope="module")
def baseline_block():
    rules = pd.DataFrame([{"rule id": "POP", "group": "baselines", "kind": "hand", "method": "POP", "run": -1,
                           "text": "POP", "knee": True},
                          {"rule id": "observed order", "group": "baselines", "kind": "control",
                           "method": "observed order", "run": -1, "text": "observed order", "knee": True}])
    return rules, rt.compute_block(rules, ["2019-01-28"], 1, SETTINGS)


@pytest.mark.parametrize("name, change", [
    ("peaks", lambda t: t.assign(site=t["site"].replace({t["site"].iloc[0]: 999}))),       # not a site
    ("hourly", lambda t: t.assign(**{"hour 10": np.inf})),
    ("observed_hourly", lambda t: t.assign(**{"hour 10": np.nan})),
    ("visitors", lambda t: t.assign(**{"change C": np.nan})),
    ("visitors", lambda t: t.assign(stops=np.inf)),
])
def test_the_block_check_refuses_more(baseline_block, name, change):
    rules, tables = baseline_block
    t = {k: v.copy() for k, v in tables.items()}
    t[name] = change(t[name])
    with pytest.raises(ValueError):
        rt.check_block(t, rules, ["2019-01-28"], 1)


def test_decision_logs_are_checked_and_an_empty_one_is_legal():
    from gp_rule import from_text
    front = pd.read_csv(ROOT / "outputs" / "search" / "pilot" / "full" / "GP-adaptive_910" / "results" / "front.csv.gz")
    rules = pd.DataFrame([{"rule id": "k", "group": "g", "kind": "gp", "method": "GP-adaptive", "run": 910,
                           "text": front.loc[front["knee"], "text"].iloc[0], "knee": True}])
    tables = rt.compute_block(rules, ["2019-01-28"], 1, SETTINGS)
    rt.check_block(tables, rules, ["2019-01-28"], 1)
    for change in (lambda t: t.assign(date="2019-01-29"),                  # outside the block
                   lambda t: t.assign(**{"jw site": 999}),                  # not a site
                   lambda t: t.assign(visitor=-1),                          # not one of the knee's visitors
                   lambda t: t.assign(w=np.nan)):
        bad = {k: v.copy() for k, v in tables.items()}
        bad["decisions"] = change(bad["decisions"])
        with pytest.raises(ValueError):
            rt.check_block(bad, rules, ["2019-01-28"], 1)
    empty = {k: v.copy() for k, v in tables.items()}
    empty["decisions"] = empty["decisions"].iloc[:0]
    rt.check_block(empty, rules, ["2019-01-28"], 1)  # no decision that day is legal
    assert from_text(rules["text"].iloc[0])  # the rule itself is a strict rule


def test_files_outside_the_checked_blocks_are_never_read_and_a_lock_stops_the_reading(tmp_path):
    rules = pd.DataFrame({"rule id": ["a"], "group": ["g"]})
    r = object.__new__(rs.Results)
    r.out, r.rules, r._tables = tmp_path, rules, {}
    r.blocks = {"g": [tmp_path / "blocks" / "g" / "d1"]}
    for folder in (tmp_path / "blocks" / "g" / "d1", tmp_path / "blocks" / "g" / "stray", tmp_path / "blocks" / "x" / "d1"):
        folder.mkdir(parents=True)
        pd.DataFrame({"rule id": ["a"], "visitors": [1]}).to_csv(folder / "groups.csv.gz", index=False)
    assert r.files("groups") == [tmp_path / "blocks" / "g" / "d1" / "groups.csv.gz"]
    (tmp_path / "launcher.lock").write_text("1")
    with pytest.raises(SystemExit, match="may still be running"):
        rs.load(tmp_path, tmp_path / "rules.csv")
