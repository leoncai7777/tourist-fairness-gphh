"""Tests of analysis/reproduce.py and analysis/step5.py on pilot outputs and made-up folders (no formal result)."""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import evaluate6  # noqa: E402
import reproduce as rp  # noqa: E402
import step5  # noqa: E402

from gp_rule import from_text  # noqa: E402
from search import complete, hypervolume, normalise, validation_dates, validation_means  # noqa: E402

PILOT = ROOT / "outputs" / "search" / "pilot" / "full"
METHODS = ("GP-fixed", "GP-prop", "GP-adaptive")
needs_pilot = pytest.mark.skipif(not (PILOT / "GP-fixed_910" / "results").exists(), reason="no pilot outputs")
SETTINGS = complete({"population": 200, "generations": 100, "crossover": 0.8, "mutation": 0.15,
                     "monitor every": 10, "validation replicates": 5})


def pilot_summary(folder):
    return json.loads((folder / "results" / "summary.json").read_text())


def pilot_range(folder):
    s = pilot_summary(folder)
    return {"R": tuple(s["range"]["R"]), "G": tuple(s["range"]["G"])}


def pilot_identity(folder):
    s = pilot_summary(folder)
    return {k: s[k] for k in ("settings hash", "code version", "data version")}


@needs_pilot
@pytest.mark.parametrize("method", METHODS)
def test_the_static_check_passes_on_the_pilot(method):
    folder = PILOT / f"{method}_910"
    out = rp.static_check(folder, pilot_range(folder))
    assert out["same rules and copies"] and out["same knee"]
    assert out["hypervolume difference"] <= rp.TOLERANCE


@needs_pilot
def test_the_static_check_sees_a_changed_population(tmp_path):
    folder = tmp_path / "run"
    shutil.copytree(PILOT / "GP-fixed_910" / "results", folder / "results")
    population = pd.read_csv(folder / "results" / "final_population.csv")
    front = set(pd.read_csv(folder / "results" / "front.csv.gz")["text"])
    i = population.index[~population["text"].isin(front)][0]  # a member that is not on the front
    population.loc[i, "mean R_v"] = population["mean R_v"].max() + 1  # now it dominates
    population.loc[i, "G_hat"] = population["G_hat"].min() - 0.01
    population.to_csv(folder / "results" / "final_population.csv", index=False)
    out = rp.static_check(folder, pilot_range(PILOT / "GP-fixed_910"))
    assert not (out["same rules and copies"] and out["same knee"] and out["hypervolume difference"] <= rp.TOLERANCE)


@needs_pilot
def test_the_input_check_passes_on_the_pilot_and_sees_bad_tables(tmp_path):
    source = PILOT / "GP-adaptive_910"
    expected = pilot_identity(source)
    assert rp.check_inputs(source, "GP-adaptive", 910, 200, expected) == []
    assert rp.check_inputs(source, "GP-prop", 910, 200, expected)  # another method's tables
    folder = tmp_path / "run"
    shutil.copytree(source / "results", folder / "results")
    front = pd.read_csv(folder / "results" / "front.csv.gz")
    front["knee"] = True  # every rule a knee
    front.to_csv(folder / "results" / "front.csv.gz", index=False)
    population = pd.read_csv(folder / "results" / "final_population.csv").iloc[:-1]  # 199 rows
    population.to_csv(folder / "results" / "final_population.csv", index=False)
    problems = rp.check_inputs(folder, "GP-adaptive", 910, 200, expected)
    assert any("exactly one knee" in p for p in problems) and any("199 rows" in p for p in problems)


def made_up_run(tmp_path, texts, knees, dates):
    means = validation_means([from_text(t) for t in texts], dates, 1, rp.days_for(SETTINGS), {})
    area, _ = hypervolume(normalise(means, rp.value_range(SETTINGS)))
    results = tmp_path / "run" / "results"
    results.mkdir(parents=True)
    table = pd.DataFrame({"text": texts, "mean R_v": [m[0] for m in means], "G_hat": [m[1] for m in means],
                          "knee": knees})
    table.to_csv(results / "front.csv.gz", index=False)
    (results / "summary.json").write_text(json.dumps({"hypervolume": area}))
    return table, results


@needs_pilot
def test_reevaluation_goes_through_the_new_path_and_sees_a_difference(tmp_path, monkeypatch):
    front = pd.read_csv(PILOT / "GP-adaptive_910" / "results" / "front.csv.gz")
    texts = [front.loc[front["knee"], "text"].iloc[0], front.loc[~front["knee"], "text"].iloc[0]]
    dates = validation_dates()[:1]
    table, results = made_up_run(tmp_path, texts, [True, False], dates)
    out = rp.reevaluate(tmp_path / "run", SETTINGS, dates=dates, k=1)
    assert out["rules"] == 2 and out["largest mean difference"] == 0 and out["hypervolume difference"] == 0
    assert out["tidy always agrees"] and out["knee decisions"] > 0 and 0 <= out["display agreement"] <= 1

    table.loc[1, "mean R_v"] += 1e-9  # a saved mean that is off
    table.to_csv(results / "front.csv.gz", index=False)
    assert rp.reevaluate(tmp_path / "run", SETTINGS, dates=dates, k=1)["largest mean difference"] > rp.TOLERANCE

    table.loc[1, "mean R_v"] -= 1e-9
    table.to_csv(results / "front.csv.gz", index=False)
    real = evaluate6.score

    def off(*args, **kwargs):  # the new evaluator returning a wrong main objective
        out = real(*args, **kwargs)
        out["objectives"][rp.MEAN] += 1e-6
        return out
    monkeypatch.setattr(evaluate6, "score", off)
    assert rp.reevaluate(tmp_path / "run", SETTINGS, dates=dates, k=1)["largest mean difference"] > rp.TOLERANCE


@needs_pilot
def test_a_tidy_form_that_differs_fails_the_check(tmp_path, monkeypatch):
    import forms
    front = pd.read_csv(PILOT / "GP-adaptive_910" / "results" / "front.csv.gz")
    texts = [front.loc[front["knee"], "text"].iloc[0]]
    dates = validation_dates()[:1]
    made_up_run(tmp_path, texts, [True], dates)
    least = from_text("GP-adaptive | h1: sub(TA, AV) | h2: GDAY")  # another rule as the "tidy" form
    monkeypatch.setattr(forms, "tidy", lambda rule: least)
    out = rp.reevaluate(tmp_path / "run", SETTINGS, dates=dates, k=1)
    assert not out["tidy always agrees"] and out["largest mean difference"] == 0  # the full rule is untouched
    assert not rp.passed([], {"same rules and copies": True, "same knee": True, "hypervolume difference": 0}, out)


def mark(folder, **fields):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "complete.json").write_text(json.dumps({**step5.STEP5, **fields}))


def test_check_complete_needs_every_run_with_the_step5_identity(tmp_path):
    for method in METHODS:
        for run in range(2):
            mark(step5.run_folder(method, run, tmp_path), method=method, run=run)
    mark(tmp_path / "tuned", rule="tuned weighted")
    folders = step5.check_complete(tmp_path, runs=range(2))
    assert len(folders) == 6
    mark(step5.run_folder("GP-prop", 1, tmp_path), method="GP-prop", run=1, **{"data content": "something else"})
    with pytest.raises(SystemExit, match="GP-prop_01: another identity"):
        step5.check_complete(tmp_path, runs=range(2))
    shutil.rmtree(step5.run_folder("GP-prop", 1, tmp_path))
    with pytest.raises(SystemExit, match="GP-prop_01: not complete"):
        step5.check_complete(tmp_path, runs=range(2))


def test_the_analysis_version_names_the_analysis_code():
    assert len(step5.analysis_version()) == 10
    versions = step5.package_versions()
    assert versions["numpy"] and versions["pandas"]
