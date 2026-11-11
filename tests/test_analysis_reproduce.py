"""Tests of analysis/reproduce.py and analysis/step5.py on pilot outputs and made-up folders (no formal result)."""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import reproduce as rp  # noqa: E402
import step5  # noqa: E402

from search import complete, hypervolume, normalise, validation_dates, validation_means  # noqa: E402
from gp_rule import from_text  # noqa: E402

PILOT = ROOT / "outputs" / "search" / "pilot" / "full"
METHODS = ("GP-fixed", "GP-prop", "GP-adaptive")
needs_pilot = pytest.mark.skipif(not (PILOT / "GP-fixed_910" / "results").exists(), reason="no pilot outputs")


def pilot_range(folder):
    s = json.loads((folder / "results" / "summary.json").read_text())
    return {"R": tuple(s["range"]["R"]), "G": tuple(s["range"]["G"])}


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
def test_reevaluation_matches_and_sees_a_difference(tmp_path):
    settings = complete({"population": 200, "generations": 100, "crossover": 0.8, "mutation": 0.15,
                         "monitor every": 10, "validation replicates": 5})
    texts = list(pd.read_csv(PILOT / "GP-adaptive_910" / "results" / "front.csv.gz")["text"])[:2]
    dates = validation_dates()[:1]
    means = validation_means([from_text(t) for t in texts], dates, 1, rp.days_for(settings), rp.scoring(settings))
    area, _ = hypervolume(normalise(means, rp.value_range(settings)))
    results = tmp_path / "run" / "results"
    results.mkdir(parents=True)
    table = pd.DataFrame({"text": texts, "mean R_v": [m[0] for m in means], "G_hat": [m[1] for m in means]})
    table.to_csv(results / "front.csv.gz", index=False)
    (results / "summary.json").write_text(json.dumps({"hypervolume": area}))
    out = rp.reevaluate(tmp_path / "run", settings, dates=dates, k=1)
    assert out["rules"] == 2 and out["largest mean difference"] == 0 and out["hypervolume difference"] == 0
    table.loc[0, "mean R_v"] += 1e-9
    table.to_csv(results / "front.csv.gz", index=False)
    again = rp.reevaluate(tmp_path / "run", settings, dates=dates, k=1)
    assert again["largest mean difference"] > rp.TOLERANCE


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
