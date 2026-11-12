"""Tests of analysis/evaluate6.py, on validation days only (no test day is read)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import evaluate6 as e6  # noqa: E402

from evaluate import Days, evaluate, grouped, share_at_least, visitor_rows  # noqa: E402
from gp_rule import from_text  # noqa: E402
from rules import RULES, Tuned  # noqa: E402
from search import validation_dates  # noqa: E402

PILOT_FRONT = ROOT / "outputs" / "search" / "pilot" / "full" / "GP-adaptive_910" / "results" / "front.csv.gz"
DAYS = Days()
DATE = validation_dates()[3]


def gp_rule():
    if not PILOT_FRONT.exists():
        pytest.skip("no pilot outputs")
    return from_text(pd.read_csv(PILOT_FRONT)["text"].iloc[0])


@pytest.fixture(scope="module")
def knee_out():
    return e6.score(gp_rule(), DATE, 1, DAYS, knee=True)


def test_the_main_alpha_is_the_search_scoring(knee_out):
    plain = evaluate(gp_rule(), DATE, 1, DAYS)
    assert knee_out["objectives"]["mean R_v (alpha 0.25)"] == plain.mean_r
    assert knee_out["objectives"]["G_hat"] == plain.g_hat
    day = DAYS.get(DATE)[0]
    main = grouped(visitor_rows(day, plain))
    assert knee_out["groups"][main.columns].equals(main)


def test_every_alpha_follows_the_formula(knee_out):
    v = knee_out["visitors"]
    for alpha in e6.ALPHAS:
        expected = 1 + (v["change A"] - alpha * v["change C"] - 0.5 * v["change T"]) / v["A observed"]
        assert np.allclose(v[f"R_v (alpha {e6.label(alpha)})"], expected, rtol=0, atol=1e-12)
        assert knee_out["objectives"][f"mean R_v (alpha {e6.label(alpha)})"] == pytest.approx(
            v[f"R_v (alpha {e6.label(alpha)})"].mean(), abs=1e-12)
    assert (v["A"] - v["A observed"]).equals(v["change A"]) or np.allclose(v["A"] - v["A observed"], v["change A"])


def test_less_weight_on_crowding_moves_r_v_one_way_only(knee_out):
    v = knee_out["visitors"]
    higher = v["change C"] > 0  # more crowded than the observed route: a larger alpha lowers R_v
    assert (v.loc[higher, "R_v (alpha 1)"] <= v.loc[higher, "R_v (alpha 0)"]).all()
    assert (v.loc[~higher, "R_v (alpha 1)"] >= v.loc[~higher, "R_v (alpha 0)"]).all()


def test_the_threshold_counts_of_each_alpha_add_up(knee_out):
    groups, v = knee_out["groups"], knee_out["visitors"]
    for alpha in (0, 0.5, 1):
        a = e6.label(alpha)
        assert groups[f"sum R_v (alpha {a})"].sum() == pytest.approx(v[f"R_v (alpha {a})"].sum(), abs=1e-9)
        for theta in e6.THETAS:
            assert groups[f"R_v >= {theta:g} (alpha {a})"].sum() == (v[f"R_v (alpha {a})"] >= theta).sum()
    for theta in e6.THETAS:  # at the main alpha the histogram gives the same counts
        assert share_at_least(groups, theta) * groups["visitors"].sum() == pytest.approx(
            (v["R_v (alpha 0.25)"] >= theta).sum())


def test_a_knee_keeps_its_details_and_a_front_rule_does_not(knee_out):
    assert {"visitors", "hourly", "decisions"} <= set(knee_out)
    assert len(knee_out["hourly"]) == 15 and len(knee_out["decisions"]) > 0
    plain = e6.score(gp_rule(), DATE, 1, DAYS, knee=False)
    assert set(plain) == {"objectives", "groups", "peaks"}
    assert plain["objectives"] == knee_out["objectives"]  # the log changes nothing


@pytest.mark.parametrize("policy", [RULES["POP"](), RULES["LEAST"](), Tuned(0.1)])
def test_hand_and_tuned_rules_go_through_evaluate(policy):
    out = e6.score(policy, DATE, 0, DAYS, knee=True)
    assert out["objectives"]["mean R_v (alpha 0.25)"] == evaluate(policy, DATE, 0, DAYS).mean_r
    assert "decisions" not in out


def test_a_day_with_no_evaluated_visitor_is_refused_before_it_is_simulated():
    day = SimpleNamespace(visitors=pd.DataFrame({"evaluated": [False, False]}))
    stub = SimpleNamespace(get=lambda date: (day, None))
    with pytest.raises(ValueError, match="no evaluated visitor"):
        e6.score(gp_rule(), "2022-03-07", 0, stub)


def test_the_decision_log_holds_card_ids_and_site_ids(knee_out):
    day = DAYS.get(DATE)[0]
    log, visitors = knee_out["decisions"], knee_out["visitors"]
    assert {"decision", "visitor", "j0 site", "jw site"} <= set(log.columns) and not {"j0", "jw"} & set(log.columns)
    assert list(log["decision"]) == list(range(len(log)))
    assert set(log["visitor"]) <= set(visitors["visitor"])  # joins the per-visitor rows by card id
    assert set(log["j0 site"]) | set(log["jw site"]) <= set(day.site_ids)
    plain = evaluate(gp_rule(), DATE, 1, DAYS, log=True)
    raw = pd.DataFrame(plain.run.decisions)
    ids = day.visitors["visitor"].to_numpy()
    assert (log["visitor"].to_numpy() == ids[raw["visitor"].to_numpy()]).all()
    assert (log["jw site"].to_numpy() == day.site_ids[raw["jw"].to_numpy()]).all()
    assert len(log) == len(raw)


def test_watched_forms_add_one_column_each_and_change_nothing():
    import forms
    rule = gp_rule()
    out = e6.score(rule, DATE, 1, DAYS, knee=True,
                   watch={"tidy": forms.tidy(rule), "display": forms.display(rule)[0]})
    assert out["decisions"]["tidy agrees"].all()
    assert out["decisions"]["display agrees"].isin([True, False]).all()
    assert out["objectives"] == e6.score(rule, DATE, 1, DAYS, knee=True)["objectives"]
