"""Tests of analysis/forms.py (Step 6 plan, section 7), on validation days only."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import forms  # noqa: E402

from evaluate import Days, evaluate  # noqa: E402
from gp_rule import H1_SET, H2_SET, from_text, make_rule, to_text, tree_from_text  # noqa: E402
from search import validation_dates  # noqa: E402

PILOT = ROOT / "outputs" / "search" / "pilot" / "full"
needs_pilot = pytest.mark.skipif(not (PILOT / "GP-adaptive_910" / "results").exists(), reason="no pilot outputs")
DAYS = Days()


def h2(text):
    return tree_from_text(text, H2_SET)


def pilot_rules(method):
    return [from_text(t) for t in pd.read_csv(PILOT / f"{method}_910" / "results" / "front.csv.gz")["text"]]


@pytest.mark.parametrize("text, expected", [
    ("add(mul(2.0, 3.0), GDAY)", "add(6.0, GDAY)"),
    ("div(5.0, sub(2.0, 2.0))", "1.0"),                       # protected division by zero gives 1
    ("max(GDAY, min(0.5, 0.25))", "max(GDAY, 0.25)"),
    ("mul(mul(1e300, 1e300), GDAY)", "mul(mul(1e+300, 1e+300), GDAY)"),  # inf: kept as it is
    ("add(mul(1e300, 1e300), add(1.0, 2.0))", "add(mul(1e+300, 1e+300), 3.0)"),  # inside a kept one
    ("add(GDAY, ARR)", "add(GDAY, ARR)"),
])
def test_only_finite_constant_subtrees_are_folded(text, expected):
    assert str(forms.fold(h2(text), H2_SET)) == expected


def test_a_folded_value_may_leave_the_search_range():
    assert str(forms.fold(h2("mul(1000.0, 1000.0)"), H2_SET)) == "1000000.0"


@needs_pilot
def test_the_tidy_form_computes_the_same_numbers():
    rng = np.random.default_rng(0)
    inputs = [rng.normal(size=200) * 10 for _ in H2_SET.arguments] + []
    inputs[0][:5] = 0.0  # some zero denominators
    for rule in pilot_rules("GP-adaptive"):
        full = forms.GPRule(rule).functions[1]
        tidy = forms.GPRule(forms.tidy(rule)).functions[1]
        with np.errstate(all="ignore"):
            a, b = np.asarray(full(*inputs), float), np.asarray(tidy(*inputs), float)
        assert np.array_equal(a, b, equal_nan=True)
        assert forms.nodes(forms.tidy(rule)) <= forms.nodes(rule)


@needs_pilot
def test_the_shadow_changes_nothing_and_the_tidy_form_always_agrees():
    rule = pilot_rules("GP-adaptive")[0]
    date = validation_dates()[5]
    plain = evaluate(rule, date, 0, DAYS, log=True)
    shadow = forms.Shadow(rule, forms.tidy(rule))
    watched = evaluate(shadow, date, 0, DAYS)
    assert (watched.mean_r, watched.g_hat) == (plain.mean_r, plain.g_hat)
    assert watched.run.fallbacks == plain.run.fallbacks  # the shadow counts nothing
    assert pd.DataFrame(watched.run.decisions).equals(pd.DataFrame(plain.run.decisions))
    assert len(shadow.agrees) == len(plain.run.decisions) and all(shadow.agrees)


@needs_pilot
def test_the_display_form_is_rounded_and_only_watched():
    rule = pilot_rules("GP-adaptive")[0]
    shown, reason = forms.display(rule)
    assert reason is None
    for part in shown[1:]:
        for node in part:
            if forms.is_constant(node):
                assert node.value == float(f"{node.value:.3g}")
    date = validation_dates()[5]
    shadow = forms.Shadow(rule, shown)
    watched = evaluate(shadow, date, 0, DAYS)
    assert watched.mean_r == evaluate(rule, date, 0, DAYS).mean_r
    assert 0 <= np.mean(shadow.agrees) <= 1


def test_c_is_rounded_and_a_constant_that_rounds_to_infinity_has_no_display_form():
    h1 = tree_from_text("add(AV, TA)", H1_SET)
    fixed = make_rule("GP-fixed", [h1, 0.123456])
    assert forms.display(fixed)[0][1] == 0.123
    huge = make_rule("GP-adaptive", [h1, h2("add(GDAY, 1.7976931348623157e308)")])
    shown, reason = forms.display(huge)
    assert shown is None and "rounds to inf" in reason


def test_the_readable_text_names_protected_division():
    rule = make_rule("GP-adaptive", [tree_from_text("div(AV, TA)", H1_SET), h2("div(GDAY, 2.0)")])
    assert forms.readable(rule) == "GP-adaptive | h1: pdiv(AV, TA) | h2: pdiv(GDAY, 2.0)"
    assert "pdiv" not in to_text(rule)  # the rule's own text is unchanged
    assert math.isfinite(forms.nodes(rule))
