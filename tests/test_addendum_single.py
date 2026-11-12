"""Tests of addendum/single.py (addendum to the Step 6 plan, section 3): a small GP-single batch and a small main
batch of pilot rules on two validation days (no test day, no formal run), and hand-made cases."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "addendum"))
import forms  # noqa: E402
import results as rs  # noqa: E402
import run_test as rt  # noqa: E402
import single  # noqa: E402
import stats  # noqa: E402

from evaluate import Days, evaluate  # noqa: E402
from gp_rule import from_text, to_text  # noqa: E402
from rules import TUNED_WEIGHTS  # noqa: E402
from search import load_settings  # noqa: E402

PILOT = ROOT / "outputs" / "search" / "pilot"
FULL, SHORT = PILOT / "full", PILOT / "short"
DAYS = ["2019-01-28", "2019-01-29"]  # validation days
RUN = 910  # the pilot run number used for both batches, so that they pair
SETTINGS = load_settings(rt.SETTINGS_FILE)
have_pilot = (FULL / "GP-fixed_910" / "results").exists() and (SHORT / "GP-single_900" / "results").exists()
needs_pilot = pytest.mark.skipif(not have_pilot, reason="no pilot outputs")


def two_days(_data):
    return list(DAYS)


def with_days(fn):
    original = rt.dates_of
    rt.dates_of = rs.dates_of = two_days
    try:
        return fn()
    finally:
        rt.dates_of = rs.dates_of = original


def write_settings(folder):
    data = {**rt.load_test_settings(), "split": "validation", "replicates": 2, "dates per block": 1}
    (folder / "test.json").write_text(json.dumps(data))
    return folder / "test.json"


def single_rules(folder):
    """Pilot GP-single rules (the short pilot's run 900, relabelled as run 910 in a folder of its own)."""
    root = folder / "single"
    root.mkdir()
    (root / f"GP-single_{RUN}").symlink_to(SHORT / "GP-single_900")
    front = pd.read_csv(SHORT / "GP-single_900" / "results" / "front.csv.gz")
    keep = pd.concat([front[front["knee"]], front[~front["knee"]].iloc[:2]])
    rows = [{"rule id": f"GP-single_{RUN}_{n:03d}", "group": f"GP-single_{RUN}", "kind": "gp", "method": "GP-single",
             "run": RUN, "text": t, "knee": bool(k), "nodes": int(s)}
            for n, (t, k, s) in enumerate(zip(keep["text"], keep["knee"], keep["nodes"]))]
    pd.DataFrame(rows).to_csv(folder / "single_rules.csv", index=False)
    return folder / "single_rules.csv", root


def main_rules(folder):
    rows = []
    for method in single.MAIN_METHODS:
        front = pd.read_csv(FULL / f"{method}_910" / "results" / "front.csv.gz")
        texts = [front.loc[front["knee"], "text"].iloc[0]] + list(front.loc[~front["knee"], "text"].iloc[:2])
        group = f"{method}_910"
        for n, text in enumerate(texts):
            rows.append({"rule id": f"{group}_{n:03d}", "group": group, "kind": "gp", "method": method, "run": RUN,
                         "text": text, "knee": n == 0})
    for n, w in enumerate(TUNED_WEIGHTS[:3]):
        rows.append({"rule id": f"tuned_{n:02d}", "group": "baselines", "kind": "tuned", "method": "tuned", "run": -1,
                     "text": repr(w), "knee": n == 1, "validation front": True})
    for name in ["POP", "LEAST", "SOT", "SOS", "observed order"]:
        rows.append({"rule id": name, "group": "baselines", "kind": "control" if name == "observed order" else "hand",
                     "method": name, "run": -1, "text": name, "knee": True})
    pd.DataFrame(rows).to_csv(folder / "main_rules.csv", index=False)
    return folder / "main_rules.csv"


@pytest.fixture(scope="module")
def batches(tmp_path_factory):
    if not have_pilot:
        pytest.skip("no pilot outputs")
    tmp = tmp_path_factory.mktemp("batches")
    test_path = write_settings(tmp)
    s_rules, s_root = single_rules(tmp)
    m_rules = main_rules(tmp)

    def make():
        single.run_main(2, tmp / "single_out", s_rules, test_path, formal=False)
        rt.main(["--workers", "2", "--not-formal", "--rules", str(m_rules), "--test-settings", str(test_path),
                 "--out", str(tmp / "main_out")])
        s = single.load_single(tmp / "single_out", s_rules, test_path, formal=False)
        m = rs.load(tmp / "main_out", m_rules, test_path, formal=False)
        return s, m
    s, m = with_days(make)
    return {"single": s, "main": m, "tmp": tmp, "test": test_path, "single rules": s_rules, "single root": s_root,
            "main rules": m_rules}


def hashes(folder):
    """The content of every block file (decompressed: a gzip header carries the time it was written)."""
    import gzip
    return {str(p.relative_to(folder)): hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()
            for p in sorted(Path(folder).glob("blocks/*/*/*.csv.gz"))}


# ---------------------------------------------------------------------- #
# The batch, end to end
# ---------------------------------------------------------------------- #

@needs_pilot
def test_the_analysis_version_is_unchanged():
    assert single.analysis_version() == "1d12673c64"  # the frozen step6-formal analysis/


@needs_pilot
def test_the_gp_single_means_are_the_evaluations(batches):
    s = batches["single"]
    rule_id, text = s.rules["rule id"].iloc[0], s.rules["text"].iloc[0]
    direct = [evaluate(from_text(text), d, r, Days()) for d in DAYS for r in range(2)]
    means = s.rule_means()
    assert means.loc[rule_id, "mean R_v (alpha 0.25)"] == pytest.approx(np.mean([e.mean_r for e in direct]), abs=1e-15)


@needs_pilot
def test_a_knee_block_passes_the_frozen_check_with_a_trivial_log(batches):
    s = batches["single"]
    decisions = s.table("decisions")
    assert len(decisions) > 0
    assert (decisions["w"] == 0).all() and not decisions["changed"].any()
    for folder in s.blocks[f"GP-single_{RUN}"]:  # the blocks were checked by the frozen check_block when written
        assert (folder / "done.json").exists() and (folder / "decisions.csv.gz").exists()


@needs_pilot
def test_a_resumed_batch_gives_the_same_outputs(batches, tmp_path):
    first = batches["tmp"] / "single_out"
    copy = tmp_path / "single_out"
    shutil.copytree(first, copy)
    block = sorted(copy.glob("blocks/*/*"))[0]
    shutil.rmtree(block)  # as if stopped before this block was done
    (copy / "launcher.lock").unlink(missing_ok=True)
    with_days(lambda: single.run_main(2, copy, batches["single rules"], batches["test"], formal=False))
    assert hashes(copy) == hashes(first)


@needs_pilot
def test_refused_inputs(batches, tmp_path):
    s_rules, test = batches["single rules"], batches["test"]
    # a missing block
    copy = tmp_path / "missing"
    shutil.copytree(batches["tmp"] / "single_out", copy)
    shutil.rmtree(sorted(copy.glob("blocks/*/*"))[0])
    with pytest.raises(SystemExit):
        with_days(lambda: single.load_single(copy, s_rules, test, formal=False))
    # a damaged completion mark (another identity in done.json) stops a restart before anything runs
    copy = tmp_path / "damaged"
    shutil.copytree(batches["tmp"] / "single_out", copy)
    mark = sorted(copy.glob("blocks/*/*/done.json"))[0]
    done = json.loads(mark.read_text())
    done["addendum version"] = "something else"
    mark.write_text(json.dumps(done))
    with pytest.raises(SystemExit):
        with_days(lambda: single.run_main(1, copy, s_rules, test, formal=False))
    # a file of another size
    copy = tmp_path / "changed"
    shutil.copytree(batches["tmp"] / "single_out", copy)
    f = sorted(copy.glob("blocks/*/*/objectives.csv.gz"))[0]
    f.write_bytes(f.read_bytes()[:-3])
    with pytest.raises(SystemExit):
        with_days(lambda: single.load_single(copy, s_rules, test, formal=False))


@needs_pilot
def test_a_duplicate_key_is_refused_by_the_block_check(batches):
    s = batches["single"]
    group = f"GP-single_{RUN}"
    folder = s.blocks[group][0]
    tables = {p.name.split(".")[0]: rs.read_table(p) for p in folder.glob("*.csv.gz")}
    tables["visitors"] = pd.concat([tables["visitors"], tables["visitors"].iloc[:1]], ignore_index=True)
    done = json.loads((folder / "done.json").read_text())
    with pytest.raises(ValueError):
        rt.check_block(tables, s.rules, done["dates"], 2)


@needs_pilot
def test_a_main_batch_of_another_setting_is_refused(batches):
    import copy as cp
    main = cp.copy(batches["main"])
    main.identity = {**main.identity, "step 6 settings hash": "another"}
    with pytest.raises(SystemExit):
        single.check_batches(batches["single"], main)
    main = cp.copy(batches["main"])
    main.data = {**main.data, "scenario": {"stop limit": False}}
    with pytest.raises(SystemExit):
        single.check_batches(batches["single"], main)


@needs_pilot
def test_the_tables(batches, tmp_path):
    folder = single.tables_main(batches["single"], batches["main"], SETTINGS, tmp_path / "t",
                                single_root=batches["single root"], step5_root=FULL)
    s1 = pd.read_csv(folder / "table_s1.csv")
    assert list(s1["method"])[:4] == list(single.ALL_METHODS)
    s2 = pd.read_csv(folder / "table_s2.csv")
    assert len(s2) == 3 and "p" not in s2.columns and (s2["n"] == 1).all()
    s3 = pd.read_csv(folder / "table_s3.csv")
    assert set(s3["method"]) == set(single.ALL_METHODS)
    assert s3.loc[s3["method"] == "GP-single", "share of knees with VS"].between(0, 1).all()
    assert len(pd.read_csv(folder / "other_ranges.csv")) == 3
    assert len(pd.read_csv(folder / "hand_rules_dominated.csv")) == 5
    sources = json.loads((folder / "sources.json").read_text())
    assert sources["GP-single"]["rule list hash"] != sources["main"]["rule list hash"]
    assert (folder / "figure_s1.png").exists()


def test_table_s2_is_the_frozen_paired_difference():
    fronts = pd.DataFrame({"method": ["GP-single"] * 7 + ["GP-fixed"] * 7 + ["GP-prop"] * 7 + ["GP-adaptive"] * 7,
                           "run": list(range(7)) * 4,
                           "hypervolume": [0.5, 0.6, 0.55, 0.7, 0.4, 0.65, 0.6] + [0.52, 0.58, 0.55, 0.6, 0.45, 0.6, 0.7]
                           + [0.5] * 7 + [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.61]})
    table = single.table_s2(fronts).set_index("pair")
    for other in single.MAIN_METHODS:
        x = dict(zip(range(7), fronts.loc[fronts["method"] == "GP-single", "hypervolume"]))
        y = dict(zip(range(7), fronts.loc[fronts["method"] == other, "hypervolume"]))
        out = stats.compare(x, y)
        row = table.loc[f"GP-single - {other}"]
        assert row["median difference"] == out["median median"] and row["95% low"] == out["median low"]
        assert row["95% high"] == out["median high"] and row["rank-biserial r"] == out["r"] and row["n"] == 7


# ---------------------------------------------------------------------- #
# GP-single trees
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("text", ["GP-single | tree: GDAY",  # scalar terminals only: broadcast
                                  "GP-single | tree: sub(AV, mul(VS, max(GDAY, MCR)))",
                                  "GP-single | tree: div(min(ARR, AV), div(VS, STOPS))"])
def test_gp_single_trees_round_trip_and_choose_alike(text):
    rule = from_text(text)
    again = from_text(to_text(rule))
    days = Days()
    a, b = evaluate(rule, DAYS[0], 0, days), evaluate(again, DAYS[0], 0, days)
    assert to_text(again) == text
    assert a.mean_r == b.mean_r and a.g_hat == b.g_hat


@needs_pilot
def test_schedules_compare_saved_checkpoints(tmp_path):
    (tmp_path / "single").mkdir()
    (tmp_path / "single" / "GP-single_910").symlink_to(FULL / "GP-adaptive_910")  # any saved checkpoint will do
    assert single.schedules_match(tmp_path / "single", FULL, runs=[910]) == []
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "GP-single_910").symlink_to(PILOT / "parallel" / "GP-fixed_920")
    assert single.schedules_match(tmp_path / "other", FULL, runs=[910]) == [910]


# ---------------------------------------------------------------------- #
# The walking weight
# ---------------------------------------------------------------------- #

def toy_rows():
    """Two days with different numbers of visitors; walking time on both routes; one visitor with T_N = T_O."""
    return pd.DataFrame({
        "rule id": "k", "date": ["d1", "d1", "d1", "d2"], "replicate": 0, "visitor": [1, 2, 3, 4],
        "A": [10.0, 8.0, 6.0, 5.0], "C": [2.0, 1.0, 0.0, 4.0], "T": [1.5, 0.5, 1.0, 2.0],
        "A observed": [9.0, 9.0, 5.0, 6.0], "C observed": [4.0, 0.0, 1.0, 2.0], "T observed": [0.5, 1.0, 1.0, 1.0]})


def test_beta_by_hand():
    v = toy_rows()
    for beta in (0.0, 1.0):
        hand = [1 + ((a - ao) - 0.25 * (c - co) - beta * (t - to)) / ao
                for a, c, t, ao, co, to in v[["A", "C", "T", "A observed", "C observed", "T observed"]].to_numpy()]
        assert np.allclose(single.r_at_beta(v, beta), hand, rtol=0, atol=1e-15)


def test_beta_acts_on_both_routes_and_not_where_walking_is_equal():
    v = toy_rows()
    r0, r1 = single.r_at_beta(v, 0.0), single.r_at_beta(v, 1.0)
    assert r0[2] == r1[2]  # T_N = T_O
    slope = -(v["T"] - v["T observed"]) / v["A observed"]
    assert np.allclose(r1 - r0, slope, rtol=0, atol=1e-15)  # dR/dbeta = -(T_N - T_O)/A_O: both routes count


def test_beta_means_weight_days_equally_and_shares_pool_visitors():
    v = toy_rows()
    m = single.knee_beta_measures(v, betas=(0.5,))[0.5]
    r = single.r_at_beta(v, 0.5)
    assert m["mean R_v"] == pytest.approx((r[:3].mean() + r[3]) / 2, abs=1e-15)
    assert m["share R_v < 1"] == pytest.approx((r < 1).mean(), abs=1e-15)


def test_knee_rows_are_checked_before_re_scoring():
    v = toy_rows()
    single.check_knee_rows(v, ["k"], ["d1", "d2"], 1)
    with pytest.raises(ValueError):
        single.check_knee_rows(pd.concat([v, v.iloc[:1]]), ["k"], ["d1", "d2"], 1)
    with pytest.raises(ValueError):
        single.check_knee_rows(v, ["k"], ["d1", "d2", "d3"], 1)
    with pytest.raises(ValueError):
        single.check_knee_rows(v.assign(**{"A observed": [9.0, 0.0, 5.0, 6.0]}), ["k"], ["d1", "d2"], 1)


@needs_pilot
def test_table_s4_reproduces_the_main_outputs_at_beta_half(batches, tmp_path):
    pytest.importorskip("scipy")
    folder = with_days(lambda: single.beta_main(batches["main"], tmp_path / "beta"))
    rows = pd.read_csv(folder / "knees_beta.csv")
    assert set(rows["beta"]) == set(single.BETAS)
    t = pd.read_csv(folder / "table_s4.csv")
    assert "order of methods by median knee mean R_v" in t.columns and "knee G_hat (median)" in t.columns
    assert not t.loc[t["beta"] == 0.5, "order differs from beta 0.5"].any()
    # with GP-single: the main-batch values must be the first version's
    folder = with_days(lambda: single.beta_main(batches["main"], tmp_path / "beta", batches["single"]))
    assert (folder / "table_s4_with_gp_single.csv").exists()
    assert (folder / "table_s4_with_gp_single_sources.json").exists()
