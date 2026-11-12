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
    assert sources["main path"] == str(Path(sources["main path"]).resolve()) and Path(sources["main path"]).is_absolute()
    assert (folder / "figure_s1.png").exists()
    with pytest.raises(SystemExit):  # made once
        single.tables_main(batches["single"], batches["main"], SETTINGS, folder, single_root=batches["single root"],
                           step5_root=FULL)


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
    import pickle
    # the pilot's own methods with run 900 share a schedule
    assert single.schedules_match(SHORT, SHORT, runs=[900]) == []
    # a checkpoint of another run, or not at its last generation, is refused
    (tmp_path / "wrong").mkdir()
    (tmp_path / "wrong" / "GP-single_900").symlink_to(PILOT / "parallel" / "GP-fixed_920")
    with pytest.raises(SystemExit):
        single.schedules_match(tmp_path / "wrong", SHORT, runs=[900])
    # a finished run of the right number with another schedule is reported
    state = pickle.loads((SHORT / "GP-single_900" / "checkpoint" / "checkpoint.pkl").read_bytes())
    state["schedule"] = list(reversed(state["schedule"]))
    folder = tmp_path / "other" / "GP-single_900" / "checkpoint"
    folder.mkdir(parents=True)
    (folder / "checkpoint.pkl").write_bytes(pickle.dumps(state))
    assert single.schedules_match(tmp_path / "other", SHORT, runs=[900]) == [900]


# ---------------------------------------------------------------------- #
# The formal entry: rule list, reproduction, pass record, identity
# ---------------------------------------------------------------------- #

PILOT_IDENTITY = {"settings hash": "d1e47312ea", "code version": "b9ea72de0e", "data version": "a6a88b365c11"}


def pilot_settings():
    out = dict(SETTINGS)
    out.update({"population": 40, "generations": 20, "monitor every": 5, "validation replicates": 1,
                "R range": (0.0, 1.5), "G range": (0.0, 1.0)})
    return out


@needs_pilot
def test_the_rule_list_is_the_saved_front():
    table = single.make_rule_list(SHORT, runs=[900])
    front = pd.read_csv(SHORT / "GP-single_900" / "results" / "front.csv.gz")
    assert len(table) == len(front)
    assert list(table["text"]) == list(front["text"]) and list(table["knee"]) == list(front["knee"])
    assert list(table["nodes"]) == list(front["nodes"]) and table["rule id"].is_unique
    two = table.copy()
    two.loc[0, "knee"] = True  # a second knee
    with pytest.raises(SystemExit):
        single.check_rule_list(two, runs=[900])
    with pytest.raises(SystemExit):  # a run missing
        single.check_rule_list(table, runs=[900, 901])


@pytest.fixture(scope="module")
def reproduced(tmp_path_factory):
    """The reproduction check run for real on the pilot GP-single run (its own settings, identity and 35
    validation days); only the Step 5 code check and the completion marks are replaced, as a pilot has none."""
    if not have_pilot:
        pytest.skip("no pilot outputs")
    base = tmp_path_factory.mktemp("reproduced")
    root = base / "root"  # a root as a formal batch has it: the pilot run's outputs, with its marks
    run = root / "GP-single_900"
    run.mkdir(parents=True)
    for name in ("results", "checkpoint"):
        (run / name).symlink_to(SHORT / "GP-single_900" / name)
    (run / "complete.json").write_text(json.dumps({**PILOT_IDENTITY, "method": "GP-single", "run": 900}))
    (root / "task.json").write_text(json.dumps(PILOT_IDENTITY))
    out = base / "out"
    keep = single.check_here, single.check_single
    single.check_here = lambda: {"stand-in": "the Step 5 identity"}
    single.check_single = lambda *a, **k: None
    try:
        single.reproduce_main(1, root=root, runs=[900], out=out, settings=pilot_settings(), expected=PILOT_IDENTITY,
                              main_root=SHORT)
    finally:
        single.check_here, single.check_single = keep
    return out


def formal_identity(out, settings, root, **change):
    keep = single.check_here, single.check_single, single.addendum_version
    single.check_here = lambda: {"stand-in": "the Step 5 identity"}
    single.check_single = lambda *a, **k: None
    if "addendum version" in change:
        single.addendum_version = lambda: change["addendum version"]
    try:
        rules = rs.read_table(out / "rules.csv")
        return single.identity_of(settings, rt.TEST_SETTINGS, out / "rules.csv", rules, formal=True, root=root,
                                  out=out, runs=[900], expected={"settings hash": single.settings_hash(settings)})
    finally:
        single.check_here, single.check_single, single.addendum_version = keep


@needs_pilot
def test_the_reproduction_writes_a_matching_pass_record(reproduced):
    record = json.loads((reproduced / "reproduced.json").read_text())
    assert record["runs reproduced"] == 1 and record["largest difference"] <= 1e-12
    assert record["rule list hash"] == rt.rule_list_hash(reproduced / "rules.csv") and record["schedules hash"]
    identity = formal_identity(reproduced, pilot_settings(), reproduced.parent / "root")
    assert identity["gp-single search"] == record["gp-single search"]
    with pytest.raises(SystemExit):  # the check is run once
        single.reproduce_main(1, root=reproduced.parent / "root", runs=[900], out=reproduced, settings=pilot_settings())


@needs_pilot
def test_a_start_is_refused_when_the_pass_record_does_not_match(reproduced, tmp_path):
    """Each case changes one thing in a copy that is first shown to pass unchanged, with the same search root as
    the pass record, and must fail for that thing."""
    root, settings = reproduced.parent / "root", pilot_settings()

    def copy():
        c = tmp_path / f"c{len(list(tmp_path.iterdir()))}"
        shutil.copytree(reproduced, c)
        formal_identity(c, settings, root)  # unchanged, it passes
        return c
    c = copy()  # the rule list changed
    t = rs.read_table(c / "rules.csv")
    t.loc[0, "nodes"] = t.loc[0, "nodes"] + 1
    t.to_csv(c / "rules.csv", index=False)
    with pytest.raises(SystemExit, match=r"\['rule list hash'\]"):
        formal_identity(c, settings, root)
    c = copy()  # another search batch in the record
    record = json.loads((c / "reproduced.json").read_text())
    record["gp-single search"]["results hash"] = "000000000000"
    (c / "reproduced.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match=r"\['gp-single search'\]"):
        formal_identity(c, settings, root)
    c = copy()  # another addendum version
    with pytest.raises(SystemExit, match=r"\['addendum version'\]"):
        formal_identity(c, settings, root, **{"addendum version": "another"})
    c = copy()  # no pass record
    (c / "reproduced.json").unlink()
    with pytest.raises(SystemExit, match="has not passed"):
        formal_identity(c, settings, root)


# ---------------------------------------------------------------------- #
# A real interruption
# ---------------------------------------------------------------------- #

RUNNER = """
import sys
sys.path.insert(0, {analysis!r}); sys.path.insert(0, {addendum!r})
import run_test as rt, results as rs, single
rt.dates_of = rs.dates_of = lambda d: {days!r}
if __name__ == "__main__":
    single.run_main(2, {out!r}, {rules!r}, {test!r}, formal=False)
"""


@needs_pilot
def test_a_real_interruption_stops_cleanly_and_resumes(tmp_path):
    import os
    import signal
    import subprocess
    import time
    days = ["2019-01-28", "2019-01-29", "2019-01-30", "2019-01-31"]
    data = {**rt.load_test_settings(), "split": "validation", "replicates": 3, "dates per block": 1}
    (tmp_path / "test.json").write_text(json.dumps(data))
    front = pd.read_csv(SHORT / "GP-single_900" / "results" / "front.csv.gz")
    rows = [{"rule id": f"GP-single_{RUN}_{n:03d}", "group": f"GP-single_{RUN}", "kind": "gp", "method": "GP-single",
             "run": RUN, "text": t, "knee": bool(k), "nodes": int(s)}
            for n, (t, k, s) in enumerate(zip(front["text"], front["knee"], front["nodes"]))][:8]
    pd.DataFrame(rows).to_csv(tmp_path / "rules.csv", index=False)

    def script(out):
        path = tmp_path / f"run_{Path(out).name}.py"
        path.write_text(RUNNER.format(analysis=str(ROOT / "analysis"), addendum=str(ROOT / "addendum"), days=days,
                                      out=str(out), rules=str(tmp_path / "rules.csv"), test=str(tmp_path / "test.json")))
        return [sys.executable, str(path)]
    whole, cut = tmp_path / "whole", tmp_path / "cut"
    assert subprocess.run(script(whole)).returncode == 0
    proc = subprocess.Popen(script(cut), start_new_session=True)
    deadline = time.time() + 120
    while not list(cut.glob("blocks/*/*.partial")) and time.time() < deadline:
        time.sleep(0.05)
    assert list(cut.glob("blocks/*/*.partial")), "no block was being written"
    time.sleep(0.3)
    os.killpg(proc.pid, signal.SIGINT)  # as one Ctrl-C in the terminal reaches the launcher and its workers
    assert proc.wait(timeout=60) != 0
    for _ in range(100):  # every process of the launcher's group has exited
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("a worker is still running after the stop")
    assert not (cut / "launcher.lock").exists()
    assert len(list(cut.glob("blocks/*/*/done.json"))) < len(days)  # it really stopped part way
    assert subprocess.run(script(cut)).returncode == 0  # the same command resumes
    assert hashes(cut) == hashes(whole)
    original = rt.dates_of
    rt.dates_of = rs.dates_of = lambda d: list(days)
    try:
        single.load_single(cut, tmp_path / "rules.csv", tmp_path / "test.json", formal=False)  # verified
    finally:
        rt.dates_of = rs.dates_of = original


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


def test_a_saved_r_v_that_differs_is_refused_even_if_the_summaries_agree():
    v = toy_rows()
    v["R_v (alpha 0.25)"] = single.r_at_beta(v, 0.5)
    single.check_saved_r(v)
    changed = v.copy()
    changed.loc[0, "R_v (alpha 0.25)"] += 1e-9  # one up, one down: the mean and the share below 1 do not move
    changed.loc[1, "R_v (alpha 0.25)"] -= 1e-9
    assert changed["R_v (alpha 0.25)"].mean() == pytest.approx(v["R_v (alpha 0.25)"].mean(), abs=1e-15)
    assert ((changed["R_v (alpha 0.25)"] < 1) == (v["R_v (alpha 0.25)"] < 1)).all()
    with pytest.raises(AssertionError):
        single.check_saved_r(changed)


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
    sources = json.loads((folder / "table_s4_with_gp_single_sources.json").read_text())
    assert sources["dates"] == DAYS and Path(sources["main path"]).is_absolute() and sources["knees"]
    with pytest.raises(SystemExit):  # each version is made once
        with_days(lambda: single.beta_main(batches["main"], tmp_path / "beta"))
