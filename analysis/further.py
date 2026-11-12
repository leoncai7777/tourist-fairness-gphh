"""
The further analyses (Step 6 plan, section 12)
==============================================

For one scenario's outputs beside the main test outputs: the knees' mean R_v
(main alpha), G_hat and completion, as the median and quartiles over the knees
of each method, and the values of the tuned knee, the hand rules and the
control, each beside the same number under the main setting. For a compliance
scenario, followers and non-followers also separately. For the h2 diagnostic,
each rule against its GP-adaptive knee. No tests.

  python3 analysis/further.py --scenario configs/further_1_no_stop_limit.json --out outputs/test/further/1_no_stop_limit
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results as rs  # noqa: E402
from rq1 import METHODS  # noqa: E402
from run_test import OUT, RULE_LIST, TEST_SETTINGS  # noqa: E402

MEASURES = ["mean R_v", "G_hat", "completion"]


def knee_values(results):
    """mean R_v (main alpha), G_hat and completion of every rule of the outputs, with its method and kind."""
    means, done = results.rule_means(), results.completion()
    t = results.rules[["rule id", "method", "kind", "knee"]].set_index("rule id")
    t["mean R_v"] = means[rs.r_column()]
    t["G_hat"] = means["G_hat"]
    t["completion"] = done["completion"]
    if "parent" in results.rules:
        t["parent"] = results.rules.set_index("rule id")["parent"]
    return t


def followers_split(results):
    """Mean R_v (main alpha) of followers, of non-followers and of all evaluated visitors, per rule, each pooled
    over days and replicates the same way (visitors together), so that the overall one lies between the two.
    (The mean R_v of the summary is the other kind: the daily means averaged.)"""
    s = results.sums("groups", ["rule id", "follows"], ["sum R_v", "visitors"])
    out = (s["sum R_v"] / s["visitors"]).unstack("follows").rename(columns={True: "followers", False: "non-followers"})
    pooled = s.groupby(level="rule id").sum()
    out["all evaluated visitors (pooled)"] = pooled["sum R_v"] / pooled["visitors"]
    out["followers' share"] = s["visitors"].unstack("follows")[True] / pooled["visitors"]
    return out


def main_rule(scenario, main):
    """For each rule of the scenario, the rule it is compared with under the main setting: itself, or for an
    h2-diagnostic rule its GP-adaptive knee. A rule without its match in the main outputs is an error."""
    match = pd.Series(scenario.index, index=scenario.index)
    if "parent" in scenario:
        built = scenario["kind"] == "diagnostic"
        match[built] = scenario.loc[built, "parent"]
    missing = sorted(set(match) - set(main.index))
    if missing:
        raise SystemExit(f"rules without their match in the main outputs: {missing[:5]}")
    return match


def summary(scenario, main):
    """Per method (GP knees, or the h2-diagnostic rules) the median and quartiles; per baseline its value, the
    tuned rule's validation knee marked; each beside the same number under the main setting."""
    match = main_rule(scenario, main)
    rows = []
    gp = scenario[scenario["kind"].isin(["gp", "diagnostic"])]
    for method in METHODS:
        ids = gp.index[gp["method"] == method]
        if len(ids) == 0:
            continue
        row = {"rules": method, "knees": len(ids)}
        for m in MEASURES:
            q, base = rs.quartiles(scenario.loc[ids, m]), rs.quartiles(main.loc[match[ids], m])
            row.update({f"{m} median": q["median"], f"{m} q1": q["q1"], f"{m} q3": q["q3"], f"{m} n": q["n"],
                        f"{m} main median": base["median"]})
        rows.append(row)
    for rule_id, r in scenario[~scenario["kind"].isin(["gp", "diagnostic"])].iterrows():
        label = f"{rule_id} (validation knee)" if r["kind"] == "tuned" and bool(r["knee"]) else rule_id
        row = {"rules": label, "knees": 1}
        for m in MEASURES:
            row.update({f"{m} median": r[m], f"{m} main median": main.loc[match[rule_id], m]})
        rows.append(row)
    return pd.DataFrame(rows)


def diagnostic(scenario, main):
    d = scenario[scenario["kind"] == "diagnostic"]
    rows = [{"rule id": i, "parent": r["parent"], **{f"{m} (h2 constant) - knee": r[m] - main.loc[r["parent"], m]
                                                     for m in MEASURES}} for i, r in d.iterrows()]
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True, help="the scenario's settings file")
    parser.add_argument("--out", required=True, help="the scenario's outputs")
    parser.add_argument("--main-out", default=str(OUT))
    parser.add_argument("--main-settings", default=str(TEST_SETTINGS))
    parser.add_argument("--rules", default=str(RULE_LIST))
    parser.add_argument("--not-formal", action="store_true")
    args = parser.parse_args(argv)
    formal = not args.not_formal
    results = rs.load(args.out, args.rules, args.scenario, formal=formal)
    scenario = knee_values(results)
    main_values = knee_values(rs.load(args.main_out, args.rules, args.main_settings, formal=formal))
    folder = Path(args.out) / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    summary(scenario, main_values).to_csv(folder / "summary.csv", index=False)
    if "compliance" in results.data.get("scenario", {}):
        followers_split(results).to_csv(folder / "followers.csv")
    if (scenario["kind"] == "diagnostic").any():
        diagnostic(scenario, main_values).to_csv(folder / "h2_diagnostic.csv", index=False)
    print(folder)


if __name__ == "__main__":
    main()
