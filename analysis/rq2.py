"""
RQ2: how the fairness term changes choices (Step 6 plan, section 7)
===================================================================

From the decision logs of the GP knees on the test days:

  - per knee: the share of decisions where the fairness term changes the choice,
    over all decisions and over decisions with two or more candidates; among the
    changed ones, the mean change in VS and the median h1 priority given up
    (read within the rule only); by hour of day, the changed share and, for
    GP-adaptive, the median logged w;
  - GP-adaptive: the Spearman correlation of w with GDAY and with ARR per knee
    (not defined when w or the state is constant: counted, left out);
  - terminal frequencies in the h2 trees of GP-adaptive's front rules and knees,
    with the number of rules and their nodes;
  - reading a knee: the display form's agreement on the validation days (from the
    reproduction pass record) and the test days, display minus full in mean R_v
    and G_hat, the tidy form's nodes; the example of each method is the knee of
    the 15th run by (validation hypervolume, run number).

Measures are summarised per method by the median and quartiles over its knees,
with the number for which each is defined. Descriptions, not tests.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results as rs  # noqa: E402
from rq1 import COLOURS, INK, METHODS, MARKERS, check_step5_source, style, validation_hypervolumes  # noqa: E402
from run_test import OUT, PASS_RECORD, RULE_LIST, SETTINGS_FILE, TEST_SETTINGS  # noqa: E402
from step5 import FORMAL  # noqa: E402

from deap import gp  # noqa: E402
from gp_rule import from_text  # noqa: E402
from search import load_settings  # noqa: E402

EXAMPLE_POSITION = 15  # the 15th of 30 runs sorted by validation hypervolume, then run number (plan, section 7)


def knee_measures(decisions, knee_ids):
    """One row per knee of the fixed list, whether or not it made a decision: with no decision the shares
    have no denominator (NA); with no changed decision the changed share is 0 and the measures on changed
    decisions are NA; nothing is filled with 0 that is not a 0."""
    rows = []
    for rule_id in knee_ids:
        d = decisions[decisions["rule id"] == rule_id]
        changed = d[d["changed"].astype(bool)]
        several = d[d["candidates"] >= 2]
        agrees = d["display agrees"].dropna() if "display agrees" in d else pd.Series(dtype=bool)
        rows.append({"rule id": rule_id, "decisions": len(d), "changed decisions": len(changed),
                     "changed share": d["changed"].astype(bool).mean() if len(d) else None,
                     "changed share, 2+ candidates": several["changed"].astype(bool).mean() if len(several) else None,
                     "VS change when changed": changed["VS change"].mean() if len(changed) else None,
                     "h1 given up when changed (within rule)": changed["h1 given up"].median() if len(changed) else None,
                     "display agreement (test)": agrees.astype(bool).mean() if len(agrees) else None})
    return pd.DataFrame(rows)


def spearman(x, y):
    """The correlation and, where it is not defined, why."""
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None, "fewer than 3 decisions"
    if np.ptp(x) == 0:
        return None, "w constant"
    if np.ptp(y) == 0:
        return None, "state constant"
    return float(spearmanr(x, y).statistic), ""


def weight_and_state(decisions, knees):
    """GP-adaptive knees: Spearman of the logged w with GDAY and with ARR, with the reason where not defined."""
    rows = []
    for rule_id in knees:
        d = decisions[decisions["rule id"] == rule_id]
        g, g_why = spearman(d["w"], d["GDAY"])
        a, a_why = spearman(d["w"], d["ARR"])
        rows.append({"rule id": rule_id, "decisions": len(d), "w vs GDAY": g, "w vs GDAY, not defined because": g_why,
                     "w vs ARR": a, "w vs ARR, not defined because": a_why})
    return pd.DataFrame(rows)


def knee_hours(decisions, rules):
    """Per knee and hour with decisions: the changed share, the mean VS change when changed (NA without a
    change) and the median logged w."""
    if decisions.empty:
        return pd.DataFrame(columns=["method", "rule id", "hour", "changed", "w", "vs"])
    d = decisions.merge(rules[["rule id", "method"]], on="rule id")
    d["hour"] = (d["time"] // 60).astype(int)
    d["vs if changed"] = d["VS change"].where(d["changed"].astype(bool))
    d["changed"] = d["changed"].astype(bool)
    return d.groupby(["method", "rule id", "hour"]).agg(
        changed=("changed", "mean"), w=("w", "median"), vs=("vs if changed", "mean")).reset_index()


def by_hour(per_knee):
    """Per method and hour: the median over its knees of the changed share, of the mean VS change when
    changed and of the median logged w (read for GP-adaptive), each with the number of knees it is defined for."""
    return per_knee.groupby(["method", "hour"]).agg(
        changed_share=("changed", "median"), changed_share_knees=("changed", "count"),
        vs_change=("vs", "median"), vs_change_knees=("vs", "count"),
        median_w=("w", "median"), median_w_knees=("w", "count")).reset_index()


def terminal_counts(rules):
    """Terminals in the h2 trees of GP-adaptive's distinct front rules, and of its knees."""
    out = []
    adaptive = rules[(rules["kind"] == "gp") & (rules["method"] == "GP-adaptive")]
    for name, subset in (("front rules", adaptive), ("knees", adaptive[adaptive["knee"].astype(bool)])):
        counts, nodes = Counter(), 0
        for text in subset["text"]:
            tree = from_text(text)[1]
            nodes += len(tree)
            counts.update(n.value for n in tree if isinstance(n, gp.Terminal) and isinstance(n.value, str))
        out.append({"set": name, "rules": len(subset), "h2 nodes": nodes, **dict(counts)})
    return pd.DataFrame(out).fillna(0)


def display_measures(results, pass_record):
    """Per GP knee with a display form: agreement on validation days (pass record, per run), display minus
    full in mean R_v (main alpha) and G_hat on the test days, the tidy form's nodes."""
    rules, means = results.rules, results.rule_means()
    shown = rules[rules["kind"] == "display"]
    validation = (pass_record or {}).get("display agreement on validation days", {})
    rows = []
    for _, row in shown.iterrows():
        parent = row["parent"]
        rows.append({"rule id": parent, "group": row["group"], "readable": row["readable"],
                     "tidy nodes": row["tidy nodes"], "display agreement (validation)": validation.get(row["group"]),
                     "display - full, mean R_v": means.loc[row["rule id"], rs.r_column()] - means.loc[parent, rs.r_column()],
                     "display - full, G_hat": means.loc[row["rule id"], "G_hat"] - means.loc[parent, "G_hat"]})
    return pd.DataFrame(rows)


def summarise(table, columns, rules):
    """Median and quartiles per method over its knees, with the number of knees where each is defined."""
    t = table.merge(rules[["rule id", "method"]], on="rule id")
    rows = []
    for method in METHODS:
        m = t[t["method"] == method]
        if m.empty:
            continue
        row = {"method": method, "knees": len(m)}
        for c in columns:
            q = rs.quartiles([None if pd.isna(v) else v for v in m[c]])
            row.update({f"{c} median": q["median"], f"{c} q1": q["q1"], f"{c} q3": q["q3"], f"{c} n": q["n"]})
        rows.append(row)
    return pd.DataFrame(rows)


def examples(rules, validation):
    """For each method, the knee of the run at the 15th place by (validation hypervolume, run number)."""
    rows = []
    for method in METHODS:
        runs = sorted((validation[k], k[1]) for k in validation if k[0] == method)
        if not runs:
            continue
        position = min(EXAMPLE_POSITION, len(runs)) - 1  # fewer runs only when developing on the pilot
        run = runs[position][1]
        knee = rules[(rules["kind"] == "gp") & (rules["method"] == method) & (rules["run"] == run)
                     & rules["knee"].astype(bool)].iloc[0]
        rows.append({"method": method, "run": run, "place": position + 1, "rule id": knee["rule id"],
                     "display": knee.get("display")})
    return pd.DataFrame(rows)


def figure3(hours, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, column, label in ((axes[0], "changed_share", "share of decisions changed by the fairness term"),
                              (axes[1], "vs_change", "mean change in VS when changed (negative: less visited)")):
        style(ax)
        for method in METHODS:
            h = hours[hours["method"] == method]
            if not h.empty:
                ax.plot(h["hour"], h[column], color=COLOURS[method], marker=MARKERS[method], markersize=4,
                        linewidth=1.5, label=method)
        ax.set_xlabel("hour of day", color=INK)
        ax.set_ylabel(label, color=INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Knee rules on the test days: how the fairness term changes choices, by hour (median over knees)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_all(results, step5_root, pass_record, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rules = results.rules
    gp_groups = [g for g in results.groups() if g != "baselines"]
    per_knee, w, hours = [], [], []
    for group in gp_groups:  # one GP knee per group: its decisions are read a group at a time
        decisions = results.table("decisions", group)
        knees = rules[(rules["group"] == group) & (rules["kind"] == "gp") & rules["knee"].astype(bool)]
        per_knee.append(knee_measures(decisions, list(knees["rule id"])))
        hours.append(knee_hours(decisions, rules))
        w.append(weight_and_state(decisions, list(knees.loc[knees["method"] == "GP-adaptive", "rule id"])))
        del decisions
    per_knee = pd.concat(per_knee, ignore_index=True)
    per_knee.to_csv(folder / "knees.csv", index=False)
    w = pd.concat([x for x in w if len(x)], ignore_index=True) if any(len(x) for x in w) else pd.DataFrame(
        columns=["rule id", "decisions", "w vs GDAY", "w vs GDAY, not defined because", "w vs ARR",
                 "w vs ARR, not defined because"])
    w.to_csv(folder / "weight_and_state.csv", index=False)
    shown = display_measures(results, pass_record)
    shown.to_csv(folder / "display_forms.csv", index=False)
    table = summarise(per_knee.merge(shown.drop(columns=["group", "readable"]), on="rule id", how="left"),
                      ["changed share", "changed share, 2+ candidates", "VS change when changed",
                       "display agreement (test)", "display agreement (validation)", "display - full, mean R_v",
                       "display - full, G_hat", "tidy nodes"], rules)
    table.to_csv(folder / "table4.csv", index=False)
    correlations = {"knees": len(w)}
    for state in ("GDAY", "ARR"):
        values = [v for v in w[f"w vs {state}"] if pd.notna(v)]
        reasons = w[f"w vs {state}, not defined because"]
        correlations[f"w vs {state}"] = {**rs.quartiles(values), "positive": sum(v > 0 for v in values),
                                         "not defined": {r: int((reasons == r).sum()) for r in set(reasons) if r}}
    (folder / "weight_and_state_summary.json").write_text(json.dumps(correlations, indent=1))
    terminal_counts(rules).to_csv(folder / "terminals.csv", index=False)
    runs = rules[rules["kind"] == "gp"].drop_duplicates(["method", "run"])
    examples(rules, validation_hypervolumes(step5_root, list(zip(runs["method"], runs["run"])))).to_csv(
        folder / "examples.csv", index=False)
    hours = by_hour(pd.concat(hours, ignore_index=True))
    hours.to_csv(folder / "by_hour.csv", index=False)
    figure3(hours, folder / "figure3.png")
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--rules", default=str(RULE_LIST))
    parser.add_argument("--test-settings", default=str(TEST_SETTINGS))
    parser.add_argument("--step5", default=str(FORMAL))
    parser.add_argument("--not-formal", action="store_true")
    args = parser.parse_args(argv)
    results = rs.load(args.out, args.rules, args.test_settings, formal=not args.not_formal)
    check_step5_source(results, args.step5)
    record = json.loads(PASS_RECORD.read_text()) if PASS_RECORD.exists() and not args.not_formal else None
    print(run_all(results, args.step5, record, Path(args.out) / "analysis" / "rq2"))


if __name__ == "__main__":
    main()
