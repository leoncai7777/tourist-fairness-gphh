"""
RQ1: the comparison of the three methods (Step 6 plan, sections 6 and 10)
=========================================================================

  - Each run's test front: its validation-selected GP rules (not their display
    forms), each with its test means; the non-dominated ones on the raw
    objectives; normalised with a range, unclipped; hypervolume by the frozen
    search.hypervolume. The tuned rule likewise, from its weights that are
    non-dominated on the validation days.
  - Table 1: per method, the test hypervolumes (median, quartiles), the
    validation hypervolumes beside them and the number of test-front rules;
    the tuned rule's single value.
  - Table 2: the three pairwise comparisons (stats.compare, Holm over three).
  - Hand rules and the control: for each, how many runs of each method have a
    test front that dominates it.
  - Table 3: the wider and narrower ranges, and alpha 0, 0.5 and 1: ranking by
    median hypervolume and median paired differences with intervals; no tests.
  - Figures 1 and 2.

  python3 analysis/rq1.py                       # the formal outputs, after the batch
  python3 analysis/rq1.py --not-formal --out O --rules R --test-settings S --step5 P   # development
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results as rs  # noqa: E402
import stats  # noqa: E402
from run_test import OUT, RULE_LIST, SETTINGS_FILE, TEST_SETTINGS  # noqa: E402
from step5 import FORMAL  # noqa: E402

from search import hypervolume, load_settings, non_dominated, normalise  # noqa: E402

METHODS = ("GP-fixed", "GP-prop", "GP-adaptive")
PAIRS = (("GP-adaptive", "GP-fixed"), ("GP-adaptive", "GP-prop"), ("GP-fixed", "GP-prop"))
WIDER = {"R": (0.3, 1.4), "G": (0.05, 0.9)}       # search design, section 15
NARROWER = {"R": (0.55, 1.2), "G": (0.12, 0.7)}
COLOURS = {"GP-fixed": "#2a78d6", "GP-prop": "#eb6834", "GP-adaptive": "#1baf7a", "tuned": "#eda100"}
MARKERS = {"GP-fixed": "o", "GP-prop": "s", "GP-adaptive": "^"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def main_range(settings):
    return {"R": tuple(settings["R range"]), "G": tuple(settings["G range"])}


def front_of(points):
    """Positions of the non-dominated points (raw objectives: R up, G down)."""
    return non_dominated(list(points))


def score_front(means, ids, alpha, value_range):
    """The test front of a set of rules: its hypervolume, size and points behind the reference."""
    points = [(float(means.loc[i, rs.r_column(alpha)]), float(means.loc[i, "G_hat"])) for i in ids]
    front = front_of(points)
    area, outside = hypervolume(normalise([points[j] for j in front], value_range))
    return {"hypervolume": area, "front rules": len(front), "behind the reference": outside,
            "front": [ids[j] for j in front]}


def run_fronts(results, alpha, value_range):
    """One row per GP run, and one for the tuned rule."""
    rules, means = results.rules, results.rule_means()
    rows = []
    for (method, run), members in rules[rules["kind"] == "gp"].groupby(["method", "run"]):
        out = score_front(means, list(members["rule id"]), alpha, value_range)
        rows.append({"method": method, "run": int(run), **{k: v for k, v in out.items() if k != "front"}})
    tuned = rules[(rules["kind"] == "tuned") & rules["validation front"].astype(bool)]
    out = score_front(means, list(tuned["rule id"]), alpha, value_range)
    rows.append({"method": "tuned", "run": -1, **{k: v for k, v in out.items() if k != "front"}})
    return pd.DataFrame(rows)


def validation_hypervolumes(root, runs):
    """Each run's validation hypervolume, from Step 5's summary.json (as saved)."""
    out = {}
    for method, run in runs:
        folder = Path(root) / (f"{method}_{run:02d}" if run < 100 else f"{method}_{run}")
        out[method, run] = json.loads((folder / "results" / "summary.json").read_text())["hypervolume"]
    return out


def table1(fronts, validation):
    rows = []
    for method in METHODS:
        m = fronts[fronts["method"] == method]
        if m.empty:
            continue
        test, front_size = rs.quartiles(m["hypervolume"]), rs.quartiles(m["front rules"])
        val = rs.quartiles([validation.get((method, r)) for r in m["run"]])
        rows.append({"method": method, "runs": test["n"], "test HV median": test["median"], "test HV q1": test["q1"],
                     "test HV q3": test["q3"], "validation HV median": val["median"], "validation HV q1": val["q1"],
                     "validation HV q3": val["q3"], "test front rules median": front_size["median"],
                     "test front rules q1": front_size["q1"], "test front rules q3": front_size["q3"]})
    tuned = fronts[fronts["method"] == "tuned"].iloc[0]
    rows.append({"method": "tuned (one value, not tested)", "runs": 1, "test HV median": tuned["hypervolume"],
                 "test front rules median": tuned["front rules"]})
    return pd.DataFrame(rows)


def by_run(fronts, method):
    m = fronts[fronts["method"] == method]
    return dict(zip(m["run"], m["hypervolume"]))


def table2(fronts):
    rows = []
    for a, b in PAIRS:
        out = stats.compare(by_run(fronts, a), by_run(fronts, b))
        rows.append({"pair": f"{a} - {b}", "n": out["n"], "zeros": out["zeros"], "m": out["m"],
                     "median difference": out["median median"], "95% low": out["median low"],
                     "95% high": out["median high"], "coverage": out["median coverage"], "rank-biserial r": out["r"],
                     "p": out["p"], "note": out["note"]})
    table = pd.DataFrame(rows)
    table["Holm p"] = stats.holm(list(table["p"]))
    table["significant (Holm p < 0.05)"] = table["Holm p"] < 0.05
    return table


def dominated_counts(results, alpha, value_range):
    """For each hand rule and the control: how many runs of each method have a test front that dominates it
    (weakly better in both objectives, strictly in one)."""
    rules, means = results.rules, results.rule_means()
    r, g = rs.r_column(alpha), "G_hat"
    fronts = {}
    for (method, run), members in rules[rules["kind"] == "gp"].groupby(["method", "run"]):
        ids = score_front(means, list(members["rule id"]), alpha, value_range)["front"]
        fronts[method, run] = [(means.loc[i, r], means.loc[i, g]) for i in ids]
    rows = []
    for _, row in rules[rules["kind"].isin(["hand", "control"])].iterrows():
        p = (means.loc[row["rule id"], r], means.loc[row["rule id"], g])
        counts = {m: sum(any(q[0] >= p[0] and q[1] <= p[1] and (q[0] > p[0] or q[1] < p[1]) for q in pts)
                         for (method, _), pts in fronts.items() if method == m) for m in METHODS}
        rows.append({"rule": row["text"], "mean R_v": p[0], "G_hat": p[1],
                     **{f"runs of {m} dominating it": counts[m] for m in METHODS},
                     "runs per method": {m: sum(1 for k in fronts if k[0] == m) for m in METHODS}[METHODS[0]]})
    return pd.DataFrame(rows)


def ranking_of(medians):
    """Methods by median hypervolume, best first; equal medians are shown as a tie ("="), not ordered."""
    present = sorted((v, m) for m, v in medians.items() if v is not None)[::-1]
    out = ""
    for i, (v, m) in enumerate(present):
        out += m if i == 0 else (" = " if v == present[i - 1][0] else " > ") + m
    return out


def is_main_test(data):
    """The 2019 test days without a scenario: the only batch whose comparison is tested (plan, sections 6, 11)."""
    return data["year"] == 2019 and "scenario" not in data


def check_step5_source(results, step5_root):
    """For the formal outputs, the Step 5 results read here must be those the reproduction check passed on."""
    if results.identity.get("mode") == "formal" and str(step5_root) != results.identity.get("step 5 results"):
        raise SystemExit(f"--step5 {step5_root} is not the Step 5 results of the pass record")


def table3(results, settings):
    """Ranking and median paired differences under other ranges and other alphas; no tests."""
    settings_list = [("main range, alpha 0.25", rs.MAIN, main_range(settings)),
                     ("wider range", rs.MAIN, WIDER), ("narrower range", rs.MAIN, NARROWER)]
    settings_list += [(f"alpha {a:g}", a, main_range(settings)) for a in rs.ALPHAS if a != rs.MAIN]
    rows = []
    for name, alpha, value_range in settings_list:
        fronts = run_fronts(results, alpha, value_range)
        medians = {m: rs.quartiles(fronts.loc[fronts["method"] == m, "hypervolume"])["median"] for m in METHODS}
        row = {"setting": name, "ranking by median HV": ranking_of(medians),
               "behind the reference": int(fronts["behind the reference"].sum()),
               **{f"{m} median HV": medians[m] for m in METHODS}}
        for a, b in PAIRS:
            _, d = stats.paired_differences(by_run(fronts, a), by_run(fronts, b))
            iv = stats.median_interval(d)
            row[f"{a} - {b}"] = iv["median"]
            row[f"{a} - {b} 95%"] = (iv["low"], iv["high"])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- #
# Figures
# ---------------------------------------------------------------------- #

def style(ax):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def figure1(fronts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    style(ax)
    data = [fronts.loc[fronts["method"] == m, "hypervolume"].to_numpy() for m in METHODS]
    ax.boxplot(data, positions=range(len(METHODS)), widths=0.5, showfliers=False,
               medianprops={"color": INK}, boxprops={"color": MUTED}, whiskerprops={"color": MUTED},
               capprops={"color": MUTED})
    rng = np.random.default_rng(0)
    for i, (m, values) in enumerate(zip(METHODS, data)):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(values)), values, s=18, color=COLOURS[m], marker=MARKERS[m],
                   zorder=3, label=f"{m} ({len(values)} runs)")
    tuned = fronts.loc[fronts["method"] == "tuned", "hypervolume"].iloc[0]
    ax.axhline(tuned, color=COLOURS["tuned"], linestyle="--", linewidth=1.5, label="tuned weighted rule (one value)")
    ax.set_xticks(range(len(METHODS)), METHODS)
    ax.set_ylabel("test hypervolume", color=INK)
    ax.set_title("Test hypervolume of each run, by method", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure2(results, alpha, value_range, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rules, means = results.rules, results.rule_means()
    r = rs.r_column(alpha)
    fig, ax = plt.subplots(figsize=(8, 5.6))
    style(ax)
    for method in METHODS:
        first = True
        for _, members in rules[(rules["kind"] == "gp") & (rules["method"] == method)].groupby("run"):
            ids = score_front(means, list(members["rule id"]), alpha, value_range)["front"]
            pts = sorted((means.loc[i, "G_hat"], means.loc[i, r]) for i in ids)
            ax.plot(*zip(*pts), color=COLOURS[method], linewidth=1, alpha=0.35,
                    label=f"{method} test fronts" if first else None)
            first = False
    tuned = rules[(rules["kind"] == "tuned") & rules["validation front"].astype(bool)]
    ids = score_front(means, list(tuned["rule id"]), alpha, value_range)["front"]
    pts = sorted((means.loc[i, "G_hat"], means.loc[i, r]) for i in ids)
    ax.plot(*zip(*pts), color=COLOURS["tuned"], linewidth=2, linestyle="--", marker="D", markersize=4,
            label="tuned weighted rule")
    for _, row in rules[rules["kind"].isin(["hand", "control"])].iterrows():
        x, y = means.loc[row["rule id"], "G_hat"], means.loc[row["rule id"], r]
        ax.scatter([x], [y], s=60, facecolor="white", edgecolor=INK, linewidth=1.3, zorder=4)
        ax.annotate(row["text"], (x, y), xytext=(6, -11), textcoords="offset points", fontsize=8, color=INK)
    seen = results.observed()
    ax.scatter([seen["G_hat"].mean()], [1.0], s=130, marker="*", color=INK, zorder=5, label="observed days")
    ax.set_xlabel("G_hat, unevenness of visits over sites (lower is more even)", color=INK)
    ax.set_ylabel(f"mean R_v (alpha {alpha:g}; higher is better)", color=INK)
    ax.set_title("Test fronts of every run, the tuned rule, the hand rules and the observed days",
                 color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_all(results, settings, step5_root, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fronts = run_fronts(results, rs.MAIN, main_range(settings))
    gp = fronts[fronts["method"] != "tuned"]
    validation = validation_hypervolumes(step5_root, list(zip(gp["method"], gp["run"])))
    fronts.to_csv(folder / "test_fronts.csv", index=False)
    table1(fronts, validation).to_csv(folder / "table1.csv", index=False)
    if is_main_test(results.data):  # other years and scenarios are checks of transfer: no test (plan, section 11)
        table2(fronts).to_csv(folder / "table2.csv", index=False)
    dominated_counts(results, rs.MAIN, main_range(settings)).to_csv(folder / "hand_rules_dominated.csv", index=False)
    table3(results, settings).to_csv(folder / "table3.csv", index=False)
    figure1(fronts, folder / "figure1.png")
    figure2(results, rs.MAIN, main_range(settings), folder / "figure2.png")
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
    print(run_all(results, load_settings(SETTINGS_FILE), args.step5, Path(args.out) / "analysis" / "rq1"))


if __name__ == "__main__":
    main()
