"""
RQ3: satisfaction, fairness and who is affected (Step 6 plan, sections 8 and 10)
================================================================================

  - How uneven the days are: the observed G_hat of every 2019 day by split, and
    of 2022, from the day tables (no rule involved).
  - The test envelope of the fixed rule set: for each run and each level g from
    0.15 to 0.65, the best mean R_v among its validation-selected rules with test
    G_hat <= g (ties: the first by rule text; the tuned rule: the smaller
    weight), with that rule's completion; "not reached" is never imputed. Per
    method and level: runs reaching it, and the median and quartiles over them.
    Built again at every alpha.
  - Who bears the losses, at the knees (visitors pooled over test days and
    replicates): the share with R_v < 1 and with R_v >= 0.8, 0.85, 0.9 at every
    alpha; by arrival time, budget and observed stops: mean R_v, those shares,
    completion, the share ending early, and what moved R_v in value units and as
    the exact split of mean(R_v - 1), checked to add up within each knee and group.
    The share with R_v < 1 at the main alpha is checked against the histograms.
  - Visit completion beside every G_hat; knees below 0.95 marked (a display aid).
  - Peak crowding: per site and test day, the knee's highest hourly C averaged
    over replicates against the observed day's.
Per method: medians and quartiles over the knees, with the number defined.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate6  # noqa: E402
import results as rs  # noqa: E402
from rq1 import COLOURS, INK, METHODS, MARKERS, MUTED, style  # noqa: E402
from run_test import OUT, RULE_LIST, SETTINGS_FILE, TEST_SETTINGS  # noqa: E402

from day import PROCESSED  # noqa: E402
from evaluate import LOWEST_BIN  # noqa: E402
from search import load_settings  # noqa: E402

LEVELS = [round(0.15 + 0.05 * i, 2) for i in range(11)]  # 0.15 .. 0.65 (plan, section 8)
THETAS = (0.8, 0.85, 0.9)
COMPLETION_MARK = 0.95
DIMENSIONS = {"arrival group": "arrival", "budget group": "budget", "stop group": "observed stops"}


def observed_unevenness():
    rows = []
    days = pd.read_csv(PROCESSED / "days_2019.csv")
    for split, d in days.groupby("split"):
        rows.append({"year": 2019, "days": split, **rs.quartiles(d["G_hat"])})
    if (PROCESSED / "days_2022.csv").exists():
        rows.append({"year": 2022, "days": "all", **rs.quartiles(pd.read_csv(PROCESSED / "days_2022.csv")["G_hat"])})
    return pd.DataFrame(rows)


def envelope(results, alpha=rs.MAIN):
    """One row per run (and the tuned rule) and level."""
    rules, means, done = results.rules, results.rule_means(), results.completion()
    r = rs.r_column(alpha)
    sets = [(m, run, members.sort_values("text")) for (m, run), members in
            rules[rules["kind"] == "gp"].groupby(["method", "run"])]
    tuned = rules[(rules["kind"] == "tuned") & rules["validation front"].astype(bool)]
    sets.append(("tuned", -1, tuned.assign(w=tuned["text"].astype(float)).sort_values("w")))
    rows = []
    for method, run, members in sets:
        ids = list(members["rule id"])  # in tie order: rule text, or the weight for the tuned rule
        for g in LEVELS:
            eligible = [i for i in ids if means.loc[i, "G_hat"] <= g]
            if not eligible:
                rows.append({"method": method, "run": run, "level": g, "reached": False})  # never imputed
                continue
            best = max(means.loc[i, r] for i in eligible)
            chosen = next(i for i in eligible if means.loc[i, r] == best)
            rows.append({"method": method, "run": run, "level": g, "reached": True, "rule id": chosen,
                         "best mean R_v": best, "G_hat": means.loc[chosen, "G_hat"], **done.loc[chosen].to_dict()})
    return pd.DataFrame(rows)


ENVELOPE_COLUMNS = ["method", "run", "level", "reached", "rule id", "best mean R_v", "G_hat", "completion",
                    "ended early", "stops lost", "end: limit", "end: none feasible", "end: no observed stop left",
                    "end: not a follower"]


def envelope_summary(env):
    env = env.reindex(columns=ENVELOPE_COLUMNS)  # the same columns even when no run reaches any level
    rows = []
    for (method, g), e in env.groupby(["method", "level"], sort=False):
        reached = e[e["reached"]]
        q, c = rs.quartiles(reached["best mean R_v"]), rs.quartiles(reached["completion"])
        rows.append({"method": method, "level": g, "runs": len(e), "runs reaching it": len(reached),
                     "best mean R_v median": q["median"], "q1": q["q1"], "q3": q["q3"],
                     "completion of chosen rules, median": c["median"],
                     f"chosen rules with completion below {COMPLETION_MARK}": int((reached["completion"] <
                                                                                   COMPLETION_MARK).sum())})
    return pd.DataFrame(rows)


def knee_visitors(v, beta):
    """The measures of one knee from its per-visitor rows, overall and by each dimension's groups."""
    keys = evaluate6.main_keys(v)
    rows = []
    for dimension in [None] + list(DIMENSIONS):
        groups = [("all", v)] if dimension is None else list(v.groupby(keys[dimension]))
        for value, g in groups:
            row = {"dimension": dimension or "all", "group": value, "visitors": len(g),
                   "completion": g["stops"].sum() / g["observed stops"].sum(),
                   "ended early": (g["stops"] < g["observed stops"]).mean()}
            for a in rs.ALPHAS:
                rv = g[f"R_v (alpha {evaluate6.label(a)})"]
                row[f"mean R_v (alpha {a:g})"] = rv.mean()
                row[f"R_v < 1 (alpha {a:g})"] = (rv < 1).mean()
                for t in THETAS:
                    row[f"R_v >= {t:g} (alpha {a:g})"] = evaluate6.at_least(rv, t).mean()
            a = rs.MAIN
            row.update({"change A": g["change A"].mean(), "-alpha change C": -a * g["change C"].mean(),
                        "-beta change T": -beta * g["change T"].mean()})
            parts = {"A": g["change A"] / g["A observed"], "C": -a * g["change C"] / g["A observed"],
                     "T": -beta * g["change T"] / g["A observed"]}
            row.update({"change A / A_O": parts["A"].mean(), "-alpha change C / A_O": parts["C"].mean(),
                        "-beta change T / A_O": parts["T"].mean()})
            total = row["change A / A_O"] + row["-alpha change C / A_O"] + row["-beta change T / A_O"]
            if abs(total - (row[f"mean R_v (alpha {a:g})"] - 1)) > 1e-9:
                raise AssertionError("the split of mean(R_v - 1) does not add up")
            rows.append(row)
    return rows


def below_one_from_histograms(results, group, ids):
    """Visitors with R_v < 1 at the main alpha, from the grouped histograms (bins below 1.0)."""
    t = results.table("groups", group)
    t = t[t["rule id"].isin(ids)]
    bins = [c for c in t.columns if c.startswith("bin ") and int(c.split()[1]) < 20]  # bin 20 starts at 1.0
    return t.groupby("rule id")[bins].sum().sum(axis=1)


def knee_tables(results, beta):
    """Per knee measures (all knees: GP, tuned, hand rules, control), read a group at a time."""
    rules = results.rules
    out = []
    for group in results.groups():
        knees = list(rules.loc[(rules["group"] == group) & rules["knee"].astype(bool), "rule id"])
        if not knees:
            continue
        v = results.table("visitors", group)
        below = below_one_from_histograms(results, group, knees)
        for rule_id, k in v.groupby("rule id"):
            if int((k[f"R_v (alpha {evaluate6.label(rs.MAIN)})"] < 1).sum()) != int(below.loc[rule_id]):
                raise AssertionError(f"{rule_id}: R_v < 1 differs between the visitor rows and the histograms")
            for row in knee_visitors(k, beta):
                out.append({"rule id": rule_id, **row})
        del v
    return pd.DataFrame(out).merge(rules[["rule id", "method", "kind"]], on="rule id")


def table5(knees):
    """Per method, dimension and group: median and quartiles over the knees."""
    measures = [c for c in knees.columns if c not in ("rule id", "dimension", "group", "method", "kind")]
    rows = []
    for (method, dimension, group), k in knees.groupby(["method", "dimension", "group"], sort=False):
        row = {"method": method, "dimension": dimension, "group": group, "knees": len(k)}
        for c in measures:
            q = rs.quartiles([x for x in k[c] if pd.notna(x)])
            row.update({c: q["median"], f"{c} q1": q["q1"], f"{c} q3": q["q3"], f"{c} n": q["n"]})
        rows.append(row)
    return pd.DataFrame(rows)


def peaks(results):
    """Per knee and site: the mean over test days of its peak C (averaged over replicates first) and of the
    observed day's, and the share of test days on which the knee's is higher."""
    rules = results.rules
    seen = results.table("observed_peaks").set_index(["date", "site"])["peak C"]
    rows = []
    for group in results.groups():
        knees = list(rules.loc[(rules["group"] == group) & rules["knee"].astype(bool), "rule id"])
        if not knees:
            continue
        p = results.table("peaks", group)
        p = p[p["rule id"].isin(knees)].groupby(["rule id", "date", "site"])["peak C"].mean().reset_index()
        p["observed"] = seen.loc[list(zip(p["date"], p["site"]))].to_numpy()
        for (rule_id, site), d in p.groupby(["rule id", "site"]):
            rows.append({"rule id": rule_id, "site": site, "knee peak C": d["peak C"].mean(),
                         "observed peak C": d["observed"].mean(), "days knee higher": (d["peak C"] > d["observed"]).mean()})
    return pd.DataFrame(rows).merge(rules[["rule id", "method"]], on="rule id")


def end_reasons(knees):
    """Per method: completion, the share ending early and the share of visitors by end reason over its knees
    (median), with the knees marked below the completion mark."""
    reasons = [c for c in knees.columns if c.startswith("end: ")]
    rows = []
    for method, k in knees.groupby("method", sort=False):
        row = {"method": method, "knees": len(k), f"knees with completion below {COMPLETION_MARK} (mark)":
               int(k["completion below 0.95 (mark)"].sum())}
        for c in ["completion", "ended early"] + reasons:
            row[c] = rs.quartiles(k[c])["median"]
        rows.append(row)
    return pd.DataFrame(rows)


def peaks_summary(per_site):
    """Per method and site: the median over its knees of the knee's peak C, of the observed peak C and of the
    share of test days on which the knee's is higher, with the number of knees."""
    return per_site.groupby(["method", "site"]).agg(
        knees=("rule id", "nunique"), knee_peak_C=("knee peak C", "median"),
        observed_peak_C=("observed peak C", "median"), share_of_days_knee_higher=("days knee higher", "median"),
        share_q1=("days knee higher", lambda s: s.quantile(0.25)),
        share_q3=("days knee higher", lambda s: s.quantile(0.75))).reset_index()


def figure4(summary, path, hand=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    style(ax)
    for method in METHODS:
        s = summary[(summary["method"] == method) & summary["best mean R_v median"].notna()]
        if s.empty:
            continue
        ax.plot(s["level"], s["best mean R_v median"], color=COLOURS[method], linewidth=1.8, label=method)
        low = s[f"chosen rules with completion below {COMPLETION_MARK}"] > 0
        ax.scatter(s.loc[~low, "level"], s.loc[~low, "best mean R_v median"], color=COLOURS[method],
                   marker=MARKERS[method], s=22, zorder=3)
        ax.scatter(s.loc[low, "level"], s.loc[low, "best mean R_v median"], facecolor="white",
                   edgecolor=COLOURS[method], marker=MARKERS[method], s=28, zorder=3)
        ax.fill_between(s["level"], s["q1"], s["q3"], color=COLOURS[method], alpha=0.15, linewidth=0)
        for _, row in s.iterrows():
            if row["runs reaching it"] < row["runs"]:
                ax.annotate(f"{int(row['runs reaching it'])}/{int(row['runs'])}", (row["level"], row["best mean R_v median"]),
                            xytext=(0, 6), textcoords="offset points", fontsize=7, color=COLOURS[method], ha="center")
    t = summary[(summary["method"] == "tuned") & summary["best mean R_v median"].notna()]
    ax.plot(t["level"], t["best mean R_v median"], color=COLOURS["tuned"], linestyle="--", linewidth=1.8,
            label="tuned weighted rule")
    if hand is not None:
        for _, row in hand.iterrows():
            filled = row["completion"] >= COMPLETION_MARK
            ax.scatter([row["G_hat"]], [row["mean R_v"]], s=55, marker="o", zorder=4, linewidth=1.3,
                       facecolor=INK if filled else "white", edgecolor=INK)
            ax.annotate(row["rule"], (row["G_hat"], row["mean R_v"]), xytext=(6, -10), textcoords="offset points",
                        fontsize=8, color=INK)
    ax.scatter([], [], facecolor="white", edgecolor=INK, s=28,
               label=f"hollow: completion below {COMPLETION_MARK} (hand rules: their own; levels: any chosen rule)")
    ax.set_xlabel("level g: test G_hat at most g", color=INK)
    ax.set_ylabel("best mean R_v among the run's rules", color=INK)
    ax.set_title("Test envelope of the fixed rule sets (median and quartiles over runs; n/30 where not all reach g)",
                 color=INK, loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure5(per_site, sites, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    style(ax)
    order = sorted(per_site["site"].unique())
    x = np.arange(len(order))
    observed = per_site.groupby("site")["observed peak C"].first().reindex(order)
    ax.bar(x - 0.3, observed, width=0.2, color=MUTED, label="observed days")
    for n, method in enumerate(METHODS):
        m = per_site[per_site["method"] == method].groupby("site")["knee peak C"].median().reindex(order)
        ax.bar(x - 0.1 + 0.2 * n, m, width=0.2, color=COLOURS[method], label=f"{method} knees (median)")
    ax.set_xticks(x, [sites.get(s, s) for s in order], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("busiest-hour crowding C (mean over test days)", color=INK)
    ax.set_title("Peak hourly crowding per site: knees against the observed days", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_all(results, settings, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    observed_unevenness().to_csv(folder / "observed_unevenness.csv", index=False)
    for a in rs.ALPHAS:
        env = envelope(results, a)
        env.to_csv(folder / f"envelope_alpha_{a:g}.csv", index=False)
        summary = envelope_summary(env)
        summary.to_csv(folder / f"envelope_summary_alpha_{a:g}.csv", index=False)
        if a == rs.MAIN:
            means, done = results.rule_means(), results.completion()
            h = results.rules[results.rules["kind"].isin(["hand", "control"])]
            hand = pd.DataFrame({"rule": h["text"].to_numpy(), "G_hat": means.loc[h["rule id"], "G_hat"].to_numpy(),
                                 "mean R_v": means.loc[h["rule id"], rs.r_column()].to_numpy(),
                                 "completion": done.loc[h["rule id"], "completion"].to_numpy()})
            figure4(summary, folder / "figure4.png", hand)
    knees = knee_tables(results, settings["beta"])
    knees.to_csv(folder / "knees.csv", index=False)
    table5(knees).to_csv(folder / "table5.csv", index=False)
    means, done = results.rule_means(), results.completion()
    k = results.rules[results.rules["knee"].astype(bool)][["rule id", "method", "kind"]]
    k = pd.concat([k.reset_index(drop=True), done.loc[k["rule id"]].reset_index(drop=True)], axis=1)
    k.insert(3, "G_hat", means.loc[k["rule id"], "G_hat"].to_numpy())
    k["completion below 0.95 (mark)"] = k["completion"] < COMPLETION_MARK
    k.to_csv(folder / "knee_completion.csv", index=False)
    end_reasons(k).to_csv(folder / "end_reasons.csv", index=False)
    per_site = peaks(results)
    per_site.to_csv(folder / "peaks.csv", index=False)
    peaks_summary(per_site).to_csv(folder / "peaks_summary.csv", index=False)
    names = dict(zip(*pd.read_csv(PROCESSED / "sites.csv")[["site", "name"]].T.values))
    figure5(per_site, names, folder / "figure5.png")
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--rules", default=str(RULE_LIST))
    parser.add_argument("--test-settings", default=str(TEST_SETTINGS))
    parser.add_argument("--not-formal", action="store_true")
    args = parser.parse_args(argv)
    results = rs.load(args.out, args.rules, args.test_settings, formal=not args.not_formal)
    print(run_all(results, load_settings(SETTINGS_FILE), Path(args.out) / "analysis" / "rq3"))


if __name__ == "__main__":
    main()
