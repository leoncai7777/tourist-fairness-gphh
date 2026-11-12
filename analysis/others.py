"""
Other measures (Step 6 plan, section 9)
=======================================

  - Rule size: nodes and depth of each tree of the front rules and of the knees,
    per method (median and quartiles).
  - Convergence: the monitored validation front of every 10th generation, kept in
    each run's checkpoint (7 monitoring days, replicate 0), scored by its
    hypervolume under the frozen range; per method and generation, the median
    and quartiles. There is no point at generation 0. Figure 6.
  - Time per run: the generation seconds of the search log summed (a resumed
    run's run.json holds only its last attempt), and the end of run from run.json.

Reads Step 5's checkpoints and run.json (validation information only); no test day.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results as rs  # noqa: E402
from rq1 import COLOURS, INK, METHODS, MARKERS, main_range, style  # noqa: E402
from run_test import OUT, RULE_LIST, SETTINGS_FILE  # noqa: E402
from step5 import FORMAL, read_table  # noqa: E402

from gp_rule import LABELS, from_text, size  # noqa: E402
from search import hypervolume, load_settings, non_dominated, normalise  # noqa: E402


def run_folder(root, method, run):
    return Path(root) / (f"{method}_{run:02d}" if run < 100 else f"{method}_{run}")


def rule_sizes(rules):
    """Nodes and depth of every tree of every GP rule, with its knee flag."""
    rows = []
    for _, row in rules[rules["kind"] == "gp"].iterrows():
        rule = from_text(row["text"])
        entry = {"rule id": row["rule id"], "method": row["method"], "knee": bool(row["knee"])}
        for label, (nodes, depth) in zip(LABELS[row["method"]], size(rule)):
            entry[f"{label} nodes"], entry[f"{label} depth"] = nodes, depth
        entry["nodes"] = sum(n for n, _ in size(rule))
        rows.append(entry)
    return pd.DataFrame(rows)


def size_summary(sizes):
    rows = []
    for method in METHODS:
        for name, subset in (("front rules", sizes[sizes["method"] == method]),
                             ("knees", sizes[(sizes["method"] == method) & sizes["knee"]])):
            if subset.empty:
                continue
            row = {"method": method, "rules": name, "count": len(subset)}
            for column in ("nodes", "h1 nodes", "h1 depth", "h2 nodes", "h2 depth"):
                if column in subset and subset[column].notna().any():
                    q = rs.quartiles(subset[column].dropna())
                    row.update({f"{column} median": q["median"], f"{column} q1": q["q1"], f"{column} q3": q["q3"]})
            rows.append(row)
    return pd.DataFrame(rows)


def convergence(root, runs, value_range):
    """Monitored validation hypervolume of every run at every monitored generation."""
    rows = []
    for method, run in runs:
        state = pickle.loads((run_folder(root, method, run) / "checkpoint" / "checkpoint.pkl").read_bytes())
        for entry in state["monitor log"]:
            points = [tuple(p) for p in entry["front"]]
            front = [points[i] for i in non_dominated(points)]
            area, _ = hypervolume(normalise(front, value_range))
            rows.append({"method": method, "run": run, "generation": entry["generation"] + 1, "hypervolume": area})
    return pd.DataFrame(rows)


def run_times(root, runs):
    rows = []
    for method, run in runs:
        folder = run_folder(root, method, run)
        state = pickle.loads((folder / "checkpoint" / "checkpoint.pkl").read_bytes())
        report = json.loads((folder / "run.json").read_text()) if (folder / "run.json").exists() else {}
        attempts = report.get("attempts", [])
        rows.append({"method": method, "run": run, "attempts": len(attempts),
                     "search minutes (generation log)": sum(e["seconds"] for e in state["log"]) / 60,
                     "end of run minutes (last attempt)": attempts[-1]["end seconds"] / 60 if attempts else None})
    return pd.DataFrame(rows)


def figure6(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2))
    style(ax)
    for method in METHODS:
        c = curves[curves["method"] == method]
        if c.empty:
            continue
        s = c.groupby("generation")["hypervolume"].quantile([0.25, 0.5, 0.75]).unstack()
        ax.plot(s.index, s[0.5], color=COLOURS[method], marker=MARKERS[method], markersize=5, linewidth=1.8,
                label=f"{method} ({c['run'].nunique()} runs)")
        ax.fill_between(s.index, s[0.25], s[0.75], color=COLOURS[method], alpha=0.15, linewidth=0)
    ax.set_xlim(0, None)
    ax.set_xlabel("generation", color=INK)
    ax.set_ylabel("monitored validation hypervolume", color=INK)
    ax.set_title("Convergence: median and quartiles over runs (7 monitoring days; no point at generation 0)",
                 color=INK, loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_all(rules, settings, step5_root, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sizes = rule_sizes(rules)
    sizes.to_csv(folder / "rule_sizes.csv", index=False)
    size_summary(sizes).to_csv(folder / "rule_size_summary.csv", index=False)
    runs = list(rules[rules["kind"] == "gp"].drop_duplicates(["method", "run"])[["method", "run"]].itertuples(
        index=False, name=None))
    curves = convergence(step5_root, runs, main_range(settings))
    curves.to_csv(folder / "convergence.csv", index=False)
    figure6(curves, folder / "figure6.png")
    run_times(step5_root, runs).to_csv(folder / "run_times.csv", index=False)
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--rules", default=str(RULE_LIST))
    parser.add_argument("--step5", default=str(FORMAL))
    args = parser.parse_args(argv)
    print(run_all(read_table(args.rules), load_settings(SETTINGS_FILE), args.step5, Path(args.out) / "analysis" / "others"))


if __name__ == "__main__":
    main()
