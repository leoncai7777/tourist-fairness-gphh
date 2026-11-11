"""
Two figures of the pilot, for the reviews (F1)
==============================================

  1. On the 35 validation days with K = 5: the pilot fronts of the three GP
     methods (one full run each, run 910), the tuned weighted rule over its
     fixed grid, the four hand rules, the observed-order control and the
     observed days, in the plane of mean R_v and G_hat.
  2. The monitored validation hypervolume of the three full pilot runs, every
     10 generations (provisional range of the pilot, only to watch the search).

Pilot, validation days only: not a result of the study. Kept in tools/.

Run:  python3 tools/pilot_figures.py   (from the repository root)
Writes outputs/search/pilot/figures/ (two PNG files and the baseline table).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PILOT = ROOT / "outputs" / "search" / "pilot" / "full"
OUT = ROOT / "outputs" / "search" / "pilot" / "figures"
COLOURS = {"GP-fixed": "#2a78d6", "GP-prop": "#eb6834", "GP-adaptive": "#1baf7a", "tuned": "#eda100"}
MARKERS = {"GP-fixed": "o", "GP-prop": "s", "GP-adaptive": "^"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def baselines():
    """The tuned rule, the hand rules and the control on the validation days, K = 5; cached in a table."""
    path = OUT / "baselines.csv"
    if path.exists():
        return pd.read_csv(path)
    from engine import ObservedOrder
    from evaluate import Days
    from preprocess import gini_hat
    from rules import RULES
    from search import complete, days_for, scoring, tuned_front, validation_dates, validation_means
    settings = complete({"population": 200, "generations": 100, "crossover": 0.8, "mutation": 0.15,
                         "monitor every": 10, "validation replicates": 5})
    table, _ = tuned_front(settings)
    rows = [("tuned", f"lambda {w:.3g}", r, g, f) for w, r, g, f in
            zip(table["weight"], table["mean R_v"], table["G_hat"], table["front"])]
    policies = []
    for name, cls in RULES.items():
        policy = cls()
        policy.name = name
        policies.append(policy)
    control = ObservedOrder()
    control.name = "observed order"
    policies.append(control)
    means = validation_means(policies, validation_dates(), 5, days_for(settings), scoring(settings))
    rows += [("hand", p.name, r, g, True) for p, (r, g) in zip(policies, means)]
    days = Days()
    observed = np.mean([gini_hat(days.get(d)[1].entries_per_site[days.get(d)[0].available])
                        for d in validation_dates()])
    rows.append(("observed", "observed days", 1.0, float(observed), True))
    out = pd.DataFrame(rows, columns=["kind", "name", "mean R_v", "G_hat", "front"])
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def style(ax):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def fronts_figure(base):
    fig, ax = plt.subplots(figsize=(8, 5.6))
    style(ax)
    for method in ("GP-fixed", "GP-prop", "GP-adaptive"):
        front = pd.read_csv(PILOT / f"{method}_910" / "results" / "front.csv.gz").drop_duplicates("text")
        front = front.sort_values("G_hat")
        ax.plot(front["G_hat"], front["mean R_v"], color=COLOURS[method], linewidth=2, marker=MARKERS[method],
                markersize=4, label=f"{method} front (pilot run)")
    tuned = base[(base["kind"] == "tuned") & base["front"]].sort_values("G_hat")
    ax.plot(tuned["G_hat"], tuned["mean R_v"], color=COLOURS["tuned"], linewidth=2, linestyle="--", marker="D",
            markersize=5, label="tuned weighted rule A - λ·VS (fixed grid)")
    hand = base[base["kind"] == "hand"]
    ax.scatter(hand["G_hat"], hand["mean R_v"], s=70, facecolor="white", edgecolor=INK, linewidth=1.5, zorder=3,
               label="hand rules and observed-order control")
    for _, h in hand.iterrows():
        ax.annotate(h["name"], (h["G_hat"], h["mean R_v"]), xytext=(6, -12), textcoords="offset points",
                    fontsize=8.5, color=INK)
    obs = base[base["kind"] == "observed"].iloc[0]
    ax.scatter([obs["G_hat"]], [obs["mean R_v"]], s=140, marker="*", color=INK, zorder=4, label="observed days")
    ax.annotate("observed days", (obs["G_hat"], obs["mean R_v"]), xytext=(8, 6), textcoords="offset points",
                fontsize=8.5, color=INK)
    ax.set_xlabel("G_hat, unevenness of visits over sites (lower is more even)", fontsize=10, color=INK)
    ax.set_ylabel("mean R_v over evaluated visitors (higher is better)", fontsize=10, color=INK)
    ax.set_title("Pilot on the 35 validation days, K = 5: the trade-off and where each rule lies\n"
                 "Pilot only (one run per method); not a result of the study", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "pilot_fronts_validation.png", dpi=160)
    plt.close(fig)


def convergence_figure():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    style(ax)
    for method in ("GP-fixed", "GP-prop", "GP-adaptive"):
        report = json.loads((PILOT / f"{method}_910" / "pilot.json").read_text())
        gens = [m["generation"] + 1 for m in report["monitoring"]]
        volume = [m["hypervolume"] for m in report["monitoring"]]
        ax.plot(gens, volume, color=COLOURS[method], linewidth=2, marker=MARKERS[method], markersize=6, label=method)
    ax.set_xlim(0, 105)
    ax.set_xlabel("generation", fontsize=10, color=INK)
    ax.set_ylabel("hypervolume on 7 validation days", fontsize=10, color=INK)
    ax.set_title("Monitored validation hypervolume of the full pilot runs: rising to about generation 50-70,\n"
                 "then flat within about 0.005 (y axis zoomed; provisional range, only to watch the search)",
                 fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "pilot_convergence.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = baselines()
    fronts_figure(base)
    convergence_figure()
    print(base.to_string(index=False))


if __name__ == "__main__":
    main()
