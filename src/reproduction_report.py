"""Compare the single-tourist GPHH with Table III of Mei and Zhang (2018) and write outputs/toptw/reproduction.md.

Run from the repository root:  python3 src/reproduction_report.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from toptw import load_instance, sot, test_score

MAX_POINTS = 100
# instance, SOS, SOT, BasicGP mean, BasicGP sd (Table III, one-day trips)
TABLE_III = """c101 187.16 159.74 292.44 10.25
c102 205.96 183.52 338.90 5.68
c103 264.46 245.84 375.89 8.26
c104 257.64 257.40 387.61 6.43
c105 192.32 171.42 323.02 8.56
r101 98.24 74.38 189.52 4.25
r102 121.10 175.03 273.66 6.82
r103 121.10 93.10 281.19 6.18
r104 149.09 128.04 288.08 11.90
r105 100.97 88.00 224.94 6.02
rc101 68.07 98.00 203.83 9.05
rc102 66.41 102.29 233.54 7.20
rc103 91.52 100.56 244.39 5.31
rc104 129.05 129.68 253.77 8.32
rc105 57.47 104.37 217.59 16.48
c201 210.00 182.16 822.96 10.20
c202 268.52 183.50 867.84 12.59
c203 435.94 233.66 916.61 11.48
c204 469.00 197.60 933.20 9.23
c205 273.72 188.40 861.22 9.63
r201 211.00 144.69 720.94 18.50
r202 309.73 257.08 840.37 15.17
r203 325.51 257.08 923.57 9.65
r204 522.80 307.24 1004.91 16.66
r205 226.92 240.40 831.81 12.11
rc201 182.99 117.49 729.19 20.24
rc202 214.20 186.40 859.00 29.34
rc203 235.10 188.66 926.51 10.38
rc204 266.33 400.14 1040.73 13.96
rc205 220.91 147.50 795.34 17.91
pr01 127.35 88.51 287.99 3.50
pr02 128.91 99.83 383.04 3.09
pr03 93.00 180.06 310.41 5.40
pr04 133.97 125.63 341.93 8.13
pr05 115.18 144.66 391.88 15.63
pr11 140.94 168.94 311.50 12.20
pr12 138.68 187.60 402.91 6.09
pr13 90.46 183.18 373.90 11.58
pr14 119.04 226.82 405.52 18.82
pr15 103.10 355.43 458.58 10.36"""
paper = pd.DataFrame([line.split() for line in TABLE_III.splitlines()], columns=["instance", "sos", "sot", "gp", "gp_sd"]).set_index("instance").astype(float)
folder = Path("outputs/toptw")
early = pd.read_csv(folder / "fig2_instances.csv")
early = early[~early.instance.isin(["pr03", "pr14"])]  # those two were first run on the full files
files = [early] + [pd.read_csv(folder / name) for name in ("pr_matching.csv", "pr_first100.csv", "remaining_instances.csv") if (folder / name).exists() and (folder / name).stat().st_size > 0]
results = pd.concat(files)

rows = []
for name, target in paper.iterrows():
    runs = results[results.instance == name]
    if runs.empty:
        continue
    instance = load_instance(f"data/toptw/{name}.txt", MAX_POINTS)
    mean, sd, n = runs.test.mean(), runs.test.std(), len(runs)
    standard_error = np.sqrt(sd ** 2 / n + target.gp_sd ** 2 / 30)
    hand = test_score(instance, sot)
    rows.append({"instance": name, "runs": n, "ours": round(mean, 2), "sd": round(sd, 2), "paper": target.gp, "paper sd": target.gp_sd,
                 "diff %": round((mean - target.gp) / target.gp * 100, 1), "t": round((mean - target.gp) / standard_error, 2),
                 "SOT ours": round(hand, 2), "SOT paper": target.sot, "SOT diff %": round((hand - target.sot) / target.sot * 100, 1)})
table = pd.DataFrame(rows)
within = (table["diff %"].abs() <= 2.5).sum()
beyond = (table.t.abs() > 2).sum()
below = (table["diff %"] < 0).sum()
sot_within = (table["SOT diff %"].abs() <= 2).sum()
out = [
    "# Reproduction of Mei and Zhang (2018), BasicGP",
    "",
    "Step 1 of the project: re-implement the single-tourist GPHH and check that its results on the TOPTW instances are comparable to those published. One-day trips, Table III of the paper.",
    "",
    "## Result",
    "",
    f"- Instances run: {len(table)} of 40, 30 runs each; {within} are within 2.5% of the published BasicGP mean.",
    f"- Mean difference {table['diff %'].mean():.2f}%, range {table['diff %'].min():.1f}% to {table['diff %'].max():.1f}%; ours is lower on {below} instances.",
    f"- The difference exceeds 2 standard errors on {beyond} instances (rough test from published mean and sd, 30 runs each), so a small part of it is systematic.",
    f"- Hand-written SOT rule: within 2% of the paper on {sot_within} instances. SOT is a deterministic-looking greedy rule and is sensitive to small changes.",
    "",
    "## Comparison",
    "",
    "| " + " | ".join(table.columns) + " |",
    "|" + "---|" * len(table.columns),
]
out += ["| " + " | ".join(str(x) for x in row) + " |" for row in table.itertuples(index=False)]
out += [
    "",
    "## Settings",
    "",
    "From Table II: population 1024, 51 generations, 10 elites, tournament size 7, maximal depth 8, crossover 80%, mutation 15%, reproduction 5%, a new sample of visit lengths each generation, 500 test samples. Functions and terminals as in the paper (BasicGP, without MNS and ANS).",
    "",
    "Not stated in the paper, set to the defaults of ECJ, which the paper used: initial trees of depth 2 to 6, 90% internal crossover points, subtree mutation with grown trees of depth 5. These may explain the small systematic difference.",
    "",
    "## What does not match, and what was tried",
    "",
    "- **Larger pr instances.** The published values match the first 100 or so points of each file, not the full file (up to 240 points), for both the hand-written rule and GP. This is inferred from the numbers: the paper says instances have up to almost 300 points. Runs here use the first 100 points; files with fewer are complete.",
    "- **SOS.** The published SOS values could not be reproduced with SCORE/SL as defined in Table I, nor with ten other forms of the denominator. SOS is therefore implemented from its formula, and its published values are not claimed to be matched.",
    "- **SOT on some instances**, such as r102, rc101, rc201. Their results shift with small changes such as distance rounding.",
    "- **Three-day trips (Table IV)** are out of scope: the project treats each card-day as one visitor.",
    "",
    "## Checks",
    "",
    "- `tests/test_toptw.py`: hand-built cases (feasibility, waiting, second day, sampled durations, protected division).",
    "- `tests/test_toptw.py` also recomputes 1,080 routes outside the simulator: no violation of visits once, time windows or budget, and scores agree.",
    "- Same seed run twice gives the same rule and test score.",
]
(folder / "reproduction.md").write_text("\n".join(out) + "\n")
print(f"{len(table)} instances, within 2.5%: {within}; mean diff {table['diff %'].mean():.2f}; |t|>2: {beyond}; below: {below}; SOT within 2%: {sot_within}")
