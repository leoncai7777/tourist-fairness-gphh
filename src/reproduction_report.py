"""
Reproduction record for Step 1
==============================

Compares the single-tourist GPHH (src/single_gphh.py) with Table III of
Mei and Zhang (2018), one-day trips, and writes the record of what matches,
what does not, and how it was checked.

Run:  python3 src/reproduction_report.py   (from the repository root)
Reads outputs/toptw/all40.csv and the TOPTW instances in data/toptw/.
Writes outputs/toptw/reproduction.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from single_gphh import fingerprint
from toptw import TEST_SEED, load_instance, sot, test_score

FOLDER = Path("outputs/toptw")
MAX_POINTS = 100    # every run uses at most the first 100 candidate points after the depot
TEST_SAMPLES = 500
HASHES = {"input_sha1": str, "code_sha1": str}  # read as text: "80781e6702" would parse as a number
SAME_RUN = ["train", "test", "height", "nodes", "non_finite", "mutation_unchanged", "crossover_unchanged", "rule"]

# ---------------------------------------------------------------------- #
# 1. Published values: instance, SOS, SOT, BasicGP mean and sd (Table III)
# ---------------------------------------------------------------------- #

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


def published_table():
    rows = [line.split() for line in TABLE_III.splitlines()]
    paper = pd.DataFrame(rows, columns=["instance", "sos", "sot", "gp", "gp_sd"])
    return paper.set_index("instance").astype(float)


# ---------------------------------------------------------------------- #
# 2. Comparison
# ---------------------------------------------------------------------- #

def compare(paper, results):
    """One row per instance: our GP mean and sd against the paper's, and the hand-written SOT.
    The standardized difference uses the standard error from both means and sds (30 runs each)."""
    rows = []
    for name, target in paper.iterrows():
        runs = results[results.instance == name]
        if runs.empty:
            continue
        instance = load_instance(f"data/toptw/{name}.txt", MAX_POINTS)
        assert runs.points.eq(len(instance.score) - 1).all()
        mean, sd, n = runs.test.mean(), runs.test.std(), len(runs)
        standard_error = np.sqrt(sd ** 2 / n + target.gp_sd ** 2 / 30)
        hand = test_score(instance, sot)
        rows.append({"instance": name, "points": int(runs.points.iloc[0]), "ours": mean, "sd": sd,
                     "paper": target.gp, "paper sd": target.gp_sd,
                     "diff %": (mean - target.gp) / target.gp * 100,
                     "std diff": (mean - target.gp) / standard_error,
                     "SOT ours": hand, "SOT paper": target.sot,
                     "SOT diff %": (hand - target.sot) / target.sot * 100})
    return pd.DataFrame(rows)


def summary(part, label):
    """One line on a group of instances. Statistics are computed before rounding."""
    d = part["diff %"]
    return (f"- {label}: {len(part)} instances, mean difference {d.mean():.2f}%, "
            f"range {d.min():.2f}% to {d.max():.2f}%; "
            f"within 1%: {(d.abs() <= 1).sum()}, within 2.5%: {(d.abs() <= 2.5).sum()}; "
            f"below the paper on {(d < 0).sum()}; "
            f"standardized difference above 2 in size on {(part['std diff'].abs() > 2).sum()}")


def markdown_table(table):
    """The comparison table, numbers shown with two decimals."""
    shown = table.copy()
    for column in shown.columns[2:]:
        shown[column] = shown[column].map(lambda x: f"{x:.2f}")
    lines = ["| " + " | ".join(shown.columns) + " |", "|" + "---|" * len(shown.columns)]
    for row in shown.itertuples(index=False):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return lines


# ---------------------------------------------------------------------- #
# 3. The record
# ---------------------------------------------------------------------- #

def record(table, results, main, supplement, truncated, checks):
    out = ["# Reproduction of Mei and Zhang (2018), BasicGP", ""]
    out += ["Step 1 of the project: re-implement the single-tourist GPHH and check its results "
            "on the TOPTW instances against Table III of the paper (one-day trips).", ""]

    out += ["## Conclusion", ""]
    out += [f"The independent DEAP implementation achieved broadly comparable BasicGP performance "
            f"on the reconstructed one-day benchmarks, with a mean relative difference of "
            f"{table['diff %'].mean():.2f}% across {len(table)} instances. "
            "Exact numerical reproduction is not claimed, because some implementation details "
            "and the construction of the larger instances remain unresolved.", ""]

    out += ["## Result", ""]
    out += [summary(main, "Main evidence, c, r and rc instances"),
            summary(supplement, "Supplementary, pr instances (construction unconfirmed)"),
            f"- All statistics are computed before rounding. Each instance has 30 runs with different seeds; "
            f"{len(results):,} runs in all. Test score: the same 500 sampled instances for every run.",
            "- The standardized difference divides the difference in means by the standard error from the "
            "reported means and standard deviations. It is descriptive: it is not the Wilcoxon test of the "
            "paper and is not corrected for 40 comparisons.",
            f"- Hand-written SOT, within 2% of the paper on {(table['SOT diff %'].abs() <= 2).sum()} "
            f"of {len(table)} instances.", ""]

    out += ["## Comparison", ""]
    out += markdown_table(table)

    out += ["", "## Settings", ""]
    out += ["From Table II of the paper: population 1024, 51 generations, 10 elites, tournament size 7, "
            "maximal depth 8, crossover 80%, mutation 15%, reproduction 5%, a new sample of visit lengths "
            "each generation, 500 test samples. Functions and terminals as in the paper (BasicGP, without "
            "MNS and ANS).", ""]
    out += ["Not stated in the paper, and set here as a declared DEAP variant modelled on ECJ's Koza defaults, "
            "the paper having used ECJ: initial trees ramped half-and-half, with a target depth drawn from 2 to 6 "
            "(the grow half may stop earlier, down to a single terminal); 90% internal crossover points, never "
            "the root, and single-node trees are not crossed; subtree mutation with trees grown to at most "
            "depth 5, where any node may stop early. ECJ's own code and the paper's parameter file were not "
            "available, so these operators are not claimed to be identical. A crossover child over the depth "
            "limit is replaced by one of its two parents, chosen at random; a mutation child over the limit "
            "reverts to its parent. The probabilities 80%, 15% and 5% are per breeding operation: a crossover "
            "adds two offspring, so more than 80% of the offspring come from crossover.", ""]
    out += [f"Operators in practice: mutation left the tree unchanged in {results.mutation_unchanged.mean():.1%} "
            f"of cases and crossover in {results.crossover_unchanged.mean():.1%} (depth limit, or parents that "
            f"are identical or too small). Priorities that were not finite: {int(results.non_finite.sum()):,} "
            f"over all runs.", ""]
    out += ["Implementation conventions not specified by the paper: negative sampled durations are set to 0; "
            "ties in priority go to the earliest point; non-finite priorities are set to 0 and counted; "
            "distances are rounded down to one decimal (Solomon) or two (Cordeau). Algorithm 2 is implemented "
            "following the definitions in the text, with one current time, rather than line by line.", ""]

    out += ["## What does not match, and what remains unresolved", ""]
    out += [f"- **Instance construction.** Every run uses at most the first {MAX_POINTS} candidate points "
            f"after the depot, which truncates only {', '.join(truncated)}. A first-{MAX_POINTS}-candidate "
            f"reconstruction produced closer numerical agreement than the full files, for both SOT and GP. "
            f"This is inferred from the numbers; the paper states that instances have almost 300 points, and "
            f"the authors' instances or parser were not available.",
            "- **SOS.** SOS was implemented as SCORE/(TC-TA), following the published definition. Its reported "
            "numerical results could not be reproduced under the documented reconstruction. The discrepancy "
            "remains unresolved; agreement with the published SOS results is not claimed.",
            f"- **SOT** differs by more than 2% on {(table['SOT diff %'].abs() > 2).sum()} instances, up to "
            f"{table['SOT diff %'].abs().max():.1f}%. SOT involves no GP training, so these differences point "
            f"to details of the instances, ties or sampling that are not documented.",
            "- **Three-day trips (Table IV)** are out of scope: the project treats each card-day as one visitor.",
            ""]

    out += ["## Checks", ""]
    out += ["Checked by `src/reproduction_report.py` each time this record is generated:", ""]
    out += [f"- {line}" for line in checks]
    out += ["", "Other checks, not run by this script:", ""]
    out += ["- `tests/test_toptw.py`, run with `python3 -m pytest tests -q`: hand-built cases; ties and "
            "non-finite priorities; fresh fitness for every individual each generation; grown and initial "
            "tree shapes and the depth limit; and a recomputation of 1,080 routes outside the simulator, in "
            "which every selection satisfied the paper's expected-duration feasibility conditions, each point "
            "was visited once and scores agreed. Realised sampled durations may exceed the nominal return "
            "budget, as in the paper.",
            "- Earlier record, no log kept: on 2026-11-10, with code 3b25478088, a run repeated with the same "
            "seed gave the same rule and test score."]
    return out


def verify(paper, results):
    """Checks on the results file before anything is written. Returns them as lines of the record."""
    expected = set(paper.index)
    assert set(results.instance) == expected, "the results must cover exactly the instances of Table III"
    assert not results.duplicated(["instance", "seed"]).any()
    for name, runs in results.groupby("instance"):
        assert sorted(runs.seed) == list(range(30)), name
    assert (results.max_points == MAX_POINTS).all()
    assert (results.test_seed == TEST_SEED).all() and (results.test_samples == TEST_SAMPLES).all()
    for name, runs in results.groupby("instance"):
        assert (runs.input_sha1 == fingerprint(f"data/toptw/{name}.txt")).all(), name
    assert results.code_sha1.nunique() == 1
    run_code = results.code_sha1.iloc[0]
    current_code = fingerprint("src/toptw.py", "src/single_gphh.py")

    regression = pd.read_csv(FOLDER / "regression_check.csv", dtype=HASHES).set_index(["instance", "seed"])
    recorded = results.set_index(["instance", "seed"]).loc[regression.index]
    for column in SAME_RUN:
        if column == "rule":
            assert (regression[column] == recorded[column]).all(), column
        else:
            assert np.allclose(regression[column], recorded[column], rtol=0, atol=1e-9), column
    regression_code = regression.code_sha1.iloc[0]
    assert (regression.code_sha1 == current_code).all(), "regression_check.csv must come from the current code"
    pairs = ", ".join(sorted(regression.index.get_level_values(0).unique()))
    seeds = sorted(regression.index.get_level_values(1).unique())

    return [f"The results cover exactly the {len(expected)} instances of Table III, each with 30 runs, "
            f"seeds 0 to 29 and no duplicates.",
            f"Every run used max_points = {MAX_POINTS}, test seed {TEST_SEED} and {TEST_SAMPLES} test samples; "
            f"each instance file still has the hash recorded with its runs.",
            f"All runs were made by one code version, {run_code}. The current code is {current_code}.",
            f"`regression_check.csv` holds {len(regression)} runs ({pairs}; seeds "
            f"{', '.join(str(s) for s in seeds)}) made with code {regression_code}. They give the same rules, "
            f"training and test scores, tree sizes and operator counts as the matching runs above."]


# ---------------------------------------------------------------------- #
# 4. Main
# ---------------------------------------------------------------------- #

def main():
    paper = published_table()
    results = pd.read_csv(FOLDER / "all40.csv", dtype=HASHES)
    checks = verify(paper, results)

    table = compare(paper, results)
    is_pr = table.instance.str.startswith("pr")
    main_part, supplement = table[~is_pr], table[is_pr]
    truncated = [name for name in paper.index if len(load_instance(f"data/toptw/{name}.txt").score) - 1 > MAX_POINTS]

    lines = record(table, results, main_part, supplement, truncated, checks)
    (FOLDER / "reproduction.md").write_text("\n".join(lines) + "\n")
    print(f"{len(table)} instances; mean {table['diff %'].mean():.3f}; "
          f"within 2.5%: {(table['diff %'].abs() <= 2.5).sum()}; below {(table['diff %'] < 0).sum()}")
    print("main", summary(main_part, "m"))
    print("pr", summary(supplement, "s"))


if __name__ == "__main__":
    main()
