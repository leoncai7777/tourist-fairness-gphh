# Reproduction of Mei and Zhang (2018), BasicGP

Step 1 of the project: re-implement the single-tourist GPHH and check its results on the TOPTW instances against Table III of the paper (one-day trips).

## Conclusion

The independent DEAP implementation achieved broadly comparable BasicGP performance on the reconstructed one-day benchmarks, with a mean relative difference of -0.60% across 40 instances. Exact numerical reproduction is not claimed, because some implementation details and the construction of the larger instances remain unresolved.

## Result

- Main evidence, c, r and rc instances: 30 instances, mean difference -0.42%, range -3.33% to 1.41%; within 1%: 17, within 2.5%: 29; below the paper on 21; standardized difference above 2 in size on 7
- Supplementary, pr instances (construction unconfirmed): 10 instances, mean difference -1.13%, range -3.02% to -0.16%; within 1%: 5, within 2.5%: 9; below the paper on 10; standardized difference above 2 in size on 3
- All statistics are computed before rounding. Each instance has 30 runs with different seeds; 1,200 runs in all. Test score: the same 500 sampled instances for every run.
- The standardized difference divides the difference in means by the standard error from the reported means and standard deviations. It is descriptive: it is not the Wilcoxon test of the paper and is not corrected for 40 comparisons.
- Hand-written SOT, within 2% of the paper on 31 of 40 instances.

## Comparison

| instance | points | ours | sd | paper | paper sd | diff % | std diff | SOT ours | SOT paper | SOT diff % |
|---|---|---|---|---|---|---|---|---|---|---|
| c101 | 100 | 294.61 | 10.17 | 292.44 | 10.25 | 0.74 | 0.82 | 159.54 | 159.74 | -0.13 |
| c102 | 100 | 327.63 | 13.21 | 338.90 | 5.68 | -3.33 | -4.29 | 183.14 | 183.52 | -0.21 |
| c103 | 100 | 373.48 | 10.97 | 375.89 | 8.26 | -0.64 | -0.96 | 245.16 | 245.84 | -0.28 |
| c104 | 100 | 393.07 | 4.75 | 387.61 | 6.43 | 1.41 | 3.74 | 256.38 | 257.40 | -0.40 |
| c105 | 100 | 327.22 | 4.82 | 323.02 | 8.56 | 1.30 | 2.34 | 170.84 | 171.42 | -0.34 |
| r101 | 100 | 187.37 | 5.94 | 189.52 | 4.25 | -1.13 | -1.61 | 78.96 | 74.38 | 6.16 |
| r102 | 100 | 269.11 | 6.37 | 273.66 | 6.82 | -1.66 | -2.67 | 123.37 | 175.03 | -29.51 |
| r103 | 100 | 282.35 | 4.48 | 281.19 | 6.18 | 0.41 | 0.83 | 93.03 | 93.10 | -0.07 |
| r104 | 100 | 286.99 | 11.53 | 288.08 | 11.90 | -0.38 | -0.36 | 128.00 | 128.04 | -0.03 |
| r105 | 100 | 222.25 | 8.48 | 224.94 | 6.02 | -1.19 | -1.41 | 88.00 | 88.00 | 0.00 |
| rc101 | 100 | 200.35 | 11.56 | 203.83 | 9.05 | -1.71 | -1.30 | 88.00 | 98.00 | -10.20 |
| rc102 | 100 | 232.97 | 12.33 | 233.54 | 7.20 | -0.25 | -0.22 | 95.38 | 102.29 | -6.76 |
| rc103 | 100 | 241.34 | 9.39 | 244.39 | 5.31 | -1.25 | -1.55 | 100.34 | 100.56 | -0.22 |
| rc104 | 100 | 250.72 | 8.40 | 253.77 | 8.32 | -1.20 | -1.41 | 130.27 | 129.68 | 0.45 |
| rc105 | 100 | 219.49 | 12.63 | 217.59 | 16.48 | 0.87 | 0.50 | 99.35 | 104.37 | -4.81 |
| c201 | 100 | 814.13 | 24.52 | 822.96 | 10.20 | -1.07 | -1.82 | 181.66 | 182.16 | -0.27 |
| c202 | 100 | 865.46 | 9.77 | 867.84 | 12.59 | -0.27 | -0.82 | 183.62 | 183.50 | 0.07 |
| c203 | 100 | 911.34 | 11.64 | 916.61 | 11.48 | -0.57 | -1.76 | 232.72 | 233.66 | -0.40 |
| c204 | 100 | 926.32 | 14.58 | 933.20 | 9.23 | -0.74 | -2.18 | 197.66 | 197.60 | 0.03 |
| c205 | 100 | 856.72 | 9.86 | 861.22 | 9.63 | -0.52 | -1.79 | 188.08 | 188.40 | -0.17 |
| r201 | 100 | 729.70 | 15.99 | 720.94 | 18.50 | 1.22 | 1.96 | 152.35 | 144.69 | 5.29 |
| r202 | 100 | 844.69 | 10.92 | 840.37 | 15.17 | 0.51 | 1.26 | 257.17 | 257.08 | 0.03 |
| r203 | 100 | 922.92 | 14.57 | 923.57 | 9.65 | -0.07 | -0.20 | 257.17 | 257.08 | 0.03 |
| r204 | 100 | 1008.20 | 21.87 | 1004.91 | 16.66 | 0.33 | 0.65 | 307.26 | 307.24 | 0.01 |
| r205 | 100 | 824.10 | 18.23 | 831.81 | 12.11 | -0.93 | -1.93 | 245.73 | 240.40 | 2.22 |
| rc201 | 100 | 728.47 | 19.25 | 729.19 | 20.24 | -0.10 | -0.14 | 104.07 | 117.49 | -11.42 |
| rc202 | 100 | 865.32 | 17.54 | 859.00 | 29.34 | 0.74 | 1.01 | 188.39 | 186.40 | 1.07 |
| rc203 | 100 | 914.62 | 20.47 | 926.51 | 10.38 | -1.28 | -2.84 | 189.89 | 188.66 | 0.65 |
| rc204 | 100 | 1036.66 | 20.74 | 1040.73 | 13.96 | -0.39 | -0.89 | 399.89 | 400.14 | -0.06 |
| rc205 | 100 | 783.44 | 22.95 | 795.34 | 17.91 | -1.50 | -2.24 | 147.54 | 147.50 | 0.03 |
| pr01 | 48 | 284.45 | 11.88 | 287.99 | 3.50 | -1.23 | -1.56 | 88.49 | 88.51 | -0.02 |
| pr02 | 96 | 377.06 | 10.05 | 383.04 | 3.09 | -1.56 | -3.11 | 100.39 | 99.83 | 0.56 |
| pr03 | 100 | 305.83 | 5.87 | 310.41 | 5.40 | -1.48 | -3.15 | 179.80 | 180.06 | -0.14 |
| pr04 | 100 | 340.99 | 6.88 | 341.93 | 8.13 | -0.27 | -0.48 | 124.20 | 125.63 | -1.14 |
| pr05 | 100 | 390.11 | 14.62 | 391.88 | 15.63 | -0.45 | -0.45 | 142.61 | 144.66 | -1.42 |
| pr11 | 48 | 306.71 | 18.55 | 311.50 | 12.20 | -1.54 | -1.18 | 162.67 | 168.94 | -3.71 |
| pr12 | 96 | 400.19 | 9.13 | 402.91 | 6.09 | -0.68 | -1.36 | 187.75 | 187.60 | 0.08 |
| pr13 | 100 | 362.61 | 13.69 | 373.90 | 11.58 | -3.02 | -3.45 | 182.90 | 183.18 | -0.15 |
| pr14 | 100 | 401.98 | 12.51 | 405.52 | 18.82 | -0.87 | -0.86 | 225.67 | 226.82 | -0.51 |
| pr15 | 100 | 457.84 | 16.07 | 458.58 | 10.36 | -0.16 | -0.21 | 354.00 | 355.43 | -0.40 |

## Settings

From Table II of the paper: population 1024, 51 generations, 10 elites, tournament size 7, maximal depth 8, crossover 80%, mutation 15%, reproduction 5%, a new sample of visit lengths each generation, 500 test samples. Functions and terminals as in the paper (BasicGP, without MNS and ANS).

Not stated in the paper, and set here as a declared DEAP variant modelled on ECJ's Koza defaults, the paper having used ECJ: initial trees ramped half-and-half, with a target depth drawn from 2 to 6 (the grow half may stop earlier, down to a single terminal); 90% internal crossover points, never the root, and single-node trees are not crossed; subtree mutation with trees grown to at most depth 5, where any node may stop early. ECJ's own code and the paper's parameter file were not available, so these operators are not claimed to be identical. A crossover child over the depth limit is replaced by one of its two parents, chosen at random; a mutation child over the limit reverts to its parent. The probabilities 80%, 15% and 5% are per breeding operation: a crossover adds two offspring, so more than 80% of the offspring come from crossover.

Operators in practice: mutation left the tree unchanged in 11.7% of cases and crossover in 18.9% (depth limit, or parents that are identical or too small). Priorities that were not finite: 0 over all runs.

Implementation conventions not specified by the paper: negative sampled durations are set to 0; ties in priority go to the earliest point; non-finite priorities are set to 0 and counted; distances are rounded down to one decimal (Solomon) or two (Cordeau). Algorithm 2 is implemented following the definitions in the text, with one current time, rather than line by line.

## What does not match, and what remains unresolved

- **Instance construction.** Every run uses at most the first 100 candidate points after the depot, which truncates only pr03, pr04, pr05, pr13, pr14, pr15. A first-100-candidate reconstruction produced closer numerical agreement than the full files, for both SOT and GP. This is inferred from the numbers; the paper states that instances have almost 300 points, and the authors' instances or parser were not available.
- **SOS.** SOS was implemented as SCORE/(TC-TA), following the published definition. Its reported numerical results could not be reproduced under the documented reconstruction. The discrepancy remains unresolved; agreement with the published SOS results is not claimed.
- **SOT** differs by more than 2% on 9 instances, up to 29.5%. SOT involves no GP training, so these differences point to details of the instances, ties or sampling that are not documented.
- **Three-day trips (Table IV)** are out of scope: the project treats each card-day as one visitor.

## Checks

Checked by `src/reproduction_report.py` each time this record is generated:

- The results cover exactly the 40 instances of Table III, each with 30 runs, seeds 0 to 29 and no duplicates.
- Every run used max_points = 100, test seed 20180101 and 500 test samples; each instance file still has the hash recorded with its runs.
- All runs were made by one code version, 3b25478088. The current code is 80781e6702.
- `regression_check.csv` holds 8 runs (c103, c201, pr13, r101; seeds 0, 1) made with code 80781e6702. They give the same rules, training and test scores, tree sizes and operator counts as the matching runs above.

Other checks, not run by this script:

- `tests/test_toptw.py`, run with `python3 -m pytest tests -q`: hand-built cases; ties and non-finite priorities; fresh fitness for every individual each generation; grown and initial tree shapes and the depth limit; and a recomputation of 1,080 routes outside the simulator, in which every selection satisfied the paper's expected-duration feasibility conditions, each point was visited once and scores agreed. Realised sampled durations may exceed the nominal return budget, as in the paper.
- Earlier record, no log kept: on 2026-11-10, with code 3b25478088, a run repeated with the same seed gave the same rule and test score.
