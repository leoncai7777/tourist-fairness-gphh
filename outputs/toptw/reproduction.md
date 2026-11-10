# Reproduction of Mei and Zhang (2018), BasicGP

Step 1 of the project: re-implement the single-tourist GPHH and check that its results on the TOPTW instances are comparable to those published. One-day trips, Table III of the paper.

## Result

- Instances run: 40 of 40, 30 runs each; 40 are within 2.5% of the published BasicGP mean.
- Mean difference -0.60%, range -2.5% to 1.1%; ours is lower on 31 instances.
- The difference exceeds 2 standard errors on 10 instances (rough test from published mean and sd, 30 runs each), so a small part of it is systematic.
- Hand-written SOT rule: within 2% of the paper on 31 instances. SOT is a deterministic-looking greedy rule and is sensitive to small changes.

## Comparison

| instance | runs | ours | sd | paper | paper sd | diff % | t | SOT ours | SOT paper | SOT diff % |
|---|---|---|---|---|---|---|---|---|---|---|
| c101 | 30 | 293.9 | 9.63 | 292.44 | 10.25 | 0.5 | 0.57 | 159.54 | 159.74 | -0.1 |
| c102 | 30 | 333.21 | 10.27 | 338.9 | 5.68 | -1.7 | -2.65 | 183.14 | 183.52 | -0.2 |
| c103 | 30 | 374.23 | 10.55 | 375.89 | 8.26 | -0.4 | -0.68 | 245.16 | 245.84 | -0.3 |
| c104 | 30 | 391.96 | 6.51 | 387.61 | 6.43 | 1.1 | 2.6 | 256.38 | 257.4 | -0.4 |
| c105 | 30 | 326.34 | 7.26 | 323.02 | 8.56 | 1.0 | 1.62 | 170.84 | 171.42 | -0.3 |
| r101 | 30 | 184.75 | 9.64 | 189.52 | 4.25 | -2.5 | -2.48 | 78.96 | 74.38 | 6.2 |
| r102 | 30 | 272.71 | 3.83 | 273.66 | 6.82 | -0.3 | -0.67 | 123.37 | 175.03 | -29.5 |
| r103 | 30 | 282.07 | 5.27 | 281.19 | 6.18 | 0.3 | 0.59 | 93.03 | 93.1 | -0.1 |
| r104 | 30 | 287.45 | 6.49 | 288.08 | 11.9 | -0.2 | -0.25 | 128.0 | 128.04 | -0.0 |
| r105 | 30 | 221.37 | 8.38 | 224.94 | 6.02 | -1.6 | -1.9 | 88.0 | 88.0 | 0.0 |
| rc101 | 30 | 200.74 | 10.82 | 203.83 | 9.05 | -1.5 | -1.2 | 88.0 | 98.0 | -10.2 |
| rc102 | 30 | 232.34 | 8.74 | 233.54 | 7.2 | -0.5 | -0.58 | 95.38 | 102.29 | -6.8 |
| rc103 | 30 | 239.38 | 10.98 | 244.39 | 5.31 | -2.1 | -2.25 | 100.34 | 100.56 | -0.2 |
| rc104 | 30 | 249.87 | 12.09 | 253.77 | 8.32 | -1.5 | -1.46 | 130.27 | 129.68 | 0.5 |
| rc105 | 30 | 217.07 | 14.41 | 217.59 | 16.48 | -0.2 | -0.13 | 99.35 | 104.37 | -4.8 |
| c201 | 30 | 821.88 | 12.0 | 822.96 | 10.2 | -0.1 | -0.37 | 181.66 | 182.16 | -0.3 |
| c202 | 30 | 861.71 | 11.37 | 867.84 | 12.59 | -0.7 | -1.98 | 183.62 | 183.5 | 0.1 |
| c203 | 30 | 908.27 | 14.02 | 916.61 | 11.48 | -0.9 | -2.52 | 232.72 | 233.66 | -0.4 |
| c204 | 30 | 925.18 | 10.47 | 933.2 | 9.23 | -0.9 | -3.15 | 197.66 | 197.6 | 0.0 |
| c205 | 30 | 861.28 | 7.78 | 861.22 | 9.63 | 0.0 | 0.03 | 188.08 | 188.4 | -0.2 |
| r201 | 30 | 724.92 | 12.56 | 720.94 | 18.5 | 0.6 | 0.98 | 152.35 | 144.69 | 5.3 |
| r202 | 30 | 837.62 | 16.65 | 840.37 | 15.17 | -0.3 | -0.67 | 257.17 | 257.08 | 0.0 |
| r203 | 30 | 926.65 | 24.39 | 923.57 | 9.65 | 0.3 | 0.64 | 257.17 | 257.08 | 0.0 |
| r204 | 30 | 1005.41 | 20.79 | 1004.91 | 16.66 | 0.1 | 0.1 | 307.26 | 307.24 | 0.0 |
| r205 | 30 | 830.06 | 14.31 | 831.81 | 12.11 | -0.2 | -0.51 | 245.73 | 240.4 | 2.2 |
| rc201 | 30 | 733.15 | 18.66 | 729.19 | 20.24 | 0.5 | 0.79 | 104.07 | 117.49 | -11.4 |
| rc202 | 30 | 858.23 | 22.52 | 859.0 | 29.34 | -0.1 | -0.11 | 188.39 | 186.4 | 1.1 |
| rc203 | 30 | 920.65 | 9.44 | 926.51 | 10.38 | -0.6 | -2.29 | 189.89 | 188.66 | 0.7 |
| rc204 | 30 | 1034.15 | 23.21 | 1040.73 | 13.96 | -0.6 | -1.33 | 399.89 | 400.14 | -0.1 |
| rc205 | 30 | 789.05 | 16.86 | 795.34 | 17.91 | -0.8 | -1.4 | 147.54 | 147.5 | 0.0 |
| pr01 | 30 | 287.27 | 6.5 | 287.99 | 3.5 | -0.2 | -0.53 | 88.49 | 88.51 | -0.0 |
| pr02 | 30 | 378.77 | 5.92 | 383.04 | 3.09 | -1.1 | -3.5 | 100.39 | 99.83 | 0.6 |
| pr03 | 30 | 306.55 | 5.35 | 310.41 | 5.4 | -1.2 | -2.78 | 179.8 | 180.06 | -0.1 |
| pr04 | 30 | 340.61 | 10.96 | 341.93 | 8.13 | -0.4 | -0.53 | 124.2 | 125.63 | -1.1 |
| pr05 | 30 | 387.64 | 19.73 | 391.88 | 15.63 | -1.1 | -0.92 | 142.61 | 144.66 | -1.4 |
| pr11 | 30 | 304.03 | 16.15 | 311.5 | 12.2 | -2.4 | -2.02 | 162.67 | 168.94 | -3.7 |
| pr12 | 30 | 401.36 | 7.13 | 402.91 | 6.09 | -0.4 | -0.9 | 187.75 | 187.6 | 0.1 |
| pr13 | 30 | 367.73 | 13.43 | 373.9 | 11.58 | -1.6 | -1.91 | 182.9 | 183.18 | -0.2 |
| pr14 | 30 | 396.55 | 23.51 | 405.52 | 18.82 | -2.2 | -1.63 | 225.67 | 226.82 | -0.5 |
| pr15 | 30 | 457.98 | 11.65 | 458.58 | 10.36 | -0.1 | -0.21 | 354.0 | 355.43 | -0.4 |

## Settings

From Table II: population 1024, 51 generations, 10 elites, tournament size 7, maximal depth 8, crossover 80%, mutation 15%, reproduction 5%, a new sample of visit lengths each generation, 500 test samples. Functions and terminals as in the paper (BasicGP, without MNS and ANS).

Not stated in the paper, set to the defaults of ECJ, which the paper used: initial trees of depth 2 to 6, 90% internal crossover points, subtree mutation with grown trees of depth 5. These may explain the small systematic difference.

## What does not match, and what was tried

- **Larger pr instances.** The published values match the first 100 or so points of each file, not the full file (up to 240 points), for both the hand-written rule and GP. This is inferred from the numbers: the paper says instances have up to almost 300 points. Runs here use the first 100 points; files with fewer are complete.
- **SOS.** The published SOS values could not be reproduced with SCORE/SL as defined in Table I, nor with ten other forms of the denominator. SOS is therefore implemented from its formula, and its published values are not claimed to be matched.
- **SOT on some instances**, such as r102, rc101, rc201. Their results shift with small changes such as distance rounding.
- **Three-day trips (Table IV)** are out of scope: the project treats each card-day as one visitor.

## Checks

- `tests/test_toptw.py`: hand-built cases (feasibility, waiting, second day, sampled durations, protected division).
- `tests/test_toptw.py` also recomputes 1,080 routes outside the simulator: no violation of visits once, time windows or budget, and scores agree.
- Same seed run twice gives the same rule and test score.
