# tourist-fairness-gphh

**Learning the Fairness Trade-off in Tourist Trip Design: A Genetic Programming Hyper-Heuristic**

Leon Cai, Master of Artificial Intelligence, Victoria University of Wellington. Supervisor: Dr Fangfang Zhang.

A next-stop rule for each visitor weighs the visitor's satisfaction against fairness among attractions (how evenly visits are spread). This project learns the rule with a genetic programming hyper-heuristic (GPHH), with a second evolved tree that sets the fairness weight from the state of the day, and tests it in a simulator of whole days built from Verona's city card records.

## Status (November 2026)

All experiments are done and frozen. The report is in its final draft and will be added here when it is submitted.

## Folders

| Folder | Content |
|---|---|
| `src/` | Data preparation, simulator, rules, GP search, evaluation, and the re-implementation of the single-tourist GPHH (Step 1) |
| `tools/` | Launchers for the pilot and the formal runs |
| `analysis/` | Test evaluation and the analyses of RQ1–RQ3 (Step 6) |
| `addendum/` | GP-single and the walking-weight re-scoring (addendum to Step 6) |
| `configs/` | Frozen settings of the search, the test evaluation and the further checks |
| `tests/` | Tests: `python3 -m pytest tests -q` |
| `data/` | VeronaCard 2019 and how to get the other files (`data/README.md`) |
| `outputs/` | Records of the checks and the pilot, and the data-exploration figures; large outputs (runs, test blocks) are not committed |
| `results/` | Frozen result tables and figures behind Chapter 10 of the report (see below) |

## Requirements

- Python 3.11
- `pip install -r requirements.txt` (exact versions in `requirements-lock.txt`)

## Data

`data/verona/veronacard_2019_opendata.csv` is included (CC BY 4.0). The 2022 and 2023 records and the tables of sites and walking times come from [itinerary-drl](https://github.com/smigliorini/itinerary-drl), whose data licence is not stated, so they are not copied here. Download them into `data/verona/` as shown in `data/README.md`.

## How to reproduce

Run from the repository root, at the tag of each step.

1. Re-implementation of the single-tourist GPHH (Step 1): put the TOPTW instances from the [KU Leuven orienteering page](https://www.mech.kuleuven.be/en/cib/op) into `data/toptw/`, then `python3 src/single_gphh.py --instances <the 40 instances listed in outputs/toptw/reproduction.md> --runs 30 --max-points 100 --out outputs/toptw/all40.csv` and `python3 src/reproduction_report.py`
2. Prepare the data: `python3 src/preprocess.py`
3. Formal runs, 30 per method (about 9 hours with 12 processes): `python3 tools/run_formal.py`
4. Test evaluation (about 12 hours): `python3 analysis/reproduce.py`, then `python3 analysis/run_test.py --workers 12`
5. Analyses: `python3 analysis/rq1.py`, `rq2.py`, `rq3.py`, `others.py`; further checks with `analysis/further.py --scenario configs/further_*.json`
6. GP-single and the walking weight: `python3 addendum/single.py check | run | reproduce | tables | beta`

Every run checks the hashes of its code, data and settings, and stops if they differ from those recorded. Comments in the code refer to the design documents (`search_design.md`, `simulator_design.md`, `implementation_protocol.md`, the Step 6 plan); they are not in this repository, and their content is in Chapters 8 and 9 and Appendix B of the report.

## Frozen snapshots and the report

| Tag | What it fixes | Report |
|---|---|---|
| `step5-formal` | Code and settings of the 90 formal runs and the tuned rule | Chapters 8 and 9 |
| `step6-formal` | Test evaluation, the RQ1–RQ3 analyses, other years and further checks | Chapter 10 (10.1–10.4) |
| `step6-single` | GP-single and the walking-weight re-scoring | Sections 10.4 and 10.5 |

## Results

| Folder | Report |
|---|---|
| `results/2019/rq1`, `rq2`, `rq3`, `others` | Sections 10.1–10.3; Appendices C.2–C.5 |
| `results/2022`, `results/2023` | Section 10.4, other years |
| `results/further/` | Section 10.4, further checks and the constant $h_2$ |
| `results/walking_weight/` | Section 10.4, walking weight |
| `results/gp_single/` | Section 10.5 |

The figures in the report were redrawn in one style from these tables, without changing their data.

## Licence

Code: MIT (`LICENSE`). Data: see `data/README.md`.
