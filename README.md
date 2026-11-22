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
| `outputs/` | Small records and figures; large outputs (runs, test blocks) are not committed |

## Requirements

- Python 3.11
- `pip install -r requirements.txt` (exact versions in `requirements-lock.txt`)

## Data

`data/verona/veronacard_2019_opendata.csv` is included (CC BY 4.0). The 2022 and 2023 records and the tables of sites and walking times come from [itinerary-drl](https://github.com/smigliorini/itinerary-drl), whose data licence is not stated, so they are not copied here. Download them into `data/verona/` as shown in `data/README.md`.

## How to reproduce

Run from the repository root, at the tag of each step.

1. Prepare the data: `python3 src/preprocess.py`
2. Formal runs, 30 per method (about 9 hours with 12 processes): `python3 tools/run_formal.py`
3. Test evaluation (about 12 hours): `python3 analysis/reproduce.py`, then `python3 analysis/run_test.py --workers 12`
4. Analyses: `python3 analysis/rq1.py`, `rq2.py`, `rq3.py`, `others.py`; further checks with `analysis/further.py --scenario configs/further_*.json`
5. GP-single and the walking weight: `python3 addendum/single.py check | run | reproduce | tables | beta`

Every run checks the hashes of its code, data and settings, and stops if they differ from those recorded.

## Frozen snapshots and the report

| Tag | What it fixes | Report |
|---|---|---|
| `step5-formal` | Code and settings of the 90 formal runs and the tuned rule | Chapters 8 and 9 |
| `step6-formal` | Test evaluation, the RQ1–RQ3 analyses, other years and further checks | Chapter 10 (10.1–10.4) |
| `step6-single` | GP-single and the walking-weight re-scoring | Sections 10.4 and 10.5 |
