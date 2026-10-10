# tourist-fairness-gphh

**Learning the Fairness Trade-off in Tourist Trip Design: A Genetic Programming Hyper-Heuristic**

Leon Cai, Master of Artificial Intelligence capstone project, Victoria University of Wellington. Supervisor: Dr Fangfang Zhang.

## The problem

Many visitors follow similar day plans, so they reach the same attractions at the same time while others stay quiet. Recommending "the next best place" to everyone only moves the crowd. This project evolves a readable rule that, each time a visitor finishes a stop, picks their next stop and learns from the current state how to balance spreading visits more evenly across attractions against visitors' satisfaction.

## Two objectives

**Visitors: satisfaction (a proxy).** Following Expectancy Disconfirmation Theory (Oliver 1980), the route a visitor actually took stands for what they expected and the suggested route for what they get:

$$R_v = \frac{V(r_v^N)}{V(r_v^O)}, \qquad V(r) = \sum_{i \in r} A_i - \alpha \sum_{i \in r} C_{i,t_i} - \beta\, T(r)$$

where $A_i$ is how popular attraction $i$ is (past visitors relative to the most visited one), $C_{i,t_i}$ how crowded it is when the visitor arrives (people inside relative to its usual peak), and $T(r)$ the walking time. A suggestion is acceptable if $R_v \ge \theta$ (e.g. 0.8).

**Attractions: fairness.** The Gini index of each attraction's visits $x_i$ over the day ($N$ attractions, mean $\mu$), also reported normalised so that the maximum is 1:

$$G = \frac{\sum_{i<j} |x_i - x_j|}{N^2 \mu}, \qquad \hat{G} = \frac{G}{1 - 1/N}$$

Equal counts are not the goal: a lower Gini is better, but more appealing attractions can stay busier.

## Method

A genetic programming hyper-heuristic (GPHH) evolves the rule. Each simulated day replays a real day of visitors. Each visitor keeps their real first stop, start time and time budget; after each visit the rule scores the attractions still reachable and sends the visitor to the highest. The evolved rule is compared with hand-written rules and with rules that keep one fixed trade-off.

## Data

VeronaCard, Verona's city pass: every attraction entry with its time, for 2019 (open data, CC BY 4.0) and 2022 (released with Dalla Vecchia et al. 2024, MIT). See [`data/README.md`](data/README.md) for files, sources, licences and summary numbers.

## Status (October 2026)

- [x] Problem definition and the two objectives agreed with the supervisor
- [x] Data collected: VeronaCard 2019 and 2022, attraction list, walking times
- [ ] Data exploration: data quality, 2019 vs 2022, baseline Gini per day, sanity checks of the objectives
- [ ] Simulator that replays a real day, with hand-written baseline rules
- [ ] GPHH implementation and experiments
- [ ] Planned extension: capacity and environmental limits (send visitors only where there is spare capacity)

## Repository layout

| Folder | Content |
|---|---|
| `data/` | Datasets and their sources (`data/README.md`) |
| `src/` | Code (to come) |
| `tests/` | Tests (to come) |
| `configs/` | Experiment settings (to come) |
| `outputs/` | Results (to come) |
