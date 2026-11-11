"""
One evaluation in Step 6 (Step 6 plan, section 5)
=================================================

A rule on one day and replicate, simulated once by the frozen evaluate(), then
scored at every alpha of the grid from the same simulation: a rule's choices
never read alpha, so routes, stops and G_hat are the same for every alpha and
only R_v changes. R_v(alpha) is computed with the frozen route_value() and
relative_value() from the parts of the new route and of the replay baseline of
the same date, so that at alpha = 0.25 it is the same computation as the
search's. That is checked in every evaluation, exactly: every visitor's R_v,
the mean (taken with the frozen scoring.summary, the same Series.mean() as
evaluate()) and the grouped table (made by the frozen grouped()). A difference
stops the run.

Saved for every rule, day and replicate: the mean R_v at each alpha and G_hat;
the grouped table at alpha = 0.25, with, for each other alpha, the sum of R_v
and the visitors with R_v >= 0.8, 0.85 and 0.9 per group; each site's highest
hourly C and its hour. For a knee rule also the per-visitor rows with the six
route parts and R_v at every alpha, the site x hour table of C and, for a GP
knee, the decision log.

A day with no evaluated visitor has no mean R_v; it is refused before it is
simulated (requirement 12), never caught after an error.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import evaluate, grouped, histogram_bin, hourly_peaks, visitor_rows  # noqa: E402
from scoring import relative_value, route_value, summary  # noqa: E402

ALPHAS = (0.0, 0.25, 0.5, 1.0)
MAIN_ALPHA = 0.25
THETAS = (0.8, 0.85, 0.9)
GROUP_KEYS = ["follows", "arrival group", "budget group", "stop group"]


def has_evaluated_visitors(day):
    return bool(day.visitors["evaluated"].any())


def label(alpha):
    return f"{alpha:g}"


def rescored(v, b, alpha, beta):
    """The visitors' table v of an evaluation, scored at another alpha against the replay baseline b."""
    out = v.copy()
    out["V"] = route_value(v["A"], v["C"], v["T"], alpha, beta)
    out["V observed"] = route_value(b["A"], b["C"], b["T"], alpha, beta)
    out["R_v"] = relative_value(out["V"], out["V observed"], b["A"])
    return out


def at_least(r, theta):
    """R_v >= theta, decided by the same bin edges as the grouped histograms."""
    return histogram_bin(r) >= round(theta * 20)


def score(rule, date, replicate, days, beta=0.5, stop_limit=True, knee=False):
    """Every output of one evaluation, as a dict of small tables and numbers."""
    day, observed = days.get(date)
    if not has_evaluated_visitors(day):
        raise ValueError(f"{date}: no evaluated visitor, so no mean R_v; this day is left out (plan, section 11)")
    result = evaluate(rule, date, replicate, days, log=knee, alpha=MAIN_ALPHA, beta=beta, stop_limit=stop_limit)
    v = result.visitors
    b = observed.baseline.loc[v.index]
    rows = visitor_rows(day, result)
    main = grouped(rows)

    means, groups_extra, r_by_alpha = {}, [], {}
    for alpha in ALPHAS:
        va = rescored(v, b, alpha, beta)
        r_by_alpha[alpha] = va["R_v"].to_numpy()
        means[alpha] = float(summary(va, alpha, beta)["all"]["mean R_v"])
        if alpha == MAIN_ALPHA:
            if not np.array_equal(va["R_v"].to_numpy(), v["R_v"].to_numpy()):
                raise AssertionError(f"{date} r{replicate}: R_v at alpha {alpha} differs from evaluate()")
            if means[alpha] != result.mean_r:
                raise AssertionError(f"{date} r{replicate}: mean R_v at alpha {alpha} differs from evaluate()")
            if not grouped(rows.assign(R_v=va["R_v"].to_numpy())).equals(main):
                raise AssertionError(f"{date} r{replicate}: grouped table at alpha {alpha} differs")
            continue
        extra = rows[[]].assign(**{"sum R_v": va["R_v"].to_numpy()},
                                **{f"R_v >= {t:g}": at_least(va["R_v"], t).astype(int) for t in THETAS})
        keys = main_keys(rows)
        extra = pd.concat([keys, extra], axis=1).groupby(GROUP_KEYS).sum().reset_index()
        if not np.array_equal(extra[GROUP_KEYS].to_numpy(int), main[GROUP_KEYS].to_numpy(int)):
            raise AssertionError(f"{date} r{replicate}: the groups at alpha {alpha} are not those of grouped()")
        groups_extra.append(extra.drop(columns=GROUP_KEYS).rename(columns=lambda c: f"{c} (alpha {label(alpha)})"))
    groups = pd.concat([main] + groups_extra, axis=1)

    out = {"objectives": {**{f"mean R_v (alpha {label(a)})": means[a] for a in ALPHAS}, "G_hat": result.g_hat},
           "groups": groups, "peaks": hourly_peaks(day, result.crowding)}
    if knee:
        detail = rows.assign(**{"A": v["A"].to_numpy(), "C": v["C"].to_numpy(), "T": v["T"].to_numpy(),
                                "A observed": b["A"].to_numpy(), "C observed": b["C"].to_numpy(),
                                "T observed": b["T"].to_numpy()},
                             **{f"R_v (alpha {label(a)})": r_by_alpha[a] for a in ALPHAS})
        out["visitors"] = detail
        out["hourly"] = pd.DataFrame(result.crowding, columns=[f"hour {h}" for h in range(24)]).assign(
            site=day.site_ids)
        if isinstance(rule, list):
            out["decisions"] = pd.DataFrame(result.run.decisions)
    return out


def main_keys(rows):
    """The group keys of each visitor, made as grouped() makes them: the main table is grouped() itself,
    and its groups are checked to be the same as these."""
    from evaluate import ARRIVAL_EDGES, STOP_GROUPS, budget_cuts
    low, high = budget_cuts()
    return pd.DataFrame({"follows": rows["follows"].to_numpy(),
                         "arrival group": np.searchsorted(ARRIVAL_EDGES, rows["arrival"], side="right"),
                         "budget group": np.searchsorted([low, high], rows["budget"], side="right"),
                         "stop group": np.minimum(rows["observed stops"].to_numpy(), STOP_GROUPS[-1])},
                        index=rows.index)
