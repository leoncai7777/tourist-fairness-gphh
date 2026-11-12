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
evaluate()); the grouped table at 0.25 is the frozen grouped() of those rows. A
difference stops the run. At the other alphas the mean is the same
Series.mean(), without the rest of summary(), which only the main alpha needs.

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


def r_at(new, observed, alpha, beta):
    """Each visitor's R_v at alpha, from the parts (A, C, T) of the new route and of the replay baseline, as
    arrays, by the frozen route_value() and relative_value(): element by element the same arithmetic as
    score_day(), which applies them to pandas columns."""
    return relative_value(route_value(*new, alpha, beta), route_value(*observed, alpha, beta), observed[0])


def at_least(r, theta):
    """R_v >= theta, decided by the same bin edges as the grouped histograms."""
    return histogram_bin(r) >= round(theta * 20)


def score(rule, date, replicate, days, beta=0.5, stop_limit=True, knee=False, watch=None):
    """Every output of one evaluation, as a dict of small tables and numbers. For a GP knee, watch may map
    names to other forms of it (its tidy and display forms): the full rule acts and logs, and at every
    decision each form's choice is computed as a shadow (forms.Shadow); the log gets a column "<name> agrees"
    for each. The decision log is written with card ids and site ids (plan, section 2), not day indices."""
    day, observed = days.get(date)
    if not has_evaluated_visitors(day):
        raise ValueError(f"{date}: no evaluated visitor, so no mean R_v; this day is left out (plan, section 11)")
    shadow = None
    if watch:
        if not (knee and isinstance(rule, list)):
            raise ValueError("only a GP knee is watched")
        from forms import Shadow
        shadow = Shadow(rule, watch)
    result = evaluate(shadow or rule, date, replicate, days, log=knee, alpha=MAIN_ALPHA, beta=beta,
                      stop_limit=stop_limit)
    v = result.visitors
    b = observed.baseline.loc[v.index]
    rows = visitor_rows(day, result)
    main = grouped(rows)

    new = tuple(v[c].to_numpy() for c in ("A", "C", "T"))
    old = tuple(b[c].to_numpy() for c in ("A", "C", "T"))
    r_by_alpha = {alpha: r_at(new, old, alpha, beta) for alpha in ALPHAS}

    # the main alpha: the same numbers as evaluate(), and the frozen summary() gives evaluate()'s mean;
    # the main table is grouped() of rows holding exactly these R_v
    r = r_by_alpha[MAIN_ALPHA]
    if not np.array_equal(r, v["R_v"].to_numpy()):
        raise AssertionError(f"{date} r{replicate}: R_v at alpha {MAIN_ALPHA} differs from evaluate()")
    means = {MAIN_ALPHA: float(summary(v.assign(R_v=r), MAIN_ALPHA, beta)["all"]["mean R_v"])}
    if means[MAIN_ALPHA] != result.mean_r:
        raise AssertionError(f"{date} r{replicate}: mean R_v at alpha {MAIN_ALPHA} differs from evaluate()")

    # the other alphas: the mean by the same Series.mean(), and per group the sum of R_v and the threshold counts
    others = [a for a in ALPHAS if a != MAIN_ALPHA]
    columns = {}
    for alpha in others:
        means[alpha] = float(pd.Series(r_by_alpha[alpha]).mean())
        columns[f"sum R_v (alpha {label(alpha)})"] = r_by_alpha[alpha]
        for t in THETAS:
            columns[f"R_v >= {t:g} (alpha {label(alpha)})"] = at_least(r_by_alpha[alpha], t).astype(int)
    extra = main_keys(rows).assign(**columns).groupby(GROUP_KEYS).sum().reset_index()
    if not np.array_equal(extra[GROUP_KEYS].to_numpy(int), main[GROUP_KEYS].to_numpy(int)):
        raise AssertionError(f"{date} r{replicate}: the groups at the other alphas are not those of grouped()")
    groups = pd.concat([main, extra.drop(columns=GROUP_KEYS)], axis=1)

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
            out["decisions"] = decisions_table(day, result.run.decisions)
            if shadow is not None:
                for name, agrees in shadow.agrees.items():
                    if len(agrees) != len(out["decisions"]):
                        raise AssertionError("a shadow did not see every decision")
                    out["decisions"][f"{name} agrees"] = agrees
    return out


def decisions_table(day, rows):
    """The frozen decision log with the day's visitor index replaced by the card id and the site indices j0,
    jw by site ids (columns "j0 site", "jw site"); "decision" numbers the rows in the order they were made."""
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=["decision", "visitor", "j0 site", "jw site"])
    table.insert(0, "decision", np.arange(len(table)))
    table["visitor"] = day.visitors["visitor"].to_numpy()[table["visitor"].to_numpy(int)]
    table["j0 site"] = day.site_ids[table.pop("j0").to_numpy(int)]
    table["jw site"] = day.site_ids[table.pop("jw").to_numpy(int)]
    return table


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
