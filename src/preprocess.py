"""Step 2: clean the VeronaCard entries, split the days, compute site statistics, check observed routes.

Definitions follow 589_project/research_design/implementation_protocol.md, section 1.
Run from the repository root: python3 src/preprocess.py
Writes data/verona/processed/ (visits, visitors, days, sites, hours, usual GDAY) and its diagnostics.txt.
Availability: an inferred calendar, open if entries on at least half of the matching days. 2019: site by weekday, training days.
2022 and 2023: site by month and weekday, from that year's own days (a stated exception to using no target-year data).
Museo Miniscalchi has too few entries to infer and is left out of the candidates and of N; its visits stay as background. Hours by site and month.
The o and c columns of sites.csv are yearly values, kept only for design_audit.py.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SITE_ID = {"Arena": 49, "Casa Giulietta": 61, "Torre Lamberti": 59, "Castelvecchio": 71,
           "Santa Anastasia": 54, "Duomo": 52, "Teatro Romano": 42, "Palazzo della Ragione": 58,
           "San Fermo": 62, "Tomba Giulietta": 202, "San Zeno": 63, "Museo Lapidario": 76,
           "Giardino Giusti": 75, "Museo Storia": 201, "Museo Miniscalchi": 300}
VALIDATION_WEEKS = {5, 15, 25, 35, 45}  # every fifth odd week, so that each season is represented
LOW_COVERAGE = {300}  # Museo Miniscalchi
DATA = Path(__file__).resolve().parents[1] / "data" / "verona"
OUT = DATA / "processed"
lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


def load(path, date_format):
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d.data_visita.astype(str), format=date_format)
    clock = d.ora_visita.str.split(":", expand=True).astype(float)
    d["t"] = clock[0] * 60 + clock[1] + (clock[2] / 60 if 2 in clock else 0)
    return d[["id_veronacard", "date", "t", "sito_nome"]]


def clean(d, name):
    """Keep the 15 sites shared with 2019; keep only the first entry of a site in a visitor's day."""
    rows = len(d)
    d = d[d.sito_nome.isin(SITE_ID)].copy()
    other = rows - len(d)
    d["site"] = d.sito_nome.map(SITE_ID)
    d["visitor"] = d.id_veronacard + "_" + d.date.dt.strftime("%Y%m%d")
    d = d.sort_values(["visitor", "t"], kind="stable")
    repeats = d.duplicated(["visitor", "site"])
    say(f"{name}: {rows} entries, {other} at other sites removed, {repeats.sum()} repeat entries removed, "
        f"{d[~repeats].visitor.nunique()} visitors")
    d = d[~repeats].copy()
    d["stop"] = d.groupby("visitor").cumcount() + 1
    return d[["visitor", "date", "stop", "site", "t"]].reset_index(drop=True)


def add_stays(d, walk):
    """Stay = next entry minus walking time. Last stops have no estimate."""
    d["next_site"] = d.groupby("visitor").site.shift(-1)
    d["next_t"] = d.groupby("visitor").t.shift(-1)
    has_next = d.next_site.notna()
    d["walk"] = np.nan
    d.loc[has_next, "walk"] = [walk[(a, int(b))] for a, b in zip(d.site[has_next], d.next_site[has_next])]
    d["gap"] = d.next_t - d.t
    d["stay"] = d.gap - d.walk
    return d


def site_statistics(fit):
    s = pd.DataFrame({"n": fit.groupby("site").visitor.nunique()})
    s["A"] = s.n / s.n.max()
    stays = fit[fit.stay > 0].groupby("site").stay
    s["d"] = stays.median()
    s["o"] = fit.groupby("site").t.min()
    s["c"] = fit.groupby("site").t.max() + s.d
    s["usual_peak"] = usual_peak(fit, s.d)
    return s


def observed_occupancy_intervals(visits, d_i):
    """One occupancy interval per observed entry: [entry, min(entry + d_i, next entry - walk)], never negative.
    A last stop or one-stop visit stays d_i. The usual peak, the EDA and the replay all use this."""
    out = visits[["visitor", "date", "site", "t", "next_site", "walk"]].rename(columns={"t": "entry"}).copy()
    out["exit"] = np.fmin(visits.t + visits.site.map(d_i), visits.next_t - visits.walk).clip(lower=visits.t)
    return out


def usual_peak(fit, d_i):
    """People inside in an hour = person-minutes in that hour / 60. Each entry occupies [t, min(t + d_i, next entry - walk)].
    Usual peak = 95th percentile over (day, hour) pairs with at least one entry."""
    f = observed_occupancy_intervals(fit, d_i).rename(columns={"entry": "t", "exit": "end"})
    peaks = {}
    entry_hour = f.t // 60
    for site, g in f.groupby("site"):
        values = []
        for hour in range(int(entry_hour.min()), int(entry_hour.max()) + 1):
            overlap = (g.end.clip(upper=60 * (hour + 1)) - g.t.clip(lower=60 * hour)).clip(lower=0)
            had_entry = (g.t // 60 == hour)
            inside = overlap.groupby(g.date).sum() / 60
            values.append(inside[had_entry.groupby(g.date).any()])
        peaks[site] = pd.concat(values).quantile(0.95)
    return pd.Series(peaks)


def site_hours(fit, s):
    """Hours by site and month: earliest entry, and latest entry plus d."""
    h = fit.assign(month=fit.date.dt.month).groupby(["site", "month"]).t.agg(o="min", c="max").reset_index()
    h["c"] = h.c + h.site.map(s.d)
    return h


def window(hours, s, site, month):
    """A site with no training entries in that month falls back to its yearly hours."""
    return hours.get((site, month), (s.o[site], s.c[site]))


def retime(sites_, start, month, hours, s, walk):
    """Entry times of a route under expected visit lengths, waiting for opening where needed."""
    times = [start]
    for a, b in zip(sites_, sites_[1:]):
        arrive = times[-1] + s.d[a] + walk[(a, b)]
        times.append(max(arrive, window(hours, s, b, month)[0]))
    return times


def visitor_table(d, s, hours, walk):
    """Budget = observed span plus d of the last site, widened where the route needs longer under expected d."""
    rows = []
    for visitor, g in d.groupby("visitor", sort=False):
        sites_, start, last = list(g.site.values), g.t.iloc[0], g.t.iloc[-1]
        date = g.date.iloc[0]
        span = last - start + s.d[sites_[-1]]
        finish = retime(sites_, start, date.month, hours, s, walk)[-1] + s.d[sites_[-1]] - start
        rows.append((visitor, date, sites_[0], start, sites_[-1], len(sites_), span, max(span, finish)))
    return pd.DataFrame(rows, columns=["visitor", "date", "first_site", "start", "last_site", "stops", "span", "budget"])


def check_routes(d, v, s, hours, walk, name):
    """Do observed routes pass their own constraints under the inferred hours, d and budget?"""
    multi = v[v.stops >= 2].set_index("visitor")
    entry_ok = closing = 0
    for visitor, g in d[d.visitor.isin(multi.index)].groupby("visitor", sort=False):
        sites_, month = list(g.site.values), g.date.iloc[0].month
        entry_ok += all(window(hours, s, i, month)[0] <= t <= window(hours, s, i, month)[1] - s.d[i]
                        for i, t in zip(sites_, g.t.values))
        times = retime(sites_, g.t.iloc[0], month, hours, s, walk)
        closing += any(t > window(hours, s, i, month)[1] - s.d[i] for i, t in zip(sites_, times))
    total = len(multi)
    hops = d[d.next_site.notna()]
    say(f"{name}: {total} multi-stop visitors")
    say(f"  every observed entry inside the monthly [o, c-d]: {entry_ok / total:.1%}")
    say(f"  budget widened so that the route fits: {(multi.budget > multi.span + 1e-9).mean():.1%}")
    say(f"  route re-timed with expected d runs past closing somewhere: {closing / total:.1%}")
    say(f"  observed hops with gap >= d + walk: {(hops.gap >= hops.site.map(s.d) + hops.walk).mean():.1%}")


def gini_hat(x):
    """Normalised Gini of the visit counts over the sites open that day."""
    x = np.asarray(x, float)
    n, mean = len(x), x.mean()
    if n < 2 or mean == 0:
        return np.nan
    return np.abs(x[:, None] - x[None, :]).sum() / 2 / (n * n * mean) / (1 - 1 / n)


def weekday_shares(fit, ids):
    """Site by weekday: share of the matching days on which the site has entries."""
    days_of_weekday = fit[["date"]].drop_duplicates().date.dt.dayofweek.value_counts()
    seen = fit.assign(dow=fit.date.dt.dayofweek).groupby(["site", "dow"]).date.nunique().unstack(fill_value=0)
    return seen.reindex(index=ids, columns=range(7), fill_value=0).div(days_of_weekday, axis=1)


def weekday_opening(shares, dates):
    """Available sites on each date, from a site-by-weekday table."""
    return {date: [i for i in shares.index if shares.loc[i, date.dayofweek] >= 0.5 and i not in LOW_COVERAGE]
            for date in dates}


def month_weekday_shares(d):
    """Site by month and weekday: share of the matching days on which the site has entries, and the number of those days."""
    x = d.assign(month=d.date.dt.month, dow=d.date.dt.dayofweek)
    days = x[["date", "month", "dow"]].drop_duplicates().groupby(["month", "dow"]).size().rename("days")
    seen = x.groupby(["site", "month", "dow"]).date.nunique().rename("seen").reset_index().merge(days.reset_index())
    seen["share"] = seen.seen / seen.days
    return seen, days


def month_weekday_opening(d, ids):
    seen, _ = month_weekday_shares(d)
    open_cells = {(i, m, w) for i, m, w, x in zip(seen.site, seen.month, seen.dow, seen.share) if x >= 0.5}
    return {date: [i for i in ids if (i, date.month, date.dayofweek) in open_cells and i not in LOW_COVERAGE]
            for date in d.date.unique()}


def day_table(d, opened):
    """G_hat over the available sites, zero-visit sites included; G_hat_same_day is the sensitivity version."""
    counts = d.groupby(["date", "site"]).size().unstack(fill_value=0).reindex(columns=sorted(SITE_ID.values()), fill_value=0)
    rows = []
    for date, x in counts.iterrows():
        sites_ = opened[date]
        rows.append((date, x.sum(), len(sites_), int((x[sites_] == 0).sum()), x.sum() - x[sites_].sum(),
                     gini_hat(x[sites_].values), gini_hat(x[(x > 0) & ~x.index.isin(LOW_COVERAGE)].values)))
    return pd.DataFrame(rows, columns=["date", "visits", "open_sites", "open_with_no_visits", "visits_at_closed_sites",
                                       "G_hat", "G_hat_same_day"])


def closed_visit_report(d, opened, s, label):
    """Where do observed visits fall on sites the calendar marks unavailable, and whom does that touch?"""
    is_open = {(date, i) for date, sites_ in opened.items() for i in sites_}
    x = d.assign(dow=d.date.dt.dayofweek, stops=d.groupby("visitor").stop.transform("max"))
    x["closed"] = [(date, i) not in is_open for date, i in zip(x.date, x.site)]
    closed = x[x.closed]
    multi = x[x.stops >= 2]
    say(f"{label}: visits at unavailable sites {len(closed)} ({len(closed) / len(x):.2%}); visitors touched "
        f"{closed.visitor.nunique() / x.visitor.nunique():.2%}; multi-stop visitors touched "
        f"{multi[multi.closed].visitor.nunique() / multi.visitor.nunique():.2%}; closed first stops {(closed.stop == 1).sum()}, "
        f"closed later stops {(closed.stop > 1).sum()}")
    say(closed.groupby(["site", "dow"]).size().unstack(fill_value=0).rename(index=s.name.to_dict()).to_string())


def budget_report(v, label):
    x = v[v.stops >= 2].assign(added=lambda g: g.budget - g.span)
    w = x[x.added > 1e-9]
    q = w.added.quantile([0.5, 0.9, 0.95])
    say(f"{label}: widened {len(w) / len(x):.1%}; minutes added median {q[0.5]:.0f}, p90 {q[0.9]:.0f}, p95 {q[0.95]:.0f}, max {w.added.max():.0f}; "
        f"relative median {(w.added / w.span).median():.1%}; total added / total span {x.added.sum() / x.span.sum():.1%}")
    by_stops = x.assign(stops=x.stops.clip(upper=7)).groupby("stops").added
    say(pd.DataFrame({"visitors": by_stops.size(), "widened": by_stops.apply(lambda a: (a > 1e-9).mean()).round(3),
                      "median_added": by_stops.apply(lambda a: a[a > 1e-9].median()).round(0)}).T.to_string())


FIRST_MINUTE, LAST_MINUTE = 480, 1260  # 08:00 to 21:00


def day_type_dates(fit):
    """Training dates, Mondays apart: fewer sites are available on them."""
    dates = [pd.Timestamp(date) for date in fit.date.unique()]
    return (("monday", [d for d in dates if d.dayofweek == 0]), ("other", [d for d in dates if d.dayofweek != 0]))


def usual_gday(fit, open_sites):
    """For each minute, the mean over training days of G_hat of the visits up to that minute. The denominator of GDAY.
    An entry counts from the first minute not earlier than its true time; entries after the last minute are not in the table."""
    minutes = np.arange(FIRST_MINUTE, LAST_MINUTE + 1)
    rows = []
    for day_type, chosen in day_type_dates(fit):
        by_day = []
        for date in chosen:
            sites_ = open_sites[date]
            part = fit[(fit.date == date) & fit.site.isin(sites_)]
            counts = np.zeros((minutes.size, len(sites_)))
            position = {site: i for i, site in enumerate(sites_)}
            nodes = np.searchsorted(minutes, part.t.values, side="left")
            for node, site in zip(nodes, part.site.values):
                if node < minutes.size:
                    counts[node, position[site]] += 1
            by_day.append([gini_hat(row) for row in counts.cumsum(axis=0)])
        rows.append(pd.DataFrame({"day_type": day_type, "minute": minutes, "usual_gday": np.nanmean(np.array(by_day), axis=0)}))
    return pd.concat(rows, ignore_index=True)


def usual_arrivals(fit):
    """For each minute, the mean over training days of the entries at all sites in the 60 minutes up to it. The denominator of ARR."""
    minutes = np.arange(FIRST_MINUTE, LAST_MINUTE + 1)
    rows = []
    for day_type, chosen in day_type_dates(fit):
        by_day = []
        for date in chosen:
            t = fit.t[fit.date == date].values  # true times, seconds included
            by_day.append([((t > minute - 60) & (t <= minute)).sum() for minute in minutes])
        rows.append(pd.DataFrame({"day_type": day_type, "minute": minutes, "usual_arrivals": np.mean(by_day, axis=0)}))
    return pd.concat(rows, ignore_index=True)


def file_hash(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def write_manifest(inputs):
    """One version id for these tables: the code and the input files. Experiments record it."""
    hashes = {str(Path(p).name): file_hash(p) for p in [__file__, *inputs]}
    version = hashlib.sha1(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12]
    (OUT / "manifest.json").write_text(json.dumps({"version": version, "files": hashes, "python": sys.version.split()[0],
                                                   "pandas": pd.__version__, "numpy": np.__version__}, indent=2) + "\n")
    return version


def split_days(dates):
    """Even ISO weeks test; every fifth odd week validates; the other odd weeks train."""
    days = pd.DataFrame({"date": sorted(dates)})
    days["week"] = days.date.dt.isocalendar().week.astype(int)
    days["split"] = np.where(days.week % 2 == 0, "test",
                             np.where(days.week.isin(VALIDATION_WEEKS), "validation", "train"))
    return days


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- load and clean
    walk_table = pd.read_csv(DATA / "poi_time_travel.csv")
    walk = {(a, b): t for a, b, t in zip(walk_table.poi_start, walk_table.poi_dest, walk_table.time_travel)}
    d19 = add_stays(clean(load(DATA / "veronacard_2019_opendata.csv", "%d/%m/%y"), "2019"), walk)
    d22 = add_stays(clean(load(DATA / "veronacard_2022_original.csv", "%d/%m/%y"), "2022"), walk)
    d23 = add_stays(clean(load(DATA / "data_2023.csv", "%Y-%m-%d"), "2023"), walk)

    # ---- split the 2019 days
    days = split_days(d19.date.unique())
    d19 = d19.merge(days[["date", "split"]], on="date")
    fit = d19[d19.split == "train"]
    say()
    say(days.split.value_counts().to_string())

    # ---- statistics from training days only
    sites = site_statistics(fit)
    sites.insert(0, "name", sites.index.map({v: k for k, v in SITE_ID.items()}))
    assert sites.d.notna().all() and len(sites) == 15

    # ---- gaps
    say()
    stays = d19[d19.next_site.notna()].stay
    say(f"2019 gaps: {len(stays)}; stay <= 0: {(stays <= 0).sum()}; over 180 min: {(stays > 180).mean():.1%}; "
        f"over 240 min: {(stays > 240).mean():.1%}")
    capped = fit[(fit.stay > 0) & (fit.stay <= 240)].groupby("site").stay.median()
    sites["d_no_long_gaps"] = capped
    say(f"largest change in d if gaps over 240 min are dropped: {(sites.d - capped).max():.1f} min")

    # ---- tables and checks
    hours_table = site_hours(fit, sites)
    hours = {(i, m): (o, c) for i, m, o, c in zip(hours_table.site, hours_table.month, hours_table.o, hours_table.c)}
    v19, v22, v23 = (visitor_table(x, sites, hours, walk) for x in (d19, d22, d23))
    v19 = v19.merge(days[["date", "split"]], on="date")
    for label, x in (("2019", v19), ("2022", v22), ("2023", v23)):
        multi = x[x.stops >= 2]
        a_sum = d19 if label == "2019" else (d22 if label == "2022" else d23)
        total = a_sum[a_sum.visitor.isin(multi.visitor)].groupby("visitor").site.apply(lambda c: c.map(sites.A).sum())
        assert (total > 0).all(), label
        say(f"{label}: min sum of A over multi-stop visitors {total.min():.3f}")

    say()
    check_routes(d19[d19.split == "train"], v19[v19.split == "train"], sites, hours, walk, "2019 training days")
    check_routes(d19[d19.split == "test"], v19[v19.split == "test"], sites, hours, walk, "2019 test days")

    # ---- are fixed yearly opening hours believable? (training days)
    say()
    fit_days = fit.date.nunique()
    per_day = fit.groupby(["site", "date"]).t.agg(["min", "max"])
    hours = pd.DataFrame({"days_with_entries": per_day.groupby("site").size(),
                          "median_first": per_day["min"].groupby("site").median(),
                          "median_last": per_day["max"].groupby("site").median()})
    hours["o"], hours["latest_start"] = sites.o, sites.c - sites.d
    hours["name"] = sites.name
    hours["no_entry_days"] = fit_days - hours.days_with_entries
    weekday = fit.assign(dow=fit.date.dt.dayofweek).groupby(["site", "dow"]).date.nunique().unstack(fill_value=0)
    weekday.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    say(hours[["name", "days_with_entries", "no_entry_days", "o", "median_first", "median_last", "latest_start"]]
        .round(0).to_string())
    say()
    say(weekday.assign(name=sites.name).to_string())

    # ---- daily fairness and usual GDAY
    ids = list(sites.index)
    calendar = weekday_shares(fit, ids)
    open19 = weekday_opening(calendar, days.date)
    open22, open23 = month_weekday_opening(d22, ids), month_weekday_opening(d23, ids)
    days = days.merge(day_table(d19, open19), on="date")
    days22, days23 = day_table(d22, open22), day_table(d23, open23)
    gday = usual_gday(fit, open19)
    arrivals = usual_arrivals(fit)
    say()
    say("share of training days with entries, by weekday (Mon = 0); available if at least 0.5")
    say(calendar.round(2).assign(name=sites.name).to_string())
    say()
    for label, x in (("2019", days), ("2022", days22), ("2023", days23)):
        say(f"{label}: G_hat median {x.G_hat.median():.3f}, same-day version {x.G_hat_same_day.median():.3f}; available sites median "
            f"{x.open_sites.median():.0f}, min {x.open_sites.min()}; available site-days with no visits {x.open_with_no_visits.sum() / x.open_sites.sum():.1%}; "
            f"visits at unavailable sites {x.visits_at_closed_sites.sum() / x.visits.sum():.2%}")
    say()
    closed_visit_report(d19, open19, sites, "2019")
    closed_visit_report(d22, open22, sites, "2022")
    closed_visit_report(d23, open23, sites, "2023")
    for label, x in (("2022", d22), ("2023", d23)):
        say(f"{label}: days per month-weekday cell from {month_weekday_shares(x)[1].min()} to {month_weekday_shares(x)[1].max()}")
    say()
    budget_report(v19[v19.split != "validation"], "2019 training and test")
    budget_report(v22, "2022")
    budget_report(v23, "2023")
    say()
    cards = d19.assign(card=d19.visitor.str.split("_").str[0]).groupby("card").split.agg(set)
    both = cards.map(lambda x: "test" in x and len(x) > 1)
    test_visitors = v19[v19.split == "test"].visitor.str.split("_").str[0]
    say(f"cards in both test and other days: {both.mean():.1%} of cards, {test_visitors.isin(both[both].index).mean():.1%} of test visitors")
    say(gday[gday.minute % 60 == 0].round(3).to_string(index=False))

    # ---- save
    say()
    say(sites.round(3).to_string())
    sites.round(4).to_csv(f"{OUT}/sites.csv")
    days.to_csv(f"{OUT}/days_2019.csv", index=False)
    days22.to_csv(f"{OUT}/days_2022.csv", index=False)
    days23.to_csv(f"{OUT}/days_2023.csv", index=False)
    calendar.round(3).to_csv(f"{OUT}/open_calendar.csv")
    month_weekday_shares(d22)[0].to_csv(f"{OUT}/open_calendar_2022.csv", index=False)
    month_weekday_shares(d23)[0].to_csv(f"{OUT}/open_calendar_2023.csv", index=False)
    for year, opened in ((2019, open19), (2022, open22), (2023, open23)):
        pd.DataFrame([(date, i) for date, sites_ in opened.items() for i in sites_], columns=["date", "site"]).to_csv(f"{OUT}/open_sites_{year}.csv", index=False)
    hours_table.round(2).to_csv(f"{OUT}/site_hours.csv", index=False)
    gday.round(4).to_csv(f"{OUT}/usual_gday.csv", index=False)
    arrivals.round(3).to_csv(f"{OUT}/usual_arrivals.csv", index=False)
    columns = ["visitor", "date", "stop", "site", "t", "stay", "next_site", "next_t", "walk"]
    d19[columns + ["split"]].to_csv(f"{OUT}/visits_2019.csv.gz", index=False)
    d22[columns].to_csv(f"{OUT}/visits_2022.csv.gz", index=False)
    d23[columns].to_csv(f"{OUT}/visits_2023.csv.gz", index=False)
    v19.to_csv(f"{OUT}/visitors_2019.csv.gz", index=False)
    v22.to_csv(f"{OUT}/visitors_2022.csv.gz", index=False)
    v23.to_csv(f"{OUT}/visitors_2023.csv.gz", index=False)
    say()
    say("manifest version " + write_manifest([DATA / "veronacard_2019_opendata.csv", DATA / "veronacard_2022_original.csv", DATA / "data_2023.csv", DATA / "poi_time_travel.csv"]))
    with open(f"{OUT}/diagnostics.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
