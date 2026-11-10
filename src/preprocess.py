"""
VeronaCard data preparation (Step 2)
====================================

Cleans the card entries, splits the 2019 days into training, validation and
test days, and computes what the simulator needs: site statistics, opening
hours, availability calendars, time budgets, and the usual levels behind the
GDAY and ARR inputs. Definitions follow implementation_protocol.md, section 1.

All statistics come from 2019 training days. The one stated exception is the
availability calendar of 2022 and Jan-Feb 2023, inferred from those years' own
entries because the sites open changed after 2019.

Run:  python3 src/preprocess.py   (from the repository root)
Reads data/verona/: the entries of 2019, 2022 and Jan-Feb 2023, and walking times.
Writes data/verona/processed/: the tables, diagnostics.txt and manifest.json.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "verona"
OUT = DATA / "processed"

SITE_ID = {"Arena": 49, "Casa Giulietta": 61, "Torre Lamberti": 59, "Castelvecchio": 71,
           "Santa Anastasia": 54, "Duomo": 52, "Teatro Romano": 42, "Palazzo della Ragione": 58,
           "San Fermo": 62, "Tomba Giulietta": 202, "San Zeno": 63, "Museo Lapidario": 76,
           "Giardino Giusti": 75, "Museo Storia": 201, "Museo Miniscalchi": 300}
LOW_COVERAGE = {300}                    # Museo Miniscalchi: too few entries to infer its calendar
VALIDATION_WEEKS = {5, 15, 25, 35, 45}  # every fifth odd week, so that each season is represented
FIRST_MINUTE, LAST_MINUTE = 480, 1260   # the usual tables run from 08:00 to 21:00
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

lines = []


def say(text=""):
    """Print a diagnostic line and keep it for diagnostics.txt."""
    print(text)
    lines.append(str(text))


# ---------------------------------------------------------------------- #
# 1. Loading and cleaning
# ---------------------------------------------------------------------- #

def load(path, date_format):
    """Card id, date, entry time in minutes after midnight, and site name."""
    entries = pd.read_csv(path)
    entries["date"] = pd.to_datetime(entries.data_visita.astype(str), format=date_format)
    clock = entries.ora_visita.str.split(":", expand=True).astype(float)
    seconds = clock[2] / 60 if 2 in clock else 0  # 2019 has minutes only
    entries["t"] = clock[0] * 60 + clock[1] + seconds
    return entries[["id_veronacard", "date", "t", "sito_nome"]]


def clean(entries, name):
    """Keep the 15 sites shared with 2019, and only the first entry of a site in a visitor's day.
    A visitor is one card on one day."""
    rows = len(entries)
    d = entries[entries.sito_nome.isin(SITE_ID)].copy()
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
    """Stay = next entry minus the walking time. A last stop has no estimate."""
    d["next_site"] = d.groupby("visitor").site.shift(-1)
    d["next_t"] = d.groupby("visitor").t.shift(-1)
    has_next = d.next_site.notna()
    d["walk"] = np.nan
    d.loc[has_next, "walk"] = [walk[(a, int(b))] for a, b in zip(d.site[has_next], d.next_site[has_next])]
    d["gap"] = d.next_t - d.t
    d["stay"] = d.gap - d.walk
    return d


# ---------------------------------------------------------------------- #
# 2. Split of the days
# ---------------------------------------------------------------------- #

def split_days(dates):
    """Even ISO weeks test; every fifth odd week validates; the other odd weeks train."""
    days = pd.DataFrame({"date": sorted(dates)})
    days["week"] = days.date.dt.isocalendar().week.astype(int)
    days["split"] = np.where(days.week % 2 == 0, "test",
                             np.where(days.week.isin(VALIDATION_WEEKS), "validation", "train"))
    return days


# ---------------------------------------------------------------------- #
# 3. Site statistics (training days)
# ---------------------------------------------------------------------- #

def site_statistics(fit):
    """Visitors n, attractiveness A, median stay d and usual peak of each site.
    The yearly o and c columns are kept only for design_audit.py; hours come from site_hours()."""
    s = pd.DataFrame({"n": fit.groupby("site").visitor.nunique()})
    s["A"] = s.n / s.n.max()
    s["d"] = fit[fit.stay > 0].groupby("site").stay.median()  # non-positive stays are left out
    s["o"] = fit.groupby("site").t.min()
    s["c"] = fit.groupby("site").t.max() + s.d
    s["usual_peak"] = usual_peak(fit, s.d)
    return s


def observed_occupancy_intervals(visits, d_i):
    """One interval per observed entry: [entry, min(entry + d_i, next entry - walk)], never negative.
    A last stop or one-stop visit stays d_i. The usual peak, the EDA and the replay all use this."""
    out = visits[["visitor", "date", "site", "t", "next_site", "walk"]].rename(columns={"t": "entry"}).copy()
    out["exit"] = np.fmin(visits.t + visits.site.map(d_i), visits.next_t - visits.walk).clip(lower=visits.t)
    return out


def usual_peak(fit, d_i):
    """95th percentile of people inside, over (day, hour) pairs with at least one entry.
    People inside in an hour = person-minutes in that hour / 60."""
    f = observed_occupancy_intervals(fit, d_i).rename(columns={"entry": "t", "exit": "end"})
    first_hour, last_hour = int((f.t // 60).min()), int((f.t // 60).max())
    peaks = {}
    for site, g in f.groupby("site"):
        values = []
        for hour in range(first_hour, last_hour + 1):
            overlap = (g.end.clip(upper=60 * (hour + 1)) - g.t.clip(lower=60 * hour)).clip(lower=0)
            inside = overlap.groupby(g.date).sum() / 60
            had_entry = (g.t // 60 == hour).groupby(g.date).any()
            values.append(inside[had_entry])
        peaks[site] = pd.concat(values).quantile(0.95)
    return pd.Series(peaks)


def site_hours(fit, s):
    """Opening hours by site and month: the earliest entry, and the latest entry plus d."""
    h = fit.assign(month=fit.date.dt.month).groupby(["site", "month"]).t.agg(o="min", c="max").reset_index()
    h["c"] = h.c + h.site.map(s.d)
    return h


# ---------------------------------------------------------------------- #
# 4. Visitors and their time budgets
# ---------------------------------------------------------------------- #

def window(hours, s, site, month):
    """Opening and closing of a site in a month; its yearly hours if it had no training entries that month."""
    return hours.get((site, month), (s.o[site], s.c[site]))


def retime(stops, start, month, hours, s, walk):
    """Entry times of a route under expected stays, waiting for opening where needed."""
    times = [start]
    for a, b in zip(stops, stops[1:]):
        arrive = times[-1] + s.d[a] + walk[(a, b)]
        times.append(max(arrive, window(hours, s, b, month)[0]))
    return times


def visitor_table(d, s, hours, walk):
    """One row per visitor. The budget is the observed span plus d of the last site,
    widened once if the route needs longer under expected stays."""
    rows = []
    for visitor, g in d.groupby("visitor", sort=False):
        stops = list(g.site.values)
        start, last, date = g.t.iloc[0], g.t.iloc[-1], g.date.iloc[0]
        span = last - start + s.d[stops[-1]]
        finish = retime(stops, start, date.month, hours, s, walk)[-1] + s.d[stops[-1]] - start
        rows.append((visitor, date, stops[0], start, stops[-1], len(stops), span, max(span, finish)))
    columns = ["visitor", "date", "first_site", "start", "last_site", "stops", "span", "budget"]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------- #
# 5. Availability and daily fairness
# ---------------------------------------------------------------------- #

def gini_hat(x):
    """Normalised Gini of visit counts, so that the maximum is 1."""
    x = np.asarray(x, float)
    n, mean = len(x), x.mean()
    if n < 2 or mean == 0:
        return np.nan
    return np.abs(x[:, None] - x[None, :]).sum() / 2 / (n * n * mean) / (1 - 1 / n)


def weekday_shares(fit, ids):
    """Site by weekday: share of the matching days on which the site has entries."""
    days_of_weekday = fit[["date"]].drop_duplicates().date.dt.dayofweek.value_counts()
    seen = fit.assign(dow=fit.date.dt.dayofweek).groupby(["site", "dow"]).date.nunique().unstack(fill_value=0)
    seen = seen.reindex(index=ids, columns=range(7), fill_value=0)
    return seen.div(days_of_weekday, axis=1)


def weekday_opening(shares, dates):
    """Available sites on each date: entries on at least half of the matching days, low-coverage sites out."""
    opened = {}
    for date in dates:
        opened[date] = [i for i in shares.index if shares.loc[i, date.dayofweek] >= 0.5 and i not in LOW_COVERAGE]
    return opened


def month_weekday_shares(d):
    """Site by month and weekday: share of the matching days with entries, and the number of those days."""
    x = d.assign(month=d.date.dt.month, dow=d.date.dt.dayofweek)
    days = x[["date", "month", "dow"]].drop_duplicates().groupby(["month", "dow"]).size().rename("days")
    seen = x.groupby(["site", "month", "dow"]).date.nunique().rename("seen").reset_index()
    seen = seen.merge(days.reset_index())
    seen["share"] = seen.seen / seen.days
    return seen, days


def month_weekday_opening(d, ids):
    """Available sites on each date of 2022 or 2023, from that year's own month-by-weekday shares."""
    seen, _ = month_weekday_shares(d)
    open_cells = set()
    for site, month, dow, share in zip(seen.site, seen.month, seen.dow, seen.share):
        if share >= 0.5:
            open_cells.add((site, month, dow))
    opened = {}
    for date in d.date.unique():
        cell = (date.month, date.dayofweek)
        opened[date] = [i for i in ids if (i, *cell) in open_cells and i not in LOW_COVERAGE]
    return opened


def day_table(d, opened):
    """Visits and G_hat of each day over the available sites, zero-visit sites included.
    G_hat_same_day is the sensitivity version: sites with entries that day, low-coverage sites out."""
    counts = d.groupby(["date", "site"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=sorted(SITE_ID.values()), fill_value=0)
    rows = []
    for date, x in counts.iterrows():
        sites_ = opened[date]
        same_day = x[(x > 0) & ~x.index.isin(LOW_COVERAGE)]
        rows.append((date, x.sum(), len(sites_), int((x[sites_] == 0).sum()), x.sum() - x[sites_].sum(),
                     gini_hat(x[sites_].values), gini_hat(same_day.values)))
    columns = ["date", "visits", "open_sites", "open_with_no_visits", "visits_at_closed_sites",
               "G_hat", "G_hat_same_day"]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------- #
# 6. Usual levels for GDAY and ARR (one value per minute)
# ---------------------------------------------------------------------- #

def day_type_dates(fit):
    """Training dates, Mondays apart: fewer sites are available on them."""
    dates = [pd.Timestamp(date) for date in fit.date.unique()]
    mondays = [d for d in dates if d.dayofweek == 0]
    others = [d for d in dates if d.dayofweek != 0]
    return (("monday", mondays), ("other", others))


def usual_gday(fit, open_sites):
    """For each minute, the mean over training days of G_hat of the visits up to that minute (denominator of GDAY).
    An entry counts from the first minute not earlier than its true time; later entries are not in the table."""
    minutes = np.arange(FIRST_MINUTE, LAST_MINUTE + 1)
    tables = []
    for day_type, dates in day_type_dates(fit):
        by_day = []
        for date in dates:
            sites_ = open_sites[date]
            part = fit[(fit.date == date) & fit.site.isin(sites_)]
            position = {site: i for i, site in enumerate(sites_)}
            counts = np.zeros((minutes.size, len(sites_)))
            nodes = np.searchsorted(minutes, part.t.values, side="left")
            for node, site in zip(nodes, part.site.values):
                if node < minutes.size:
                    counts[node, position[site]] += 1
            by_day.append([gini_hat(row) for row in counts.cumsum(axis=0)])
        mean = np.nanmean(np.array(by_day), axis=0)
        tables.append(pd.DataFrame({"day_type": day_type, "minute": minutes, "usual_gday": mean}))
    return pd.concat(tables, ignore_index=True)


def usual_arrivals(fit):
    """For each minute, the mean over training days of the entries at all sites
    in the 60 minutes up to it (denominator of ARR). Days without entries count as zero."""
    minutes = np.arange(FIRST_MINUTE, LAST_MINUTE + 1)
    tables = []
    for day_type, dates in day_type_dates(fit):
        by_day = []
        for date in dates:
            t = fit.t[fit.date == date].values  # true times, seconds included
            by_day.append([((t > minute - 60) & (t <= minute)).sum() for minute in minutes])
        mean = np.mean(by_day, axis=0)
        tables.append(pd.DataFrame({"day_type": day_type, "minute": minutes, "usual_arrivals": mean}))
    return pd.concat(tables, ignore_index=True)


# ---------------------------------------------------------------------- #
# 7. Diagnostics
# ---------------------------------------------------------------------- #

def report_gaps(d19, fit, sites):
    """How many stays are non-positive or long, and how much long gaps move d."""
    stays = d19[d19.next_site.notna()].stay
    say(f"2019 gaps: {len(stays)}; stay <= 0: {(stays <= 0).sum()}; over 180 min: {(stays > 180).mean():.1%}; "
        f"over 240 min: {(stays > 240).mean():.1%}")
    capped = fit[(fit.stay > 0) & (fit.stay <= 240)].groupby("site").stay.median()
    say(f"largest change in d if gaps over 240 min are dropped: {(sites.d - capped).max():.1f} min")
    return capped


def report_sum_of_a(years, sites):
    """R_v divides by the sum of A over the observed route, so it must be positive."""
    for label, visits, visitors in years:
        multi = visitors[visitors.stops >= 2]
        on_route = visits[visits.visitor.isin(multi.visitor)]
        total = on_route.groupby("visitor").site.apply(lambda route: route.map(sites.A).sum())
        assert (total > 0).all(), label
        say(f"{label}: min sum of A over multi-stop visitors {total.min():.3f}")


def check_routes(d, v, s, hours, walk, name):
    """Do observed routes pass their own constraints under the inferred hours, stays and budget?"""
    multi = v[v.stops >= 2].set_index("visitor")
    entry_ok = 0
    past_closing = 0
    for _, g in d[d.visitor.isin(multi.index)].groupby("visitor", sort=False):
        stops, month = list(g.site.values), g.date.iloc[0].month
        entry_ok += all(window(hours, s, i, month)[0] <= t <= window(hours, s, i, month)[1] - s.d[i]
                        for i, t in zip(stops, g.t.values))
        times = retime(stops, g.t.iloc[0], month, hours, s, walk)
        past_closing += any(t > window(hours, s, i, month)[1] - s.d[i] for i, t in zip(stops, times))
    total = len(multi)
    hops = d[d.next_site.notna()]
    say(f"{name}: {total} multi-stop visitors")
    say(f"  every observed entry inside the monthly [o, c-d]: {entry_ok / total:.1%}")
    say(f"  budget widened so that the route fits: {(multi.budget > multi.span + 1e-9).mean():.1%}")
    say(f"  route re-timed with expected d runs past closing somewhere: {past_closing / total:.1%}")
    say(f"  observed hops with gap >= d + walk: {(hops.gap >= hops.site.map(s.d) + hops.walk).mean():.1%}")


def report_yearly_hours(fit, sites):
    """Are fixed yearly opening hours believable? Entries by site, and days with entries by weekday."""
    per_day = fit.groupby(["site", "date"]).t.agg(["min", "max"])
    hours = pd.DataFrame({"days_with_entries": per_day.groupby("site").size(),
                          "median_first": per_day["min"].groupby("site").median(),
                          "median_last": per_day["max"].groupby("site").median()})
    hours["o"], hours["latest_start"] = sites.o, sites.c - sites.d
    hours["name"] = sites.name
    hours["no_entry_days"] = fit.date.nunique() - hours.days_with_entries
    columns = ["name", "days_with_entries", "no_entry_days", "o", "median_first", "median_last", "latest_start"]
    say(hours[columns].round(0).to_string())
    say()
    weekday = fit.assign(dow=fit.date.dt.dayofweek).groupby(["site", "dow"]).date.nunique().unstack(fill_value=0)
    weekday.columns = WEEKDAYS
    say(weekday.assign(name=sites.name).to_string())


def report_days(label, x):
    say(f"{label}: G_hat median {x.G_hat.median():.3f}, same-day version {x.G_hat_same_day.median():.3f}; "
        f"available sites median {x.open_sites.median():.0f}, min {x.open_sites.min()}; "
        f"available site-days with no visits {x.open_with_no_visits.sum() / x.open_sites.sum():.1%}; "
        f"visits at unavailable sites {x.visits_at_closed_sites.sum() / x.visits.sum():.2%}")


def closed_visit_report(d, opened, s, label):
    """Where do observed visits fall on sites the calendar marks unavailable, and whom does that touch?"""
    is_open = {(date, i) for date, sites_ in opened.items() for i in sites_}
    x = d.assign(dow=d.date.dt.dayofweek, stops=d.groupby("visitor").stop.transform("max"))
    x["closed"] = [(date, i) not in is_open for date, i in zip(x.date, x.site)]
    closed = x[x.closed]
    multi = x[x.stops >= 2]
    say(f"{label}: visits at unavailable sites {len(closed)} ({len(closed) / len(x):.2%}); "
        f"visitors touched {closed.visitor.nunique() / x.visitor.nunique():.2%}; "
        f"multi-stop visitors touched {multi[multi.closed].visitor.nunique() / multi.visitor.nunique():.2%}; "
        f"closed first stops {(closed.stop == 1).sum()}, closed later stops {(closed.stop > 1).sum()}")
    say(closed.groupby(["site", "dow"]).size().unstack(fill_value=0).rename(index=s.name.to_dict()).to_string())


def budget_report(v, label):
    """How often, and by how much, the budget correction adds time."""
    x = v[v.stops >= 2].assign(added=lambda g: g.budget - g.span)
    w = x[x.added > 1e-9]
    q = w.added.quantile([0.5, 0.9, 0.95])
    say(f"{label}: widened {len(w) / len(x):.1%}; minutes added median {q[0.5]:.0f}, p90 {q[0.9]:.0f}, "
        f"p95 {q[0.95]:.0f}, max {w.added.max():.0f}; relative median {(w.added / w.span).median():.1%}; "
        f"total added / total span {x.added.sum() / x.span.sum():.1%}")
    by_stops = x.assign(stops=x.stops.clip(upper=7)).groupby("stops").added
    table = pd.DataFrame({"visitors": by_stops.size(),
                          "widened": by_stops.apply(lambda a: (a > 1e-9).mean()).round(3),
                          "median_added": by_stops.apply(lambda a: a[a > 1e-9].median()).round(0)})
    say(table.T.to_string())


def report_cards(d19, v19):
    """Cards valid for two days can fall on both sides of the split."""
    cards = d19.assign(card=d19.visitor.str.split("_").str[0]).groupby("card").split.agg(set)
    both = cards.map(lambda splits: "test" in splits and len(splits) > 1)
    test_cards = v19[v19.split == "test"].visitor.str.split("_").str[0]
    say(f"cards in both test and other days: {both.mean():.1%} of cards, "
        f"{test_cards.isin(both[both].index).mean():.1%} of test visitors")


# ---------------------------------------------------------------------- #
# 8. Saving
# ---------------------------------------------------------------------- #

def file_hash(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def write_manifest(inputs):
    """One version id for these tables, from the code and the input files. Experiments record it."""
    hashes = {str(Path(p).name): file_hash(p) for p in [__file__, *inputs]}
    version = hashlib.sha1(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12]
    manifest = {"version": version, "files": hashes, "python": sys.version.split()[0],
                "pandas": pd.__version__, "numpy": np.__version__}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return version


def save_open_sites(opened, year):
    rows = [(date, site) for date, sites_ in opened.items() for site in sites_]
    pd.DataFrame(rows, columns=["date", "site"]).to_csv(f"{OUT}/open_sites_{year}.csv", index=False)


# ---------------------------------------------------------------------- #
# 9. Main
# ---------------------------------------------------------------------- #

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Load and clean the three years.
    walk_table = pd.read_csv(DATA / "poi_time_travel.csv")
    walk = {(a, b): t for a, b, t in zip(walk_table.poi_start, walk_table.poi_dest, walk_table.time_travel)}
    d19 = add_stays(clean(load(DATA / "veronacard_2019_opendata.csv", "%d/%m/%y"), "2019"), walk)
    d22 = add_stays(clean(load(DATA / "veronacard_2022_original.csv", "%d/%m/%y"), "2022"), walk)
    d23 = add_stays(clean(load(DATA / "data_2023.csv", "%Y-%m-%d"), "2023"), walk)

    # Split the 2019 days; every statistic below comes from the training days.
    days = split_days(d19.date.unique())
    d19 = d19.merge(days[["date", "split"]], on="date")
    fit = d19[d19.split == "train"]
    say()
    say(days.split.value_counts().to_string())

    sites = site_statistics(fit)
    sites.insert(0, "name", sites.index.map({v: k for k, v in SITE_ID.items()}))
    assert sites.d.notna().all() and len(sites) == 15
    say()
    sites["d_no_long_gaps"] = report_gaps(d19, fit, sites)

    # Visitors, budgets and checks of the observed routes.
    hours_table = site_hours(fit, sites)
    hours = {(i, m): (o, c) for i, m, o, c in zip(hours_table.site, hours_table.month, hours_table.o, hours_table.c)}
    v19, v22, v23 = (visitor_table(x, sites, hours, walk) for x in (d19, d22, d23))
    v19 = v19.merge(days[["date", "split"]], on="date")
    report_sum_of_a((("2019", d19, v19), ("2022", d22, v22), ("2023", d23, v23)), sites)
    say()
    check_routes(d19[d19.split == "train"], v19[v19.split == "train"], sites, hours, walk, "2019 training days")
    check_routes(d19[d19.split == "test"], v19[v19.split == "test"], sites, hours, walk, "2019 test days")
    say()
    report_yearly_hours(fit, sites)

    # Availability, daily fairness and the usual levels.
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
        report_days(label, x)
    say()
    closed_visit_report(d19, open19, sites, "2019")
    closed_visit_report(d22, open22, sites, "2022")
    closed_visit_report(d23, open23, sites, "2023")
    for label, x in (("2022", d22), ("2023", d23)):
        cell_days = month_weekday_shares(x)[1]
        say(f"{label}: days per month-weekday cell from {cell_days.min()} to {cell_days.max()}")
    say()
    budget_report(v19[v19.split != "validation"], "2019 training and test")
    budget_report(v22, "2022")
    budget_report(v23, "2023")
    say()
    report_cards(d19, v19)
    say(gday[gday.minute % 60 == 0].round(3).to_string(index=False))

    # Save the tables, then the version and the diagnostics.
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
        save_open_sites(opened, year)
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
    inputs = [DATA / "veronacard_2019_opendata.csv", DATA / "veronacard_2022_original.csv",
              DATA / "data_2023.csv", DATA / "poi_time_travel.csv"]
    say()
    say("manifest version " + write_manifest(inputs))
    with open(f"{OUT}/diagnostics.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
