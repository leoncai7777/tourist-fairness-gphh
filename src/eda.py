"""
VeronaCard exploratory data analysis (Step 2)
=============================================

Describes the data with the definitions of the research design, by reading the
tables made by preprocess.py: one visitor is one card on one day; statistics
come from training days; availability is an inferred calendar; crowding comes
from occupancy intervals; budgets are the corrected ones.

Sections of the summary:
  1. Sample after cleaning          5. Crowding by site and hour
  2. The split of the days          6. Routes and the time budget
  3. Sites                          7. Observed route value
  4. Availability and concentration 8. Other years

Run:  python3 src/eda.py   (from the repository root, after src/preprocess.py)
Writes outputs/eda/summary.md and seven figures next to it.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocess import WEEKDAYS, gini_hat, observed_occupancy_intervals

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "verona" / "processed"
OUT = ROOT / "outputs" / "eda"
ALPHA, BETA = 0.25, 0.5  # as in the worked example of the research design

INK, MUTED, BLUE, ORANGE, AQUA = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e7e6e1",
                     "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.dpi": 160})

lines = []


def say(text=""):
    """Print a line and keep it for summary.md."""
    print(text)
    lines.append(str(text))


def table(frame, index=True):
    """Write a pandas table as a markdown table."""
    if index:
        frame = frame.reset_index()
    say("| " + " | ".join(str(c) for c in frame.columns) + " |")
    say("|" + "---|" * len(frame.columns))
    for row in frame.itertuples(index=False):
        say("| " + " | ".join(str(x) for x in row) + " |")
    say()


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight")
    plt.close()


def season(month):
    """Northern-hemisphere seasons: Verona is in Italy."""
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def dumbbell(order, left, right, left_label, right_label, left_color, right_color, xlabel, title, name):
    """Two dots per site joined by a line, one site per row."""
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.hlines(y, order[left], order[right], color="#c9c8c2", linewidth=2)
    ax.scatter(order[left], y, s=42, color=left_color, label=left_label, zorder=3)
    ax.scatter(order[right], y, s=42, color=right_color, label=right_label, zorder=3)
    ax.set_yticks(y, order.site)
    ax.set(xlabel=xlabel, title=title)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    save(name)


# ---------------------------------------------------------------------- #
# 1. Sample after cleaning
# ---------------------------------------------------------------------- #

def section_sample(visits, visitors, multi, days):
    say("## 1. Sample after cleaning (2019)")
    say()
    say(f"- Visits {len(visits):,}; visitors {len(visitors):,}; "
        f"multi-stop visitors {len(multi):,} ({len(multi) / len(visitors):.1%}), "
        f"the only ones whose next stops a rule chooses")
    say(f"- Stops per multi-stop visitor: mean {multi.stops.mean():.2f}, median {multi.stops.median():.0f}")
    say(f"- Each site at most once per visitor; dates {visits.date.min():%d %b} to {visits.date.max():%d %b}, "
        f"{days.date.nunique()} days")
    say()


# ---------------------------------------------------------------------- #
# 2. The split of the days
# ---------------------------------------------------------------------- #

def section_split(days):
    say("## 2. The split of the days")
    say()
    days["season"] = days.date.dt.month.map(season)
    split_table = days.groupby("split").agg(days=("date", "size"), visits_per_day=("visits", "mean"),
                                            median_visits=("visits", "median"), busiest=("visits", "max"),
                                            G_hat=("G_hat", "median")).round(3)
    rounded = ["visits_per_day", "median_visits"]
    split_table[rounded] = split_table[rounded].round(0).astype(int)
    table(split_table)
    table(days.groupby(["split", "season"]).size().unstack(fill_value=0))

    fig, ax = plt.subplots(figsize=(8, 3.2))
    splits = (("train", BLUE, "Training"), ("validation", ORANGE, "Validation"), ("test", AQUA, "Test"))
    for split, color, label in splits:
        part = days[days.split == split]
        ax.scatter(part.date, part.visits, s=14, color=color, label=f"{label} ({len(part)} days)", linewidths=0)
    ax.set(ylabel="Visits that day", title="Quiet and busy days fall in all three sets, across the year")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    save("days_by_split.png")


# ---------------------------------------------------------------------- #
# 3. Sites
# ---------------------------------------------------------------------- #

def section_sites(sites, train, visitors, listed):
    """Returns the site table, ordered by attractiveness, for later sections."""
    say("## 3. Sites (statistics from training days)")
    say()
    all_share = train.groupby("site").size() / len(train)
    first = visitors[visitors.split == "train"].first_site.value_counts(normalize=True)
    site_table = pd.DataFrame({"site": sites.name, "A": sites.A.round(3), "stay d (min)": sites.d.astype(int),
                               "listed (min)": sites.index.map(listed), "usual peak": sites.usual_peak.round(1),
                               "share of visits": all_share.reindex(sites.index).mul(100).round(1),
                               "share of first stops": first.reindex(sites.index).fillna(0).mul(100).round(1)})
    site_table = site_table.sort_values("A", ascending=False)
    table(site_table, index=False)

    # The same three sites for both shares: the most visited ones.
    top3 = all_share.nlargest(3).index
    top3_all, top3_first = all_share[top3].sum(), first.reindex(top3).fillna(0).sum()
    say(f"The three most visited sites hold {top3_all:.0%} of visits and {top3_first:.0%} of first stops. "
        f"First stops stay fixed under every rule; section 4 shows what that limits. "
        f"Inferred stays are longer than listed times where the gaps also hold meals or rest.")
    say()

    dumbbell(site_table.iloc[::-1], "share of visits", "share of first stops", "All visits",
             "First stops (fixed by design)", BLUE, ORANGE, "Share (%)",
             "First stops are more concentrated than visits overall", "first_stops_vs_visits.png")
    dumbbell(site_table.sort_values("stay d (min)"), "listed (min)", "stay d (min)", "Listed visit time",
             "Inferred stay d (median)", ORANGE, BLUE, "Minutes",
             "Inferred stays exceed listed times at quick stops such as churches", "stay_vs_listed.png")
    return site_table


# ---------------------------------------------------------------------- #
# 4. Availability and concentration
# ---------------------------------------------------------------------- #

def spread_evenly(fixed, total):
    """Add the visits that are not fixed to the sites with the fewest, one level at a time."""
    x, left = fixed.astype(float), total - fixed.sum()
    while left > 1e-9:
        low = x <= x.min() + 1e-9
        if (~low).any():
            step = min(left / low.sum(), x[~low].min() - x.min())
        else:
            step = left / low.sum()
        x[low] += step
        left -= step * low.sum()
    return x


def fixed_share_of_hour(arena, hour):
    """Share of the person-minutes in this hour that come from fixed (first-stop or one-stop) entries."""
    overlap = (arena.end.clip(upper=60 * (hour + 1)) - arena.t.clip(lower=60 * hour)).clip(lower=0)
    return overlap[arena.fixed].sum() / overlap.sum()


def plot_availability(calendar, names, site_order):
    share = calendar.copy()
    share.columns = WEEKDAYS
    share.index = [names[i] for i in calendar.index]
    share = share.loc[site_order]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    image = ax.imshow(share.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(7), WEEKDAYS)
    ax.set_yticks(range(len(share)), share.index)
    ax.grid(False)
    for r, row in enumerate(share.values):
        for c, value in enumerate(row):
            if value < 0.5:
                ax.text(c, r, "n/a", ha="center", va="center", fontsize=8, color=INK)
    ax.set_title("Inferred availability: three sites close on Mondays; one has too few entries")
    fig.colorbar(image, ax=ax, label="Share of training days with entries", fraction=0.04)
    save("availability.png")


def plot_usual_gday(usual_gday):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    curves = (("other", BLUE, "Tuesday to Sunday"), ("monday", ORANGE, "Monday (fewer sites available)"))
    for day_type, color, label in curves:
        part = usual_gday[usual_gday.day_type == day_type]
        ax.plot(part.minute / 60, part.usual_gday, color=color, linewidth=2, label=label)
    ax.set(xlabel="Time of day (hour)", ylabel="Usual G_hat of visits so far", ylim=(0, 1),
           title="Visits so far look concentrated early in any day, hence GDAY is relative to its hour")
    ax.legend(frameon=False)
    save("usual_gday.png")


def section_concentration(days, train, calendar, names, site_table, usual_gday, available):
    say("## 4. Availability and concentration")
    say()
    say("Availability is an inferred calendar, not a timetable: a site counts as available on a weekday "
        "if it has entries on at least half of those training days.")
    say()
    plot_availability(calendar, names, site_table.site)

    train_days = days[days.split == "train"]
    say(f"- Available sites per training day: median {train_days.open_sites.median():.0f}, "
        f"minimum {train_days.open_sites.min()}")
    say(f"- Normalised Gini of a day, over available sites (G_hat): median {train_days.G_hat.median():.3f}; "
        f"with same-day open sites (sensitivity) {train_days.G_hat_same_day.median():.3f}")
    by_weekday = train_days.groupby(train_days.date.dt.dayofweek).agg(
        visits=("visits", "mean"), G_hat=("G_hat", "median"), sites=("open_sites", "median"))
    by_weekday.index = pd.Index(WEEKDAYS, name="Weekday")
    by_weekday["visits"] = by_weekday.visits.round(0).astype(int)
    table(by_weekday.round(3))
    plot_usual_gday(usual_gday)

    # How much do the fixed first stops limit spreading? Bound each day by
    # spreading the other visits as evenly as possible, ignoring routes and time.
    bound = []
    for date, part in train.groupby("date"):
        part = part[part.site.isin(available[date])]
        total = part.groupby("site").size().reindex(available[date], fill_value=0).to_numpy()
        fixed = part[part.fixed].groupby("site").size().reindex(available[date], fill_value=0).to_numpy()
        bound.append((gini_hat(total), gini_hat(spread_evenly(fixed, total.sum())), fixed.sum() / total.sum()))
    bound = np.array(bound)

    arena = train[train.site == 49]
    at_10 = arena[arena.t // 60 == 10].fixed.mean()
    at_11 = arena[arena.t // 60 == 11].fixed.mean()
    length = (arena.end - arena.t).clip(lower=0)
    say(f"- First stops and one-stop visits are fixed under every rule: {np.median(bound[:, 2]):.0%} of visits "
        f"on a median day and {arena.fixed.mean():.0%} of Arena's ({at_10:.0%} of its entries at 10:00, "
        f"{at_11:.0%} at 11:00)")
    say(f"- Reconstructed occupancy at Arena attributable to first-stop and one-stop entries (observed replay): "
        f"{fixed_share_of_hour(arena, 10):.0%} at 10:00, {fixed_share_of_hour(arena, 11):.0%} at 11:00, "
        f"{length[arena.fixed].sum() / length.sum():.0%} over the day. A rule does not move first-stop entries, "
        f"so its room to ease peaks they dominate is limited; in the simulation their stays are still sampled.")
    say(f"- An aggregate relaxation that keeps these counts and the same total visits, and ignores routes, "
        f"time and budgets, has a median lower bound on G_hat of {np.median(bound[:, 1]):.3f}, "
        f"against {np.median(bound[:, 0]):.3f} observed. Fixed counts alone do not explain the observed "
        f"concentration; this does not show that the bound is attainable.")
    say()


# ---------------------------------------------------------------------- #
# 5. Crowding by site and hour
# ---------------------------------------------------------------------- #

def section_crowding(train, sites, days, site_table, names):
    """Returns crowding C for each day, site and hour, used again in section 7."""
    say("## 5. Crowding by site and hour (training days)")
    say()
    rows = []
    for hour in range(8, 22):
        overlap = (train.end.clip(upper=60 * (hour + 1)) - train.t.clip(lower=60 * hour)).clip(lower=0)
        inside = (overlap.groupby([train.date, train.site]).sum() / 60).rename("inside").reset_index()
        inside["hour"] = hour
        rows.append(inside)
    occupancy = pd.concat(rows)
    occupancy["C"] = occupancy.inside / occupancy.site.map(sites.usual_peak)

    # Mean over all training days: a day with nobody inside counts as zero.
    training_days = days[days.split == "train"].date.nunique()
    mean_c = occupancy.groupby(["site", "hour"]).C.sum().unstack(fill_value=0) / training_days
    mean_c = mean_c.reindex(site_table.index)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    image = ax.imshow(mean_c.values, cmap="Blues", vmin=0, aspect="auto")
    ax.set_xticks(range(mean_c.shape[1]), mean_c.columns)
    ax.set_yticks(range(len(mean_c)), [names[i] for i in mean_c.index])
    ax.set_xlabel("Hour of the day")
    ax.grid(False)
    ax.set_title("Each site fills at its own hours, so crowding is a matter of when as well as where")
    fig.colorbar(image, ax=ax, label="Mean crowding C (1 = usual peak)", fraction=0.04)
    save("crowding_by_hour.png")
    say(f"Crowding C is reconstructed from entries (card holders only), not observed occupancy. "
        f"Averaged over all sites and training days, with empty hours counted as zero, "
        f"it is highest at {mean_c.mean().idxmax()}:00.")
    say()
    return occupancy


# ---------------------------------------------------------------------- #
# 6. Routes and the time budget
# ---------------------------------------------------------------------- #

def section_budget(multi):
    say("## 6. Routes and the time budget")
    say()
    used = multi[multi.split != "validation"].assign(added=lambda x: x.budget - x.span)
    widened = used[used.added > 1e-9]
    q = widened.added.quantile([0.5, 0.9, 0.95])
    say(f"- Budgets corrected so that the observed route fits under expected stays: "
        f"{len(widened) / len(used):.1%} of multi-stop visitors")
    say(f"- Minutes added among those: median {q[0.5]:.0f}, p90 {q[0.9]:.0f}, p95 {q[0.95]:.0f}, "
        f"maximum {widened.added.max():.0f}; {used.added.sum() / used.span.sum():.1%} of the total time span")
    say()

    by_stops = used.assign(stops=used.stops.clip(upper=7)).groupby("stops").added
    stop_table = pd.DataFrame({"visitors": by_stops.size(),
                               "widened (%)": by_stops.apply(lambda a: (a > 1e-9).mean()).mul(100).round(1),
                               "median minutes added": by_stops.apply(lambda a: a[a > 1e-9].median())})
    stop_table.index = pd.Index([f"{i}" if i < 7 else "7+" for i in stop_table.index], name="Stops")
    table(stop_table)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars = ax.bar(stop_table.index, stop_table["widened (%)"], color=BLUE, width=0.62)
    ax.bar_label(bars, fmt="%.0f%%", color=INK, padding=2)
    ax.set(xlabel="Stops on the observed route", ylabel="Budgets widened (%)", ylim=(0, 100),
           title="The longest routes need the budget correction most often")
    ax.grid(axis="x", visible=False)
    save("budget_by_stops.png")


# ---------------------------------------------------------------------- #
# 7. Observed route value
# ---------------------------------------------------------------------- #

def section_route_value(train, multi, occupancy, sites, walk):
    say("## 7. Observed route value")
    say()
    keep = train.merge(multi[["visitor"]], on="visitor").copy()
    keep["hour"] = (keep.t // 60).astype(int)
    keep = keep.merge(occupancy[["date", "site", "hour", "C"]], on=["date", "site", "hour"], how="left")
    keep["A"] = keep.site.map(sites.A)
    keep = keep.sort_values(["visitor", "stop"])
    keep["next_site"] = keep.groupby("visitor").site.shift(-1)
    has_next = keep.next_site.notna()
    keep["walk"] = 0.0
    keep.loc[has_next, "walk"] = [walk[(a, int(b))] / 60 for a, b in zip(keep.site[has_next], keep.next_site[has_next])]

    route = keep.groupby("visitor").agg(A=("A", "sum"), C=("C", "sum"), T=("walk", "sum"), stops=("A", "size"))
    route["V"] = route.A - ALPHA * route.C - BETA * route["T"]
    say(f"With alpha = {ALPHA}, beta = {BETA}, on {len(route):,} training multi-stop visitors:")
    say()
    terms = pd.Index(["sum of A", "alpha x sum of C", "beta x T", "V(r^O)"], name="Term")
    means = [route.A.mean(), ALPHA * route.C.mean(), BETA * route["T"].mean(), route.V.mean()]
    table(pd.DataFrame({"mean": means}, index=terms).round(3))
    say(f"- V(r^O) <= 0 for {(route.V <= 0).mean():.2%} of routes, which is why R_v divides by the sum of A; "
        f"that denominator has minimum {route.A.min():.3f} and median {route.A.median():.2f}")
    say()


# ---------------------------------------------------------------------- #
# 8. Other years
# ---------------------------------------------------------------------- #

def section_other_years(days):
    say("## 8. Other years (held out, 15 shared sites)")
    say()
    years = (("2019", days), ("2022", pd.read_csv(PROCESSED / "days_2022.csv")),
             ("Jan-Feb 2023", pd.read_csv(PROCESSED / "days_2023.csv")))
    rows = []
    for year, frame in years:
        unavailable = round(frame.visits_at_closed_sites.sum() / frame.visits.sum() * 100, 2)
        rows.append({"year": year, "days": len(frame), "median visits": int(frame.visits.median()),
                     "G_hat median": round(frame.G_hat.median(), 3),
                     "available sites median": int(frame.open_sites.median()),
                     "minimum": int(frame.open_sites.min()),
                     "visits at unavailable sites (%)": unavailable})
    table(pd.DataFrame(rows), index=False)
    say("Availability differs by year, and the calendars of 2022 and 2023 are inferred from those years' own entries. "
        "Compare methods within a year; do not read the G_hat of different years as equal difficulty.")


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(PROCESSED / "sites.csv", index_col=0)
    days = pd.read_csv(PROCESSED / "days_2019.csv", parse_dates=["date"])
    visits = pd.read_csv(PROCESSED / "visits_2019.csv.gz", parse_dates=["date"])
    visitors = pd.read_csv(PROCESSED / "visitors_2019.csv.gz", parse_dates=["date"])
    calendar = pd.read_csv(PROCESSED / "open_calendar.csv", index_col=0)
    usual_gday = pd.read_csv(PROCESSED / "usual_gday.csv")
    open_sites = pd.read_csv(PROCESSED / "open_sites_2019.csv", parse_dates=["date"])
    available = open_sites.groupby("date").site.apply(list).to_dict()
    listed = pd.read_csv(ROOT / "data" / "verona" / "poi_it_complete.csv").set_index("id").Time_Visit
    walk_table = pd.read_csv(ROOT / "data" / "verona" / "poi_time_travel.csv")
    walk = {(a, b): t for a, b, t in zip(walk_table.poi_start, walk_table.poi_dest, walk_table.time_travel)}
    names = sites.name.to_dict()

    train = visits[visits.split == "train"].copy()
    train["end"] = observed_occupancy_intervals(train, sites.d).exit.values
    train["fixed"] = (train.stop == 1) | (train.groupby("visitor").stop.transform("max") == 1)
    multi = visitors[visitors.stops >= 2]

    say("# VeronaCard EDA, aligned with the research design")
    say()
    say("Generated by `src/eda.py` from the tables of `preprocess.py`. One visitor = one card on one day.")
    say()
    section_sample(visits, visitors, multi, days)
    section_split(days)
    site_table = section_sites(sites, train, visitors, listed)
    section_concentration(days, train, calendar, names, site_table, usual_gday, available)
    occupancy = section_crowding(train, sites, days, site_table, names)
    section_budget(multi)
    section_route_value(train, multi, occupancy, sites, walk)
    section_other_years(days)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
