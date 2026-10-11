"""Exploratory data analysis of the VeronaCard entries, 2019 and 2022.

Run from the repository root:  python3 src/eda.py
Writes figures and tables to outputs/eda/ and a summary to outputs/eda/summary.md.

One visitor = one card on one day. Stay time at a site = time to the card's next entry
that day minus the walking time between the two sites (as in Migliorini et al. 2021);
the last stop of a day has no stay time.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA = Path("data/verona")
OUT = Path("outputs/eda")
OUT.mkdir(parents=True, exist_ok=True)

# Card site name -> id in poi_it_complete.csv. Castelvecchio and Teatro Romano each have two rows
# (site and museum); the museum rows are used, and the stay times below check that choice.
SITE_ID = {
    "Arena": 49, "Casa Giulietta": 61, "Torre Lamberti": 59, "Castelvecchio": 71, "Santa Anastasia": 54,
    "Duomo": 52, "Teatro Romano": 42, "Palazzo della Ragione": 58, "San Fermo": 62, "Tomba Giulietta": 202,
    "San Zeno": 63, "Museo Lapidario": 76, "Giardino Giusti": 75, "Museo Storia": 201, "Museo Miniscalchi": 300,
    "Palazzo Maffei Casa Museo": 301, "Museo Nazionale": 302, "Eataly Verona": 303,
}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
COLOURS = {2019: "#9aa5b1", 2022: "#2a78d6"}

lines = []  # summary.md


def say(text=""):
    print(text)
    lines.append(text)


def load(year):
    if year == 2019:
        d = pd.read_csv(DATA / "veronacard_2019_opendata.csv")
        d = d.rename(columns={"profilo": "profile"})
        time = pd.to_datetime(d.ora_visita, format="%H:%M")
    else:
        d = pd.read_csv(DATA / "veronacard_2022_original.csv")
        d = d.rename(columns={"profilo": "profile"})
        time = pd.to_datetime(d.ora_visita, format="%H:%M:%S")
    d["day"] = pd.to_datetime(d.data_visita, format="%d/%m/%y")
    d["minute"] = time.dt.hour * 60 + time.dt.minute + time.dt.second / 60
    d = d.rename(columns={"id_veronacard": "card", "sito_nome": "site"})
    d["visitor"] = d.card + "_" + d.day.dt.strftime("%Y%m%d")
    return d


def quality(year, d):
    say(f"### Data quality, {year}")
    say()
    say(f"- Entries: {len(d):,}; cards: {d.card.nunique():,}; visitors (card-days): {d.visitor.nunique():,}; "
        f"days: {d.day.nunique()}; sites: {d.site.nunique()}")
    say(f"- Missing values: {int(d[['card', 'day', 'minute', 'site']].isna().sum().sum())}")
    exact = d.duplicated(["card", "day", "minute", "site"]).sum()
    repeat = d.duplicated(["visitor", "site"]).sum()
    say(f"- Exact duplicate entries (same card, site and time): {exact:,} ({exact / len(d):.2%})")
    say(f"- Repeat entries at the same site on the same day: {repeat:,} ({repeat / len(d):.2%})")
    early = (d.minute < 8 * 60).sum()
    late = (d.minute >= 20 * 60).sum()
    say(f"- Entries before 08:00: {early:,}; from 20:00: {late:,}")
    stops = d.drop_duplicates(["visitor", "site"]).groupby("visitor").size()
    say(f"- Stops per visitor: mean {stops.mean():.2f}, median {stops.median():.0f}; "
        f"one stop only: {(stops == 1).mean():.1%}; 2 or more: {(stops >= 2).mean():.1%}")
    prof = d.drop_duplicates("card").profile.fillna("missing").value_counts(normalize=True)
    say("- Card types (share of cards): " + ", ".join(f"{k} {v:.1%}" for k, v in prof.items()))
    if "numero" in d:
        say("- 2022 field `numero` (meaning not documented): "
            + ", ".join(f"{k}: {v:,}" for k, v in d.numero.value_counts().sort_index().items()))
    say()
    return stops


def site_shares(d19, d22):
    s = pd.DataFrame({2019: d19.site.value_counts(normalize=True), 2022: d22.site.value_counts(normalize=True)})
    s = s.sort_values(2019, ascending=False)
    common = s.dropna()
    rank = common[2019].rank().corr(common[2022].rank())
    say("### Visits per site (share of entries)")
    say()
    say("| Site | 2019 | 2022 |")
    say("|---|---|---|")
    for site, r in s.iterrows():
        say(f"| {site} | {r[2019]:.1%} | {r[2022]:.1%} |".replace("nan%", "–"))
    say()
    say(f"Rank correlation over the {len(common)} sites in both years: {rank:.2f}. "
        f"Top 3 sites hold {s[2019].nlargest(3).sum():.0%} of entries in 2019 and {s[2022].nlargest(3).sum():.0%} in 2022.")
    say()

    ax = s.plot(kind="barh", figsize=(6.5, 5), color=[COLOURS[2019], COLOURS[2022]], width=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Share of entries")
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "site_shares.png", dpi=200)
    plt.close()


def time_patterns(d19, d22):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for year, d in ((2019, d19), (2022, d22)):
        days = d.groupby("day").size()
        by_wd = days.groupby(days.index.dayofweek).mean()
        axes[0].plot(WEEKDAYS, by_wd.values, marker="o", color=COLOURS[year], label=str(year))
        hour = (d.minute // 60).astype(int).value_counts(normalize=True).sort_index()
        axes[1].plot(hour.index, hour.values, marker="o", color=COLOURS[year], label=str(year))
        busiest = hour.idxmax()
        say(f"- {year}: busiest weekday {WEEKDAYS[by_wd.idxmax()]} ({by_wd.max():.0f} entries per day on average), "
            f"quietest {WEEKDAYS[by_wd.idxmin()]} ({by_wd.min():.0f}); busiest hour {busiest}:00 "
            f"({hour.max():.1%} of entries); 10:00–13:00 holds {hour.loc[10:12].sum():.0%}")
    axes[0].set_ylabel("Entries per day (mean)")
    axes[1].set_ylabel("Share of entries")
    axes[1].set_xlabel("Hour of day")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "weekday_hour.png", dpi=200)
    plt.close()
    say()


def stay_times(d, year, poi, walk):
    d = d.sort_values(["visitor", "minute"]).copy()
    d["next_site"] = d.groupby("visitor").site.shift(-1)
    d["gap"] = d.groupby("visitor").minute.shift(-1) - d.minute
    d = d.dropna(subset=["next_site"])
    d = d[d.site != d.next_site]
    d["walk"] = [walk.get((SITE_ID.get(a), SITE_ID.get(b))) for a, b in zip(d.site, d.next_site)]
    d["stay"] = d.gap - d.walk
    d = d[(d.stay > 0) & (d.stay < 300)]
    listed = poi.set_index("id").Time_Visit
    say(f"### Stay times, {year} (median minutes; listed = visit time in the attraction list)")
    say()
    say("| Site | Median stay | Listed | Stays |")
    say("|---|---|---|---|")
    rows = []
    for site, g in d.groupby("site"):
        rows.append((site, g.stay.median(), listed.get(SITE_ID.get(site)), len(g)))
    for site, med, lst, n in sorted(rows, key=lambda r: -r[1]):
        say(f"| {site} | {med:.0f} | {lst} | {n:,} |")
    say()
    return d


def transitions(d, year):
    d = d.sort_values(["visitor", "minute"])
    d = d.drop_duplicates(["visitor", "site"])
    nxt = d.groupby("visitor").site.shift(-1)
    pairs = (d.site + " → " + nxt).dropna().value_counts()
    first = d.groupby("visitor").site.first().value_counts(normalize=True)
    say(f"### Routes, {year}")
    say()
    say("- Most common first stop: " + ", ".join(f"{k} {v:.0%}" for k, v in first.head(3).items()))
    say("- Most common moves: " + "; ".join(f"{k} ({v:,})" for k, v in pairs.head(5).items()))
    say(f"- The top 5 moves are {pairs.head(5).sum() / pairs.sum():.0%} of all moves")
    say()


def crowding_heatmap(d, year, poi):
    """C = people inside in an hour / the site's usual peak, averaged over the busiest quarter of days."""
    visit = poi.set_index("id").Time_Visit
    d = d.copy()
    d["hour"] = (d.minute // 60).astype(int)
    per_hour = d.groupby(["site", "day", "hour"]).size().rename("entries").reset_index()
    per_hour["inside"] = per_hour.entries * per_hour.site.map(lambda s: visit.get(SITE_ID.get(s), 60)) / 60
    peak = per_hour.groupby("site").inside.quantile(0.95)
    per_hour["C"] = per_hour.inside / per_hour.site.map(peak)
    daily = d.groupby("day").size()
    busy = daily[daily >= daily.quantile(0.75)].index
    sites = d.site.value_counts().index
    grid = (per_hour[per_hour.day.isin(busy)]
            .pivot_table(index="site", columns="hour", values="C", aggfunc="sum")
            .reindex(index=sites, columns=range(9, 19)).fillna(0) / len(busy))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.imshow(grid.values, aspect="auto", cmap="Oranges", vmin=0, vmax=1)
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set_xticks(range(len(grid.columns)), [f"{h}" for h in grid.columns])
    ax.set_xlabel("Hour of day")
    fig.colorbar(im, ax=ax, label="Crowding C (1 = usual peak)")
    plt.tight_layout()
    plt.savefig(OUT / f"crowding_heatmap_{year}.png", dpi=200)
    plt.close()
    lines_sites = {"Arena": "#1f5fa8", "Castelvecchio": "#6fa3dc", "Casa Giulietta": "#3a9e5f",
                   "Santa Anastasia": "#d9531e", "Duomo": "#f2a35e"}
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for site, colour in lines_sites.items():
        ax.plot(grid.columns, grid.loc[site].values, marker="o", color=colour, label=site, linewidth=2)
    ax.axhline(1, color="grey", linestyle="--", linewidth=1)
    ax.text(16.3, 1.03, "usual peak (C = 1)", fontsize=9, color="grey")
    ax.set_xticks(list(grid.columns))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Crowding C")
    ax.set_ylim(0, 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / f"crowding_lines_{year}.png", dpi=200)
    plt.close()
    inside = (per_hour[per_hour.day.isin(busy) & (per_hour.hour == 11)]
              .groupby("site").inside.sum().reindex(sites).fillna(0) / len(busy)).sort_values()
    entries11 = (per_hour[per_hour.day.isin(busy) & (per_hour.hour == 11)]
                 .groupby("site").entries.sum().reindex(sites).fillna(0) / len(busy)).sort_values()
    tt = pd.read_csv(DATA / "poi_time_travel.csv")
    walk_min = {(x, y): t for x, y, t in zip(tt.poi_start, tt.poi_dest, tt.time_travel)}
    from_arena = {site: walk_min.get((SITE_ID["Arena"], SITE_ID.get(site))) for site in entries11.index}
    top2 = set(entries11.nlargest(4).index)
    quiet_near = {k for k in entries11.index if k not in top2 and from_arena[k] <= 10 and entries11[k] < entries11.max() / 4}
    shown = entries11
    colours = ["#8c2d04" if k == "Arena" else "#f28c4f" if k in top2 else "#2a78d6" if k in quiet_near else "#c8ced6"
               for k in shown.index]
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.barh(shown.index, shown.values, color=colours, height=0.75)
    ax.tick_params(axis="y", labelsize=8)
    for y, v in enumerate(shown.values):
        ax.text(v + 1.5, y, f"{v:.0f}", va="center", fontsize=9)
    ax.legend(handles=[plt.matplotlib.patches.Patch(color="#8c2d04", label="Arena (walking times measured from here)"),
                       plt.matplotlib.patches.Patch(color="#f28c4f", label="next three busiest"),
                       plt.matplotlib.patches.Patch(color="#2a78d6", label="quiet attractions, 5–10 min walk from the Arena"),
                       plt.matplotlib.patches.Patch(color="#c8ced6", label="other attractions")],
              frameon=False, loc="lower right", fontsize=9)
    ax.set_xlabel("Entries per hour at 11:00 (busiest days, 2019)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / f"inside_at_11_{year}.png", dpi=200)
    plt.close()
    say(f"- Entries 11:00-12:00 (busy days): " + ", ".join(f"{k} {v:.0f}" for k, v in entries11.sort_values(ascending=False).items()))
    say(f"- People inside at 11:00 (busy days, card holders): "
        + ", ".join(f"{k} {v:.0f}" for k, v in inside.sort_values(ascending=False).items()))
    xy = d.groupby("site")[["sito_longitudine", "sito_latitudine"]].first() if "sito_longitudine" in d else None
    if xy is not None:
        tt = pd.read_csv(DATA / "poi_time_travel.csv")
        walk = {(x, y): t for x, y, t in zip(tt.poi_start, tt.poi_dest, tt.time_travel)}
        offsets = {"Museo Miniscalchi": (-8, 6), "Santa Anastasia": (6, 6), "Duomo": (-8, 6)}
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        for site, (lon, lat) in xy.iterrows():
            n = inside.get(site, 0)
            ax.scatter(lon, lat, s=max(n, 0.5) * 18, color="#d9531e" if n >= inside.max() / 2 else "#2a78d6",
                       alpha=0.6, edgecolor="white")
            minutes = walk.get((SITE_ID["Arena"], SITE_ID.get(site)))
            label = f"{site} ({n:.0f})" if site == "Arena" else f"{site} ({n:.0f}; {minutes:.0f} min)"
            dx, dy = offsets.get(site, (6, 4))
            ax.annotate(label, (lon, lat), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                        ha="right" if dx < 0 else "left")
        ax.set_aspect(1 / 0.70)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title("Card holders inside at 11:00 on the busiest quarter of days (circle size);\n"
                     "minutes = walking time from the Arena", fontsize=9)
        plt.tight_layout()
        plt.savefig(OUT / f"map_at_11_{year}.png", dpi=200)
        plt.close()
        near = [(site, walk.get((SITE_ID["Arena"], SITE_ID.get(site))), inside.get(site, 0)) for site in xy.index if site != "Arena"]
        quiet_near = [f"{k} ({m:.0f} min, {n:.0f})" for k, m, n in sorted(near, key=lambda r: r[1]) if m <= 10 and n < 10]
        say("- Quiet sites within 10 minutes' walk of the Arena at 11:00: " + ", ".join(quiet_near))
    at11 = grid[11].sort_values(ascending=False)
    say(f"### Crowding by site and hour, {year} (busiest quarter of days, {len(busy)} days)")
    say()
    say(f"- At 11:00: {', '.join(f'{k} {v:.2f}' for k, v in at11.head(3).items())}; "
        f"meanwhile {', '.join(f'{k} {v:.2f}' for k, v in at11.tail(3).items())}")
    say(f"- Sites above 0.5 at 11:00: {(at11 > 0.5).sum()} of {len(at11)}; below 0.2: {(at11 < 0.2).sum()}")
    say()


if __name__ == "__main__":
    poi = pd.read_csv(DATA / "poi_it_complete.csv")
    tt = pd.read_csv(DATA / "poi_time_travel.csv")
    walk = {(a, b): t for a, b, t in zip(tt.poi_start, tt.poi_dest, tt.time_travel)}

    d19, d22 = load(2019), load(2022)
    say("# VeronaCard EDA, 2019 and 2022")
    say()
    say("Generated by `src/eda.py`. One visitor = one card on one day.")
    say()
    quality(2019, d19)
    quality(2022, d22)
    site_shares(d19, d22)
    say("### Weekday and hour")
    say()
    time_patterns(d19, d22)
    stay_times(d19, 2019, poi, walk)
    stay_times(d22, 2022, poi, walk)
    transitions(d19, 2019)
    transitions(d22, 2022)
    crowding_heatmap(d19, 2019, poi)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT)
