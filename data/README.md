# Data

## Verona (main dataset)

VeronaCard is Verona's city pass. Each time a visitor enters an attraction with the card, the entry is recorded.

| File | What it is | Source |
|---|---|---|
| `verona/veronacard_2019_opendata.csv` | One row per entry: card id, card type (24h/48h), activation date, visit date, visit time (to the minute), site name, latitude, longitude | Comune di Verona, "Dati Veronacard 2019", [dati.veneto.it](https://dati.veneto.it/opendata/Dati_Veronacard_2019), CC BY 4.0 |
| `verona/poi_it_complete.csv` | Attraction list with typical visit time (minutes) and capacity (`max_crowd`, given for 10 attractions) | [smigliorini/itinerary-drl](https://github.com/smigliorini/itinerary-drl) |
| `verona/poi_time_travel.csv` | Walking time between attraction pairs (minutes) | same |
| `verona/log_crowd.csv` | Estimated crowd per attraction per hour, from Nov 2022 (not checked yet) | same |
| `verona/poi_popularity_train.csv` | Popularity table used in that repository (not checked yet) | same |

Quick numbers for 2019:

- 397,562 entries, 79,825 cards (visitors), 15 sites
- Stops per visitor: mean 4.98, median 5; 97% have at least 2 stops, 89% at least 3
- Entries per day: min 113, median 1,017, max 3,981
- Gini of entries per site: 0.440 (0.472 after dividing by the maximum 1 − 1/15)

Limits: only card holders (mostly culture-focused visitors); entries only, no exit times, so stay times come from the attraction list; capacity is missing for 5 sites.

Please cite: Migliorini, Carra and Belussi (2021), IEEE TETC 9(4):1765-1779, doi 10.1109/TETC.2019.2920484; Dalla Vecchia et al. (2024), Information Technology & Tourism 26(3):449-484, doi 10.1007/s40558-024-00288-x.

## Toronto (backup, not in this repository)

Flickr trips from Lim et al. (2015), as released with Chen et al. (2016): 29 attractions, 6,057 trips, 1,395 users. Most trips have one stop, and there are no stay times or capacities, so it is kept only as a backup or a second city.

The source repository gives no licence and the data contains Flickr user ids, so it is not copied here. To get it:

```
mkdir -p data/toronto
curl -L -o data/toronto/poi-Toro.csv  https://raw.githubusercontent.com/computationalmedia/tour-cikm16/master/data/poi-Toro.csv
curl -L -o data/toronto/traj-Toro.csv https://raw.githubusercontent.com/computationalmedia/tour-cikm16/master/data/traj-Toro.csv
```

`data/toronto/` is in `.gitignore`.
