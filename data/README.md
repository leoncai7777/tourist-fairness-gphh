# Data

## Verona (main dataset)

VeronaCard is Verona's city pass. Each time a visitor enters an attraction with the card, the entry is recorded. In this project one **visitor** is one card on one day (a 48-hour card used on two days counts as two visitors).

| File | What it is | Source and licence |
|---|---|---|
| `verona/veronacard_2019_opendata.csv` | Full year 2019. One row per entry: card id, card type (24h/48h), activation date, visit date, visit time (to the minute), site name, latitude, longitude | Comune di Verona, "Dati Veronacard 2019", [dati.veneto.it](https://dati.veneto.it/opendata/Dati_Veronacard_2019), CC BY 4.0 |
| `verona/veronacard_2022_original.csv` | Full year 2022 (1 Jan to 30 Dec). One row per entry: visit date, visit time (to the second), site name, device, card id, activation date, "in", count, card type | [smigliorini/itinerary-drl](https://github.com/smigliorini/itinerary-drl), released with Dalla Vecchia et al. (2024). **Not in this repository: download it into `data/verona/`.** Its MIT licence covers its code; the data's own licence is not stated; the data originate from the VeronaCard system of the Comune di Verona |
| `verona/data_2023.csv` | 1 Jan to 27 Feb 2023 only | Same source as the 2022 file. **Not in this repository: download it into `data/verona/`.** The data's own licence is not stated |
| `verona/poi_it_complete.csv` | Attraction list with typical visit time (minutes) and capacity (`max_crowd`, given for some attractions) | Same repository (itinerary-drl). **Not in this repository: download it into `data/verona/`.** The data's own licence is not stated |
| `verona/poi_time_travel.csv` | Walking time between attraction pairs (minutes), computed along the street network with OSMnx (Dalla Vecchia et al. 2024, p.470) | Same repository (itinerary-drl). **Not in this repository: download it into `data/verona/`.** The data's own licence is not stated |
| `verona/log_crowd.csv` | Estimated people per attraction at 00, 08, 12 and 16 h, 8 Nov to 14 Dec 2022 (31 days) | Same repository (itinerary-drl). **Not in this repository: download it into `data/verona/`.** The data's own licence is not stated |
| `verona/poi_popularity_train.csv` | Popularity table used in that repository | Same repository (itinerary-drl). **Not in this repository: download it into `data/verona/`.** The data's own licence is not stated |

To get the files that are not in this repository:

```
cd data/verona
for f in veronacard_2022_original.csv data_2023.csv poi_it_complete.csv poi_time_travel.csv; do
  curl -L -O https://raw.githubusercontent.com/smigliorini/itinerary-drl/main/data/$f
done
```

Quick numbers (computed with pandas from these files):

| | 2019 | 2022 | 2023 (Jan–Feb) |
|---|---|---|---|
| Entries | 397,562 | 382,344 | 58,599 |
| Cards | 79,825 | 89,549 | 12,850 |
| Visitors (card-days) | 112,821 | 124,662 | 18,771 |
| Attractions | 15 | 18 | 18 |
| Stops per visitor (mean / median) | 3.52 / 3 | 3.07 / 3 | |
| Entries per day (median / max) | 1,017 / 3,981 | 984 / 5,080 | |

2022 has three attractions not in 2019: Eataly Verona, Museo Nazionale, Palazzo Maffei Casa Museo. January to May 2022 is still below 2019 (COVID), from June it is at or above 2019 levels.

Limits: only card holders (mostly culture-focused visitors); entries only, no exit times, so stay times are inferred from the gaps between entries (`src/preprocess.py`); capacity is missing or doubtful for several sites.

Please cite: Comune di Verona (open data, 2019); Migliorini, Carra and Belussi (2021), IEEE TETC 9(4):1765-1779, doi 10.1109/TETC.2019.2920484; Dalla Vecchia et al. (2024), Information Technology & Tourism 26(3):449-484, doi 10.1007/s40558-024-00288-x.
