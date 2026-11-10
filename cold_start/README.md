# Cold start tables

Precomputed top-10 lists for a player who has no library yet. **This workbook is from an older clustering run:** it contains cluster scopes 0–15, while the current `new_package/04_clusters/` data contains 18 clusters (0–17) with different labels and assignments. Do not join the workbook to the current clusters by ID. NextQuest reads the current assignments and recomputes the same review/player ranking from saved data at startup.

One row in `cold_start_tables.xlsx` (`tops`) is one ranked game in one list. The switch key is:

- `scope`: `all`, or `cluster_<id>`
- `budget`: `all`, `0`, `20`, or `80` (dataset price units)
- `hardware`: `all`, `low`, `medium`, or `high`
- `rank`: 1–10

`game_specs` is every clustered game (6,750), not only the tops. `game_tiers.csv` is the full 7,430-game catalog with a tier when the spec text could be read.

## What the score is trying to do

Recommend a game that is both well reviewed and actually being played. A tiny game with a perfect review ratio should not outrank Counter-Strike 2, and a huge game with poor reviews should not sit at the top only because of its player count.

```
score = 100 * sqrt(Q * P)

Q = (positive + 100 * catalog_positive_fraction) / (positive + negative + 100)
P = log(1 + current_players) / log(1 + max current_players in the catalog)
```

`Q` is review quality with a catalog-wide prior, so a handful of reviews cannot jump to 100%. `P` is log-scaled players online at fetch time. The busiest game in that fetch is 1. `peak_ccu` stays in the file as the old snapshot peak and is not used in the score. The square root means a game has to be decent on both sides.

Games with no positive count, no negative count, or no live player count get no score and never enter a top 10. A missing price is treated as 0 (free). A recorded price is kept.

## Hardware tiers

Reference PC, called medium: **16 GB RAM** and an **RTX 3060-class GPU**.

| Signal | low | medium | high |
|---|---|---|---|
| RAM | 8 GB or less | above 8 GB up to 16 GB | above 16 GB |
| GPU | below RTX 2060 / RX 5700 | RTX 2060 through RTX 4060 | RTX 3070 / RX 6700 and stronger |

The game tier is the stricter of the two. 16 GB plus an RTX 4090 is high. If only one of RAM or GPU can be read, that one sets the tier. If neither can be read, the tier stays empty.

Text comes from **Recommended** first. **Minimum** is used only when Recommended has neither RAM nor a GPU string. Minimum is never copied into the Recommended fields.

A player filter is cumulative, matching the discovery UI:

- `low` shows low games
- `medium` shows low and medium
- `high` shows low, medium, and high
- `all` does not filter on tier

Empty-tier games can still appear under `hardware=all`. They are left out of the three PC filters. We are not filling those gaps.

## Data sources

Reviews, price, and cluster ids come from `new_package`:

- `new_package/01_catalog/games_for_features.csv`
- `new_package/04_clusters/game_clusters.csv`
- `new_package/04_clusters/cluster_summary.csv`

Hardware text is `game_hardware_specs.csv`, passed as `HARDWARE_SPECS`. This script does not fetch Steam. `spec_source` is the source of the level actually used for the tier. `spec_url` is the original link. `used` is `recommended` or `minimum`.

| `spec_source` | Where it came from |
|---|---|
| `zenodo_2025` | Ready dataset, Zenodo record 17266923 |
| `hf_steam` | Ready dataset, Hugging Face `frankjc2022/steam-dataset` |
| `github_2019` | GitHub archive, `yuhkulele/Steam-Recommendations` `steam_requirements_data.csv` (2019) |
| `steam_live` | Steam store page retrieved in the browser on 30 Sep 2026 |
| empty | No readable RAM or GPU, so no tier |

On the 6,750 clustered games the counts are about: Zenodo 3,913, Steam live 1,675, Hugging Face 598, GitHub 72, and 492 with no tier.

## Rebuild

```
python cold_start/build_cold_start_tables.py
```

Needs `HARDWARE_SPECS` pointing at `game_hardware_specs.csv`, plus `pandas`, `numpy`, and `openpyxl`. Outputs land in this folder.
