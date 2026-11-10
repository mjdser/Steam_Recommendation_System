# What we built today

Steam user libraries scored into a top 10, then a game catalog built from those games plus the popular Steam list, then each game given a 93-label feature vector, then the games clustered.

This folder is the package. Review it before pushing. Do not add `gemini_api_key.txt`.

## Folders

| Folder | What it is |
|---|---|
| `01_catalog` | 7,430 games with descriptions |
| `02_taxonomy` | The 93 labels |
| `03_features` | Polished Gemini scores |
| `04_clusters` | 18 clusters |
| `06_code` | Scripts that built this package |
| `07_users` | Each user's interest top 10 |
| `08_item_neighbors` | Top-10 co-Top10 neighbors per game (item–item CF table) |

## 1. User libraries and interest top 10

Source library (not inside this folder):

`steam_user_api_data_26_9.csv` (the public-user library export; it is not inside this folder)

22,605 public users. Each row has a `game_library` JSON list. Every game has `appid`, `name`, `playtime_forever`, and `playtime_2weeks`. Those playtimes are **minutes**. The scorer converts them to hours and keeps hours for the whole formula.

Script: `06_code/score_interest_top10.py`

```
interest = 0.40*s_recent + 0.35*s_play + 0.15*s_momentum + 0.10*s_invest
```

- `s_recent` is how much this game was played in the last 2 weeks, compared with this user's other games.
- `s_play` is lifetime play, compared with this user's other games.
- `s_momentum` is recent play divided by lifetime play.
- `s_invest` is lifetime play against `H_cap` of **200 hours**.

20,064 users have at least one played game. 6,902 unique games appear in those top 10s.

| File | What it is |
|---|---|
| `07_users/libraries_top10.csv` | One row per user. Columns `top1_game`, `top1_hours`, `top1_interest`, through top 10. |
| `07_users/libraries_top10_coverage.json` | How those top 10s overlap the Steam popular list. |

## 2. Game catalog

Decision: use the Steam **top 2500** popular games, then **force-add** every user top-10 game that is not already in that 2500. Top 5000 was worse: only 61% of those extra popular games show up in anyone's top 10, and you still have to force-add.

Script: `06_code/build_game_catalog.py`

Catalog size: **7,430** games.

Description comes from, in order:

1. Steam 2026 dump (stefanotriscali)
2. March 2025 dump (artermiloff) if 2026 has no text
3. Live Steam store page, only when neither dump has text
4. A sibling edition or next yearly sports game (GTA Definitive, Forza Horizon 5, FIFA/FC, F1, NBA 2K) when the owned id has no text

The library `appid` is kept. Portal 2 stays `620`. Text may be copied from another store id.

| File | What it is |
|---|---|
| `01_catalog/games_for_features.csv` | Full descriptions. |
| `01_catalog/games_for_features_summary.json` | Counts. |

`inclusion` is `top2500` (popular candidate only), `force` (user top 10 outside that 2500), or `both`.

## 3. Feature vectors

Script: `06_code/score_games_gemini.py`

Model: `gemini-3.8-flash`. Eight games per call. Description is the score. A community genre or tag does not raise a score the text already earned. If the text never mentions it, a genre can be at most **0.4** and a tag at most **0.3**. 1.0 only when the description shows the label is central. Indie, Free to Play, and NSFW are ignored. Usually 4 to 12 labels per game.

93 labels: 9 genres (Action, Adventure, Casual, Simulation, Strategy, RPG, Sports, Racing, Massively Multiplayer) and 84 tags. Order is in `02_taxonomy/dimension_order.json`. Definitions are in `02_taxonomy/taxonomy_v2.json`.

Gemini result after the full pass: 6,731 scored, 503 with no usable description, 180 not a game, 16 failed calls.

A later polish pass fixed about 420 games from the full descriptions. Famous titles stayed as they were (Counter-Strike 2 still FPS and Shooter at 0.9). Dota 2's MOBA score is still only 0.3 because the store text never says MOBA.

| File | What it is |
|---|---|
| `03_features/game_features.csv` | The feature sheet. One row per game, 93 score columns. |

## 4. Clusters

Script: `06_code/cluster_game_features.py`

Method: L2-normalize the 93 scores, PCA down to 54 components (90% of variance), MiniBatchKMeans. Silhouette picked **18** clusters.

6,750 games clustered. 680 with no vector left unclustered.

| File | What it is |
|---|---|
| `04_clusters/game_clusters.csv` | Every game, its cluster id, and a short label. |
| `04_clusters/cluster_summary.csv` | The 18 groups, what they mean, and example titles. |
| `04_clusters/cluster_config.json` | Settings and counts. |

The groups read as shelves: FPS shooters, action RPGs, survival crafting, sports and racing, story games, and so on. Two weak placements come from the scores: Dota 2 sits with general competitive multiplayer because MOBA is only 0.3, and The Witcher 3 sits with fantasy RPGs that include turn-based games.

## Scripts

| Script | Job |
|---|---|
| `06_code/score_interest_top10.py` | User library to interest top 10. |
| `06_code/build_game_catalog.py` | Top 2500 plus force-adds, with descriptions. |
| `06_code/score_games_gemini.py` | Gemini feature scores. Needs an API key in the environment, not in git. |
| `06_code/cluster_game_features.py` | 18 clusters from the polished sheet. |

## Before you push

- This folder is the one to review.
- Do not stage an API key.
- Features: `03_features/game_features.csv`
- Clusters: `04_clusters/game_clusters.csv`
