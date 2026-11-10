# 01_catalog — games to tag

Built by `06_code/build_game_catalog.py`.

**Catalog = 2026 Steam top 2500 + every unique user top-10 game.**

| File | Job |
|------|-----|
| `games_for_features.csv` | One row per game, full description. |
| `games_for_features_summary.json` | Counts. |

`inclusion`: `top2500` candidate-only, `force` user top-10 not in that 2500, `both` in both.
