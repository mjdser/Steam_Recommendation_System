# NextQuest

NextQuest is a working web interface for three routes into the repository's game catalog: free-text descriptions, personalized recommendations from a Steam library, and cluster discovery for a new player. The server and responsive frontend live in `product_app/`; they read the existing catalog, cluster data, and saved model artifacts.

## Run locally

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m pip install -r product_app/requirements.txt
python3 -m product_app.backend.server
```

Open `http://127.0.0.1:8000/`. The server binds to localhost by default and serves the UI and API from the same origin. The catalog loads when the server starts.

Cluster discovery works without credentials. It uses the repository's current 18 game clusters and saved review, player-count, price, and hardware data.

For Steam-history recommendations, the **server owner** configures a Steam Web API key locally:

```bash
cp product_app/.env.example product_app/.env
chmod 600 product_app/.env
```

Set `STEAM_API_KEY` in `product_app/.env`, or set it in the server's environment. The environment value takes precedence, and changes to `product_app/.env` are read on the next Steam request. The key stays on the server; it is never requested from or sent to a player's browser. Each player enters **only their 17-digit SteamID64**. The account's **Game details** must be public for Steam to return its library, as [Steam's GetOwnedGames documentation](https://partner.steamgames.com/doc/webapi/IPlayerService) explains. The server uses the key to request owned games from Steam over HTTPS and fetch the player's display name through [ISteamUser/GetPlayerSummaries/v2](https://partner.steamgames.com/doc/webapi/ISteamUser). A successful recommendation shows the display name when available and falls back to the SteamID when this optional metadata cannot be retrieved. For an empty or unavailable library, choose a cluster instead.

Steam requires a key for `GetOwnedGames`. If none is configured, the Steam form shows a setup notice and cluster discovery remains available. A server owner with a Steam account can [register a user Web API key](https://steamcommunity.com/dev/apikey) under Steam's [key requirements](https://partner.steamgames.com/doc/webapi_overview/auth); players do not need keys.

For free-text descriptions, configure Gemini on the server:

```bash
cp -n converting-description-to-vector/.env.example converting-description-to-vector/.env
chmod 600 converting-description-to-vector/.env
```

Set `GEMINI_API_KEY` in that local `.env` file, or set it in the server's environment, then restart the server. The Gemini key is never sent to the browser. Without it, Steam-history recommendations still work when `STEAM_API_KEY` is configured; cluster discovery works without either key.

The Steam route runs the saved `model_rank_hybrid.json` XGBoost model with its matching 670-column bundle. It computes interest from the player's playtime, puts up to six scored catalog games in the model's history slots, scores unowned catalog games, and returns ranked games. The newer `rank_v3_interact` model cannot run from this checkout because its training-derived similarity matrix and popularity data under `data/` are absent; using zero-filled replacements would change what that model was trained to predict.

## API

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Catalog count, cluster availability, model name, and whether Gemini and Steam are configured. No key values are returned. |
| `GET /api/clusters` | Current cluster IDs, labels, characteristic features, and examples. |
| `GET /api/cluster-recommendations?cluster_id=2&budget=20&hardware=medium&limit=6` | Ranked games in one cluster, with cumulative PC filter and catalog price ceiling. |
| `POST /api/steam/recommend` | Personalized XGBoost ranking. JSON body: `{"steamid":"76561198000000000","limit":6}`. `limit` is optional. Successful responses include `steam_username` as a display-name string or `null` if the optional lookup is unavailable; the interface shows the SteamID in that case. The Steam key stays on the server and is never included in the request. |
| `GET /api/discover?preset=story&limit=6` | Real catalog ranking for a curated feature profile. Presets: `story`, `cozy`, `coop`, `strategy`, `racing`. |
| `POST /api/recommend` | Gemini interpretation and real catalog ranking. JSON body: `{"query":"A cooperative story game","limit":5}`. |

The server validates input and reports errors as JSON. Scores have **different meanings by route**: text scores measure trait match; the Steam model emits relative ranking signals; cluster discovery uses a 0–100 review/player score. None are guarantees that a player will like a game.

Cluster choices come from `new_package/04_clusters/cluster_summary.csv` and current assignments from `game_clusters.csv`. The older `cold_start_tables.xlsx` has cluster IDs and labels from another clustering run, so the web route recomputes its saved-data ranking from the current assignments, catalog reviews/prices, `current_players.jsonl`, and `game_tiers.csv`. The player counts are a saved snapshot, not live Steam traffic. Budgets are `all`, `0`, `20`, or `80` in catalog price units; PC filters are `all`, `low`, `medium`, or `high`, where higher tiers include the lower ones.

## Verify

Run the app's loopback HTTP/model tests and the existing text recommender tests from the repository root:

```bash
python3 -m unittest discover -s product_app/backend/tests -v
python3 -m unittest discover -s tests -v
```

The tests exercise the real cluster data, saved XGBoost model, Steam response parsing with a fake upstream transport, actual HTTP routes, input errors, and text ranking with a mocked Gemini response. They do not spend API quota. A live Steam check requires the server's `STEAM_API_KEY` and a player's public SteamID64; a live Gemini check requires the server-side Gemini key. Steam cover art and optional web fonts load from their providers when available; the interface has local visual fallbacks.

## Architecture

`frontend/` contains the interface. `backend/server.py` serves those files and dispatches requests to three engines: the existing `TextGameRecommender` plus Gemini converter, `SteamHistoryRecommender` with the saved XGBoost model and live owned-library request, and `ColdStartCatalog` with current cluster assignments and saved discovery signals. The server caches curated discovery rankings and never substitutes a static list of games for real catalog results.

## Design references

The interface adapts discovery patterns from [Steam Labs' Interactive Recommender and Discovery Queue](https://store.steampowered.com/labs/) and named recommendation groups discussed by [Netflix Engineering](https://netflixtechblog.com/learning-a-personalized-homepage-aa8ec670359a). Its bold color and interaction direction draws on the [Awwwards Site of the Day: Carnival with Spotify](https://www.awwwards.com/sites/carnival-with-spotify), without copying that site. Visible matching traits and an explicit live-search status follow [Nielsen Norman Group's research on recommendation clarity](https://www.nngroup.com/articles/machine-learning-ux/); loading and result messages follow [W3C guidance on status messages](https://www.w3.org/WAI/WCAG21/Understanding/status-messages.html).
