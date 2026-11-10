# Steam Game Recommender

This repository turns Steam libraries and game descriptions into personalized game recommendations. It contains prepared game data, 93 interpretable genre/tag scores, game clusters, user interest histories, three trained XGBoost models, and cold-start recommendation tables.

The dataset is `new_package/`.

**Recommend from a game description:** [recommend_games.py](converting-description-to-vector/recommend_games.py) converts free text to the shared 93-feature vector with Gemini, then ranks usable catalog games by feature match. See the [usage guide](converting-description-to-vector/README.md) for setup, CLI and Python examples. This text-only path does not require a Steam library or modify the trained models.

**Try the web app:** [NextQuest](product_app/README.md) serves three pathways through one local browser interface: Gemini-assisted text matching, personalized Steam-library recommendations with the saved XGBoost model, and discovery from the current 18 game clusters for a new player. Its frontend and backend live together in `product_app/`. For Steam recommendations, the server owner sets `STEAM_API_KEY` in `product_app/.env` or the server environment; players enter only their 17-digit SteamID64 and must make their Steam **Game details** public. Successful recommendations show the player's Steam display name when available, using the same server key with [ISteamUser/GetPlayerSummaries/v2](https://partner.steamgames.com/doc/webapi/ISteamUser); if that optional lookup fails or has no name, the app shows the SteamID instead. The key stays on the server and never goes to the browser. See the [web app setup and API guide](product_app/README.md) for details.

**How the existing pipeline fits together**

1. Download Steam game/review datasets and collect public user libraries.
2. Score each user's played games and retain their ten highest-interest games.
3. Build a catalog from popular Steam games plus games in those user histories.
4. Assign genre/tag scores with Gemini, then cluster games with similar scores.
5. Prepare user/game training rows, train XGBoost models, and evaluate rankings.
6. Recommend using user history, or use prepared discovery lists for new users.

The catalog contains 7,430 rows; 6,750 have usable polished features and cluster assignments. The remaining 680 rows lack usable game vectors or represent non-games.

Game vectors contain **93 labels**: nine genres and 84 tags. XGBoost adds `n_labels`, giving **94 game features**. Personalized models combine six history slots with candidate features, producing **670 input columns**. Load each model's matching bundle to preserve its exact column order.

**Root files**

| File | Purpose |
| --- | --- |
| [README.md](README.md) | Project overview, workflow, and complete guide to the active repository files. |
| [models_readme.md](models_readme.md) | Describes the trained models, ranking metrics, and prediction examples. |
| [training_readme.md](training_readme.md) | Explains interest prediction, data splits, history slots, and training settings. |
| [.gitignore](.gitignore) | Excludes Python caches, local environments, credentials files, and package build metadata from Git. |
| [.gitattributes](.gitattributes) | Configures Git LFS for selected files in the outdated package. |

**Data collection and modeling — `src/`**

| File | Purpose |
| --- | --- |
| [load_kaggle_data.py](src/load_kaggle_data.py) | Downloads Steam game and review datasets into the local `data/` directory. |
| [merge_review_files.py](src/merge_review_files.py) | Merges review CSV shards in chunks and checks the combined output. |
| [data_review.py](src/data_review.py) | Prints dataset structure, missing-value counts, and basic playtime/price statistics. |
| [collect_steam_data.py](src/collect_steam_data.py) | Collects public Steam profiles, libraries, and achievement statistics with resumable progress. |
| [inspect_API_data.py](src/inspect_API_data.py) | Summarizes collected user data and generates exploratory charts. |
| [inspect_data.ipynb](src/inspect_data.ipynb) | Inspects user/game data and exports user histories joined to game features. |
| [prepare_data_for_training.ipynb](src/prepare_data_for_training.ipynb) | Converts histories into user/game rows and creates cleaned train/validation/test splits. |
| [train_models.ipynb](src/train_models.ipynb) | Trains the personalized and candidate-only XGBoost models and saves their bundles. |
| [eval_xgboost.ipynb](src/eval_xgboost.ipynb) | Compares saved models using sampled candidate pools and recommendation metrics. |
| [evaluate_xgboost_script.py](src/evaluate_xgboost_script.py) | Runs the model comparison as a Python script with configured data/model paths. |
| [rec_eval.py](src/rec_eval.py) | Provides recall, precision, NDCG, reciprocal rank, coverage, novelty, and evaluation helpers. |

**Saved models — `models/`**

| File | Purpose |
| --- | --- |
| [model2_slot_reg.json](models/model2_slot_reg.json) | Primary regularized XGBoost model using six user-history slots and a candidate game. |
| [model2_slot_reg_bundle.json](models/model2_slot_reg_bundle.json) | Stores the primary model's feature order, slot count, and selected training iteration. |
| [model_slot_deep.json](models/model_slot_deep.json) | Earlier, deeper personalized model retained for comparison. |
| [model_slot_deep_bundle.json](models/model_slot_deep_bundle.json) | Stores feature layout and prediction metadata for the deeper model. |
| [model_cand_only.json](models/model_cand_only.json) | Baseline model that scores games without user-history input. |
| [model_cand_only_bundle.json](models/model_cand_only_bundle.json) | Stores the candidate-only model's 94-feature layout and prediction metadata. |

**Active data package — `new_package/`**

| File | Purpose |
| --- | --- |
| [README.md](new_package/README.md) | Documents the package's interest scoring, catalog construction, features, and clusters. |

`01_catalog/` contains game metadata and descriptions.

| File | Purpose |
| --- | --- |
| [README.md](new_package/01_catalog/README.md) | Explains catalog membership, description sources, and catalog outputs. |
| [games_for_features.csv](new_package/01_catalog/games_for_features.csv) | Current 7,430-game catalog with full descriptions and source metadata. |
| [games_for_features_summary.json](new_package/01_catalog/games_for_features_summary.json) | Records catalog counts and description coverage. |

`02_taxonomy/` defines the shared feature vocabulary.

| File | Purpose |
| --- | --- |
| [README.md](new_package/02_taxonomy/README.md) | Introduces the taxonomy and its common feature order. |
| [taxonomy_v2.json](new_package/02_taxonomy/taxonomy_v2.json) | Names and definitions of the 93 game labels. |
| [dimension_order.json](new_package/02_taxonomy/dimension_order.json) | Lists labels in their fixed vector-column order. |

`03_features/` contains the scores used for clustering and recommendations.

| File | Purpose |
| --- | --- |
| [game_features.csv](new_package/03_features/game_features.csv) | Final game feature scores, status, and label counts indexed by Steam app ID. |

`04_clusters/` groups games by feature similarity.

| File | Purpose |
| --- | --- |
| [game_clusters.csv](new_package/04_clusters/game_clusters.csv) | Assigns each game a cluster ID and descriptive cluster label when features exist. |
| [cluster_summary.csv](new_package/04_clusters/cluster_summary.csv) | Summarizes the 18 clusters with characteristic features and example games. |
| [cluster_config.json](new_package/04_clusters/cluster_config.json) | Records the clustering method, PCA dimensions, selected cluster count, and coverage. |

`06_code/` builds the package artifacts.

| File | Purpose |
| --- | --- |
| [README.md](new_package/06_code/README.md) | Explains the build scripts, scoring rules, execution notes, and prerequisites. |
| [score_interest_top10.py](new_package/06_code/score_interest_top10.py) | Scores game interest from playtime and exports each user's top ten games. |
| [build_game_catalog.py](new_package/06_code/build_game_catalog.py) | Combines popular games and user favorites, then fills missing descriptions from available sources. |
| [score_games_gemini.py](new_package/06_code/score_games_gemini.py) | Calls Gemini for taxonomy scores with resumable caching and spending limits. |
| [cluster_game_features.py](new_package/06_code/cluster_game_features.py) | Normalizes polished scores, applies PCA, selects clusters, and exports assignments. |

`07_users/` contains prepared user interests.

| File | Purpose |
| --- | --- |
| [libraries_top10.csv](new_package/07_users/libraries_top10.csv) | Stores users' top ten games, playtime, and calculated interest scores in wide format. |
| [libraries_top10_coverage.json](new_package/07_users/libraries_top10_coverage.json) | Records interest-scoring weights, user counts, and overlap with popular Steam games. |

**New-user discovery — `cold_start/`**

| File | Purpose |
| --- | --- |
| [README.md](cold_start/README.md) | Explains discovery ranking, budget filters, hardware tiers, and source data. |
| [build_cold_start_tables.py](cold_start/build_cold_start_tables.py) | Builds filtered top-ten discovery lists using review quality, player counts, and hardware tiers. |
| [cold_start_tables.xlsx](cold_start/cold_start_tables.xlsx) | Contains precomputed recommendation lists and game hardware/specification data. |
| [current_players.jsonl](cold_start/current_players.jsonl) | Caches Steam concurrent-player snapshots used for discovery ranking. |
| [game_tiers.csv](cold_start/game_tiers.csv) | Stores catalog-wide hardware tiers and the specification evidence used to assign them. |

**Working with the existing pipeline**

For model training, run the notebooks from `src/` in this order: `inspect_data.ipynb` → `prepare_data_for_training.ipynb` → `train_models.ipynb` → `eval_xgboost.ipynb`. The first notebook creates `libraries_top10_with_features.csv`; preparation creates the training CSVs under `data/`.

Rebuilding upstream datasets requires inputs that are not all included here: original user libraries, Steam dataset dumps, hardware specification sources, and generated training splits. Several older scripts contain machine-specific paths; configure them before running. Data collection and Gemini scoring also require their service credentials.

Use the polished feature CSV for existing models, and pair every model JSON with its bundle JSON. The candidate-only model supplies a general-interest baseline; the precomputed cold-start tables supply discovery lists with budget and hardware filters.
