# Scripts

Run these from the package root, in this order. Each one reads the folder the previous one wrote.

## score_interest_top10.py

Turns the owned-library export into an interest top 10 for every user.

The export stores playtime in **minutes**, inside each user's `game_library` JSON. This script converts to hours and uses hours everywhere, including `H_cap` (200 hours).

```
interest = 0.40*s_recent + 0.35*s_play + 0.15*s_momentum + 0.10*s_invest
```

Writes `07_users/libraries_top10.csv` (one row per user) and a coverage JSON that compares those games with the Steam top 2500 and top 5000.

## build_game_catalog.py

Builds the 7,430-game list used for features.

The list is the Steam 2026 **top 2500** (non-games dropped) plus every user top-10 game that is not already in that 2500. Descriptions come from the 2026 dump, then the March 2025 dump, then a live Steam page, then a sibling edition if the owned id still has no text.

```
python 06_code/build_game_catalog.py
python 06_code/build_game_catalog.py --no-fetch
```

Writes `01_catalog/games_for_features.csv`.

## score_games_gemini.py

Asks Gemini 3.8 Flash for a sparse 93-label score for each game that has a description. Eight games per call. The description decides the score. A community genre the text never mentions can be at most 0.4. A community tag in that case can be at most 0.3. Indie, Free to Play, and NSFW are ignored.

The key is an environment variable, `GEMINI_API_KEY`. Do not put the key in git.

```
python 06_code/score_games_gemini.py --limit 8
python 06_code/score_games_gemini.py
python 06_code/score_games_gemini.py --assemble
```

`--assemble` rebuilds the CSV from the saved JSONL and does not call Gemini.

## cluster_game_features.py

Clusters `03_features/game_features.csv`. Only rows with `status=ok` (6,750 games). The other 680 have no vector and stay `unclustered`.

What the script does, in order:

1. Take the 93 scores for each scored game.
2. L2-normalize each row so a game is compared by which labels are strong, not by how many numbers are non-zero.
3. PCA keeps 90% of that variance, which here is 54 components.
4. Try k from 12 to 28, step 2. Silhouette picked **18** (score 0.109).
5. MiniBatchKMeans assigns every scored game to one of those 18 groups.
6. Each group is named from the labels that are higher in that group than in the catalog overall. Example names are the first games in the group, not a hand-picked sample.

Writes `04_clusters/game_clusters.csv` and `04_clusters/cluster_summary.csv`.

## What we ran into

**No description.** About 500 catalog games have no usable store text. They are marked `insufficient_description` and are not sent to Gemini, so they are not billed. They stay in the catalog with an empty vector and are not clustered. Examples that never got text: WARDOGS, Zula Europe, Ring of Elysium, Heroes & Generals.

**Not a game.** About 180 rows were marked `non_game` (tools, soundtracks, software). They have no feature vector.

**16 calls failed.** Those games are in the score file as errors. They were not polished into `ok`.

**Same name, different app id.** The library and the Steam dump often disagree. Portal 2 is `620` in the library and `659` in the dump. FIFA 23, FC 25, and Black Ops III do the same. We keep the library id and copy the other id's text onto it. We do not treat Siege X as Siege, or a GTA Definitive Edition as the original. Those are different products.

**Sibling text is a different edition.** Forza Horizon 4 and 6 use Horizon 5's description. FIFA and F1 years use another year in the series. The polish pass corrected some of that by reading the full text. A score can still describe the sibling rather than the exact owned build.

**Community tags are weak when the text is silent.** Dota 2's MOBA score stayed at 0.3 because the description never says MOBA. Apex's FPS is the same. That is why Dota 2 landed in the general multiplayer cluster, and why The Witcher 3 sits beside turn-based RPGs even though it is not turn-based.

**Steam store pages have no community tags.** A live fetch fills the description and genres. Tag lists come only from the dumps.

**Excel cuts long descriptions.** Use the CSV, not the xlsx, as the source of text.

**Gemini 2.5 Flash was refused.** New keys have to use `gemini-3.8-flash`.

**The monthly spend cap stopped the API** while credits were still in the balance. The red spend bar and the credit balance are not the same number. The bar can run ahead of the money that actually left the account.

**Nine copies of this script ran at once** during the first scoring stretch. Each copy called Gemini. That is what emptied the first balance. The restart that finished the catalog was one process only.

**A half-written line** in `gemini_scores.jsonl` crashed one restart. The loader skips broken lines and continues from the games already saved.

## build_item_neighbors.py

Builds an item–item neighbor table from user Top-10 co-occurrence.

```
sim(i, j) = co_count(i, j) / sqrt( users(i) * users(j) )
```

- `co_count(i, j)` = users who have both games in their Top-10
- `users(i)` = users who have game i in their Top-10

For each game, keeps the top 10 neighbors by similarity. This is co-play closeness, not genre similarity.

```
python 06_code/build_item_neighbors.py
```

Writes `08_item_neighbors/` (`game_top10_neighbors.xlsx`, `.csv`, `.json`, `README.md`).
