# Recommender Models — README

This project contains seven XGBoost models for predicting a Steam user's
**interest score** for a game. They share the same 94-feature game space
(NLP-derived genre/tag/metadata features) but differ in objective, input
layout, and whether they use collaborative signal.

| Model | File | Cycle | Purpose |
|-------|------|-------|---------|
| **`slot_reg`** | `models/model2_slot_reg.json` | 1 | Regression, personalized. First working model. |
| `slot_deep` | `models/model_slot_deep.json` | 1 | Regression, less regularized. Comparison only. |
| **`cand_only`** | `models/model_cand_only.json` | 1 | Regression, candidate-only. Cold-start fallback. |
| **`rank_hybrid`** | `models/model_rank_hybrid.json` | 2 | Binary, negative-sampled, no CF. |
| `rank_v2` | `models/model_rank_v2.json` | 2 | Binary, negative-sampled, **with CF**. Superseded. |
| `rank_v2_nopop` | `models/model_rank_v2_nopop.json` | 2.5 | `rank_v2` with `candidate_popularity` dropped. Superseded. |
| **`rank_v3_interact`** | `models/model_rank_v3_interact.json` | 3 | **Current best.** Adds per-tag user profile + tag-interaction features. |

Each model ships with a `_bundle.json` storing the exact feature layout,
feature names, and best iteration. **Always load the bundle alongside the model.**

---

## Quick comparison (held-out evaluation, ~14k users, ~503 candidates/user)

**Cycle 2 models (binary, negative-sampled):**

| Metric | popularity | rank_hybrid (no CF) | rank_v2 | rank_v2_nopop | **rank_v3_interact** |
|--------|-----------:|-------------------:|--------:|--------------:|---------------------:|
| Recall@5     | 0.4752 | 0.5088 | 0.5390 | 0.5385 | **0.5398** |
| Recall@10    | 0.6067 | 0.6131 | 0.6427 | **0.6541** | 0.6540 |
| Recall@20    | 0.7149 | 0.6875 | 0.7219 | **0.7459** | 0.7440 |
| Precision@5  | 0.2828 | 0.3030 | 0.3210 | 0.3207 | **0.3214** |
| NDCG@5       | 0.5626 | 0.5893 | 0.6080 | 0.6078 | 0.6067 |
| NDCG@10      | 0.5798 | 0.6050 | 0.6261 | 0.6275 | 0.6272 |
| MRR          | 0.7031 | 0.7427 | **0.7739** | 0.7730 | 0.7714 |
| Coverage@10  | 0.0426 | 0.2729 | 0.3699 | 0.6155 | **0.6283** |
| Coverage@20  | 0.0679 | 0.4933 | 0.4920 | 0.7517 | **0.7730** |
| Novelty      |  8.89  | 10.79  | 10.41  | 10.4084 | **10.4793** |

**Cycle 1 models (regression, positives only):**

| Metric | slot_reg | slot_deep | cand_only |
|--------|---------:|----------:|----------:|
| Recall@5    | 0.1056 | 0.1015 | 0.0225 |
| Recall@10   | 0.1234 | 0.1223 | 0.0678 |
| Recall@20   | 0.1412 | 0.1429 | 0.1342 |
| Precision@5 | 0.0613 | 0.0588 | 0.0134 |
| NDCG@5      | 0.5278 | 0.5294 | 0.3341 |
| NDCG@10     | 0.4913 | 0.4824 | 0.2333 |
| MRR         | 0.2179 | 0.2069 | 0.0668 |
| Coverage@10 | 0.2893 | 0.3948 | 0.0488 |
| Novelty     | 14.82  | 14.62  | 14.10  |

> **Note on comparability:** Cycle 1 and Cycle 2 use *different training
> objectives* (regression vs. binary) and *different negative-sampling schemes*,
> so the raw numbers are not directly comparable across cycles. Within Cycle 2,
> the family progression is `rank_hybrid` → `rank_v2` → `rank_v2_nopop` →
> `rank_v3_interact`, and all four are evaluated on the same pool.

**Takeaways:**

- **`rank_v3_interact` is the current production model.** It matches
  `rank_v2_nopop` on every ranking metric (all differences within seed noise),
  while further improving coverage@10 (0.6155 → 0.6283) and coverage@20
  (0.7517 → 0.7730).
- **The single biggest quality jump in Cycle 2 came from dropping
  `candidate_popularity`** (the `rank_v2` → `rank_v2_nopop` step): coverage@10
  went from 0.37 to 0.62 with no loss in ranking quality. See the `rank_v2_nopop`
  section below for why.
- **`rank_v3_interact` is an incremental refinement on top of that**, not a new
  regime. It is the model to ship; it is not a new state of the art.
- **`rank_hybrid`** showed that the binary objective + negative sampling was the
  key change that let any of these models beat popularity on ranking metrics.
- **Cycle 1 models** are ~5× behind on recall@10 — the regression objective was
  the limiting factor, not the slot design.
- **`cand_only`** performs near chance on ranking metrics; it's a popularity
  predictor, not a recommender. Keep only as a cold-start fallback.


## Metric definitions

### Recall@K
**What it measures:** Of the user's *actual* games, what fraction appear in the
model's top-K recommendations?

**What it tells us:** The basic "did we find their games at all" metric. Values
near the random baseline (~0.06 at K=10) mean the model essentially finds
nothing. Rewards quantity, not order.

### Precision@K
**What it measures:** Of the top-K shown, what fraction are actually the user's
games?

**What it tells us:** How clean the top of the list is. Chance precision is
~0.006, so even small values are meaningful.

### NDCG@K
**What it measures:** Ranking quality with *position weighting* — a hit at rank
1 counts more than at rank K, and a highly-rated game counts more than a
weakly-rated one. Relevance is graded by interest.

**What it tells us:** The best single metric for "did the model order things
well at the top." A score of 1.0 means the ranking exactly matches the ideal
ordering.

### MRR
**What it measures:** How high the user's *first* actual game appears.

**What it tells us:** "How quickly do we hit gold." MRR = 0.77 means the first
correct recommendation lands around rank ~1.3 on average.

### Coverage@K
**What it measures:** What fraction of the whole catalog the model ever
recommends, across all users.

**What it tells us:** The diversity / anti-popularity metric. A popularity
model shows the same few games to everyone → low coverage. `rank_v3_interact`
at 0.63 means it shows 63% of the catalog to someone at K=10.

### Novelty
**What it measures:** How *unpopular* the recommended games are, on average,
on a log2 scale.

**What it tells us:** Whether the model reaches beyond the head of the
popularity distribution. +1.0 in novelty = the recommended games are ~2× rarer.

**Random baselines** (~3 relevant games/user, ~503 candidates/user):
- Random Recall@10 ≈ 0.06
- Random Precision@10 ≈ 0.006
- Random MRR ≈ 0.006

---

## What each model is

### Cycle 1 — Regression models

#### `slot_reg` — first personalized model (regression)
- **Objective:** `reg:squarederror`, trained on positives only.
- **Input:** candidate features + up to 6 of the user's training games in
  fixed slots (`slot1_*` ... `slot6_*`), sorted by interest descending.
- **Learns:** "given these games a user liked, what interest score will they
  give this candidate."
- **Regularization:** `max_depth=4`, `eta=0.05`, `colsample_bytree=0.3`,
  `min_child_weight=20`, `reg_lambda=5.0`, `reg_alpha=1.0`.
- **Feature count:** 670.

#### `slot_deep` — comparison only
- Same structure, less regularization. Overfits. Not for production.

#### `cand_only` — cold-start fallback
- **Input:** candidate features only. No user history.
- **Learns:** "how interesting is this game in general."
- **Feature count:** 94.

### Cycle 2 — Binary, negative-sampled models

#### `rank_hybrid` — first cross-entropy model (no CF)
- **Objective:** `binary:logistic`. For each (user, positive game), 15
  negatives are sampled; the model learns to score the positive above them.
- **Input:** same slot + candidate layout as `slot_reg` (670 features). **No
  collaborative signal.**
- **Learns:** "given the user's games, is this candidate more likely to be
  played than these plausible alternatives."
- **Sample weights:** positive weighted by its interest score, negatives 1.
- **Negatives:** 5 random + 5 content-similar (hard, from similarity band
  10–100) + 5 popularity-matched.
- **Use:** when you want personalization without the CF artifact dependency.

#### `rank_v2` — first CF model (superseded)
- **Objective:** same as `rank_hybrid` (`binary:logistic`).
- **Input:** 683 features = `rank_hybrid`'s 670 + 13 heuristic features:
  - **CF (8):** `cf_sim_1`...`cf_sim_6` (co-occurrence similarity between
    candidate and each slot game), `cf_score` (interest-weighted mean), `cf_max`
  - **Content (3):** `content_score` (cosine between user taste vector and
    candidate), `content_score_top3` (top-3 only), `max_sim_to_any_game`
  - **Popularity (2):** `candidate_popularity`, `user_mean_popularity`
- **Learns:** same as `rank_hybrid` plus collaborative signal.
- **Requires:** a co-occurrence matrix `data/game_sim_matrix.npz` built from a
  **disjoint set of users** (see training README for why).
- **Why it was superseded:** when the outputs were inspected manually, the
  model was recommending popular "hub" games — Counter-Strike, Garry's Mod,
  Team Fortress 2 — to users whose histories had no action or shooter signal
  at all. Feature-importance analysis showed `candidate_popularity` was the
  **single highest-gain feature by a wide margin**, meaning the model had
  learned a popularity shortcut rather than personalization. See
  `rank_v2_nopop` below.

#### `rank_v2_nopop` — `rank_v2` with popularity feature removed (superseded)
- **Objective:** identical to `rank_v2`.
- **Input:** 682 features = `rank_v2`'s 683 minus `candidate_popularity`.
- **Why:** dropping the popularity shortcut caused a large, immediate change
  in the model's behavior — the same setup retrained without that one feature
  produced **coverage@10 = 0.6155 vs. `rank_v2`'s 0.3699** (a 67% relative
  increase), while ranking metrics were statistically identical
  (recall@10 0.6427 → 0.6541, NDCG@10 0.6261 → 0.6275, MRR 0.7739 → 0.7730).
- **What this told us:** the popularity feature had been masking how much
  personalization signal the rest of the pipeline already contained. Once
  removed, the model started showing the wider catalog to more users.
- **Residual problem:** manually inspecting individual users showed that
  popular hub games were *still* being recommended to users with no relevant
  genre signal — the bias was reduced, not eliminated. This motivated the
  `rank_v3_interact` work.

#### `rank_v3_interact` — current production model
- **Objective:** same as `rank_v2_nopop` (`binary:logistic`).
- **Input:** 870 features = `rank_v2_nopop`'s 682 + 188 interaction features:
  - **`user_tag_{f}`** (94 features) — the mean value of feature `f` across
    the user's present slots. This is a per-user tag profile: "how much does
    this user like action / strategy / JRPG / etc."
  - **`interact_{f}`** (94 features) — `user_tag_{f} × cand_{f}`. This is a
    per-tag match score: "this user's action weight × this candidate's action
    weight."
- **Learns:** the same ranking objective as before, but with an explicit
  per-user, per-tag preference signal available. The intent is that the model
  can say "for users whose top slot is a JRPG, promote JRPG candidates," and
  "for users whose top slot is a strategy game, promote strategy candidates,"
  without needing to lean on popularity as a proxy for genre preference.
- **Why the interaction features exist:** the original motivation came from
  looking at the distribution of tag features in the training set. Action
  games dominate (~61% of slot-1 rows have `Action=1`), while RPG, Strategy,
  Simulation and Puzzle are far rarer. But a cross-tab of `P(candidate has tag
  T)` against `P(candidate has tag T | user's slot-1 has tag T)` showed the
  *rarer* tags have far higher lift:

  | Tag | P(cand=1) | P(cand=1 \| slot1=1) | Lift |
  |-----|----------:|---------------------:|-----:|
  | Action | 0.567 | 0.681 | 1.20 |
  | Shooter | 0.424 | 0.478 | 1.13 |
  | FPS | 0.281 | 0.408 | 1.45 |
  | Strategy | 0.162 | 0.308 | 1.90 |
  | RPG | 0.149 | 0.328 | 2.19 |
  | Simulation | 0.120 | 0.273 | 2.27 |
  | Sports | 0.031 | 0.071 | 2.24 |
  | Racing | 0.026 | 0.073 | 2.79 |
  | Puzzle | 0.024 | 0.084 | 3.49 |

  In other words: the model needs to *learn more* from the rare tags than the
  raw frequency of those tags suggests, because rare tags are much more
  informative when the user actually has them. The `interact_*` columns are
  the model's opportunity to discover that.
- **Results:** ranking metrics are essentially unchanged from `rank_v2_nopop`
  (recall@10 0.6541 → 0.6540, NDCG@10 0.6275 → 0.6272, MRR 0.7730 → 0.7714,
  all within seed noise). Coverage@10 improved 0.6155 → 0.6283, coverage@20
  0.7517 → 0.7730, novelty 10.4084 → 10.4793.
- **Qualitative impact:** the more important change is visible when the model
  is run against the full catalog and the top-20 lists are read side-by-side.
  On users whose history contains a minority thread (a JRPG user with one
  non-JRPG game mixed in, a Souls user with a couple of shooters, a
  wrestling/strategy user), `rank_v3_interact` tends to make that minority
  thread **dominate** the top-10 where `rank_v2_nopop` buried it under
  generic popular games. Roughly 10–12 of 50 manually-inspected users show
  this change clearly; ~35 show little to no change; 2–3 show a mild
  regression (typically a slightly more confident placement of a popular hub
  game at rank 1).
- **Residual problem:** the popularity-hub issue is *reduced but not solved*.
  Counter-Strike 2, Garry's Mod, and Team Fortress 2 still appear at the top
  of some recommendation lists where the user's history has no shooter or
  action signal. This is the target for the next iteration.
- **Requires:** same artifacts as `rank_v2` — `data/game_sim_matrix.npz`,
  `pop_norm`, and the `heuristic_features_v2` module at inference time
  (**not** `heuristic_features`; the v3 feature builder lives in the `_v2`
  module and adds the interaction block).
- **Use:** current production model. Load it from `model_rank_v3_interact.json`
  together with `model_rank_v3_interact_bundle.json`.

---

## Feature space

All models operate on the same **94 game features** (`GAME_FEATS`) derived from
NLP processing of game descriptions, tags, and metadata. The feature list is
stored in every bundle (`bundle['game_feats']`).

Raw game features live in `new_package/03_features/game_features.csv` (one row
per `appid`).

**Critical:** the `appid` dtype must be consistent across all data sources. A
mismatch (int vs str) silently produces all-NaN features and garbage predictions.

---

## How to load a model

```python
import os, json
import xgboost as xgb

MODEL_DIR = "models"

def load_model_and_bundle(name):
    m = xgb.Booster()
    m.load_model(os.path.join(MODEL_DIR, f"{name}.json"))
    with open(os.path.join(MODEL_DIR, f"{name}_bundle.json")) as f:
        b = json.load(f)
    return m, b

# Cycle 1
model_slot, bundle_slot = load_model_and_bundle("model2_slot_reg")
model_cand, bundle_cand = load_model_and_bundle("model_cand_only")

# Cycle 2
model_hybrid, bundle_hybrid = load_model_and_bundle("model_rank_hybrid")
model_v2,     bundle_v2     = load_model_and_bundle("model_rank_v2")
model_v2np,   bundle_v2np   = load_model_and_bundle("model_rank_v2_nopop")
model_v3,     bundle_v3     = load_model_and_bundle("model_rank_v3_interact")

# Common constants (all bundles agree on GAME_FEATS and N_SLOTS)
GAME_FEATS = bundle_slot["game_feats"]     # 94 raw game features
N_SLOTS    = bundle_slot["n_slots"]        # 6
```

Each bundle's `feature_cols` is model-specific — always use the bundle's
`feature_cols`, not a hardcoded list.

**Important:** the `feature_cols` lists differ across `rank_hybrid` (670),
`rank_v2` (683), `rank_v2_nopop` (682) and `rank_v3_interact` (870). Loading
the wrong bundle with the wrong model will produce silently wrong scores — the
`.json` model file stores only the trees, and XGBoost will not validate column
counts or column order at predict time.

---

## How to use for prediction

### Building the feature lookup

```python
import numpy as np
import pandas as pd
import xgboost as xgb

raw_games = pd.read_csv("new_package/03_features/game_features.csv")
raw_games["appid"] = raw_games["appid"].astype("int64")
game_feats_df = raw_games.drop_duplicates("appid").set_index("appid")[GAME_FEATS]

# For rank_v2 / rank_v2_nopop / rank_v3_interact: load the co-occurrence matrix
sim_data = np.load("data/game_sim_matrix.npz")
sim = sim_data["sim"].astype(np.float32)
sim_appids = sim_data["appids"]
sim_idx = {int(a): i for i, a in enumerate(sim_appids)}

# Popularity normalizer (from training data)
train = pd.read_csv("data/training_data_train_final.csv")
pop = train["appid"].value_counts()
pop_log = np.log1p(pop)
pop_norm = (pop_log / pop_log.max()).to_dict()
```

---

### Predicting with `slot_reg` (Cycle 1 regression)

```python
def predict_interest_slot(played_games, candidate_appid,
                          model=model_slot, bundle=bundle_slot,
                          feats_df=game_feats_df):
    """
    played_games: DataFrame with ['appid', 'interest'].
    Returns predicted interest (float).
    """
    FEATS = bundle["feature_cols"]
    played = played_games.sort_values("interest", ascending=False).head(N_SLOTS)
    rows = {}
    for i in range(1, N_SLOTS + 1):
        for f in GAME_FEATS:
            rows[f"slot{i}_{f}"] = np.full(1, np.nan, dtype=np.float32)
        rows[f"slot{i}_interest"] = np.full(1, np.nan, dtype=np.float32)
        rows[f"slot{i}_present"]  = np.zeros(1, dtype=np.float32)
        if i <= len(played):
            g  = played.iloc[i - 1]
            gf = feats_df.loc[g["appid"]]
            for f in GAME_FEATS:
                rows[f"slot{i}_{f}"] = np.array([gf[f]], dtype=np.float32)
            rows[f"slot{i}_interest"] = np.array([g["interest"]], dtype=np.float32)
            rows[f"slot{i}_present"]  = np.ones(1, dtype=np.float32)
    cf = feats_df.loc[candidate_appid]
    for f in GAME_FEATS:
        rows[f"cand_{f}"] = np.array([cf[f]], dtype=np.float32)
    X = pd.DataFrame(rows)[FEATS]
    return float(model.predict(
        xgb.DMatrix(X),
        iteration_range=(0, bundle["best_iteration"] + 1),
    )[0])
```

---

### Predicting with `cand_only` (Cycle 1 cold-start)

```python
def predict_interest_coldstart(candidate_appid,
                               model=model_cand, bundle=bundle_cand,
                               feats_df=game_feats_df):
    FEATS = bundle["feature_cols"]   # just cand_* columns
    cf = feats_df.loc[candidate_appid]
    X = pd.DataFrame([{f: cf[f] for f in GAME_FEATS}])[FEATS]
    return float(model.predict(
        xgb.DMatrix(X),
        iteration_range=(0, bundle["best_iteration"] + 1),
    )[0])
```

---

### Predicting with `rank_hybrid` (Cycle 2, no CF)

```python
def predict_rank_hybrid(played_games, candidate_appid,
                        model=model_hybrid, bundle=bundle_hybrid,
                        feats_df=game_feats_df):
    """Same feature layout as slot_reg, but the output is a ranking score
    (binary:logistic probability), not an interest value."""
    FEATS = bundle["feature_cols"]
    rows = _build_slot_cand_inference(played_games, candidate_appid,
                                       feats_df, GAME_FEATS, N_SLOTS)
    X = pd.DataFrame(rows)[FEATS]
    return float(model.predict(
        xgb.DMatrix(X),
        iteration_range=(0, bundle["best_iteration"] + 1),
    )[0])
```

---

### Predicting with `rank_v2_nopop` (Cycle 2, with CF, no popularity)

```python
from heuristic_features import add_heuristic_features, HEURISTIC_COLS

def predict_rank_v2_nopop(played_games, candidate_appid,
                          model=model_v2np, bundle=bundle_v2np,
                          feats_df=game_feats_df):
    FEATS = bundle["feature_cols"]  # 682

    row_dict = _build_slot_cand_inference(played_games, candidate_appid,
                                           feats_df, GAME_FEATS, N_SLOTS)

    ctx = {
        "sim":          sim,
        "appid_to_idx": sim_idx,
        "pop_norm":     pop_norm,
        "n_slots":      N_SLOTS,
    }
    played = played_games.sort_values("interest", ascending=False).head(N_SLOTS)
    add_heuristic_features(row_dict, uid=None, cands=np.array([candidate_appid]),
                            played=played, ctx=ctx,
                            game_feats=GAME_FEATS, n_slots=N_SLOTS)

    X = pd.DataFrame(row_dict)[FEATS]
    return float(model.predict(
        xgb.DMatrix(X),
        iteration_range=(0, bundle["best_iteration"] + 1),
    )[0])
```

---

### Predicting with `rank_v3_interact` (current production model)

Same as `rank_v2_nopop`, but the heuristic module is `heuristic_features_v2`
(which adds the per-tag user profile and interaction columns), and the bundle's
`feature_cols` has 870 entries.

```python
from heuristic_features_v2 import add_heuristic_features, HEURISTIC_COLS

def predict_rank_v3(played_games, candidate_appid,
                    model=model_v3, bundle=bundle_v3,
                    feats_df=game_feats_df):
    FEATS = bundle["feature_cols"]  # 870

    row_dict = _build_slot_cand_inference(played_games, candidate_appid,
                                           feats_df, GAME_FEATS, N_SLOTS)

    ctx = {
        "sim":          sim,
        "appid_to_idx": sim_idx,
        "pop_norm":     pop_norm,
        "n_slots":      N_SLOTS,
    }
    played = played_games.sort_values("interest", ascending=False).head(N_SLOTS)
    add_heuristic_features(row_dict, uid=None, cands=np.array([candidate_appid]),
                            played=played, ctx=ctx,
                            game_feats=GAME_FEATS, n_slots=N_SLOTS)

    X = pd.DataFrame(row_dict)[FEATS]
    return float(model.predict(
        xgb.DMatrix(X),
        iteration_range=(0, bundle["best_iteration"] + 1),
    )[0])
```

**Critical for `rank_v2*` and `rank_v3*`:** the co-occurrence matrix `sim` must
be the **same matrix used at training time** (the one built from the disjoint
user set, saved at `data/game_sim_matrix.npz`). Do not rebuild it from all
users — that reintroduces the leak.

---

### Ranking a set of candidates

```python
def top_n_rank_v3(played_games, candidate_appids, n=10):
    scores = [predict_rank_v3(played_games, aid) for aid in candidate_appids]
    ranked = pd.DataFrame({"appid": candidate_appids, "score": scores})
    return ranked.sort_values("score", ascending=False).head(n).reset_index(drop=True)
```

---

### Routing logic (production)

```python
def recommend_score(user_history, candidate_appid,
                    use_cf=True, sim_available=True):
    """
    Pick the right model based on user state and artifact availability.
    """
    if user_history is None or len(user_history) == 0:
        return predict_interest_coldstart(candidate_appid)   # cold start
    if use_cf and sim_available:
        return predict_rank_v3(user_history, candidate_appid)  # best
    return predict_rank_hybrid(user_history, candidate_appid)  # no CF
```

**Decision guide:**

| Situation | Model |
|---|---|
| New user, no history | `cand_only` |
| History available, sim matrix available | `rank_v3_interact` |
| History available, no sim matrix | `rank_hybrid` |
| Legacy — need interpretable interest *value* | `slot_reg` |
| Legacy — reproducing pre-v2_nopop results | `rank_v2` |

---

## Artifacts required per model

| Model | Needs `sim` matrix? | Needs popularity normalizer? | Feature count |
|----------------|--------------------|------------------------------|-----|
| `slot_reg`        | no  | no  | 670 |
| `slot_deep`       | no  | no  | 670 |
| `cand_only`       | no  | no  | 94  |
| `rank_hybrid`     | no  | no  | 670 |
| `rank_v2`         | yes | yes | 683 |
| `rank_v2_nopop`   | yes | yes | 682 |
| **`rank_v3_interact`** | **yes** | **yes** | **870** |

**Inference-time module requirement:** `rank_v2`, `rank_v2_nopop` and
`rank_v3_interact` all need `heuristic_features` on the Python path.
`rank_v3_interact` specifically needs `heuristic_features_v2` (the v2 module
adds the interaction block that the v3 model expects).

---

## What's next

The following items are queued for the next iteration. They are ordered
roughly by expected impact.

1. **Popularity-gap metric in `rec_eval`.** Add
   `mean(pop_norm[top-K]) − mean(pop_norm[user_history])` per user and report
   the aggregate. This is the metric that would have flagged the original
   Garry's Mod / CS:GO problem automatically.
2. **Inference-time popularity penalty.** A post-hoc reranking term of the
   form `score' = score − α * log(pop_norm[cand] / user_mean_popularity)`,
   with α tuned on a validation split. Cheap, orthogonal to the model, and
   directly targets the residual hub problem.
3. **More aggressive popularity-matched negatives.** Currently 5 of 15
   negatives are popularity-matched. Try 10 of 15 and hold out the rest as
   random; if recall holds while the popularity gap shrinks, we have
   localized the residual bias to the sampling scheme.

These items will become the backbone of the v4 write-up.

---

## Reproducing the evaluation

The `rank_v2` family (`rank_v2` and `rank_v2_nopop`) is trained by a single
driver, `src/rank_pipeline.py`. Whether the popularity feature is included is
controlled by a command-line flag:

```bash
# rank_v2 (with candidate_popularity)
python src/rank_pipeline.py --stage all --force

# rank_v2_nopop (identical, but candidate_popularity is dropped)
python src/rank_pipeline.py --stage all --force --drop-popularity

# rank_v3 
python src/rank_pipeline_v3_interact.py --stage all --force

Each pipeline rebuilds the training parquet, retrains the model, and evaluates
against the same set of baselines (popularity, `rank_hybrid`, and any earlier
model in the family). Results are saved under `results/`.