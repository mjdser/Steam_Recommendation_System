# Training — README

This document describes how the recommender models were trained, across
**three development cycles**:

- **Cycle 1** — regression models on positives only (`slot_reg`, `slot_deep`,
  `cand_only`).
- **Cycle 2** — binary cross-entropy models trained with negative sampling,
  first without CF (`rank_hybrid`), then with CF (`rank_v2`), then with the
  popularity feature removed (`rank_v2_nopop`).
- **Cycle 3** — `rank_v3_interact`, which adds a per-user per-tag profile and
  per-tag interaction features on top of `rank_v2_nopop`.

For how to *use* the models, see `models_readme.md`.

---

## Input data

We started from two prepared datasets:

- **User histories** — one row per (user, game) with `steamid`, `appid`, and
  the user's `interest` score. Each user has ~6–10 games.
- **Game features** — `new_package/03_features/game_features.csv`, one row per
  `appid`, containing 94 NLP-derived features (genre indicators, theme tags,
  metadata such as `n_labels`).

From these, the pipeline produces three **split frames** by holding out games
per user:

| File | Content |
|---|---|
| `training_data_train_final.csv` | One row per (user, training game) |
| `training_data_val_final.csv`   | Held-out val games (3 per user on average) |
| `training_data_test_final.csv`  | Held-out test games (1 per user) |

All three are **slot-expanded** — each row has `slot1_*` ... `slot6_*` and
`cand_*` columns. All cycles consume these same split frames.

---

## Feature space

The **94 game features** (`GAME_FEATS`) are the same across every model. They
are looked up by `appid` at training and inference. Dtype consistency matters —
an int-vs-str mismatch silently produces all-NaN features.

---

## Train / val / test split

Split at the **(user, game) row level**, not by user. Rule per user (by game
count `n`):

| `n` | Val held out | Test held out | Train kept |
|-----|--------------|---------------|------------|
| ≤ 1 | 0 | 0 | n |
| 2–3 | 1 | 0 | n − 1 |
| > 3 | 3 | 1 | n − 4 |

Users have ~10 games, so most contribute **3 val + 1 test**, leaving ~6 in
train. The split is by (user, game) so val/test users still have context.

---

# Cycle 1 — Regression models

**Objective:** `reg:squarederror` on the positive games only (their interest
scores).

**Models produced:** `slot_reg`, `slot_deep`, `cand_only`.

## Slot design

For each candidate game, we assemble a row:

```
[ slot1 features, slot1 interest, slot1 present,
  slot2 features, slot2 interest, slot2 present,
  ...
  slot6 features, slot6 interest, slot6 present,
  candidate features ]
→ predict interest
```

Design choices:

- **6 slots.** Users have ~6 train games, so 6 covers the common case.
- **Sorted by interest, descending.** Slot 1 = the user's favourite, etc.
- **`present` flags.** Empty slots are NaN + `present=0`.

### Held-out candidate exclusion
For each training row, the positive game is **excluded from the slots** — it
appears only as the candidate. This must hold at both training and inference
time.

### Slot pool restriction
- Train rows draw slots from **train games only**.
- Val rows draw slots from **train games only**.
- Test rows draw slots from **train + val games**.

## Training config (`slot_reg`)

```python
params = {
    'objective': 'reg:squarederror',
    'eval_metric': ['rmse', 'mae'],
    'tree_method': 'hist',
    'eta': 0.05, 'max_depth': 4,
    'subsample': 0.7, 'colsample_bytree': 0.3,
    'min_child_weight': 20, 'reg_lambda': 5.0, 'reg_alpha': 1.0,
    'seed': 0,
}
```

`slot_deep` uses `eta=0.03, max_depth=6, colsample_bytree=0.5,
min_child_weight=5, reg_lambda=1.0` — overfits, kept for comparison.
`cand_only` uses the same params as `slot_reg`.

## Results

See `models_readme.md` → Quick comparison. Cycle 1's best model reaches
recall@10 = 0.123 — well below popularity's 0.61 on the same pool.

**Why we moved to Cycle 2:** the regression objective treats every positive as
a target to fit, but the ranking task asks "put this game above plausible
alternatives." Those are different. Regression models cannot learn *what not to
recommend*. The fix is a binary objective trained with sampled negatives.

---

# Cycle 2 — Binary cross-entropy with negative sampling

**Objective:** `binary:logistic`. For each (user, positive game), sample 15
negatives and train the model to score the positive above them.

## Dataset structure

For each **(user, training-game)** pair (one "group"):

- 1 positive row → `label=1`, `sample_weight = interest`
- 15 negative rows → `label=0`, `sample_weight = 1`
- **All 16 rows share the same slot columns** (the user's context, excluding
  the positive)
- **Only the `cand_*` columns and the label differ across the 16 rows**

Total groups = 117,489 (one per train positive). Total rows ≈ 1.88M.

## Negative sampling scheme

For each positive, 15 negatives are drawn from three pools:

| Pool | Count | What it teaches |
|------|-------|-----------------|
| Random | 5 | "distinguish from any catalog game" |
| Hard (content-similar) | 5 | "distinguish from plausible alternatives" |
| Popularity-matched | 5 | "don't win via popularity" |

**Hard negatives** are drawn from the games ranked **10th–100th** most similar
to the positive by feature cosine. Skipping the top 10 avoids **false
negatives** — games so similar to the positive that the user would plausibly
like them too.

**Popularity-matched** negatives are drawn from the same popularity decile as
the median popularity of the user's played games.

After dedup, if fewer than 15 negatives remain, the group is topped up with
extra random draws to exactly 15.

## Feature set — `rank_hybrid` (no CF)

Same 670-feature layout as `slot_reg`. No collaborative signal.

**Training config:** same as `slot_reg` but with `objective='binary:logistic'`,
`eval_metric=['logloss', 'auc']`.

**Result:** recall@10 = 0.6131, NDCG@10 = 0.6050, MRR = 0.7427,
coverage@10 = 0.2729. **First model to beat popularity on ranking metrics.**

## Feature set — `rank_v2` (with CF)

`rank_hybrid`'s 670 features + **13 heuristic features** (683 total):

### CF features (8)
- `cf_sim_1` ... `cf_sim_6` — co-occurrence similarity between candidate and
  each slot game
- `cf_score` — interest-weighted mean of the six `cf_sim_i`
- `cf_max` — max over the six slots

### Content features (3)
- `content_score` — cosine between interest-weighted user taste vector and
  candidate features
- `content_score_top3` — same but using only the user's top-3 games
- `max_sim_to_any_game` — max cosine between candidate and any slot game

### Popularity features (2)
- `candidate_popularity` — frequency of the candidate
- `user_mean_popularity` — mean popularity of the user's games

### Building the co-occurrence matrix

The matrix `sim(i, j)` is a **cosine similarity over user sets**:

```
sim(i, j) = co_count(i, j) / sqrt(users(i) * users(j))
```

where `co_count(i, j)` = number of users who have both `i` and `j` in their
Top-10, and `users(k)` = number of users who have `k`.

Built from a **sparse user×game matrix U** as `C = Uᵀ U`, then normalized:

```python
C = (U.T @ U).tocsr()                              # co-occurrence counts
inv_sqrt = 1.0 / np.sqrt(users_per_game)
sim = diags(inv_sqrt) @ C @ diags(inv_sqrt)        # cosine
np.fill_diagonal(sim, 0.0)
```

Matrix is `(n_games, n_games)`, stored as float32.

**Result:** recall@10 = 0.6427, NDCG@10 = 0.6261, MRR = 0.7739, coverage@10 =
0.3699. Best ranking model at the time, but see below.

## Feature set — `rank_v2_nopop` (popularity feature removed)

**Motivation.** When we inspected `rank_v2`'s outputs against individual user
histories, we noticed something the aggregate metrics did not show. The model
was recommending the same handful of popular hub games — Counter-Strike 2,
Garry's Mod, Team Fortress 2 — to users whose histories contained no action or
shooter games at all. A user whose history was entirely puzzle games, or
entirely JRPGs, would still get CS2 in the top 5.

**Diagnosis.** Feature-importance analysis on `rank_v2` showed that
`candidate_popularity` had the **highest gain of any single feature by a wide
margin**. The model was not learning "which game fits this user"; it was
learning "which game is popular, and which of these popular games fits this
user enough to be plausible." The popularity signal was masking how much
personalization signal the rest of the pipeline actually contained.

**Fix.** Drop `candidate_popularity` and retrain with everything else
identical. Result: 682 features instead of 683.

**Consequence.** Coverage@10 jumped from **0.3699 → 0.6155** — a 67% relative
increase — while every ranking metric stayed statistically identical
(recall@10 0.6427 → 0.6541, NDCG@10 0.6261 → 0.6275, MRR 0.7739 → 0.7730).
The model started showing a much wider slice of the catalog to more users,
without sacrificing ranking quality.

**Residual problem.** Even after removing `candidate_popularity`, manual
inspection still showed hub games reaching the top of some users'
recommendation lists. The bias was reduced, not eliminated. This motivated
Cycle 3.

## Feature set — `rank_v3_interact` (per-tag profile + interaction)

**Motivation.** The residual popularity bias meant that some users were still
seeing head-catalog games that did not match their history. We hypothesized
that the model was leaning on popularity as an implicit proxy for genre
preference, and that giving it an explicit per-tag preference signal would let
it stop doing that.

**Supporting evidence.** The distribution of tag features in the training set
is heavily skewed toward Action:

```python
train['slot1_Action'].mean() == 0.6076   # 61% of users have an Action game at slot 1
train['slot1_Casual'].mean() == 0.0487   # 5% for Casual
```

But when we computed, for each tag, the **marginal** probability of a candidate
having the tag versus the **conditional** probability given that the user's
slot-1 game has the same tag, the rare tags had far higher lift:

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

Read: Action is common and its lift is small (1.20), while Puzzle is rare and
its lift is large (3.49). In other words, if a user has Puzzle in their history,
that is a much stronger signal about what they want than if they have Action.
But the model had no direct way to condition on this — it only saw raw tag
values and popularity.

**New features.** We added two blocks of per-tag features, for each `f` in the
94 game features:

- **`user_tag_{f}`** — the mean of `f` across the user's *present* slots. A
  per-user tag profile: "how much does this user like this tag." This is a
  *per-user scalar* — it does not vary across candidates within a ranking group.
- **`interact_{f}`** — `user_tag_{f} × cand_{f}`. A per-candidate match score:
  "this user's tag weight × this candidate's tag presence."

That is 2 × 94 = 188 new features. Combined with `rank_v2_nopop`'s 682, the
model has **870 features** and trains to `best_iteration=2688`.

**Training config:** identical to `rank_v2_nopop`.

**Results:** ranking metrics essentially unchanged from `rank_v2_nopop` (all
differences within seed noise). Coverage@10 improved 0.6155 → 0.6283,
coverage@20 0.7517 → 0.7730, novelty 10.4084 → 10.4793.

**Qualitative impact.** The main effect is visible only when reading top-20
lists side-by-side. For users with a distinctive minority thread in their
history, `rank_v3_interact` surfaces that thread more aggressively than
`rank_v2_nopop` did. A few examples from manual inspection:

- **A JRPG user** (Persona 4 Golden, Trails through Daybreak,
  etc.) saw `rank_v2_nopop` produce a top-20 that mixed Persona titles with
  Path of Exile, GTA V and Rocket League. `rank_v3_interact` produces a top-20
  that is almost entirely JRPG: Persona 3 Portable, Persona 3 Reload, Trails
  in the Sky, Metaphor: ReFantazio, Baldur's Gate 3, Dragon Quest XI, Ni no
  Kuni, Fairy Tail, Trails of Cold Steel.
- **A Souls user** (Dark Souls III at 0.77 interest, ELDEN RING, Fallout: New
  Vegas) sees Dark Souls Remastered and Dark Souls II pulled up to ranks 1 and
  2, and Dishonored 2 into the top 10 — the action-RPG thread is preserved
  more tightly.
- **A wrestling/strategy user** (WWE 2K26, Dynasty Warriors, Civ V, Civ VI)
  sees the strategy thread dominate: Hearts of Iron IV, Total War: Rome II,
  Civ IV, Civ IV: Beyond the Sword, Civ VII, EU4, Crusader Kings II, Stellaris.
  Wrestling Empire and WWE 2K16/2K19 remain in the list at ranks 5, 17, 18.

Roughly 10–12 of 50 manually-inspected users show this change clearly; ~35
show little to no change; 2–3 show a mild regression.

**Residual problem (still open).** Popular hub games — Counter-Strike 2,
Garry's Mod, Team Fortress 2 — are still recommended to some users whose
histories contain no relevant genre signal. In two worst-case users, CS2
remains the top recommendation (though with a lower confidence score in v3
than in v2). The interaction features reduced the problem but did not
solve it. This is the target of the next cycle.

### Why the interaction features don't fully solve the problem

Worth recording for the next iteration. The interaction block is:

```
user_tag_{f}     = mean of f across the user's present slots
interact_{f}     = user_tag_{f} * cand_{f}
```

Within a single user's ranking group, `user_tag_{f}` is **constant across all
candidates** — it is a per-user scalar. Therefore `interact_{f}` is just
`cand_{f}` scaled by a per-user constant. It gives the tree a per-user
threshold on `cand_{f}`, which is *something*, but it cannot directly express
"for this candidate specifically, how strongly does the user's profile match
it" — the per-user scalar is the same regardless of which candidate is being
scored.

A more expressive version would replace `user_tag_{f}` with a
**candidate-conditional** user profile, e.g. weighting each slot by its
similarity to the candidate before averaging:

```
user_tag_conditional[f] = sum_i (w_i * cf_sim_i * slot_i[f]) / sum_i (w_i * cf_sim_i)
```

This has been noted for v4 but is not currently implemented.

---

## Data preparation pipeline (Cycle 2 / 3, final)

The pipeline (`src/v3_user_interaction_added/rank_pipeline_v3_interact.py`) runs in three stages, each checkpointed.

### Stage 1: split users

```python
sim_users, model_users = _split_users(train, sim_frac=0.30, seed=0)
```

- **30%** of users are `sim_users` — their histories build the CF matrix.
- **70%** are `model_users` — training and eval happen on these only.

This split is essential to avoid data leakage in the training stage. 

### Stage 2: build the co-occurrence matrix

```python
train_s = train[train.steamid.isin(sim_users)]
sim, appid_to_idx = _build_cooccurrence(train_s, catalog_appids)
np.savez_compressed("data/game_sim_matrix.npz", sim=sim, appids=catalog_appids)
```

The matrix is **persisted** so training and eval read the *same* matrix.

### Stage 3: build training parquet

For each `(user, positive game)` pair in `model_users`:

1. **Slots**: the user's *other* training games, sorted by interest desc, into 6 slots.
2. **Negatives**: 5 random + 5 hard + 5 popularity-matched (topped up to 15).
3. **Labels**: `[1, 0, 0, ..., 0]` (1 positive, 15 negatives).
4. **Sample weights**: `[interest, 1, 1, ..., 1]`.
5. **Heuristic features** via `add_heuristic_features` (CF, content, popularity, and for v3: `user_tag_*` and `interact_*).
6. Write to the parquet file in 5,000-group chunks.

Output: ~1.88M rows × 870 columns.

### Stage 4: train

- Load the training parquet.
- Split users 90/10 into `fit` / `es` (early-stopping) sets.
- `binary:logistic`, `max_depth=4`, `eta=0.05`, `colsample_bytree=0.3`,
  `min_child_weight=20`, `reg_lambda=5.0`, `reg_alpha=1.0`.
- Early stop on validation AUC, `num_boost_round=3000`, `early_stopping_rounds=50`.

### Stage 5: evaluate

- Reconstruct the same user split.
- Build eval pools: each val user's val games (positives) + 500 random negatives.
- Score with popularity, `rank_hybrid`, `rank_v2`, `rank_v2_nopop`, and
  `rank_v3_interact` on the **same pool**.
- Report recall, precision, NDCG, MRR, coverage, novelty at K = 5, 10, 20.

---

## Training results (final, clean run)

14,030 eval users (model_users only), ~503 candidates each.

| model | recall@10 | NDCG@10 | MRR | coverage@10 | novelty |
|-------|----------:|--------:|----:|------------:|--------:|
| popularity         | 0.6067 | 0.5798 | 0.7031 | 0.0426 | 8.8880 |
| rank_hybrid        | 0.6131 | 0.6050 | 0.7427 | 0.2729 | 10.7921 |
| rank_v2            | 0.6427 | 0.6261 | 0.7739 | 0.3699 | 10.4104 |
| rank_v2_nopop      | 0.6541 | 0.6275 | 0.7730 | 0.6155 | 10.4084 |
| **rank_v3_interact** | 0.6540 | 0.6272 | 0.7714 | **0.6283** | **10.4793** |

**Headline:** the big quality step in Cycle 2 was removing
`candidate_popularity`. Everything after that — the interaction features in
v3 — is incremental. `rank_v3_interact` is the model to ship because it is
never worse than `rank_v2_nopop` is slightly better on coverage and
novelty, and likely has less popularity bias, but the majority of the improvement over `rank_v2` came from the popularity feature drop.

---

## Saving models

Each model is saved as a `.json` file plus a `_bundle.json`:

```python
model.save_model(f"{MODEL_DIR}/{name}.json")

bundle = {
    'feature_cols':   FEATURE_COLS,
    'game_feats':     GAME_FEATS,
    'n_slots':        N_SLOTS,
    'params':         params,
    'best_iteration': int(model.best_iteration),
    'objective':      'binary:logistic',      # Cycle 2/3
    'heuristic_cols': HEURISTIC_COLS,          # v2/v2_nopop/v3
    'interaction_cols': interaction_cols(GAME_FEATS),  # v3 only
    'dropped_features': ['candidate_popularity'],       # v2_nopop/v3 only
}
json.dump(bundle, open(f"{MODEL_DIR}/{name}_bundle.json", "w"))
```

**Why the bundle:** the `.json` file stores only the trees. XGBoost does not
validate column order at predict time — feed it the wrong columns and it
returns confident garbage. The bundle stores the exact `FEATURE_COLS` and
`best_iteration`.

---

## Pipeline summary

### Cycle 1 (regression)
1. Melt wide histories into long form.
2. Merge game features by `appid`.
3. Split into train / val / test per (user, game).
4. Build slot rows.
5. Compact-sort slots by interest descending.
6. Train XGBoost `reg:squarederror`, early stop on val RMSE.
7. Save model + bundle.

### Cycle 2 (binary + negative sampling)
1. Load split frames produced by Cycle 1.
2. **Split users** into `sim_users` (30%) and `model_users` (70%).
3. **Build `sim`** from `sim_users` only. Save to disk.
4. **Build training parquet:** for each (model_user, positive), sample 15
   negatives, build slots, compute heuristic features, write in chunks.
5. **Train** `binary:logistic`, early stop on val AUC.
6. **Evaluate** on `model_users`' val games against popularity and
   `rank_hybrid`, on a shared candidate pool.

### Cycle 2.5 (`rank_v2_nopop`)
1. Reuse everything from Cycle 2.
2. Drop `candidate_popularity` from the feature list.
3. Retrain and evaluate. Coverage jumps; ranking metrics unchanged.

### Cycle 3 (`rank_v3_interact`)
1. Reuse everything from Cycle 2.5.
2. Add 188 interaction features via `heuristic_features_v2`.
3. Retrain and evaluate. Ranking metrics unchanged; coverage/novelty improve
   slightly; minority threads in user histories are surfaced more aggressively
   in the top-20.

---

## Reproducing

```bash
 # rank_v2 family
python src/rank_pipeline.py             --stage all --force 
# rank_v2_nopop — identical, but candidate_popularity is dropped
python src/rank_pipeline.py --stage all --force --drop-popularity
# rank_v3_interact
python src/rank_pipeline_v3_interact.py --stage all --force  
```

Stages are resumable — a completed parquet or `.json` model file skips its
stage unless `--force` is passed.

---

## Known limitations

- **No true negatives.** Interest is only observed for games a user played.
  Unplayed games are unknown, not disliked.
- **Selection bias.** Games in histories are user-chosen, so interest is biased
  upward.
- **Eval pool skew.** The eval uses ~3 positives + 500 random negatives. Because
  positives are user-chosen (relatively popular) and negatives are random
  (mostly niche), *popularity* gets an inflated score on this pool. The
  rankings we report are valid but the *magnitudes* are optimistic. A
  popularity-matched eval pool would be fairer.
- **CF matrix sparsity.** At 30% of users, `sim` is sparse; rare games have
  few or no co-occurrence partners. Their CF features are near-zero and
  uninformative.
- **Cold-start users** get no personalization — use `cand_only`.
- **Feature dependence.** Models are only as good as the 94 game features.
- **Residual popularity bias.** Despite dropping `candidate_popularity` in
  `rank_v2_nopop` and adding interaction features in `rank_v3_interact`,
  popular hub games (Counter-Strike 2, Garry's Mod, Team Fortress 2) still
  reach the top of some users' recommendation lists where they shouldn't. The
  interaction block reduced this but did not eliminate it. See the v4 backlog
  in `models_readme.md`.
- **Interaction block is per-user, not per-candidate.** `user_tag_{f}` is
  constant within a ranking group, so `interact_{f}` is `cand_{f}` scaled by
  a per-user constant. This limits how much the block can do; a
  candidate-conditional user profile is queued for v4.

---

## Artifacts and where they live

| Path | Contents |
|------|----------|
| `data/training_data_{train,val,test}_final.csv` | Split frames (input to all cycles) |
| `data/tr_rank_v3_interact.parquet` | Cycle 3 training set (~1.88M rows, 870 cols) |
| `data/tr_rank_v2_nopop.parquet` | Cycle 2.5 training set |
| `data/tr_rank_v2.parquet` | Cycle 2 training set |
| `data/game_sim_matrix.npz` | Co-occurrence matrix + appid index |
| `models/model2_slot_reg.json` + bundle | Cycle 1 personalized |
| `models/model_slot_deep.json` + bundle | Cycle 1 comparison |
| `models/model_cand_only.json` + bundle | Cycle 1 cold-start |
| `models/model_rank_hybrid.json` + bundle | Cycle 2 no-CF |
| `models/model_rank_v2.json` + bundle | Cycle 2 with CF (superseded) |
| `models/model_rank_v2_nopop.json` + bundle | Cycle 2.5 (superseded) |
| `models/model_rank_v3_interact.json` + bundle | **Cycle 3 — current production model** |
| `results/rank_v2_comparison.csv` | Cycle 2 eval table |
| `results/rank_v3_comparison.csv` | Cycle 3 eval table |
| `src/rank_pipeline.py` | Cycle 2 driver. Handles both `rank_v2` and `rank_v2_nopop`; the `--drop-popularity` flag selects which. |
| `src/rank_pipeline_v3_interact.py` | Cycle 3 driver (`rank_v3_interact`) |
| `src/heuristic_features.py` | CF + content + popularity feature builders |
| `src/heuristic_features_v2.py` | Adds `user_tag_*` and `interact_*` blocks |
| `src/rec_eval.py` | Metrics |