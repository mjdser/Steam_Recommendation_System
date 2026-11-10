import os, json
import numpy as np
import pandas as pd
import xgboost as xgb
from rec_eval import (
    evaluate_model, build_popularity, relevant_from_val,
    recall_at_k, ndcg_at_k, reciprocal_rank, catalog_coverage, novelty,
)

# ------------------------------------------------------------------ config
MODEL_DIR = "recommender_project/models"
DATA_DIR  = "recommender_project/data"
N_SLOTS   = 6
K_LIST    = (5, 10, 20)
N_NEG     = 500            # negatives per user for the candidate pool
SEED      = 0
rng       = np.random.default_rng(SEED)

# ------------------------------------------------------------------ load models + bundles
def load_model_and_bundle(name):
    m = xgb.Booster()
    m.load_model(os.path.join(MODEL_DIR, f"{name}.json"))
    with open(os.path.join(MODEL_DIR, f"{name}_bundle.json")) as f:
        b = json.load(f)
    return m, b

model_slot,  bundle_slot  = load_model_and_bundle("model2_slot_reg")
model_deep,  bundle_deep  = load_model_and_bundle("model_slot_deep")
model_cand,  bundle_cand  = load_model_and_bundle("model_cand_only")

FEATURE_COLS = bundle_slot['feature_cols']
GAME_FEATS   = bundle_slot['game_feats']
CAND_ONLY    = bundle_cand['feature_cols']
print("slot feature cols:", len(FEATURE_COLS), "| game feats:", len(GAME_FEATS),
      "| cand-only:", len(CAND_ONLY))

# ------------------------------------------------------------------ load data
train = pd.read_csv(os.path.join(DATA_DIR, "training_data_train_final.csv"))
val   = pd.read_csv(os.path.join(DATA_DIR, "training_data_val_final.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "training_data_test_final.csv"))
raw_games = pd.read_csv("recommender_project/new_package/03_features/game_features.csv")

# cast all appid columns to int64 (consistent across every frame)
train['appid'] = train['appid'].astype('int64')
val['appid']   = val['appid'].astype('int64')
test['appid']  = test['appid'].astype('int64')
raw_games['appid'] = raw_games['appid'].astype('int64')

print("train:", train.shape, "| val:", val.shape, "| test:", test.shape)


# check how many rows have a missing appid
print("train rows with NaN appid:", train['appid'].isna().sum())
print("val   rows with NaN appid:", val['appid'].isna().sum())
print("test  rows with NaN appid:", test['appid'].isna().sum())

# ------------------------------------------------------------------ catalog + popularity (from TRAIN only)
catalog_appids = train['appid'].unique()          # every game that appears in train
df_games_eval  = pd.DataFrame({'appid': catalog_appids})
popularity     = build_popularity(train)          # {appid: prob}

# users to evaluate = users present in val (fall back to test)
eval_users = val['steamid'].unique()
print("users to evaluate:", len(eval_users))


# ------------------------------------------------------------------ candidate pools
# For each user: their val games (positives) + N_NEG random negatives drawn from
# games the user has NEVER played (excludes train, val, AND test).
def build_pools(eval_df, catalog_appids, n_neg, rng,
                train_df=None, test_df=None):
    # per-user sets of games to exclude from the negative pool
    val_sets   = {uid: set(g['appid']) for uid, g in eval_df.groupby('steamid')}
    train_sets = ({uid: set(g['appid']) for uid, g in train_df.groupby('steamid')}
                  if train_df is not None else {})
    test_sets  = ({uid: set(g['appid']) for uid, g in test_df.groupby('steamid')}
                  if test_df is not None else {})

    pools = {}
    for uid, g in eval_df.groupby('steamid'):
        pos = list(val_sets[uid])
        exclude = (val_sets[uid]
                   | train_sets.get(uid, set())
                   | test_sets.get(uid, set()))
        pool = catalog_appids[~np.isin(catalog_appids, list(exclude))]
        neg  = rng.choice(pool, size=min(n_neg, len(pool)), replace=False)
        pools[uid] = np.array(pos + list(neg))
    return pools

pools = build_pools(val, catalog_appids, N_NEG, rng,
                    train_df=train, test_df=test)

# ground truth: {steamid: {appid: interest}} for the val games
relevant = relevant_from_val(val)

print(raw_games.shape)
print([c for c in raw_games.columns][:20])


# ------------------------------------------------------------------ game feature lookup
game_feats_df = raw_games.drop_duplicates('appid').set_index('appid')[GAME_FEATS]


# --- guard: dtype must match, and every catalog game must have features ---
assert game_feats_df.index.dtype == catalog_appids.dtype, \
    f"appid dtype mismatch: feats={game_feats_df.index.dtype}, catalog={catalog_appids.dtype}"
_catalog_missing = set(catalog_appids) - set(game_feats_df.index)
print(f"catalog appids missing from feature file: {len(_catalog_missing)} / {len(catalog_appids)}")


# user's train games: appid + interest only
train_games_by_user = {
    uid: g.sort_values('interest', ascending=False)[['appid', 'interest']]
    for uid, g in train.groupby('steamid')
}

# ------------------------------------------------------------------ slot-row builder
def build_candidate_matrix(uid, candidate_appids):
    played = train_games_by_user.get(uid)
    feats  = game_feats_df.reindex(candidate_appids)
    n = len(candidate_appids)
    rows = {}

    # ALWAYS create the slot columns, empty by default
    for i in range(1, N_SLOTS + 1):
        for f in GAME_FEATS:
            rows[f'slot{i}_{f}'] = np.full(n, np.nan)
        rows[f'slot{i}_interest'] = np.full(n, np.nan)
        rows[f'slot{i}_present']  = np.zeros(n)

    # fill them only if the user has train games
    if played is not None:
        played = played.head(N_SLOTS)
        for i in range(1, N_SLOTS + 1):
            if i <= len(played):
                g  = played.iloc[i - 1]
                gf = game_feats_df.loc[g['appid']]
                for f in GAME_FEATS:
                    rows[f'slot{i}_{f}'] = gf[f]
                rows[f'slot{i}_interest'] = g['interest']
                rows[f'slot{i}_present']  = 1

    # candidate features
    for f in GAME_FEATS:
        rows[f'cand_{f}'] = feats[f].to_numpy()

    return pd.DataFrame(rows)

# ------------------------------------------------------------------ score functions
def make_score_slot(model, best_iter):
    def score_fn(uid, candidate_appids):
        X = build_candidate_matrix(uid, candidate_appids)
        return model.predict(xgb.DMatrix(X[FEATURE_COLS]),
                             iteration_range=(0, best_iter + 1))
    return score_fn

def make_score_cand(model, best_iter):
    def score_fn(uid, candidate_appids):
        feats = game_feats_df.reindex(candidate_appids)
        X = feats.to_numpy()
        Xdf = pd.DataFrame(X, columns=CAND_ONLY)
        return model.predict(xgb.DMatrix(Xdf),
                             iteration_range=(0, best_iter + 1))
    return score_fn

score_slot = make_score_slot(model_slot, bundle_slot['best_iteration'])
score_deep = make_score_slot(model_deep, bundle_deep['best_iteration'])
score_cand = make_score_cand(model_cand, bundle_cand['best_iteration'])


# --- popularity baseline: rank candidates by training frequency ---
pop_counts = train['appid'].value_counts()

def score_popularity(uid, candidate_appids):
    """Higher = more popular in training. No personalization."""
    return pop_counts.reindex(candidate_appids).fillna(0).to_numpy()



# ------------------------------------------------------------------ evaluate
def run_eval(score_fn, label):
    res = evaluate_model(
        users=eval_users,
        score_fn=score_fn,
        df_games=df_games_eval,
        relevant_by_user=relevant,
        item_popularity=popularity,
        candidates_per_user=pools,
        K=K_LIST,
    )
    res.insert(0, 'model', label)
    return res

results = pd.concat([
    run_eval(score_slot,      'slot_reg'),
    run_eval(score_deep,      'slot_deep'),
    run_eval(score_cand,      'cand_only'),
    run_eval(score_popularity,'popularity'),
], ignore_index=True)


pd.set_option('display.float_format', lambda x: f'{x:.4f}')
print(results.to_string(index=False))