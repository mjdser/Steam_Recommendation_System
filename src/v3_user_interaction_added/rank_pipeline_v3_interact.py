"""rank_pipeline_v3_interact.py — nopop + tag-interaction features.

Usage:
    python rank_pipeline_v3_interact.py --stage all
    python rank_pipeline_v3_interact.py --stage build
    python rank_pipeline_v3_interact.py --stage train
    python rank_pipeline_v3_interact.py --stage eval
"""
import os, json, time, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rec_eval import evaluate_model
from v3_user_interaction_added.heuristic_features_v2 import (
    load_heuristic_context, add_heuristic_features,
    HEURISTIC_COLS, interaction_cols,
)

# ================================================================== config
BASE      = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = BASE / "data"
MODEL_DIR = BASE / "models"
RAW_FEATS = BASE / "new_package" / "03_features" / "game_features.csv"

SLOT_BUNDLE       = MODEL_DIR / "model2_slot_reg_bundle.json"
SIM_PATH          = DATA_DIR  / "game_sim_matrix.npz"

# v3 artefacts — distinct from v2
TR_DATA_PATH      = DATA_DIR  / "tr_rank_v3_interact.parquet"
MODEL_PATH        = MODEL_DIR / "model_rank_v3_interact.json"
BUNDLE_PATH       = MODEL_DIR / "model_rank_v3_interact_bundle.json"

# v2 models to compare against (loaded read-only)
NOPOP_MODEL_PATH  = MODEL_DIR / "model_rank_v2_nopop.json"
NOPOP_BUNDLE_PATH = MODEL_DIR / "model_rank_v2_nopop_bundle.json"

HYBRID_MODEL_PATH  = MODEL_DIR / "model_rank_hybrid.json"
HYBRID_BUNDLE_PATH = MODEL_DIR / "model_rank_hybrid_bundle.json"

N_SLOTS       = 6
N_NEG_TRAIN   = 15
SEED          = 0
VAL_FRAC      = 0.10
K_LIST        = (5, 10, 20)
N_NEG_EVAL    = 500
CHUNK_SIZE    = 5000
SIM_USER_FRAC = 0.30

DROP_FEATURE  = "candidate_popularity"


# ================================================================== helpers (shared)
def _load_base():
    with open(SLOT_BUNDLE) as f:
        slot_bundle = json.load(f)
    GAME_FEATS = slot_bundle["game_feats"]
    assert slot_bundle["n_slots"] == N_SLOTS

    train = pd.read_csv(DATA_DIR / "training_data_train_final.csv")
    val   = pd.read_csv(DATA_DIR / "training_data_val_final.csv")
    test  = pd.read_csv(DATA_DIR / "training_data_test_final.csv")
    raw   = pd.read_csv(RAW_FEATS)
    for df in (train, val, test, raw):
        df.dropna(subset=["appid"], inplace=True)
        df["appid"] = df["appid"].astype("int64")
    game_feats_df = raw.drop_duplicates("appid").set_index("appid")[GAME_FEATS]
    assert game_feats_df.index.dtype == train["appid"].dtype
    return GAME_FEATS, train, val, test, raw, game_feats_df


def _split_users(train, sim_frac=SIM_USER_FRAC, seed=SEED):
    users = train["steamid"].unique()
    rng = np.random.default_rng(seed); rng.shuffle(users)
    cut = int(sim_frac * len(users))
    sim_users   = set(users[:cut])
    model_users = set(users[cut:])
    print(f"[split] sim_users={len(sim_users):,} | model_users={len(model_users):,}")
    return sim_users, model_users


def _build_slot_cand(played_df, cands, game_feats_df, GAME_FEATS):
    feats = game_feats_df.reindex(cands)
    n = len(cands)
    rows = {}
    for i in range(1, N_SLOTS + 1):
        for f in GAME_FEATS:
            rows[f"slot{i}_{f}"] = np.full(n, np.nan, dtype=np.float32)
        rows[f"slot{i}_interest"] = np.full(n, np.nan, dtype=np.float32)
        rows[f"slot{i}_present"]  = np.zeros(n, dtype=np.float32)
    if played_df is not None and len(played_df) > 0:
        played_df = played_df.head(N_SLOTS)
        for i in range(1, N_SLOTS + 1):
            if i <= len(played_df):
                g  = played_df.iloc[i - 1]
                gf = game_feats_df.loc[g["appid"]]
                for f in GAME_FEATS:
                    rows[f"slot{i}_{f}"] = np.full(n, gf[f], dtype=np.float32)
                rows[f"slot{i}_interest"] = np.full(n, g["interest"], dtype=np.float32)
                rows[f"slot{i}_present"]  = np.ones(n, dtype=np.float32)
    for f in GAME_FEATS:
        rows[f"cand_{f}"] = feats[f].to_numpy(dtype=np.float32)
    return rows


def _build_one_group(pos, ctx, train_games_by_user, game_feats_df,
                     GAME_FEATS, rng, neg_sampler):
    uid = pos["steamid"]
    pos_appid = pos["appid"]
    user_games = train_games_by_user.get(uid)
    others = (pd.DataFrame(columns=["appid", "interest"])
              if user_games is None
              else user_games[user_games["appid"] != pos_appid])
    negs = neg_sampler(uid, pos_appid, rng)
    cands = np.concatenate([[pos_appid], negs])
    labels = np.concatenate([[1], np.zeros(len(negs))])
    n = len(cands)

    row_dict = _build_slot_cand(others, cands, game_feats_df, GAME_FEATS)
    row_dict["sample_weight"] = np.concatenate(
        [[pos["interest"]], np.ones(len(negs))]
    ).astype(np.float32)

    add_heuristic_features(row_dict, uid, cands, others, ctx,
                            GAME_FEATS, N_SLOTS)

    row_dict["label"]   = labels.astype(np.float32)
    row_dict["steamid"] = np.full(n, uid, dtype=np.int64)
    row_dict["appid"]   = cands.astype(np.int64)
    return pd.DataFrame(row_dict)


def _build_cooccurrence(sim_users_df, catalog_appids):
    import scipy.sparse as sp
    appid_to_idx = {int(a): i for i, a in enumerate(catalog_appids)}
    n_games = len(catalog_appids)
    users = sim_users_df["steamid"].unique()
    user_to_idx = {u: i for i, u in enumerate(users)}
    u_idx = sim_users_df["steamid"].map(user_to_idx).to_numpy(dtype=np.float64)
    g_idx = sim_users_df["appid"].map(lambda a: appid_to_idx.get(int(a), -1)).to_numpy()
    valid = ~(pd.isna(u_idx) | (g_idx < 0))
    u_idx = u_idx[valid].astype(np.int64); g_idx = g_idx[valid].astype(np.int64)
    pairs = pd.DataFrame({"u": u_idx, "g": g_idx}).drop_duplicates()
    U = sp.csr_matrix(
        (np.ones(len(pairs), dtype=np.float32),
         (pairs["u"].to_numpy(), pairs["g"].to_numpy())),
        shape=(len(users), n_games),
    )
    C = (U.T @ U).tocsr()
    users_per_game = np.asarray(U.sum(axis=0)).ravel().astype(np.float32)
    users_per_game[users_per_game == 0] = 1.0
    inv_sqrt = 1.0 / np.sqrt(users_per_game)
    sim_sp = sp.diags(inv_sqrt) @ C.astype(np.float32) @ sp.diags(inv_sqrt)
    sim = sim_sp.toarray().astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    print(f"  sim: {sim.shape} | nonzero={np.count_nonzero(sim):,}")
    return sim, appid_to_idx


# ================================================================== stage: build
def build_stage(cfg, force=False):
    if TR_DATA_PATH.exists() and not force:
        print(f"[build] {TR_DATA_PATH.name} exists — skipping")
        return

    print("[build] loading ...")
    GAME_FEATS, train, val, test, raw, game_feats_df = _load_base()
    sim_users, model_users = _split_users(train)

    catalog_appids = np.intersect1d(
        train["appid"].unique(), game_feats_df.index.to_numpy()
    )
    print(f"[build] catalog: {len(catalog_appids)} games")

    train_m = train[train["steamid"].isin(model_users)].copy()
    train_s = train[train["steamid"].isin(sim_users)].copy()

    sim, appid_to_idx = _build_cooccurrence(train_s, catalog_appids)
    np.savez_compressed(SIM_PATH, sim=sim, appids=catalog_appids)
    print(f"[build] saved sim to {SIM_PATH}")

    pop_counts = train_m["appid"].value_counts()
    pop_log = np.log1p(pop_counts)
    pop_norm = (pop_log / pop_log.max()).to_dict()

    ctx = {
        "sim": sim.astype(np.float32),
        "appid_to_idx": appid_to_idx,
        "pop_norm": pop_norm,
        "n_slots": N_SLOTS,
    }

    train_games_by_user = {
        uid: g.sort_values("interest", ascending=False)
               [["appid", "interest"]].reset_index(drop=True)
        for uid, g in train_m.groupby("steamid")
    }
    all_played = (
        pd.concat([train_m[["steamid", "appid"]],
                   val[["steamid", "appid"]],
                   test[["steamid", "appid"]]])
          .groupby("steamid")["appid"].apply(set).to_dict()
    )

    feats_norm = game_feats_df.to_numpy(dtype=np.float32)
    feats_norm /= (np.linalg.norm(feats_norm, axis=1, keepdims=True) + 1e-9)
    appid_arr = game_feats_df.index.to_numpy()
    TOP_SIM = 200
    print("[build] precomputing similarity cache ...")
    sim_cache = {}
    for aid in game_feats_df.index:
        p = game_feats_df.index.get_loc(aid)
        s = feats_norm @ feats_norm[p]
        sim_cache[aid] = appid_arr[np.argsort(-s)[:TOP_SIM]]

    pop_counts_m = train_m["appid"].value_counts()
    pop_bucket = pd.qcut(pop_counts_m.rank(method="first"), q=10,
                          labels=False, duplicates="drop")
    def bucket_of(a): return pop_bucket.get(a, np.nan)
    catalog_buckets = np.array([bucket_of(a) for a in catalog_appids])

    def sample_negatives(uid, pos_appid, rng,
                          n_rand=5, n_hard=5, n_pop=5, sim_band=(10, 100)):
        played = all_played.get(uid, set())
        pool_all = appid_arr[np.array([a not in played for a in appid_arr])]
        rand_neg = rng.choice(pool_all, size=min(n_rand, len(pool_all)), replace=False)

        top_sim = sim_cache[pos_appid]
        band = top_sim[sim_band[0]:sim_band[1]]
        hp = np.array([a for a in band if a not in played])
        hard_neg = (rng.choice(hp, size=min(n_hard, len(hp)), replace=False)
                    if len(hp) else np.array([], dtype=appid_arr.dtype))

        ub = [bucket_of(a) for a in played]
        ub = [b for b in ub if not np.isnan(b)]
        tb = int(np.median(ub)) if ub else None
        m = np.array([a not in played for a in catalog_appids])
        if tb is not None:
            m &= (catalog_buckets == tb)
        pool = catalog_appids[m]
        if len(pool) < n_pop:
            pool = catalog_appids[np.array([a not in played for a in catalog_appids])]
        pop_neg = (rng.choice(pool, size=min(n_pop, len(pool)), replace=False)
                   if len(pool) else np.array([], dtype=appid_arr.dtype))

        out = np.unique(np.concatenate([rand_neg, hard_neg, pop_neg]))
        if len(out) < N_NEG_TRAIN:
            extra = pool_all[~np.isin(pool_all, out)]
            if len(extra):
                top = rng.choice(extra, size=min(N_NEG_TRAIN - len(out), len(extra)), replace=False)
                out = np.concatenate([out, top])
        return out

    rng = np.random.default_rng(SEED)
    pos_df = train_m[["steamid", "appid", "interest"]].reset_index(drop=True)

    writer = None; buffer = []; done = 0
    print(f"[build] writing to {TR_DATA_PATH} ...")
    for i in range(len(pos_df)):
        buffer.append(_build_one_group(pos_df.iloc[i], ctx,
                                        train_games_by_user, game_feats_df,
                                        GAME_FEATS, rng, sample_negatives))
        done += 1
        if len(buffer) >= CHUNK_SIZE:
            df = pd.concat(buffer, ignore_index=True)
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(TR_DATA_PATH, tbl.schema, compression="snappy")
            writer.write_table(tbl)
            print(f"  {done}/{len(pos_df)} groups")
            buffer = []; del df, tbl
    if buffer:
        df = pd.concat(buffer, ignore_index=True)
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(TR_DATA_PATH, tbl.schema, compression="snappy")
        writer.write_table(tbl)
    if writer is not None:
        writer.close()
    print(f"[build] done — {done} groups → {TR_DATA_PATH}")


# ================================================================== stage: train
def train_stage(cfg, force=False):
    if MODEL_PATH.exists() and BUNDLE_PATH.exists() and not force:
        print(f"[train] {MODEL_PATH.name} exists — skipping")
        return

    with open(SLOT_BUNDLE) as f:
        slot_bundle = json.load(f)
    GAME_FEATS = slot_bundle["game_feats"]

    base_cols = (
        [f"slot{i}_{f}"      for i in range(1, N_SLOTS+1) for f in GAME_FEATS] +
        [f"slot{i}_interest" for i in range(1, N_SLOTS+1)] +
        [f"slot{i}_present"  for i in range(1, N_SLOTS+1)] +
        [f"cand_{f}"         for f in GAME_FEATS]
    )
    int_cols = interaction_cols(GAME_FEATS)

    FEATURE_COLS = base_cols + HEURISTIC_COLS + int_cols
    # nopop: remove candidate_popularity
    FEATURE_COLS = [c for c in FEATURE_COLS if c != DROP_FEATURE]

    print(f"[train] {len(FEATURE_COLS)} features "
          f"({len(base_cols)} base + {len(HEURISTIC_COLS)} heur + {len(int_cols)} interact "
          f"- 1 dropped)")

    print(f"[train] loading {TR_DATA_PATH.name} ...")
    df = pd.read_parquet(TR_DATA_PATH)
    print(f"[train] {len(df):,} rows | pos={int(df['label'].sum()):,}")

    users = df["steamid"].unique()
    rng = np.random.default_rng(SEED); rng.shuffle(users)
    cut = int((1 - VAL_FRAC) * len(users))
    val_users = set(users[cut:])
    is_val = df["steamid"].isin(val_users)
    tr_fit = df[~is_val].copy(); tr_es = df[is_val].copy()
    del df, is_val
    print(f"[train] fit={len(tr_fit):,} es={len(tr_es):,}")

    def make_dm(d):
        return xgb.DMatrix(
            d[FEATURE_COLS].to_numpy(dtype=np.float32),
            label=d["label"].to_numpy(dtype=np.float32),
            weight=d["sample_weight"].to_numpy(dtype=np.float32),
        )
    dtrain, dval = make_dm(tr_fit), make_dm(tr_es)
    del tr_fit, tr_es

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "tree_method": "hist",
        "eta": 0.05, "max_depth": 4,
        "subsample": 0.7, "colsample_bytree": 0.3,
        "min_child_weight": 20, "reg_lambda": 5.0, "reg_alpha": 1.0,
        "seed": SEED,
    }
    print("[train] training ...")
    model = xgb.train(params, dtrain, num_boost_round=3000,
                       evals=[(dtrain, "train"), (dval, "val")],
                       early_stopping_rounds=50, verbose_eval=100)
    best_iter = int(model.best_iteration)
    print(f"[train] best_iteration={best_iter}")

    model.save_model(MODEL_PATH)
    bundle = {
        "feature_cols": FEATURE_COLS,
        "game_feats": GAME_FEATS,
        "n_slots": N_SLOTS,
        "params": params,
        "best_iteration": best_iter,
        "objective": "binary:logistic",
        "heuristic_cols": HEURISTIC_COLS,
        "interaction_cols": int_cols,
        "dropped_features": [DROP_FEATURE],
    }
    with open(BUNDLE_PATH, "w") as f:
        json.dump(bundle, f)
    print(f"[train] saved {MODEL_PATH.name} + bundle")


# ================================================================== stage: eval
def eval_stage(cfg):
    print("[eval] loading models + data ...")
    with open(BUNDLE_PATH) as f:
        bundle = json.load(f)
    FEATURE_COLS = bundle["feature_cols"]
    best_iter = bundle["best_iteration"]
    GAME_FEATS = bundle["game_feats"]

    model = xgb.Booster(); model.load_model(MODEL_PATH)

    # load v2 nopop for comparison
    model_nopop = xgb.Booster()
    model_nopop.load_model(NOPOP_MODEL_PATH)
    with open(NOPOP_BUNDLE_PATH) as f:
        nopop_bundle = json.load(f)
    NOPOP_COLS = nopop_bundle["feature_cols"]
    nopop_best = nopop_bundle["best_iteration"]

    # load hybrid baseline
    model_hyb = xgb.Booster()
    model_hyb.load_model(HYBRID_MODEL_PATH)
    with open(HYBRID_BUNDLE_PATH) as f:
        hyb_bundle = json.load(f)
    HYB_COLS = hyb_bundle["feature_cols"]
    hyb_best = hyb_bundle["best_iteration"]

    GAME_FEATS_OLD, train, val, test, raw, game_feats_df = _load_base()
    assert GAME_FEATS == GAME_FEATS_OLD

    sim_users, model_users = _split_users(train)

    data = np.load(SIM_PATH)
    sim = data["sim"].astype(np.float32)
    sim_appids = data["appids"]
    appid_to_idx = {int(a): i for i, a in enumerate(sim_appids)}

    pop_counts_m = train[train["steamid"].isin(model_users)]["appid"].value_counts()
    pop_log = np.log1p(pop_counts_m)
    pop_norm = (pop_log / pop_log.max()).to_dict()

    ctx = {"sim": sim, "appid_to_idx": appid_to_idx,
           "pop_norm": pop_norm, "n_slots": N_SLOTS}

    catalog_appids = np.intersect1d(
        train["appid"].unique(), game_feats_df.index.to_numpy())
    df_games_eval = pd.DataFrame({"appid": catalog_appids})
    pop_counts = train["appid"].value_counts()
    pop_prob = (pop_counts / pop_counts.sum()).to_dict()

    train_games_by_user = {
        uid: g.sort_values("interest", ascending=False)
               [["appid", "interest"]].reset_index(drop=True)
        for uid, g in train.groupby("steamid")
    }
    val_m = val[val["steamid"].isin(model_users)].copy()
    eval_users = val_m["steamid"].unique()
    relevant = {uid: dict(zip(g["appid"], g["interest"]))
                for uid, g in val_m.groupby("steamid")}
    print(f"[eval] eval users (model_users only): {len(eval_users):,}")

    rng = np.random.default_rng(SEED)
    val_sets   = {uid: set(g["appid"]) for uid, g in val_m.groupby("steamid")}
    train_sets = {uid: set(g["appid"]) for uid, g in train.groupby("steamid")}
    test_sets  = {uid: set(g["appid"]) for uid, g in test.groupby("steamid")}
    pools = {}
    for uid in eval_users:
        pos = list(val_sets[uid])
        exc = val_sets[uid] | train_sets.get(uid, set()) | test_sets.get(uid, set())
        pool = catalog_appids[~np.isin(catalog_appids, list(exc))]
        neg = rng.choice(pool, size=min(N_NEG_EVAL, len(pool)), replace=False)
        pools[uid] = np.array(pos + list(neg))

    def make_score_v3():
        def score_fn(uid, cands):
            played = train_games_by_user.get(uid)
            rd = _build_slot_cand(played, cands, game_feats_df, GAME_FEATS)
            add_heuristic_features(rd, uid, cands, played, ctx, GAME_FEATS, N_SLOTS)
            X = pd.DataFrame(rd)[FEATURE_COLS]
            return model.predict(xgb.DMatrix(X),
                                  iteration_range=(0, best_iter + 1))
        return score_fn

    def make_score_nopop():
        def score_fn(uid, cands):
            played = train_games_by_user.get(uid)
            rd = _build_slot_cand(played, cands, game_feats_df, GAME_FEATS)
            add_heuristic_features(rd, uid, cands, played, ctx, GAME_FEATS, N_SLOTS)
            X = pd.DataFrame(rd)[NOPOP_COLS]
            return model_nopop.predict(xgb.DMatrix(X),
                                        iteration_range=(0, nopop_best + 1))
        return score_fn

    def make_score_hyb():
        def score_fn(uid, cands):
            played = train_games_by_user.get(uid)
            rd = _build_slot_cand(played, cands, game_feats_df, GAME_FEATS)
            X = pd.DataFrame(rd)[HYB_COLS]
            return model_hyb.predict(xgb.DMatrix(X),
                                      iteration_range=(0, hyb_best + 1))
        return score_fn

    def make_score_pop():
        def score_fn(uid, cands):
            return pop_counts.reindex(cands).fillna(0).to_numpy()
        return score_fn

    scorers = {
        "popularity":         make_score_pop(),
        "rank_hybrid":        make_score_hyb(),
        "rank_v2_nopop":      make_score_nopop(),
        "rank_v3_interact":   make_score_v3(),
    }

    all_res = []
    for label, fn in scorers.items():
        t0 = time.time()
        print(f"[eval] {label} ...")
        res = evaluate_model(
            users=eval_users, score_fn=fn,
            df_games=df_games_eval,
            relevant_by_user=relevant,
            item_popularity=pop_prob,
            candidates_per_user=pools,
            K=K_LIST,
        )
        res.insert(0, "model", label)
        all_res.append(res)
        print(f"  done in {time.time()-t0:.1f}s")

    results = pd.concat(all_res, ignore_index=True)
    out_csv = DATA_DIR.parent / "results" / "rank_v3_comparison.csv"
    out_csv.parent.mkdir(exist_ok=True)
    results.to_csv(out_csv, index=False)
    print(f"[eval] saved {out_csv}")

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(results.to_string(index=False))


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["build", "train", "eval", "all"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if args.stage in ("build", "all"):
        build_stage({}, force=args.force)
    if args.stage in ("train", "all"):
        train_stage({}, force=args.force)
    if args.stage in ("eval", "all"):
        eval_stage({})
    print(f"total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()