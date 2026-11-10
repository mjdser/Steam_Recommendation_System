"""heuristic_features.py — CF + content + popularity features.

Shared by training-data build and eval, so features are identical everywhere.
"""
import numpy as np
import pandas as pd


HEURISTIC_COLS = (
    [f"cf_sim_{i}" for i in range(1, 7)] +
    ["cf_score", "cf_max"] +
    ["content_score", "content_score_top3", "max_sim_to_any_game"] +
    ["candidate_popularity", "user_mean_popularity"]
)


def load_heuristic_context(sim_path, train_df, n_slots=6):
    """Load sim matrix and popularity, both from training data only."""
    data = np.load(sim_path)
    sim = data["sim"].astype(np.float32)
    sim_appids = data["appids"]
    appid_to_idx = {int(a): i for i, a in enumerate(sim_appids)}

    pop = train_df["appid"].value_counts()
    pop_log = np.log1p(pop)
    pop_norm = (pop_log / pop_log.max()).to_dict()

    users_per_game = train_df.groupby("appid")["steamid"].nunique()
    users_per_game = users_per_game.to_dict()

    return {
        "sim": sim,
        "appid_to_idx": appid_to_idx,
        "pop_norm": pop_norm,
        "n_slots": n_slots,
        "users_per_game": users_per_game,
    }


def add_heuristic_features(row_dict, uid, cands, played, ctx,
                            game_feats, n_slots=6, user_full_history=None):
    """
    Append heuristic features to row_dict IN PLACE.

    row_dict: must already contain slot{i}_{f}, slot{i}_interest,
              slot{i}_present, cand_{f} columns
    played:   DataFrame [appid, interest] sorted desc, or None
    """
    n = len(cands)
    sim = ctx["sim"]
    ati = ctx["appid_to_idx"]
    pop_norm = ctx["pop_norm"]

    # ---------- popularity ----------
    row_dict["candidate_popularity"] = np.array(
        [pop_norm.get(int(a), 0.0) for a in cands], dtype=np.float32
    )

    if played is not None and len(played) > 0:
        played = played.head(n_slots)
        user_pop = float(np.mean([pop_norm.get(int(a), 0.0)
                                   for a in played["appid"]]))
    else:
        played = None
        user_pop = 0.0
    row_dict["user_mean_popularity"] = np.full(n, user_pop, dtype=np.float32)

    # ---------- CF features ----------
    cand_idx = np.array([ati.get(int(a), -1) for a in cands])
    cand_valid = cand_idx >= 0

    cf_sims = np.zeros((n_slots, n), dtype=np.float32)
    slot_weights = np.zeros(n_slots, dtype=np.float32)

    # users_per_game = ctx["users_per_game"]

    # the set of games this user played in TRAINING — used to detect
    # whether a slot and a candidate both contributed to sim via this user
    if user_full_history is not None:
        played_appids = set(int(a) for a in user_full_history["appid"])
    elif played is not None:
        played_appids = set(int(a) for a in played["appid"])
    else:
        played_appids = set()


    if played is not None:
        for i in range(len(played)):
            g_appid = int(played.iloc[i]["appid"])
            slot_weights[i] = float(played.iloc[i]["interest"])
            g_idx = ati.get(g_appid, -1)
            if g_idx >= 0:
                cf_sims[i][cand_valid] = sim[g_idx][cand_idx[cand_valid]]

    for i in range(n_slots):
        row_dict[f"cf_sim_{i+1}"] = cf_sims[i]

    if slot_weights.sum() > 0:
        row_dict["cf_score"] = ((cf_sims * slot_weights[:, None]).sum(axis=0)
                                / (slot_weights.sum() + 1e-9)).astype(np.float32)
    else:
        row_dict["cf_score"] = np.zeros(n, dtype=np.float32)

    present = slot_weights > 0
    if present.any():
        row_dict["cf_max"] = cf_sims[present].max(axis=0)
    else:
        row_dict["cf_max"] = np.zeros(n, dtype=np.float32)

    # ---------- content features (from existing slot + cand columns) ----------
    slot_arr = np.stack(
        [np.stack([row_dict[f"slot{i}_{f}"] for f in game_feats], axis=1)
         for i in range(1, n_slots + 1)],
        axis=0,
    )  # (n_slots, n, n_feats)
    slot_arr = np.nan_to_num(slot_arr, nan=0.0).astype(np.float32)

    slot_int  = np.stack([row_dict[f"slot{i}_interest"] for i in range(1, n_slots+1)], axis=0).astype(np.float32)
    slot_pres = np.stack([row_dict[f"slot{i}_present"]  for i in range(1, n_slots+1)], axis=0).astype(np.float32)


    w = np.where(slot_pres > 0, np.nan_to_num(slot_int, nan=0.0), 0.0).astype(np.float32)
    wsum = w.sum(axis=0) + 1e-9

    p_u = (slot_arr * w[:, :, None]).sum(axis=0) / wsum[:, None]
    p_u_norm = p_u / (np.linalg.norm(p_u, axis=1, keepdims=True) + 1e-9)

    cand_arr = np.stack([row_dict[f"cand_{f}"] for f in game_feats], axis=1)
    cand_arr = np.nan_to_num(cand_arr, nan=0.0)
    cand_norm = cand_arr / (np.linalg.norm(cand_arr, axis=1, keepdims=True) + 1e-9)

    row_dict["content_score"] = (cand_norm * p_u_norm).sum(axis=1).astype(np.float32)

    w3 = w[:3, :]
    w3sum = w3.sum(axis=0) + 1e-9
    p_u3 = (slot_arr[:3] * w3[:, :, None]).sum(axis=0) / w3sum[:, None]
    p_u3_norm = p_u3 / (np.linalg.norm(p_u3, axis=1, keepdims=True) + 1e-9)
    row_dict["content_score_top3"] = (cand_norm * p_u3_norm).sum(axis=1).astype(np.float32)

    slot_norms = slot_arr / (np.linalg.norm(slot_arr, axis=2, keepdims=True) + 1e-9)
    cos_all = (slot_norms * cand_norm[None, :, :]).sum(axis=2)   # (n_slots, n)
    cos_all = np.where(slot_pres > 0, cos_all, -np.inf)
    max_sim = cos_all.max(axis=0)
    row_dict["max_sim_to_any_game"] = np.where(
        np.isfinite(max_sim), max_sim, 0.0
    ).astype(np.float32)

    return row_dict