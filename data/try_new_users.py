"""Score a few Steam users who are not in libraries_top10.csv.

User ids come from the public Kaggle Steam-200k file (steam32 ids).
Libraries are read from each public Steam profile. There is no Web API key
in this environment, so this uses the public profile games XML.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "v3_user_interaction_added"))

from rank_pipeline_v3_interact import (  # noqa: E402
    N_SLOTS,
    _build_cooccurrence,
    _build_slot_cand,
)
from heuristic_features_v2 import add_heuristic_features  # noqa: E402

STEAM64 = 76561197960265728
H_CAP = 200.0
SIM_PATH = ROOT / "data" / "game_sim_from_top10.npz"
N_USERS = 3


def norm_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def interest_rows(games):
    rows = []
    for g in games:
        forever_h = float(g["hours"])
        weeks_h = float(g["hours_2weeks"])
        if forever_h <= 0 and weeks_h <= 0:
            continue
        rows.append({**g, "forever_h": forever_h, "weeks_h": weeks_h,
                     "log_forever": math.log1p(forever_h),
                     "log_weeks": math.log1p(weeks_h)})
    if not rows:
        return pd.DataFrame()
    max_f = max(r["log_forever"] for r in rows)
    max_w = max(r["log_weeks"] for r in rows)
    invest_den = math.log1p(H_CAP)
    out = []
    for r in rows:
        s_play = r["log_forever"] / max_f if max_f else 0.0
        s_recent = r["log_weeks"] / max_w if max_w else 0.0
        s_mom = 0.0 if r["forever_h"] <= 0 else min(1.0, r["log_weeks"] / r["log_forever"])
        s_inv = 0.0 if r["forever_h"] <= 0 else min(1.0, r["log_forever"] / invest_den)
        out.append({
            "appid": int(r["appid"]),
            "name": r["name"],
            "hours": round(r["forever_h"], 2),
            "hours_2weeks": round(r["weeks_h"], 2),
            "interest": round(0.40 * s_recent + 0.35 * s_play + 0.15 * s_mom + 0.10 * s_inv, 4),
        })
    return pd.DataFrame(out).sort_values("interest", ascending=False)


def fetch_library(steamid64):
    url = f"https://steamcommunity.com/profiles/{steamid64}/games?tab=all&xml=1"
    r = requests.get(url, timeout=30, headers={"User-Agent": "recommender-check/1.0"})
    if r.status_code != 200 or "<appID>" not in r.text:
        return None
    games = []
    for block in re.findall(r"<game>(.*?)</game>", r.text, flags=re.S):
        app = re.search(r"<appID>(\d+)</appID>", block)
        if not app:
            continue
        name = re.search(r"<name><!\[CDATA\[(.*?)\]\]></name>|<name>(.*?)</name>", block)
        hours = re.search(r"<hoursOnRecord>([\d.]+)</hoursOnRecord>", block)
        weeks = re.search(r"<hoursLast2Weeks>([\d.]+)</hoursLast2Weeks>", block)
        games.append({
            "appid": int(app.group(1)),
            "name": (name.group(1) or name.group(2) or "") if name else "",
            "hours": float(hours.group(1)) if hours else 0.0,
            "hours_2weeks": float(weeks.group(1)) if weeks else 0.0,
        })
    return games


def load_context(game_feats_df):
    if SIM_PATH.exists():
        data = np.load(SIM_PATH)
        sim = data["sim"].astype(np.float32)
        appids = data["appids"]
        print(f"loaded sim {sim.shape} from {SIM_PATH.name}")
    else:
        wide = pd.read_csv(ROOT / "new_package" / "07_users" / "libraries_top10.csv")
        parts = []
        for i in range(1, 11):
            parts.append(wide[["steamid", f"top{i}_appid", f"top{i}_interest"]].rename(
                columns={f"top{i}_appid": "appid", f"top{i}_interest": "interest"}
            ))
        long = pd.concat(parts, ignore_index=True).dropna(subset=["appid"])
        long["appid"] = long["appid"].astype("int64")
        catalog = np.intersect1d(long["appid"].unique(), game_feats_df.index.to_numpy())
        long = long[long["appid"].isin(set(catalog))]
        print(f"building sim from {long['steamid'].nunique()} training users, {len(catalog)} games")
        sim, _ = _build_cooccurrence(long, catalog)
        np.savez_compressed(SIM_PATH, sim=sim, appids=catalog)
        appids = catalog
        print(f"saved {SIM_PATH.name}")
    pop = pd.Series(appids).value_counts()  # placeholder replaced below
    # popularity from the same training histories
    wide = pd.read_csv(ROOT / "new_package" / "07_users" / "libraries_top10.csv",
                       usecols=[c for c in pd.read_csv(ROOT / "new_package" / "07_users" / "libraries_top10.csv", nrows=0).columns if c.endswith("_appid") or c == "steamid"])
    apps = []
    for i in range(1, 11):
        apps.append(wide[f"top{i}_appid"])
    counts = pd.concat(apps, ignore_index=True).dropna().astype("int64").value_counts()
    pop_log = np.log1p(counts)
    pop_norm = (pop_log / pop_log.max()).to_dict()
    ati = {int(a): i for i, a in enumerate(appids)}
    return {"sim": sim, "appid_to_idx": ati, "pop_norm": pop_norm, "n_slots": N_SLOTS}, set(int(a) for a in appids)


def recommend(played, model, bundle, game_feats_df, ctx, names, n=8):
    feats = bundle["feature_cols"]
    game_feats = bundle["game_feats"]
    owned = set(played["appid"].astype(int))
    cands = np.array([a for a in game_feats_df.index if int(a) in ctx["appid_to_idx"] and int(a) not in owned])
    hist = played.sort_values("interest", ascending=False).head(N_SLOTS)
    rows = _build_slot_cand(hist, cands, game_feats_df, game_feats)
    add_heuristic_features(rows, None, cands, hist, ctx, game_feats, N_SLOTS)
    X = pd.DataFrame(rows)[feats]
    scores = model.predict(xgb.DMatrix(X), iteration_range=(0, bundle["best_iteration"] + 1))
    out = pd.DataFrame({"appid": cands, "score": scores})
    out["name"] = out["appid"].map(lambda a: names.get(int(a), ""))
    return out.sort_values("score", ascending=False).head(n)


def main():
    known = set(pd.read_csv(
        ROOT / "new_package" / "07_users" / "libraries_top10.csv", usecols=["steamid"]
    )["steamid"].astype("int64"))
    raw = pd.read_csv(ROOT / "data" / "steam-200k.csv", header=None,
                      names=["user", "game", "behavior", "hours", "extra"])
    play = raw[raw["behavior"] == "play"]
    counts = play.groupby("user").size()
    candidates = counts[counts >= 8].index.astype("int64").to_numpy().copy()
    rng = np.random.default_rng(0)
    rng.shuffle(candidates)

    bundle = pd.read_json(ROOT / "models" / "model_rank_v3_interact_bundle.json", typ="series")
    # read_json series may not keep lists well; use json
    import json
    bundle = json.load(open(ROOT / "models" / "model_rank_v3_interact_bundle.json", encoding="utf-8"))
    games = pd.read_csv(ROOT / "new_package" / "03_features" / "game_features.csv")
    games["appid"] = games["appid"].astype("int64")
    game_feats_df = games.drop_duplicates("appid").set_index("appid")[bundle["game_feats"]]
    names = pd.read_csv(ROOT / "new_package" / "01_catalog" / "games_for_features.csv",
                        usecols=["appid", "name"])
    names = dict(zip(names["appid"].astype(int), names["name"]))
    name_to_app = {}
    for appid, name in names.items():
        name_to_app.setdefault(norm_name(name), appid)
    name_to_app[norm_name("Counter-Strike:Global Offensive")] = 730

    ctx, sim_games = load_context(game_feats_df)
    model = xgb.Booster()
    model.load_model(str(ROOT / "models" / "model_rank_v3_interact.json"))

    got = 0
    tried = 0
    for uid32 in candidates:
        sid = int(uid32) + STEAM64
        if sid in known:
            continue
        tried += 1
        if tried > 25:
            break
        print(f"fetch {sid}")
        lib = fetch_library(sid)
        if not lib:
            print("  profile games are not public without a Steam Web API key; using Kaggle play hours")
            mine = play[play["user"] == int(uid32)]
            lib = []
            for _, row in mine.iterrows():
                appid = name_to_app.get(norm_name(row["game"]))
                if appid is None:
                    continue
                lib.append({"appid": appid, "name": names.get(appid, row["game"]),
                            "hours": float(row["hours"]), "hours_2weeks": 0.0})
        if not lib:
            continue
        scored = interest_rows(lib)
        if scored.empty:
            continue
        in_catalog = scored[scored["appid"].isin(game_feats_df.index)]
        if len(in_catalog) < 4:
            print(f"  only {len(in_catalog)} games in our feature table")
            continue
        print(f"\n=== {sid}  (kaggle steam32 {int(uid32)}) ===")
        print("history used by the model:")
        hist = in_catalog.sort_values("interest", ascending=False).head(6)
        print(hist[["name", "hours", "hours_2weeks", "interest"]].to_string(index=False))
        recs = recommend(in_catalog, model, bundle, game_feats_df, ctx, names)
        print("v3 recommendations:")
        print(recs[["name", "appid", "score"]].to_string(index=False))
        got += 1
        if got >= N_USERS:
            break
    print(f"\nscored {got} new users, tried {tried} profiles")


if __name__ == "__main__":
    main()
