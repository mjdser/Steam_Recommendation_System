"""Cluster candidates (150 / 100 / 50 from the user's top 3 clusters) + neighbor formula.

Builds three lists per judged user and writes a blind 3-way judging set:
  v3          : rank_v3_interact over the whole catalog
  cluster     : 300 cluster candidates scored by the formula below
  cluster_v3  : the same 300 candidates ranked by rank_v3_interact

score = W_CONTENT * content + W_NEIGHBOR * neighbor + W_CLUSTER * cluster_share
  content       cosine(candidate features, interest-weighted library profile)
  neighbor      sum over library games g of interest(g) * similarity(g, candidate),
                using the top-10 co-occurrence neighbors table
  cluster_share share of the library's interest that falls in the candidate's cluster
Each term is min-max scaled over the 300 candidates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from try_new_users import STEAM64, interest_rows, load_context, norm_name, recommend  # noqa: E402

PER_CLUSTER = (150, 100, 50)
W_CONTENT, W_NEIGHBOR, W_CLUSTER = 0.5, 0.35, 0.15
TOP_N = 10
EDITION_PREFIX = 12


def scale(x):
    x = np.asarray(x, dtype=np.float64)
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


class ClusterRecommender:
    def __init__(self, feats_df, clusters, neighbors, names):
        self.feats_df = feats_df
        vecs = feats_df.to_numpy(dtype=np.float32)
        self.unit = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        self.row = {int(a): i for i, a in enumerate(feats_df.index)}
        clusters = clusters[clusters["appid"].isin(self.row)]
        self.cluster_of = dict(zip(clusters["appid"].astype(int), clusters["cluster_id"].astype(int)))
        self.members = {c: g["appid"].astype(int).to_numpy() for c, g in clusters.groupby("cluster_id")}
        self.centroid = {}
        for c, apps in self.members.items():
            v = self.unit[[self.row[a] for a in apps]].mean(axis=0)
            self.centroid[c] = v / (np.linalg.norm(v) + 1e-9)
        self.neighbors = {}
        for g, sub in neighbors.groupby("game_appid"):
            self.neighbors[int(g)] = dict(zip(sub["neighbor_appid"].astype(int), sub["similarity"].astype(float)))
        self.names = names

    def profile(self, hist):
        rows = [self.row[a] for a in hist["appid"]]
        w = hist["interest"].to_numpy(dtype=np.float32)
        p = (self.unit[rows] * w[:, None]).sum(axis=0)
        return p / (np.linalg.norm(p) + 1e-9)

    def top_clusters(self, hist, prof):
        weight = {}
        for a, w in zip(hist["appid"], hist["interest"]):
            c = self.cluster_of.get(int(a))
            if c is not None:
                weight[c] = weight.get(c, 0.0) + float(w)
        ranked = sorted(weight, key=lambda c: -weight[c])
        for c in sorted(self.centroid, key=lambda c: -float(self.centroid[c] @ prof)):
            if len(ranked) >= len(PER_CLUSTER):
                break
            if c not in ranked:
                ranked.append(c)
        total = sum(weight.values()) or 1.0
        return ranked[:len(PER_CLUSTER)], {c: weight.get(c, 0.0) / total for c in ranked}

    def candidates(self, hist, blocked):
        prof = self.profile(hist)
        chosen, share = self.top_clusters(hist, prof)
        out, taken = [], set()
        carry = 0
        for c, quota in zip(chosen, PER_CLUSTER):
            apps = [a for a in self.members[c] if a not in blocked and a not in taken]
            apps.sort(key=lambda a: -float(self.unit[self.row[a]] @ prof))
            want = quota + carry
            pick = apps[:want]
            carry = want - len(pick)
            out += pick
            taken.update(pick)
        return np.array(out, dtype=np.int64), prof, share

    def score(self, hist, cands, prof, share):
        content = self.unit[[self.row[a] for a in cands]] @ prof
        neighbor = np.zeros(len(cands))
        idx = {a: i for i, a in enumerate(cands)}
        for g, w in zip(hist["appid"], hist["interest"]):
            for n, s in self.neighbors.get(int(g), {}).items():
                if n in idx:
                    neighbor[idx[n]] += float(w) * s
        cluster = np.array([share.get(self.cluster_of.get(int(a)), 0.0) for a in cands])
        return W_CONTENT * scale(content) + W_NEIGHBOR * scale(neighbor) + W_CLUSTER * scale(cluster)


def edition_blocklist(owned, names, all_appids):
    owned_keys = {norm_name(names.get(a, ""))[:EDITION_PREFIX] for a in owned}
    owned_keys.discard("")
    return set(owned) | {a for a in all_appids
                         if norm_name(names.get(a, ""))[:EDITION_PREFIX] in owned_keys}


def describe(appid, cat):
    row = cat.loc[appid]
    tags = [t for t in str(row.get("tags", "") or "").split(",")[:6] if t]
    return {"appid": int(appid), "name": row["name"],
            "genres": str(row.get("genres", "") or ""), "tags": ", ".join(tags)}


def main():
    key_in = json.loads((ROOT / "data" / "judge_key.json").read_text())
    raw = pd.read_csv(ROOT / "data" / "steam-200k.csv", header=None,
                      names=["user", "game", "behavior", "hours", "extra"])
    play = raw[(raw["behavior"] == "play") & (raw["hours"] > 0)]

    bundle = json.load(open(ROOT / "models" / "model_rank_v3_interact_bundle.json", encoding="utf-8"))
    games = pd.read_csv(ROOT / "new_package" / "03_features" / "game_features.csv")
    games["appid"] = games["appid"].astype("int64")
    feats_df = games.drop_duplicates("appid").set_index("appid")[bundle["game_feats"]]
    cat = pd.read_csv(ROOT / "new_package" / "01_catalog" / "games_for_features.csv",
                      usecols=["appid", "name", "genres", "tags"])
    cat["appid"] = cat["appid"].astype("int64")
    cat = cat.drop_duplicates("appid").set_index("appid")
    names = {int(a): n for a, n in cat["name"].items()}
    name_to_app = {}
    for a, n in names.items():
        name_to_app.setdefault(norm_name(n), a)
    name_to_app[norm_name("Counter-Strike:Global Offensive")] = 730

    clusters = pd.read_csv(ROOT / "new_package" / "04_clusters" / "game_clusters.csv")
    clusters = clusters[clusters["cluster_id"].notna() & (clusters["cluster_id"] >= 0)]
    neighbors = pd.read_csv(ROOT / "new_package" / "08_item_neighbors" / "game_top10_neighbors.csv")
    rec = ClusterRecommender(feats_df, clusters, neighbors, names)
    labels = clusters.drop_duplicates("cluster_id").set_index("cluster_id")["cluster_label"].to_dict()

    ctx, _ = load_context(feats_df)
    model = xgb.Booster()
    model.load_model(str(ROOT / "models" / "model_rank_v3_interact.json"))
    feat_ids = [int(a) for a in feats_df.index]

    rng = np.random.default_rng(11)
    judge, key, debug = [], {}, {}
    for label, info in key_in.items():
        uid32 = info["steamid"] - STEAM64
        lib = []
        for _, row in play[play["user"] == uid32].iterrows():
            a = name_to_app.get(norm_name(row["game"]))
            if a is not None and a in feats_df.index:
                lib.append({"appid": a, "name": names.get(a, row["game"]),
                            "hours": float(row["hours"]), "hours_2weeks": 0.0})
        hist = interest_rows(lib).drop_duplicates("appid")
        blocked = edition_blocklist(set(hist["appid"].astype(int)), names, feat_ids)

        full = recommend(hist, model, bundle, feats_df, ctx, names, n=200)
        v3 = [int(a) for a in full["appid"] if int(a) not in blocked][:TOP_N]

        cands, prof, share = rec.candidates(hist, blocked)
        formula = rec.score(hist, cands, prof, share)
        cluster = [int(cands[i]) for i in np.argsort(-formula)[:TOP_N]]

        sub = cands[np.isin(cands, list(ctx["appid_to_idx"]))]
        ranked = recommend_subset(hist, sub, model, bundle, feats_df, ctx)
        cluster_v3 = ranked[:TOP_N]

        systems = {"v3": v3, "cluster": cluster, "cluster_v3": cluster_v3}
        order = list(systems)
        rng.shuffle(order)
        letters = dict(zip("ABC", order))
        judge.append({
            "user": label,
            "library": [{**describe(int(r.appid), cat), "hours": r.hours}
                        for r in hist.sort_values("hours", ascending=False).head(12).itertuples()],
            **{f"list_{L}": [describe(a, cat) for a in systems[s]] for L, s in letters.items()},
        })
        key[label] = letters
        chosen, _ = rec.top_clusters(hist, prof)
        debug[label] = {"clusters": [labels.get(c) for c in chosen], "n_candidates": int(len(cands)),
                        **{s: [names.get(a) for a in v] for s, v in systems.items()}}
        print(f"{label}: clusters {debug[label]['clusters']}")

    out = ROOT / "data"
    (out / "judge3_set.json").write_text(json.dumps(judge, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "judge3_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    (out / "judge3_debug.json").write_text(json.dumps(debug, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(judge)} users")


def recommend_subset(hist, cands, model, bundle, feats_df, ctx):
    from rank_pipeline_v3_interact import N_SLOTS, _build_slot_cand
    from heuristic_features_v2 import add_heuristic_features
    h = hist.sort_values("interest", ascending=False).head(N_SLOTS)
    rows = _build_slot_cand(h, cands, feats_df, bundle["game_feats"])
    add_heuristic_features(rows, None, cands, h, ctx, bundle["game_feats"], N_SLOTS)
    X = pd.DataFrame(rows)[bundle["feature_cols"]]
    s = model.predict(xgb.DMatrix(X), iteration_range=(0, bundle["best_iteration"] + 1))
    return [int(cands[i]) for i in np.argsort(-s)]


if __name__ == "__main__":
    main()
