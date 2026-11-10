"""Cluster the polished Gemini feature sheet.

Uses status=ok rows only. Scores are L2-normalized, compressed with PCA,
then MiniBatchKMeans. k is chosen by silhouette.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "03_features" / "game_features.csv"
OUT = ROOT / "04_clusters"
META_COLS = {"appid", "name", "inclusion", "status", "note", "n_labels"}
K_MIN, K_MAX, K_STEP = 12, 28, 2
PCA_VARIANCE = 0.90
RANDOM_STATE = 42


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    feature_names = [c for c in df.columns if c not in META_COLS]
    ok = df["status"].eq("ok").to_numpy()
    X = df.loc[ok, feature_names].fillna(0).to_numpy(dtype=np.float32)
    names = df.loc[ok, ["appid", "name", "inclusion"]].reset_index(drop=True)
    print(f"ok games: {len(names)} / {len(df)}", flush=True)

    X_l2 = normalize(X, norm="l2")
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=RANDOM_STATE)
    Z = pca.fit_transform(X_l2)
    print(f"PCA {X.shape[1]} -> {Z.shape[1]} ({pca.explained_variance_ratio_.sum():.3f})", flush=True)

    best_k, best_score = None, -1.0
    search = []
    for k in range(K_MIN, K_MAX + 1, K_STEP):
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, batch_size=2048, n_init=10, max_iter=200)
        labels = km.fit_predict(Z)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(Z, labels, metric="euclidean", sample_size=min(4000, len(Z)), random_state=RANDOM_STATE))
        search.append({"k": k, "silhouette": round(score, 4)})
        print(f"k={k:2d} silhouette={score:.4f}", flush=True)
        if score > best_score:
            best_k, best_score = k, score
    print(f"selected k={best_k} silhouette={best_score:.4f}", flush=True)

    model = MiniBatchKMeans(n_clusters=best_k, random_state=RANDOM_STATE, batch_size=2048, n_init=20, max_iter=300)
    labels = model.fit_predict(Z)

    global_mean = X.mean(axis=0)
    summaries = []
    for c in range(best_k):
        members = X[labels == c]
        mean = members.mean(axis=0)
        lift = mean - global_mean
        top = [i for i in np.argsort(-lift) if lift[i] > 0.05][:6]
        bits = [feature_names[i] for i in top[:3]]
        examples = names.loc[labels == c, "name"].head(6).tolist()
        summaries.append(
            {
                "cluster_id": int(c),
                "n_games": int(len(members)),
                "label": " / ".join(bits) if bits else f"cluster_{c}",
                "top_features": ", ".join(f"{feature_names[i]} {mean[i]:.2f}" for i in top),
                "examples": "; ".join(examples),
            }
        )
    summaries = sorted(summaries, key=lambda s: -s["n_games"])
    label_map = {s["cluster_id"]: s["label"] for s in summaries}

    assign = df[["appid", "name", "inclusion", "status"]].copy()
    assign["cluster_id"] = -1
    assign.loc[ok, "cluster_id"] = labels
    assign["cluster_label"] = assign["cluster_id"].map(label_map).fillna("unclustered")

    assign.to_csv(OUT / "game_clusters.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "cluster_summary.csv", index=False)
    pd.DataFrame(search).to_csv(OUT / "cluster_k_selection.csv", index=False)
    meta = {
        "source": "03_features/game_features.csv",
        "method": "l2 + PCA + MiniBatchKMeans",
        "k": best_k,
        "silhouette": round(best_score, 4),
        "n_clustered": int(ok.sum()),
        "n_unclustered": int((~ok).sum()),
        "pca_components": int(Z.shape[1]),
    }
    (OUT / "cluster_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    for s in summaries:
        print(f"c{s['cluster_id']:02d} n={s['n_games']:5d}  {s['label']}")
        print(f"    {s['examples']}")


if __name__ == "__main__":
    main()
