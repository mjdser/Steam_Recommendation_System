"""
rec_eval.py — evaluation metrics for the slot-based recommender.

Usage:
    from rec_eval import recall_at_k, ndcg_at_k, mrr, catalog_coverage, novelty, evaluate_model

    # per-user ranking metrics (need a candidate pool per user)
    results = evaluate_model(users, score_fn, df_games, train_df, K=10)

Or call the primitives directly on arrays you've already built.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- primitives

def recall_at_k(ranked_ids, relevant_ids, k):
    """Fraction of the user's relevant items found in the top-k ranked ids."""
    if len(relevant_ids) == 0:
        return np.nan
    topk = set(ranked_ids[:k])
    return len(topk & set(relevant_ids)) / len(relevant_ids)


def precision_at_k(ranked_ids, relevant_ids, k):
    """Fraction of the top-k that are relevant."""
    if k == 0:
        return np.nan
    topk = set(ranked_ids[:k])
    return len(topk & set(relevant_ids)) / k


def ndcg_at_k(ranked_ids, relevance_scores, k):
    """
    NDCG@k with graded relevance.
    ranked_ids:        ordered list of item ids (best first)
    relevance_scores:  dict {item_id: relevance} (e.g. interest)
    """
    gains = np.array([relevance_scores.get(i, 0.0) for i in ranked_ids[:k]], dtype=float)
    if gains.size == 0 or gains.sum() == 0:
        return np.nan
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = np.sum((2 ** gains - 1) * discounts)

    ideal = np.sort(np.array(list(relevance_scores.values()), dtype=float))[::-1][:k]
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal.size + 2))
    idcg = np.sum((2 ** ideal - 1) * ideal_discounts)
    return dcg / idcg if idcg > 0 else np.nan


def reciprocal_rank(ranked_ids, relevant_ids):
    """1 / rank of the first relevant item (0 if none)."""
    rel = set(relevant_ids)
    for i, item in enumerate(ranked_ids, start=1):
        if item in rel:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------- aggregate metrics

def catalog_coverage(all_ranked_ids, total_catalog_size):
    """
    Fraction of the catalog that appears in at least one user's top-k.
    all_ranked_ids: iterable of per-user ranked id lists (already truncated to k).
    """
    shown = set()
    for ids in all_ranked_ids:
        shown.update(ids)
    return len(shown) / total_catalog_size


def novelty(ranked_ids, item_popularity, k):
    """
    Mean self-information of recommended items: -log2(p(i)).
    item_popularity: dict {item_id: probability} — probability of being played.
    Higher = more novel / less popular.
    """
    topk = ranked_ids[:k]
    if len(topk) == 0:
        return np.nan
    vals = []
    for i in topk:
        p = item_popularity.get(i, None)
        if p is None or p <= 0:
            continue
        vals.append(-np.log2(p))
    return np.mean(vals) if vals else np.nan


# ---------------------------------------------------------------- driver

def evaluate_model(users, score_fn, df_games, relevant_by_user,
                   item_popularity=None, candidates_per_user=None,
                   K=(5, 10, 20)):
    """
    Evaluate a model across users.

    users:                iterable of user ids to evaluate
    score_fn:             callable(user_id, candidate_appids) -> np.array of scores
                          (higher = more relevant). This is where you plug your model.
    df_games:             full catalog (needs 'appid')
    relevant_by_user:     dict {user_id: {appid: interest}} — the ground-truth games
    item_popularity:      dict {appid: probability}, for novelty. Optional.
    candidates_per_user:  dict {user_id: [appids]} to score. If None, scores the whole catalog.
    K:                    tuple of cutoffs

    Returns a DataFrame with one row per K and the aggregate metrics.
    """
    all_appids = df_games['appid'].to_numpy()
    max_k = max(K)

    per_user = {k: {'recall': [], 'precision': [], 'ndcg': [], 'mrr': []} for k in K}
    all_topk_ids = {k: [] for k in K}
    novelty_vals = []

    for uid in users:
        rel = relevant_by_user.get(uid, {})
        if len(rel) == 0:
            continue

        cands = candidates_per_user[uid] if candidates_per_user else all_appids
        scores = score_fn(uid, cands)
        order = np.argsort(-np.asarray(scores))
        ranked = np.asarray(cands)[order]

        for k in K:
            per_user[k]['recall'].append(recall_at_k(ranked, rel.keys(), k))
            per_user[k]['precision'].append(precision_at_k(ranked, rel.keys(), k))
            per_user[k]['ndcg'].append(ndcg_at_k(ranked, rel, k))
            all_topk_ids[k].append(list(ranked[:k]))

        per_user[max_k]['mrr'].append(reciprocal_rank(ranked, rel.keys()))

        if item_popularity is not None:
            novelty_vals.append(novelty(ranked, item_popularity, max_k))

    rows = []
    for k in K:
        rows.append({
            'K': k,
            'recall':    np.nanmean(per_user[k]['recall']),
            'precision': np.nanmean(per_user[k]['precision']),
            'ndcg':      np.nanmean(per_user[k]['ndcg']),
            'mrr':       np.nanmean(per_user[max_k]['mrr']),   # MRR is K-independent
            'coverage':  catalog_coverage(all_topk_ids[k], len(df_games)),
            'novelty':   np.nanmean(novelty_vals) if novelty_vals else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- helpers

def build_popularity(train_df):
    """Probability that a game appears in a user's list, from training data."""
    counts = train_df['appid'].value_counts(normalize=True)
    return counts.to_dict()


def relevant_from_val(val_df):
    """{steamid: {appid: interest}} from a val/test frame."""
    out = {}
    for sid, g in val_df.groupby('steamid'):
        out[sid] = dict(zip(g['appid'], g['interest']))
    return out