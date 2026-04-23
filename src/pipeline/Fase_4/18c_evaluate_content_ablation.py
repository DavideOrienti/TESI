from __future__ import annotations
import json
import numpy as np
import pandas as pd

from src.utils.io import load_settings
from src.utils.eval import hit_rate_at_k, ndcg_at_k_single, mrr_at_k_single, precision_at_k, recall_at_k
from src.recommenders.scoring import (
    build_user_seen_ratings,
    build_user_mean_ratings,
    score_candidate_content,
)


VARIANT = "full"
TOP_K_LIST = [5, 10, 20]
OUTPUT_DIR_NAME = "content_ablation_results"


def load_data(base):
    train = pd.read_csv(base / "ratings_train.csv")
    val = pd.read_csv(base / "ratings_val.csv")
    test = pd.read_csv(base / "ratings_test.csv")
    index_df = pd.read_csv(base / f"movie_embeddings_index_{VARIANT}.csv")
    neighbors_df = pd.read_csv(base / f"content_top_neighbors_{VARIANT}.csv")
    return train, val, test, index_df, neighbors_df


def build_index_to_movieid(index_df):
    return {int(i): int(mid) for i, mid in enumerate(index_df["movieId"].tolist())}


def build_neighbors_dict(neighbors_df, index_to_movieid):
    mapping = {}
    for movie_idx, g in neighbors_df.groupby("movie_idx"):
        cand = index_to_movieid.get(int(movie_idx))
        if cand is None:
            continue
        rows = []
        for _, row in g.iterrows():
            neigh_idx = int(row["neighbor_idx"])
            neigh_mid = index_to_movieid.get(neigh_idx)
            if neigh_mid is None:
                continue
            rows.append((neigh_mid, float(row["similarity"])))
        rows.sort(key=lambda x: (-x[1], x[0]))
        mapping[cand] = rows
    return mapping




def recommend_top_k(user_id, candidate_items, user_seen_map, user_mean_map, neighbors_dict, k):
    seen_ratings = user_seen_map.get(user_id, {})
    seen_items = set(seen_ratings.keys())
    user_mean = user_mean_map.get(user_id, 0.0)

    scored = []
    for movie_id in candidate_items:
        if movie_id in seen_items:
            continue
        score = score_candidate_content(movie_id, seen_ratings, user_mean, neighbors_dict)
        scored.append((movie_id, score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return [movie_id for movie_id, _ in scored[:k]]


def evaluate_split(train, eval_df, candidate_items, neighbors_dict):
    user_seen_map = build_user_seen_ratings(train)
    user_mean_map = build_user_mean_ratings(train)
    max_k = max(TOP_K_LIST)

    metrics = {f"HR@{k}": [] for k in TOP_K_LIST}
    metrics.update({f"NDCG@{k}": [] for k in TOP_K_LIST})
    metrics.update({f"MRR@{k}": [] for k in TOP_K_LIST})
    metrics.update({f"P@{k}": [] for k in TOP_K_LIST})
    metrics.update({f"R@{k}": [] for k in TOP_K_LIST})

    for _, row in eval_df.iterrows():
        user_id = int(row["userId"])
        gt = int(row["movieId"])

        recs = recommend_top_k(user_id, candidate_items, user_seen_map, user_mean_map, neighbors_dict, max_k)

        for k in TOP_K_LIST:
            metrics[f"HR@{k}"].append(hit_rate_at_k(recs, gt, k))
            metrics[f"NDCG@{k}"].append(ndcg_at_k_single(recs, gt, k))
            metrics[f"MRR@{k}"].append(mrr_at_k_single(recs, gt, k))
            metrics[f"P@{k}"].append(precision_at_k(recs, gt, k))
            metrics[f"R@{k}"].append(recall_at_k(recs, gt, k))

    return {m: float(np.mean(v)) for m, v in metrics.items()}


def main():
    s = load_settings()
    base = s.paths.processed
    output_dir = base / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    train, val, test, index_df, neighbors_df = load_data(base)
    index_to_movieid = build_index_to_movieid(index_df)
    neighbors_dict = build_neighbors_dict(neighbors_df, index_to_movieid)

    candidate_items = sorted(set(train["movieId"].unique()).intersection(set(index_df["movieId"].unique())))

    val_summary = evaluate_split(train, val, candidate_items, neighbors_dict)
    test_summary = evaluate_split(train, test, candidate_items, neighbors_dict)

    summary = {
        "variant": VARIANT,
        "validation": val_summary,
        "test": test_summary,
    }

    out_file = output_dir / f"summary_{VARIANT}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[18c] variant={VARIANT}")
    print(summary)


if __name__ == "__main__":
    main()