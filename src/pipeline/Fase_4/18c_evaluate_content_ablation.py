from __future__ import annotations
import json
import numpy as np
import pandas as pd

import argparse

from src.utils.io import load_settings
from src.utils.eval import hit_rate_at_k, ndcg_at_k_single, mrr_at_k_single, precision_at_k, recall_at_k
from src.recommenders.content_based import (
    build_content_neighbors_dict,
    build_index_to_movieid,
    score_content_candidates,
)
from src.recommenders.scoring import (
    build_user_seen_ratings,
    build_user_mean_ratings,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--variant",
    type=str,
    choices=["full", "no_tags", "no_overview", "genres_only"],
    required=True,
    help="Variante della text representation per l'ablation study"
)
args = parser.parse_args()
VARIANT = args.variant
TOP_K_LIST = [5, 10, 20]
OUTPUT_DIR_NAME = "content_ablation_results"


def load_data(base):
    train = pd.read_csv(base / "ratings_train.csv")
    val = pd.read_csv(base / "ratings_val.csv")
    test = pd.read_csv(base / "ratings_test.csv")
    index_df = pd.read_csv(base / f"movie_embeddings_index_{VARIANT}.csv")
    neighbors_df = pd.read_csv(base / f"content_top_neighbors_{VARIANT}.csv")
    return train, val, test, index_df, neighbors_df


def recommend_top_k(user_id, candidate_items, user_seen_map, user_mean_map, neighbors_dict, k):
    seen_ratings = user_seen_map.get(user_id, {})
    user_mean = user_mean_map.get(user_id, 0.0)

    scores = score_content_candidates(
        candidate_items=candidate_items,
        seen_ratings=seen_ratings,
        user_mean_rating=user_mean,
        content_neighbors_dict=neighbors_dict,
    )

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [movie_id for movie_id, _ in ranked[:k]]


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
    neighbors_dict = build_content_neighbors_dict(neighbors_df, index_to_movieid)

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
