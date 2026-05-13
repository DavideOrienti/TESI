from __future__ import annotations
import json
import numpy as np
import pandas as pd

from src.utils.io import load_settings
from src.utils.eval import hit_rate_at_k, ndcg_at_k_single, mrr_at_k_single, precision_at_k, recall_at_k
from src.recommenders.collaborative import build_cf_neighbors_dict, score_cf_candidates
from src.recommenders.content_based import (
    build_content_neighbors_dict,
    build_index_to_movieid,
    score_content_candidates,
)
from src.recommenders.hybrid import rank_hybrid_scores
from src.recommenders.scoring import (
    build_user_seen_ratings,
    build_user_mean_ratings,
)


TOP_K_LIST = [5, 10, 20]
ALPHAS = [0.2, 0.4, 0.5, 0.6, 0.8]

TRAIN_NAME = "ratings_train.csv"
VAL_NAME = "ratings_val.csv"
TEST_NAME = "ratings_test.csv"

# CF improved artifacts
CF_NEIGHBORS_NAME = "baseline_itemknn_cf_improved/item_similarity_top_neighbors.csv"

# Content artifacts
CONTENT_INDEX_NAME = "movie_embeddings_index_v2.csv"
CONTENT_NEIGHBORS_NAME = "content_top_neighbors_v2.csv"

OUTPUT_DIR_NAME = "hybrid_weighted_v1"


def load_data(base):
    train = pd.read_csv(base / TRAIN_NAME)
    val = pd.read_csv(base / VAL_NAME)
    test = pd.read_csv(base / TEST_NAME)

    cf_neighbors_df = pd.read_csv(base / CF_NEIGHBORS_NAME)
    content_index_df = pd.read_csv(base / CONTENT_INDEX_NAME)
    content_neighbors_df = pd.read_csv(base / CONTENT_NEIGHBORS_NAME)

    return train, val, test, cf_neighbors_df, content_index_df, content_neighbors_df


def build_user_model_scores(
    user_id: int,
    candidate_items: list[int],
    user_seen_map: dict[int, dict[int, float]],
    user_mean_map: dict[int, float],
    cf_neighbors_dict: dict[int, list[tuple[int, float]]],
    content_neighbors_dict: dict[int, list[tuple[int, float]]]
) -> tuple[dict[int, float], dict[int, float]]:
    seen_ratings = user_seen_map.get(user_id, {})
    user_mean_rating = user_mean_map.get(user_id, 0.0)

    cf_scores = score_cf_candidates(
        candidate_items=candidate_items,
        seen_ratings=seen_ratings,
        user_mean_rating=user_mean_rating,
        cf_neighbors_dict=cf_neighbors_dict,
    )
    content_scores = score_content_candidates(
        candidate_items=candidate_items,
        seen_ratings=seen_ratings,
        user_mean_rating=user_mean_rating,
        content_neighbors_dict=content_neighbors_dict,
    )

    return cf_scores, content_scores


def recommend_top_k_hybrid(
    cf_scores: dict[int, float],
    content_scores: dict[int, float],
    alpha: float,
    k: int
) -> list[int]:
    return [
        item.movie_id
        for item in rank_hybrid_scores(
            collaborative_scores=cf_scores,
            content_scores=content_scores,
            collaborative_weight=alpha,
            top_k=k,
        )
    ]


def evaluate_split_for_alpha(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    candidate_items: list[int],
    cf_neighbors_dict: dict[int, list[tuple[int, float]]],
    content_neighbors_dict: dict[int, list[tuple[int, float]]],
    alpha: float,
    top_k_list: list[int]
):
    user_seen_map = build_user_seen_ratings(train)
    user_mean_map = build_user_mean_ratings(train)
    max_k = max(top_k_list)

    metrics = {f"HR@{k}": [] for k in top_k_list}
    metrics.update({f"NDCG@{k}": [] for k in top_k_list})
    metrics.update({f"MRR@{k}": [] for k in top_k_list})
    metrics.update({f"P@{k}": [] for k in top_k_list})
    metrics.update({f"R@{k}": [] for k in top_k_list})

    rows = []

    for _, row in eval_df.iterrows():
        user_id = int(row["userId"])
        ground_truth = int(row["movieId"])

        cf_scores, content_scores = build_user_model_scores(
            user_id=user_id,
            candidate_items=candidate_items,
            user_seen_map=user_seen_map,
            user_mean_map=user_mean_map,
            cf_neighbors_dict=cf_neighbors_dict,
            content_neighbors_dict=content_neighbors_dict
        )

        recs = recommend_top_k_hybrid(
            cf_scores=cf_scores,
            content_scores=content_scores,
            alpha=alpha,
            k=max_k
        )

        result_row = {
            "userId": user_id,
            "ground_truth_movieId": ground_truth,
            "alpha": alpha,
            "num_recommendations": len(recs),
        }

        for k in top_k_list:
            hr   = hit_rate_at_k(recs, ground_truth, k)
            ndcg = ndcg_at_k_single(recs, ground_truth, k)
            mrr  = mrr_at_k_single(recs, ground_truth, k)
            p    = precision_at_k(recs, ground_truth, k)
            r    = recall_at_k(recs, ground_truth, k)

            metrics[f"HR@{k}"].append(hr)
            metrics[f"NDCG@{k}"].append(ndcg)
            metrics[f"MRR@{k}"].append(mrr)
            metrics[f"P@{k}"].append(p)
            metrics[f"R@{k}"].append(r)

            result_row[f"HR@{k}"]   = hr
            result_row[f"NDCG@{k}"] = ndcg
            result_row[f"MRR@{k}"]  = mrr
            result_row[f"P@{k}"]    = p
            result_row[f"R@{k}"]    = r

        rows.append(result_row)

    summary = {
        metric: float(np.mean(values)) if values else 0.0
        for metric, values in metrics.items()
    }
    summary["n_users_evaluated"] = int(len(eval_df))
    summary["alpha"] = float(alpha)

    per_user_df = pd.DataFrame(rows)
    return summary, per_user_df


def main():
    s = load_settings()
    base = s.paths.processed
    output_dir = base / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    train, val, test, cf_neighbors_df, content_index_df, content_neighbors_df = load_data(base)

    print(f"[15] dataset={s.dataset}")
    print(f"train: {train.shape}")
    print(f"val: {val.shape}")
    print(f"test: {test.shape}")
    print(f"cf_neighbors: {cf_neighbors_df.shape}")
    print(f"content_index: {content_index_df.shape}")
    print(f"content_neighbors: {content_neighbors_df.shape}")

    cf_neighbors_dict = build_cf_neighbors_dict(cf_neighbors_df)

    index_to_movieid = build_index_to_movieid(content_index_df)
    content_neighbors_dict = build_content_neighbors_dict(content_neighbors_df, index_to_movieid)

    candidate_items = sorted(
        set(train["movieId"].unique()).intersection(set(content_index_df["movieId"].unique()))
    )
    print(f"candidate items: {len(candidate_items)}")

    all_results = []
    best_alpha = None
    best_val_ndcg10 = -1.0

    for alpha in ALPHAS:
        print(f"\n=== ALPHA = {alpha:.2f} ===")

        val_summary, val_per_user = evaluate_split_for_alpha(
            train=train,
            eval_df=val,
            candidate_items=candidate_items,
            cf_neighbors_dict=cf_neighbors_dict,
            content_neighbors_dict=content_neighbors_dict,
            alpha=alpha,
            top_k_list=TOP_K_LIST
        )

        test_summary, test_per_user = evaluate_split_for_alpha(
            train=train,
            eval_df=test,
            candidate_items=candidate_items,
            cf_neighbors_dict=cf_neighbors_dict,
            content_neighbors_dict=content_neighbors_dict,
            alpha=alpha,
            top_k_list=TOP_K_LIST
        )

        print(
            f"VAL  HR@10={val_summary['HR@10']:.4f} | "
            f"NDCG@10={val_summary['NDCG@10']:.4f} | "
            f"MRR@10={val_summary['MRR@10']:.4f}"
        )
        print(
            f"TEST HR@10={test_summary['HR@10']:.4f} | "
            f"NDCG@10={test_summary['NDCG@10']:.4f} | "
            f"MRR@10={test_summary['MRR@10']:.4f}"
        )

        val_per_user.to_csv(output_dir / f"val_per_user_alpha_{alpha:.2f}.csv", index=False)
        test_per_user.to_csv(output_dir / f"test_per_user_alpha_{alpha:.2f}.csv", index=False)

        all_results.append({
            "alpha": alpha,
            **{f"val_{k}": v for k, v in val_summary.items()},
            **{f"test_{k}": v for k, v in test_summary.items()},
        })

        if val_summary["NDCG@10"] > best_val_ndcg10:
            best_val_ndcg10 = val_summary["NDCG@10"]
            best_alpha = alpha

    results_df = pd.DataFrame(all_results)
    results_csv = output_dir / "alpha_search_results.csv"
    results_df.to_csv(results_csv, index=False)

    summary = {
        "config": {
            "dataset": s.dataset,
            "alphas": ALPHAS,
            "top_k_list": TOP_K_LIST,
            "model": "hybrid_weighted_v1",
            "cf_source": CF_NEIGHBORS_NAME,
            "content_source": CONTENT_NEIGHBORS_NAME,
            "normalization": "per-user minmax"
        },
        "best_alpha_by_val_ndcg10": best_alpha,
        "results_table_file": str(results_csv)
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== BEST CONFIG ===")
    print(f"best alpha (by VAL NDCG@10): {best_alpha}")
    print(f"saved table -> {results_csv}")
    print(f"saved summary -> {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
