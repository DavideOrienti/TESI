from __future__ import annotations
import json
import numpy as np
import pandas as pd

from src.utils.io import load_settings
from src.utils.eval import hit_rate_at_k, ndcg_at_k_single, mrr_at_k_single, precision_at_k, recall_at_k
from src.recommenders.collaborative import get_svd_scores_for_user
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


# Selezione GAMMA: grid search su validation set
# Il test set NON è stato usato per la selezione
# per evitare data leakage nella valutazione finale.
# Il GAMMA ottimale viene poi applicato fisso in produzione (recommender_service.py).
TOP_K_LIST = [5, 10, 20]
GAMMAS = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8]

TRAIN_NAME = "ratings_train.csv"
VAL_NAME = "ratings_val.csv"
TEST_NAME = "ratings_test.csv"
SVD_MATRIX_NAME = "baseline_pure_svd/predicted_scores_matrix.csv"
CONTENT_INDEX_NAME = "movie_embeddings_index_v2.csv"
CONTENT_NEIGHBORS_NAME = "content_top_neighbors_v2.csv"

OUTPUT_DIR_NAME = "hybrid_svd_content_v1"


def load_data(base):
    train = pd.read_csv(base / TRAIN_NAME)
    val = pd.read_csv(base / VAL_NAME)
    test = pd.read_csv(base / TEST_NAME)

    svd_matrix = pd.read_csv(base / SVD_MATRIX_NAME, index_col=0)
    svd_matrix.columns = [int(c) for c in svd_matrix.columns]
    svd_matrix.index = [int(i) for i in svd_matrix.index]

    content_index_df = pd.read_csv(base / CONTENT_INDEX_NAME)
    content_neighbors_df = pd.read_csv(base / CONTENT_NEIGHBORS_NAME)

    return train, val, test, svd_matrix, content_index_df, content_neighbors_df


def evaluate_split_for_gamma(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    candidate_items: list[int],
    svd_matrix: pd.DataFrame,
    content_neighbors_dict: dict[int, list[tuple[int, float]]],
    gamma: float,
    top_k_list: list[int],
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

        seen_ratings = user_seen_map.get(user_id, {})
        seen_items = set(seen_ratings.keys())
        user_mean_rating = user_mean_map.get(user_id, 0.0)

        unseen_candidates = [m for m in candidate_items if m not in seen_items]

        svd_scores = get_svd_scores_for_user(
            user_id=user_id,
            candidate_items=unseen_candidates,
            svd_matrix=svd_matrix,
            fallback_score=user_mean_rating,
        )

        content_scores = score_content_candidates(
            candidate_items=unseen_candidates,
            seen_ratings=seen_ratings,
            user_mean_rating=user_mean_rating,
            content_neighbors_dict=content_neighbors_dict,
            exclude_seen=False,
        )

        recs = [
            item.movie_id
            for item in rank_hybrid_scores(
                collaborative_scores=svd_scores,
                content_scores=content_scores,
                collaborative_weight=gamma,
                top_k=max_k,
            )
        ]

        result_row = {
            "userId": user_id,
            "ground_truth_movieId": ground_truth,
            "gamma": gamma,
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
    summary["gamma"] = float(gamma)

    per_user_df = pd.DataFrame(rows)
    return summary, per_user_df


def main():
    s = load_settings()
    base = s.paths.processed
    output_dir = base / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    train, val, test, svd_matrix, content_index_df, content_neighbors_df = load_data(base)

    print(f"[17] dataset={s.dataset}")
    print(f"train: {train.shape}")
    print(f"val: {val.shape}")
    print(f"test: {test.shape}")
    print(f"svd_matrix: {svd_matrix.shape}")
    print(f"content_index: {content_index_df.shape}")
    print(f"content_neighbors: {content_neighbors_df.shape}")

    index_to_movieid = build_index_to_movieid(content_index_df)
    content_neighbors_dict = build_content_neighbors_dict(content_neighbors_df, index_to_movieid)

    candidate_items = sorted(
        set(train["movieId"].unique()).intersection(set(content_index_df["movieId"].unique()))
    )
    print(f"candidate items: {len(candidate_items)}")

    all_results = []
    best_gamma = None
    best_val_ndcg10 = -1.0
    best_val_summary = None
    best_val_per_user = None
    best_test_summary = None
    best_test_per_user = None

    for gamma in GAMMAS:
        print(f"\n=== GAMMA = {gamma:.2f} ===")

        val_summary, val_per_user = evaluate_split_for_gamma(
            train=train,
            eval_df=val,
            candidate_items=candidate_items,
            svd_matrix=svd_matrix,
            content_neighbors_dict=content_neighbors_dict,
            gamma=gamma,
            top_k_list=TOP_K_LIST,
        )

        test_summary, test_per_user = evaluate_split_for_gamma(
            train=train,
            eval_df=test,
            candidate_items=candidate_items,
            svd_matrix=svd_matrix,
            content_neighbors_dict=content_neighbors_dict,
            gamma=gamma,
            top_k_list=TOP_K_LIST,
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

        all_results.append({
            "gamma": gamma,
            **{f"val_{k}": v for k, v in val_summary.items()},
            **{f"test_{k}": v for k, v in test_summary.items()},
        })

        if val_summary["NDCG@10"] > best_val_ndcg10:
            best_val_ndcg10 = val_summary["NDCG@10"]
            best_gamma = gamma
            best_val_summary = val_summary
            best_val_per_user = val_per_user
            best_test_summary = test_summary
            best_test_per_user = test_per_user

    results_df = pd.DataFrame(all_results)
    results_csv = output_dir / "gamma_search_results.csv"
    results_df.to_csv(results_csv, index=False)

    best_val_per_user.to_csv(output_dir / "val_per_user_metrics.csv", index=False)
    best_test_per_user.to_csv(output_dir / "test_per_user_metrics.csv", index=False)

    summary_json = {
        "config": {
            "dataset": s.dataset,
            "gammas_searched": GAMMAS,
            "top_k_list": TOP_K_LIST,
            "model": "hybrid_pure_svd_content",
            "svd_source": SVD_MATRIX_NAME,
            "content_source": CONTENT_NEIGHBORS_NAME,
            "normalization": "per-user minmax",
            "selection_criterion": "NDCG@10 validation",
        },
        "best_gamma": best_gamma,
        "validation": best_val_summary,
        "test": best_test_summary,
    }

    summary_path = output_dir / "summary_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(f"\n=== BEST GAMMA (by VAL NDCG@10): {best_gamma} ===")

    print("\n=== VALIDATION (best gamma) ===")
    for k in TOP_K_LIST:
        print(
            f"HR@{k}: {best_val_summary[f'HR@{k}']:.4f} | "
            f"NDCG@{k}: {best_val_summary[f'NDCG@{k}']:.4f} | "
            f"MRR@{k}: {best_val_summary[f'MRR@{k}']:.4f}"
        )

    print("\n=== TEST (best gamma) ===")
    for k in TOP_K_LIST:
        print(
            f"HR@{k}: {best_test_summary[f'HR@{k}']:.4f} | "
            f"NDCG@{k}: {best_test_summary[f'NDCG@{k}']:.4f} | "
            f"MRR@{k}: {best_test_summary[f'MRR@{k}']:.4f}"
        )

    print(f"\nsaved -> {results_csv}")
    print(f"saved -> {output_dir / 'val_per_user_metrics.csv'}")
    print(f"saved -> {output_dir / 'test_per_user_metrics.csv'}")
    print(f"saved -> {summary_path}")


if __name__ == "__main__":
    main()
