from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.utils.eval import hit_rate_at_k, ndcg_at_k_single, mrr_at_k_single

DATASET = "small"
BASE = Path(f"data/processed/{DATASET}")

TRAIN_FILE = BASE / "ratings_train.csv"
VAL_FILE = BASE / "ratings_val.csv"
TEST_FILE = BASE / "ratings_test.csv"

INDEX_FILE = BASE / "movie_embeddings_index_v2.csv"
SIM_FILE = BASE / "content_similarity_matrix_v2.npy"

OUTPUT_DIR = BASE / "baseline_content_based_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K_LIST = [5, 10, 20]
MIN_SCORE_THRESHOLD = -10.0


def load_data():
    train = pd.read_csv(TRAIN_FILE)
    val = pd.read_csv(VAL_FILE)
    test = pd.read_csv(TEST_FILE)

    index_df = pd.read_csv(INDEX_FILE)
    sim_matrix = np.load(SIM_FILE)

    return train, val, test, index_df, sim_matrix


def build_movieid_to_index(index_df: pd.DataFrame) -> dict[int, int]:
    return {int(mid): int(i) for i, mid in enumerate(index_df["movieId"].tolist())}


def get_user_seen_ratings(train: pd.DataFrame) -> dict[int, dict[int, float]]:
    mapping = {}
    for user_id, g in train.groupby("userId"):
        mapping[int(user_id)] = {
            int(mid): float(r)
            for mid, r in zip(g["movieId"], g["rating"])
        }
    return mapping


def get_user_mean_ratings(train: pd.DataFrame) -> dict[int, float]:
    return {
        int(user_id): float(g["rating"].mean())
        for user_id, g in train.groupby("userId")
    }


def score_candidate(
    candidate_movie_id: int,
    seen_ratings: dict[int, float],
    user_mean_rating: float,
    movieid_to_index: dict[int, int],
    sim_matrix: np.ndarray
) -> float:
    if candidate_movie_id not in movieid_to_index:
        return 0.0

    cand_idx = movieid_to_index[candidate_movie_id]

    num = 0.0
    den = 0.0

    for seen_movie_id, rating in seen_ratings.items():
        if seen_movie_id not in movieid_to_index:
            continue

        seen_idx = movieid_to_index[seen_movie_id]
        sim = float(sim_matrix[cand_idx, seen_idx])

        if sim <= 0:
            continue

        adjusted_rating = rating - user_mean_rating

        num += sim * adjusted_rating
        den += abs(sim)

    if den == 0.0:
        return 0.0

    return num / den


def recommend_top_k_for_user(
    user_id: int,
    candidate_items: list[int],
    user_seen_map: dict[int, dict[int, float]],
    user_mean_map: dict[int, float],
    movieid_to_index: dict[int, int],
    sim_matrix: np.ndarray,
    k: int
) -> list[int]:
    seen_ratings = user_seen_map.get(user_id, {})
    seen_items = set(seen_ratings.keys())
    user_mean_rating = user_mean_map.get(user_id, 0.0)

    scored = []
    for movie_id in candidate_items:
        if movie_id in seen_items:
            continue

        score = score_candidate(
            candidate_movie_id=movie_id,
            seen_ratings=seen_ratings,
            user_mean_rating=user_mean_rating,
            movieid_to_index=movieid_to_index,
            sim_matrix=sim_matrix
        )

        if score > MIN_SCORE_THRESHOLD:
            scored.append((movie_id, score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return [movie_id for movie_id, _ in scored[:k]]


def evaluate_split(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    candidate_items: list[int],
    movieid_to_index: dict[int, int],
    sim_matrix: np.ndarray,
    top_k_list: list[int]
):
    user_seen_map = get_user_seen_ratings(train)
    user_mean_map = get_user_mean_ratings(train)
    max_k = max(top_k_list)

    metrics = {f"HR@{k}": [] for k in top_k_list}
    metrics.update({f"NDCG@{k}": [] for k in top_k_list})
    metrics.update({f"MRR@{k}": [] for k in top_k_list})

    rows = []

    for _, row in eval_df.iterrows():
        user_id = int(row["userId"])
        ground_truth = int(row["movieId"])

        recs = recommend_top_k_for_user(
            user_id=user_id,
            candidate_items=candidate_items,
            user_seen_map=user_seen_map,
            user_mean_map=user_mean_map,
            movieid_to_index=movieid_to_index,
            sim_matrix=sim_matrix,
            k=max_k
        )

        result_row = {
            "userId": user_id,
            "ground_truth_movieId": ground_truth,
            "num_recommendations": len(recs)
        }

        for k in top_k_list:
            hr = hit_rate_at_k(recs, ground_truth, k)
            ndcg = ndcg_at_k_single(recs, ground_truth, k)
            mrr = mrr_at_k_single(recs, ground_truth, k)

            metrics[f"HR@{k}"].append(hr)
            metrics[f"NDCG@{k}"].append(ndcg)
            metrics[f"MRR@{k}"].append(mrr)

            result_row[f"HR@{k}"] = hr
            result_row[f"NDCG@{k}"] = ndcg
            result_row[f"MRR@{k}"] = mrr

        rows.append(result_row)

    summary = {
        metric: float(np.mean(values)) if values else 0.0
        for metric, values in metrics.items()
    }
    summary["n_users_evaluated"] = int(len(eval_df))

    per_user_df = pd.DataFrame(rows)
    return summary, per_user_df


def main():
    train, val, test, index_df, sim_matrix = load_data()

    print("train:", train.shape)
    print("val:", val.shape)
    print("test:", test.shape)
    print("index:", index_df.shape)
    print("sim_matrix:", sim_matrix.shape)

    movieid_to_index = build_movieid_to_index(index_df)

    candidate_items = sorted(
        set(train["movieId"].unique()).intersection(set(index_df["movieId"].unique()))
    )

    print("candidate items:", len(candidate_items))

    val_summary, val_per_user = evaluate_split(
        train=train,
        eval_df=val,
        candidate_items=candidate_items,
        movieid_to_index=movieid_to_index,
        sim_matrix=sim_matrix,
        top_k_list=TOP_K_LIST
    )

    test_summary, test_per_user = evaluate_split(
        train=train,
        eval_df=test,
        candidate_items=candidate_items,
        movieid_to_index=movieid_to_index,
        sim_matrix=sim_matrix,
        top_k_list=TOP_K_LIST
    )

    val_per_user_path = OUTPUT_DIR / "val_per_user_metrics.csv"
    test_per_user_path = OUTPUT_DIR / "test_per_user_metrics.csv"
    summary_path = OUTPUT_DIR / "summary_metrics.json"

    val_per_user.to_csv(val_per_user_path, index=False)
    test_per_user.to_csv(test_per_user_path, index=False)

    summary = {
        "config": {
            "dataset": DATASET,
            "top_k_list": TOP_K_LIST,
            "model": "content_based_similarity_v2",
            "features": ["genres", "director", "actors_top5", "overview_en"],
            "title_used_in_similarity": False,
            "user_rating_centering": True
        },
        "validation": val_summary,
        "test": test_summary
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== VALIDATION METRICS ===")
    for k in TOP_K_LIST:
        print(
            f"HR@{k}: {val_summary[f'HR@{k}']:.4f} | "
            f"NDCG@{k}: {val_summary[f'NDCG@{k}']:.4f} | "
            f"MRR@{k}: {val_summary[f'MRR@{k}']:.4f}"
        )

    print("\n=== TEST METRICS ===")
    for k in TOP_K_LIST:
        print(
            f"HR@{k}: {test_summary[f'HR@{k}']:.4f} | "
            f"NDCG@{k}: {test_summary[f'NDCG@{k}']:.4f} | "
            f"MRR@{k}: {test_summary[f'MRR@{k}']:.4f}"
        )

    print(f"\nSaved val per-user metrics -> {val_per_user_path}")
    print(f"Saved test per-user metrics -> {test_per_user_path}")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()